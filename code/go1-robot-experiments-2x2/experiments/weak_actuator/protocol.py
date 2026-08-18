"""Task-side weak-actuator curriculum and training utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

from brax.training import types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import train as ppo
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

from go1_core import GO1_CONTRACT_V1
from go1_robot_experiments.constants import ENV_NAME
from go1_robot_experiments.util import json_safe


DEFAULT_MODEL = "multirank_hodge_lower"
DEFAULT_SEED = 0
DEFAULT_NUM_TIMESTEPS = 22_937_600
DEFAULT_NUM_EVALS = 2
DEFAULT_EVALUATION_GRID = "full"
ORIGINAL_FULL_NUM_TIMESTEPS = 400_000_000
ORIGINAL_FULL_NUM_EVALS = 19
ORIGINAL_FULL_ACTUAL_TIMESTEPS = 412_876_800
OFFICIAL_PPO_CONFIG_SHA256 = (
    "7df61b65d671c7f2df21ecf075143473a976a632b85c041503ad1aa52e15c407"
)

HEALTHY_PROBABILITY = 0.30
WEAK_PROBABILITY = 0.50
DEAD_PROBABILITY = 0.20
MIN_WEAK_STRENGTH = 0.05
MAX_WEAK_STRENGTH = 0.20
CATEGORY_HEALTHY = 0
CATEGORY_WEAK = 1
CATEGORY_DEAD = 2
CURRICULUM_SHA256 = (
    "d2e14a8b3eb31dc9d0728008c891ac71c69d46a739047c0c8087cb38af38898e"
)

CURRICULUM = {
    "assignment_scope": "fixed_vectorized_environment_member",
    "reset_resampling": False,
    "injuries_per_nonhealthy_environment": 1,
    "category_probabilities": {
        "healthy": HEALTHY_PROBABILITY,
        "weak": WEAK_PROBABILITY,
        "dead": DEAD_PROBABILITY,
    },
    "weak_strength_distribution": {
        "family": "log_uniform",
        "minimum": MIN_WEAK_STRENGTH,
        "maximum_exclusive": MAX_WEAK_STRENGTH,
    },
    "dead_strength": 0.0,
    "healthy_strength": 1.0,
    "actuator_sampling": "uniform_over_frozen_12_actuator_order",
    "actuator_names": list(GO1_CONTRACT_V1.action_names),
    "official_randomizer": ENV_NAME,
}


def canonical_json(value: Any) -> str:
  def plain(item):
    if callable(item):
      return f"{item.__module__}.{item.__qualname__}"
    if isinstance(item, Mapping):
      return {str(key): plain(child) for key, child in item.items()}
    if isinstance(item, (tuple, list)):
      return [plain(child) for child in item]
    if isinstance(item, np.ndarray):
      return item.item() if item.ndim == 0 else item.tolist()
    if isinstance(item, np.generic):
      return item.item()
    if item is None or isinstance(item, (str, int, float, bool)):
      return item
    return repr(item)

  return json.dumps(plain(value), sort_keys=True, separators=(",", ":"))


def config_sha256(value: Any) -> str:
  return hashlib.sha256(canonical_json(value).encode()).hexdigest()


if config_sha256(CURRICULUM) != CURRICULUM_SHA256:
  raise AssertionError("weak-actuator curriculum identity changed")


class TrainingKeys(NamedTuple):
  official: jax.Array
  category: jax.Array
  actuator: jax.Array
  strength: jax.Array


@dataclass(frozen=True)
class InjurySamples:
  category: jax.Array
  actuator_index: jax.Array
  sampled_weak_strength: jax.Array
  effective_strength: jax.Array


def split_training_keys(rng: jax.Array) -> TrainingKeys:
  keys = jax.vmap(lambda key: jax.random.split(key, 4))(rng)
  return TrainingKeys(keys[:, 0], keys[:, 1], keys[:, 2], keys[:, 3])


def sample_injuries(keys: TrainingKeys) -> InjurySamples:
  category_uniform = jax.vmap(jax.random.uniform)(keys.category)
  category = jnp.where(
      category_uniform < HEALTHY_PROBABILITY,
      CATEGORY_HEALTHY,
      jnp.where(
          category_uniform < HEALTHY_PROBABILITY + WEAK_PROBABILITY,
          CATEGORY_WEAK,
          CATEGORY_DEAD,
      ),
  ).astype(jnp.int32)
  actuator_index = jax.vmap(
      lambda key: jax.random.randint(
          key, (), 0, len(GO1_CONTRACT_V1.action_names)
      )
  )(keys.actuator)
  log_min = jnp.log(jnp.asarray(MIN_WEAK_STRENGTH, dtype=jnp.float32))
  log_max = jnp.log(jnp.asarray(MAX_WEAK_STRENGTH, dtype=jnp.float32))
  uniform = jax.vmap(jax.random.uniform)(keys.strength)
  sampled = jnp.exp(log_min + uniform * (log_max - log_min))
  effective = jnp.where(
      category == CATEGORY_HEALTHY,
      1.0,
      jnp.where(category == CATEGORY_WEAK, sampled, 0.0),
  ).astype(jnp.float32)
  return InjurySamples(category, actuator_index, sampled, effective)


def apply_injury(force_ranges: jax.Array, samples: InjurySamples) -> jax.Array:
  batch_size = samples.category.shape[0]
  if force_ranges.ndim == 2:
    force_ranges = jnp.broadcast_to(
        force_ranges, (batch_size,) + force_ranges.shape,
    )
  elif force_ranges.ndim != 3 or force_ranges.shape[0] != batch_size:
    raise ValueError("forcerange must be unbatched or match the injury batch")
  return force_ranges.at[
      jnp.arange(batch_size), samples.actuator_index, :
  ].multiply(samples.effective_strength[:, None])


def weak_actuator_randomize(
    model, rng: jax.Array, official_randomizer: Callable | None = None,
):
  if official_randomizer is None:
    official_randomizer = registry.get_domain_randomizer(ENV_NAME)
  keys = split_training_keys(rng)
  randomized, in_axes = official_randomizer(model, keys.official)
  randomized = randomized.tree_replace({
      "actuator_forcerange": apply_injury(
          randomized.actuator_forcerange, sample_injuries(keys),
      ),
  })
  return randomized, in_axes.tree_replace({"actuator_forcerange": 0})


def step_accounting(config: Mapping[str, Any]) -> dict[str, int]:
  rollout_steps = (
      int(config["batch_size"])
      * int(config["num_minibatches"])
      * int(config["unroll_length"])
      * int(config["action_repeat"])
  )
  eval_intervals = max(int(config["num_evals"]) - 1, 1)
  resets = max(int(config["num_resets_per_eval"]), 1)
  steps_per_epoch = math.ceil(
      int(config["num_timesteps"])
      / (eval_intervals * resets * rollout_steps)
  )
  rollout_batches = eval_intervals * resets * steps_per_epoch
  return {
      "requested_environment_timesteps": int(config["num_timesteps"]),
      "actual_environment_timesteps": rollout_batches * rollout_steps,
      "rollout_steps": rollout_steps,
      "eval_intervals": eval_intervals,
      "resets_per_interval": resets,
      "steps_per_epoch": steps_per_epoch,
      "fresh_rollout_batches": rollout_batches,
      "optimizer_steps": (
          rollout_batches * int(config["num_updates_per_batch"])
          * int(config["num_minibatches"])
      ),
  }


def resolved_ppo_config(num_timesteps: int, seed: int, num_evals: int) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, int]
]:
  if num_timesteps <= 0:
    raise ValueError("num_timesteps must be positive")
  if seed < 0:
    raise ValueError("seed must be nonnegative")
  if num_evals < 2:
    raise ValueError("num_evals must include initial and terminal evaluations")
  official = {}
  for name, parameter in inspect.signature(ppo.train).parameters.items():
    if name != "environment" and parameter.default is not inspect.Parameter.empty:
      official[name] = parameter.default
  official.update(locomotion_params.brax_ppo_config(ENV_NAME, "jax").to_dict())
  observed_hash = config_sha256(official)
  if observed_hash != OFFICIAL_PPO_CONFIG_SHA256:
    raise ValueError(f"official PPO configuration changed: {observed_hash}")
  effective = dict(official)
  effective.update({
      "num_timesteps": num_timesteps,
      "num_evals": num_evals,
      "seed": seed,
  })
  return official, effective, step_accounting(effective)


def observation_preprocessor(config: Mapping[str, Any]):
  if config["normalize_observations"]:
    return running_statistics.normalize
  return types.identity_observation_preprocessor


def checkpoint_observation_size(observation_size):
  def make_spec(size):
    if isinstance(size, Mapping) and "shape" in size:
      shape = tuple(size["shape"])
    elif hasattr(size, "shape") and not isinstance(size, (tuple, list)):
      shape = tuple(size.shape)
    elif isinstance(size, (tuple, list)):
      shape = tuple(size)
    else:
      shape = (int(size),)
    return specs.Array(shape, jnp.float32)

  if isinstance(observation_size, Mapping):
    return {key: make_spec(size) for key, size in observation_size.items()}
  return make_spec(observation_size)


def assert_numeric_finite(value: Any, path: str = "value") -> None:
  if isinstance(value, Mapping):
    for key, item in value.items():
      assert_numeric_finite(item, f"{path}.{key}")
    return
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
    for index, item in enumerate(value):
      assert_numeric_finite(item, f"{path}[{index}]")
    return
  if hasattr(value, "dtype") and hasattr(value, "shape"):
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
      raise FloatingPointError(f"non-finite numeric array at {path}")
    return
  if isinstance(value, (int, float, np.number)):
    if not math.isfinite(float(value)):
      raise FloatingPointError(f"non-finite numeric value at {path}: {value}")


def assert_finite_tree(tree: Any, label: str = "tree") -> None:
  for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
    assert_numeric_finite(leaf, f"{label}.leaf[{index}]")


def parameter_count(tree: Any) -> int:
  return sum(
      int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(tree)
  )


def assert_exact_parameter_tree(left: Any, right: Any) -> None:
  if isinstance(left, (list, tuple)):
    left = tuple(left)
  if isinstance(right, (list, tuple)):
    right = tuple(right)
  if jax.tree.structure(left) != jax.tree.structure(right):
    raise AssertionError("restored parameter tree structure changed")
  for index, (left_leaf, right_leaf) in enumerate(zip(
      jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right),
  )):
    if not np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf)):
      raise AssertionError(f"restored parameter leaf {index} changed")


def changed_leaf_count(left: Any, right: Any) -> int:
  if jax.tree.structure(left) != jax.tree.structure(right):
    raise AssertionError("parameter tree structure changed during training")
  return sum(
      not np.array_equal(np.asarray(a), np.asarray(b))
      for a, b in zip(
          jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right),
      )
  )


def validate_output_path(
    root: str | Path, experiment_commit: str, model: str, seed: int,
    num_timesteps: int, num_evals: int, evaluation_grid: str,
    repository: str | Path | None = None,
) -> Path:
  if evaluation_grid not in {"smoke", "full"}:
    raise ValueError(f"unknown evaluation grid: {evaluation_grid!r}")
  if num_evals < 2:
    raise ValueError("num_evals must include initial and terminal evaluations")
  root = Path(root).resolve()
  if repository is not None and root.is_relative_to(
      Path(repository).resolve()
  ):
    raise ValueError("experiment output root must be outside the repository")
  expected = (
      root / experiment_commit / "weak-actuator" / model / f"seed-{seed}"
      / f"steps-{num_timesteps}" / f"evals-{num_evals}"
      / f"grid-{evaluation_grid}"
  ).resolve()
  relative = expected.relative_to(root)
  if relative.parts[:2] != (experiment_commit, "weak-actuator"):
    raise ValueError("weak-actuator output identity is malformed")
  if expected.exists():
    raise FileExistsError(f"immutable experiment output exists: {expected}")
  return expected


def _device_put_replicated(value, devices):
  mesh = Mesh(np.array(devices), ("weak_actuator_device",))
  sharding = NamedSharding(mesh, PartitionSpec("weak_actuator_device"))
  return jax.tree.map(
      lambda leaf: jax.device_put(jnp.stack([leaf] * len(devices)), sharding),
      value,
  )


def install_jax_compatibility() -> str:
  try:
    jax.device_put_replicated
  except AttributeError:
    jax.device_put_replicated = _device_put_replicated
    return "official_device_put_replicated_drop_in"
  return "native_device_put_replicated"


class TerminalCheckpointCallbacks:
  """Captures step zero and saves only the terminal checkpoint."""

  def __init__(self, output: Path, config: Any, terminal_step: int):
    self.progress_path = output / "progress.jsonl"
    self.checkpoint_root = output / "checkpoints"
    self.config = config
    self.terminal_step = terminal_step
    self.initial_params = None
    self.initial_metrics = None
    self.terminal_params = None
    self.terminal_metrics = None
    self.pending_step = None
    self.pending_params = None
    self.events: list[int] = []
    self.parameter_events: list[int] = []
    self.saved_steps: list[int] = []
    self._handle = self.progress_path.open("x", buffering=1)

  def policy_params_fn(self, step, unused_make_policy, params) -> None:
    del unused_make_policy
    step = int(step)
    assert_finite_tree(params, f"parameters[{step}]")
    copied = jax.device_get(params)
    if step == 0:
      if self.initial_params is not None:
        raise RuntimeError("duplicate step-0 parameters")
      self.initial_params = copied
      self.parameter_events.append(step)
      return
    if self.pending_params is not None:
      raise RuntimeError(f"unexpected parameter callback step: {step}")
    self.pending_step = step
    self.pending_params = copied
    self.parameter_events.append(step)

  def progress_fn(self, step, metrics) -> None:
    step = int(step)
    assert_numeric_finite(metrics, f"progress_metrics[{step}]")
    copied = jax.device_get(metrics)
    self._handle.write(json.dumps({
        "step": step, "metrics": json_safe(copied),
    }, sort_keys=True) + "\n")
    self._handle.flush()
    self.events.append(step)
    if step == 0:
      if self.initial_metrics is not None:
        raise RuntimeError("duplicate step-0 metrics")
      self.initial_metrics = copied
      return
    if self.pending_step != step or self.pending_params is None:
      raise RuntimeError(f"unpaired progress callback step: {step}")
    if step == self.terminal_step:
      self.terminal_params = self.pending_params
      self.terminal_metrics = copied
      ppo_checkpoint.save(
          self.checkpoint_root, step, self.terminal_params, self.config,
      )
      self.saved_steps.append(step)
    self.pending_step = None
    self.pending_params = None

  def assert_complete(self) -> None:
    checks = {
        "events": (
            len(self.events) >= 2
            and self.events[0] == 0
            and self.events[-1] == self.terminal_step
            and self.events == sorted(set(self.events))
            and self.parameter_events == self.events
        ),
        "initial_params": self.initial_params is not None,
        "initial_metrics": self.initial_metrics is not None,
        "terminal_params": self.terminal_params is not None,
        "terminal_metrics": self.terminal_metrics is not None,
        "no_pending": self.pending_step is None and self.pending_params is None,
        "saved": self.saved_steps == [self.terminal_step],
    }
    if not all(checks.values()):
      raise AssertionError(f"incomplete training callbacks: {checks}")

  def close(self) -> None:
    self._handle.close()
