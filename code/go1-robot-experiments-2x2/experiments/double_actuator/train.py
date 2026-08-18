"""Checkpoint-only PPO training for the frozen double-actuator study."""

from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path
import shutil
import time
import traceback
from typing import Any

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import train as ppo
import jax
import numpy as np
from mujoco_playground import registry, wrapper

from go1_core import MODEL_SPECS, load_checkpoint, make_go1_ppo_networks
from go1_robot_experiments.constants import ENV_NAME
from go1_robot_experiments.dependencies import verify_runtime_dependencies
from go1_robot_experiments.protocol import assert_environment_contract
from go1_robot_experiments.runtime import experiment_identity, runtime_information
from go1_robot_experiments.util import write_json
from experiments.weak_actuator.protocol import (
    TerminalCheckpointCallbacks,
    assert_exact_parameter_tree,
    assert_finite_tree,
    assert_numeric_finite,
    changed_leaf_count,
    checkpoint_observation_size,
    config_sha256,
    install_jax_compatibility,
    observation_preprocessor,
    parameter_count,
    resolved_ppo_config,
)
from experiments.weak_actuator.train import (
    _array_sha256,
    _policy_outputs,
    _serialized_factory_kwargs,
)
from experiments.double_actuator.protocol import (
    CURRICULUM,
    CURRICULUM_SHA256,
    DEFAULT_BATCH_SIZE,
    MODEL_PARAMETERS,
    MODELS,
    PAIR_MANIFEST,
    PAIR_MANIFEST_SHA256,
    prepare_training_randomization,
)

FULL_REQUESTED_STEPS = 400_000_000
FULL_ACTUAL_STEPS = 412_876_800
FULL_NUM_EVALS = 19
SMOKE_REQUESTED_STEPS = 22_937_600
SMOKE_NUM_EVALS = 2


def output_path(
    root: str | Path, commit: str, model: str, seed: int,
    requested_steps: int, num_evals: int,
) -> Path:
  root = Path(root).resolve()
  home = Path.home().resolve()
  if root == home or root.is_relative_to(home):
    raise ValueError("campaign outputs and caches must not use home quota")
  if not root.is_relative_to(Path("/scratch/network/dy0130")):
    raise ValueError("campaign output root must be under /scratch/network/dy0130")
  return (
      root / commit / "double-actuator" / model / f"seed-{seed}"
      / f"steps-{requested_steps}" / f"evals-{num_evals}"
  )


def _save_assignments(path: Path, arrays: dict[str, Any]) -> None:
  np.savez_compressed(
      path,
      **{name: np.asarray(jax.device_get(value)) for name, value in arrays.items()},
  )


def _randomization_batch(
    randomized_model: Any, randomization_axes: Any, batch_size: int,
) -> Any:
  if batch_size <= 0 or batch_size > DEFAULT_BATCH_SIZE:
    raise ValueError(f"invalid randomization batch size: {batch_size}")
  return jax.tree.map(
      lambda value, axis: value if axis is None else value[:batch_size],
      randomized_model,
      randomization_axes,
      is_leaf=lambda value: value is None,
  )


def run(
    output: Path,
    identity: dict[str, Any],
    model: str,
    seed: int,
    num_timesteps: int,
    num_evals: int,
    require_gpu: bool,
) -> dict[str, Any]:
  if require_gpu and jax.default_backend() != "gpu":
    raise RuntimeError(f"GPU required, got {jax.default_backend()!r}")
  dependencies = verify_runtime_dependencies()
  compatibility = install_jax_compatibility()
  official, effective, accounting = resolved_ppo_config(
      num_timesteps, seed, num_evals,
  )
  actual_steps = accounting["actual_environment_timesteps"]
  base_env = registry.load(
      ENV_NAME,
      config=registry.get_default_config(ENV_NAME),
      config_overrides={"impl": "jax"},
  )
  assert_environment_contract(base_env)

  training_params = dict(effective)
  network_kwargs = dict(training_params.pop("network_factory"))
  network_factory = functools.partial(
      make_go1_ppo_networks, model_name=model, **network_kwargs,
  )
  networks = network_factory(
      base_env.observation_size,
      base_env.action_size,
      preprocess_observations_fn=observation_preprocessor(effective),
  )
  actor_parameters = parameter_count(
      networks.policy_network.init(jax.random.PRNGKey(0))
  )
  if actor_parameters != MODEL_PARAMETERS[model]:
    raise AssertionError(
        f"actor count {actor_parameters} differs from frozen {MODEL_PARAMETERS[model]}"
    )
  if MODEL_SPECS[model].actor_parameters != actor_parameters:
    raise AssertionError("installed core registry differs from initialized actor")

  checkpoint_config = ppo_checkpoint.network_config(
      checkpoint_observation_size(base_env.observation_size),
      base_env.action_size,
      effective["normalize_observations"],
      network_factory,
  )
  serialized_kwargs = _serialized_factory_kwargs(checkpoint_config, model)
  official_randomizer = registry.get_domain_randomizer(ENV_NAME)
  randomized_model, randomization_axes, assignments, assignment_hashes = (
      prepare_training_randomization(
          base_env.mjx_model, seed, official_randomizer, DEFAULT_BATCH_SIZE,
      )
  )

  write_json(output / "pair_manifest.json", PAIR_MANIFEST)
  write_json(output / "curriculum.json", {
      **CURRICULUM, "sha256": CURRICULUM_SHA256,
  })
  _save_assignments(output / "assignments.npz", assignments)
  write_json(output / "assignment_manifest.json", {
      "seed": seed,
      "batch_size": DEFAULT_BATCH_SIZE,
      "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
      "curriculum_sha256": CURRICULUM_SHA256,
      "hashes": assignment_hashes,
  })
  write_json(output / "ppo_config.json", {
      "official": official,
      "official_sha256": config_sha256(official),
      "effective": effective,
      "effective_sha256": config_sha256(effective),
      "step_accounting": accounting,
  })
  write_json(output / "run.json", {
      "experiment": "double-actuator-capacity-matched",
      **identity,
      "model": model,
      "actor_parameters": actor_parameters,
      "seed": seed,
      "environment": ENV_NAME,
      "environment_impl": "jax",
      "requested_steps": num_timesteps,
      "actual_steps": actual_steps,
      "num_evals": num_evals,
      "terminal_checkpoint": f"checkpoints/{actual_steps:012d}",
      "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
      "curriculum_sha256": CURRICULUM_SHA256,
      "official_ppo_config_sha256": config_sha256(official),
      "effective_ppo_config_sha256": config_sha256(effective),
      "serialized_factory_kwargs": serialized_kwargs,
      "dependencies": dependencies,
      "slurm": {
          "job_id": os.environ.get("SLURM_JOB_ID"),
          "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
          "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
      },
  })

  def fixed_randomization(unused_model, rng):
    del unused_model
    return (
        _randomization_batch(
            randomized_model, randomization_axes, int(rng.shape[0]),
        ),
        randomization_axes,
    )

  callbacks = TerminalCheckpointCallbacks(output, checkpoint_config, actual_steps)
  training_params.update({
      "seed": seed,
      "network_factory": network_factory,
      "wrap_env_fn": wrapper.wrap_for_brax_training,
      "randomization_fn": fixed_randomization,
      "save_checkpoint_path": None,
      "policy_params_fn": callbacks.policy_params_fn,
      "progress_fn": callbacks.progress_fn,
  })
  started = time.monotonic()
  try:
    _, params, final_metrics = ppo.train(environment=base_env, **training_params)
  finally:
    callbacks.close()
  wall_time = time.monotonic() - started
  callbacks.assert_complete()
  assert_finite_tree(params, "terminal_parameters")
  assert_numeric_finite(final_metrics, "final_metrics")
  assert_exact_parameter_tree(callbacks.terminal_params, params)

  changed_actor_leaves = changed_leaf_count(callbacks.initial_params[1], params[1])
  if changed_actor_leaves == 0:
    raise AssertionError("no actor tensors changed during training")
  initial_logits, initial_actions = _policy_outputs(networks, callbacks.initial_params)
  terminal_logits, terminal_actions = _policy_outputs(networks, params)
  checkpoint = output / "checkpoints" / f"{actual_steps:012d}"
  restored_networks, restored_params, restored_config = load_checkpoint(
      checkpoint, expected_model=model,
  )
  assert_finite_tree(restored_params, "restored_parameters")
  assert_exact_parameter_tree(params, restored_params)
  _serialized_factory_kwargs(restored_config, model)
  restored_logits, restored_actions = _policy_outputs(restored_networks, restored_params)
  if not np.array_equal(terminal_logits, restored_logits):
    raise AssertionError("restored logits differ from terminal logits")
  if not np.array_equal(terminal_actions, restored_actions):
    raise AssertionError("restored actions differ from terminal actions")
  if not np.isfinite(restored_logits).all() or not np.isfinite(restored_actions).all():
    raise AssertionError("restored policy outputs are non-finite")
  if not np.all(np.abs(restored_actions) <= 1.0):
    raise AssertionError("restored actions are outside [-1, 1]")

  initial_reward = float(callbacks.initial_metrics["eval/episode_reward"])
  terminal_reward = float(callbacks.terminal_metrics["eval/episode_reward"])
  initial_length = float(callbacks.initial_metrics["eval/avg_episode_length"])
  terminal_length = float(callbacks.terminal_metrics["eval/avg_episode_length"])
  steps_per_second = actual_steps / wall_time
  result = {
      "status": "COMPLETE",
      **identity,
      "model": model,
      "seed": seed,
      "requested_steps": num_timesteps,
      "actual_steps": actual_steps,
      "num_evals": num_evals,
      "runtime": runtime_information(),
      "jax_compatibility": compatibility,
      "wall_time_seconds": wall_time,
      "environment_steps_per_second": steps_per_second,
      "projected_full_training_seconds": FULL_ACTUAL_STEPS / steps_per_second,
      "initial_metrics": callbacks.initial_metrics,
      "terminal_metrics": callbacks.terminal_metrics,
      "learning": {
          "reward_delta": terminal_reward - initial_reward,
          "average_episode_length_delta": terminal_length - initial_length,
      },
      "actor_parameters": actor_parameters,
      "changed_actor_leaves": changed_actor_leaves,
      "assignment_hashes": assignment_hashes,
      "fixed_probe": {
          "initial_logits_sha256": _array_sha256(initial_logits),
          "terminal_logits_sha256": _array_sha256(terminal_logits),
          "restored_logits_sha256": _array_sha256(restored_logits),
          "initial_actions_sha256": _array_sha256(initial_actions),
          "terminal_actions_sha256": _array_sha256(terminal_actions),
          "restored_actions_sha256": _array_sha256(restored_actions),
      },
      "checkpoint": {
          "path": str(checkpoint.relative_to(output)),
          "exact_parameter_roundtrip": True,
          "exact_logits_roundtrip": True,
          "exact_actions_roundtrip": True,
      },
  }
  write_json(output / "result.json", result)
  return result


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-root", default=os.environ.get("GO1_EXPERIMENT_OUTPUT_ROOT"))
  parser.add_argument("--model", choices=MODELS, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--num-timesteps", type=int, default=FULL_REQUESTED_STEPS)
  parser.add_argument("--num-evals", type=int, default=FULL_NUM_EVALS)
  parser.add_argument("--require-gpu", action="store_true")
  return parser


def main() -> None:
  parser = make_parser()
  args = parser.parse_args()
  if args.output_root is None:
    parser.error("--output-root or GO1_EXPERIMENT_OUTPUT_ROOT is required")
  if args.seed < 0 or args.num_timesteps <= 0 or args.num_evals < 2:
    parser.error("seed, timesteps, or evaluations are invalid")
  identity = experiment_identity()
  output = output_path(
      args.output_root, identity["experiment_commit"], args.model, args.seed,
      args.num_timesteps, args.num_evals,
  )
  if output.exists():
    raise FileExistsError(f"immutable campaign output exists: {output}")
  output.mkdir(parents=True)
  try:
    run(
        output, identity, args.model, args.seed, args.num_timesteps,
        args.num_evals, args.require_gpu,
    )
  except Exception as error:
    if not (output / "failure.json").exists():
      write_json(output / "failure.json", {
          "status": "FAILED",
          **identity,
          "model": args.model,
          "seed": args.seed,
          "requested_steps": args.num_timesteps,
          "num_evals": args.num_evals,
          "error_type": type(error).__name__,
          "error": str(error),
          "traceback": traceback.format_exc(),
          "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
      })
    raise


if __name__ == "__main__":
  main()
