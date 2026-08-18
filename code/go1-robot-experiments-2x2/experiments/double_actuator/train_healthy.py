"""Checkpoint-only PPO training for the frozen healthy-only 2x2 control."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
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
    CURRICULUM_SHA256,
    DEFAULT_BATCH_SIZE,
    MODEL_PARAMETERS,
    MODELS,
    PAIR_MANIFEST_SHA256,
    _array_hash,
    _tree_hash,
)
from experiments.double_actuator.healthy_protocol import (
    HEALTHY_PROTOCOL,
    HEALTHY_PROTOCOL_SHA256,
    prepare_healthy_randomization,
)

FULL_REQUESTED_STEPS = 400_000_000
FULL_ACTUAL_STEPS = 412_876_800
FULL_NUM_EVALS = 19
SMOKE_REQUESTED_STEPS = 22_937_600
SMOKE_NUM_EVALS = 2
DAMAGE_TRAINING_COMMIT = "9a125fd245874b1d3be50238f3cdddc510832ce0"


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
      root / commit / "double-actuator-healthy" / model / f"seed-{seed}"
      / f"steps-{requested_steps}" / f"evals-{num_evals}"
  )


def _read_json(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def verify_damage_randomization_reference(
    damage_reference: Path,
    base_model: Any,
    model: str,
    seed: int,
    official_randomizer: Any,
    healthy_randomized: Any,
    healthy_assignments: dict[str, Any],
) -> dict[str, Any]:
  """Reconstructs D before injury and proves it equals healthy official DR."""
  run_path = damage_reference / "run.json"
  result_path = damage_reference / "result.json"
  assignment_manifest_path = damage_reference / "assignment_manifest.json"
  assignments_path = damage_reference / "assignments.npz"
  for path in (run_path, result_path, assignment_manifest_path, assignments_path):
    if not path.is_file():
      raise FileNotFoundError(f"missing D randomization evidence: {path}")
  run = _read_json(run_path)
  result = _read_json(result_path)
  manifest = _read_json(assignment_manifest_path)
  checks = {
      "training_commit": run.get("experiment_commit") == DAMAGE_TRAINING_COMMIT,
      "model": run.get("model") == model == result.get("model"),
      "seed": run.get("seed") == seed == result.get("seed") == manifest.get("seed"),
      "status": result.get("status") == "COMPLETE",
      "curriculum": (
          run.get("curriculum_sha256") == CURRICULUM_SHA256
          == manifest.get("curriculum_sha256")
      ),
      "pair_manifest": (
          run.get("pair_manifest_sha256") == PAIR_MANIFEST_SHA256
          == manifest.get("pair_manifest_sha256")
      ),
  }
  if not all(checks.values()):
    raise ValueError(f"D randomization reference validation failed: {checks}")
  with np.load(assignments_path, allow_pickle=False) as archive:
    if "domain_keys" not in archive.files:
      raise ValueError("D assignments do not contain domain_keys")
    damage_domain_keys = np.asarray(archive["domain_keys"])
  healthy_domain_keys = np.asarray(
      jax.device_get(healthy_assignments["domain_keys"])
  )
  if not np.array_equal(damage_domain_keys, healthy_domain_keys):
    raise AssertionError("healthy and D domain-randomization keys differ")
  damage_keys_hash = _array_hash(damage_domain_keys)
  if damage_keys_hash != manifest["hashes"]["domain_keys"]:
    raise AssertionError("saved D domain-key hash does not match assignments.npz")
  reconstructed, unused_axes = official_randomizer(
      base_model, jax.device_put(damage_domain_keys),
  )
  del unused_axes
  reconstructed_hash = _tree_hash(reconstructed)
  healthy_hash = _tree_hash(healthy_randomized)
  if reconstructed_hash != healthy_hash:
    raise AssertionError(
        "healthy official DR differs from reconstructed pre-injury D model"
    )
  return {
      "status": "EXACT",
      "damage_training_commit": DAMAGE_TRAINING_COMMIT,
      "damage_reference": str(damage_reference),
      "damage_run_json_sha256": _file_sha256(run_path),
      "damage_result_json_sha256": _file_sha256(result_path),
      "damage_assignment_manifest_sha256": _file_sha256(
          assignment_manifest_path
      ),
      "damage_assignments_npz_sha256": _file_sha256(assignments_path),
      "domain_keys_sha256": damage_keys_hash,
      "healthy_pre_actuator_randomized_model_tree_sha256": healthy_hash,
      "reconstructed_damage_pre_injury_model_tree_sha256": reconstructed_hash,
      "checks": checks,
  }


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
    damage_reference: Path,
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
      prepare_healthy_randomization(
          base_env.mjx_model, seed, official_randomizer, DEFAULT_BATCH_SIZE,
      )
  )
  cross_regime = verify_damage_randomization_reference(
      damage_reference,
      base_env.mjx_model,
      model,
      seed,
      official_randomizer,
      randomized_model,
      assignments,
  )

  write_json(output / "healthy_protocol.json", {
      **HEALTHY_PROTOCOL, "sha256": HEALTHY_PROTOCOL_SHA256,
  })
  _save_assignments(output / "assignments.npz", assignments)
  write_json(output / "assignment_manifest.json", {
      "seed": seed,
      "batch_size": DEFAULT_BATCH_SIZE,
      "training_regime": "healthy_only",
      "healthy_protocol_sha256": HEALTHY_PROTOCOL_SHA256,
      "hashes": assignment_hashes,
  })
  write_json(output / "cross_regime_randomization.json", cross_regime)
  write_json(output / "ppo_config.json", {
      "official": official,
      "official_sha256": config_sha256(official),
      "effective": effective,
      "effective_sha256": config_sha256(effective),
      "step_accounting": accounting,
  })
  write_json(output / "run.json", {
      "experiment": "double-actuator-healthy-capacity-matched",
      "training_regime": "healthy_only",
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
      "healthy_protocol_sha256": HEALTHY_PROTOCOL_SHA256,
      "reference_damage_curriculum_sha256": CURRICULUM_SHA256,
      "cross_regime_randomization_status": cross_regime["status"],
      "damage_reference": cross_regime,
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
      "healthy_protocol_sha256": HEALTHY_PROTOCOL_SHA256,
      "cross_regime_randomization": cross_regime,
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
  parser.add_argument("--damage-reference-root", required=True)
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
        Path(args.damage_reference_root).resolve(),
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
          "training_regime": "healthy_only",
          "healthy_protocol_sha256": HEALTHY_PROTOCOL_SHA256,
          "damage_reference_root": str(
              Path(args.damage_reference_root).resolve()
          ),
          "error_type": type(error).__name__,
          "error": str(error),
          "traceback": traceback.format_exc(),
          "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
      })
    raise


if __name__ == "__main__":
  main()
