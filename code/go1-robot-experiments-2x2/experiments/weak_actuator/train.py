"""Train, restore, and evaluate a weak-actuator Go1 policy."""

from __future__ import annotations

import argparse
import functools
import hashlib
import os
from pathlib import Path
import time
import traceback
from typing import Any

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground import registry, wrapper

from go1_core import (
    CANONICAL_MODELS,
    MODEL_SPECS,
    load_checkpoint,
    make_go1_ppo_networks,
)
from go1_robot_experiments.constants import (
    COMMANDS,
    CONDITIONS,
    ENV_NAME,
    HORIZON,
    MODEL_TIERS,
    ORIGINAL_WEAK_ACTUATOR_COMMIT,
    RESET_KEYS,
    SUMMARY_METRICS,
)
from go1_robot_experiments.dependencies import verify_runtime_dependencies
from go1_robot_experiments.protocol import (
    assert_environment_contract,
    condition_force_ranges,
    fixed_condition_randomize,
    paired_reset_keys,
    patch_batched_command,
)
from go1_robot_experiments.rollout import make_weak_actuator_rollout
from go1_robot_experiments.runtime import (
    experiment_identity,
    runtime_information,
)
from go1_robot_experiments.util import write_csv, write_json

from experiments.weak_actuator.protocol import (
    CURRICULUM,
    CURRICULUM_SHA256,
    DEFAULT_EVALUATION_GRID,
    DEFAULT_MODEL,
    DEFAULT_NUM_EVALS,
    DEFAULT_NUM_TIMESTEPS,
    DEFAULT_SEED,
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
    validate_output_path,
    weak_actuator_randomize,
)


def _array_sha256(value: Any) -> str:
  array = np.asarray(value)
  dtype = array.dtype.newbyteorder("<")
  canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
  return hashlib.sha256(canonical.tobytes()).hexdigest()


def _policy_probe() -> dict[str, jax.Array]:
  return {
      "state": jnp.stack((
          jnp.zeros((48,), dtype=jnp.float32),
          jnp.linspace(-1.0, 1.0, 48, dtype=jnp.float32),
          jnp.sin(jnp.arange(48, dtype=jnp.float32)),
      )),
      "privileged_state": jnp.zeros((3, 123), dtype=jnp.float32),
  }


def _policy_outputs(networks, params) -> tuple[np.ndarray, np.ndarray]:
  observation = _policy_probe()
  logits = networks.policy_network.apply(params[0], params[1], observation)
  policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
  actions, _ = policy(observation, jax.random.PRNGKey(0))
  return np.asarray(jax.device_get(logits)), np.asarray(jax.device_get(actions))


def _serialized_factory_kwargs(config: Any, model: str) -> dict[str, Any]:
  raw_kwargs = config.network_factory_kwargs.to_dict()
  kwargs = {
      key: list(value) if isinstance(value, tuple) else value
      for key, value in raw_kwargs.items()
  }
  expected = {
      "model_name": model,
      "policy_hidden_layer_sizes": [512, 256, 128],
      "policy_obs_key": "state",
      "value_hidden_layer_sizes": [512, 256, 128],
      "value_obs_key": "privileged_state",
  }
  if kwargs != expected:
    raise AssertionError(f"serialized factory kwargs changed: {kwargs}")
  return kwargs


def evaluation_axes(
    evaluation_grid: str,
) -> tuple[tuple[tuple[str, tuple[float, float, float]], ...], tuple[int, ...]]:
  if evaluation_grid == "full":
    return tuple(COMMANDS.items()), RESET_KEYS
  if evaluation_grid == "smoke":
    return (("stand", COMMANDS["stand"]),), (0,)
  raise ValueError(f"unknown evaluation grid: {evaluation_grid!r}")


def _rollout_rows(
    networks, params, base_env, model: str, step: int, evaluation_grid: str,
):
  policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
  force_ranges = condition_force_ranges(base_env.mjx_model.actuator_forcerange)
  force_limits = jnp.max(jnp.abs(force_ranges), axis=-1)
  vectorized_env = wrapper.BraxDomainRandomizationVmapWrapper(
      base_env, fixed_condition_randomize,
  )
  reset = jax.jit(vectorized_env.reset)
  rollout = jax.jit(
      make_weak_actuator_rollout(base_env, vectorized_env, policy, force_limits)
  )
  selected_commands, selected_resets = evaluation_axes(evaluation_grid)
  rows = []
  damage_only = {
      "damaged_actuator_force_utilization",
      "damaged_joint_tracking_rmse",
  }
  for command_name, command_value in selected_commands:
    command = jnp.asarray(command_value, dtype=jnp.float32)
    for reset_key in selected_resets:
      state = patch_batched_command(
          reset(paired_reset_keys(reset_key)), command,
      )
      result = jax.device_get(rollout(state, command))
      for index, condition in enumerate(CONDITIONS):
        terminated = bool(result["was_terminated"][index])
        termination_step = int(result["termination_step"][index])
        active_steps = int(result["active_steps"][index])
        metrics = {name: float(result[name][index]) for name in SUMMARY_METRICS}
        if not all(np.isfinite(value) for value in metrics.values()):
          raise FloatingPointError(
              f"non-finite rollout row {command_name}/{reset_key}/{index}"
          )
        row = {
            "model": model,
            "model_tier": MODEL_TIERS[model],
            "checkpoint_step": step,
            "condition": condition.condition,
            "actuator_index": condition.actuator_index,
            "actuator_name": condition.actuator_name,
            "strength": condition.strength,
            "command": command_name,
            "command_value": list(command_value),
            "reset_key": reset_key,
            **{
                name: (
                    None
                    if condition.actuator_index is None and name in damage_only
                    else value
                )
                for name, value in metrics.items()
            },
            "was_terminated": terminated,
            "termination_step": termination_step if terminated else None,
            "active_steps": active_steps,
        }
        valid_termination = (
            terminated
            and termination_step == active_steps
            and 1 <= active_steps <= HORIZON
        ) or (
            not terminated
            and row["termination_step"] is None
            and active_steps == HORIZON
        )
        if not (
            0.0 <= metrics["survival"] <= 1.0
            and metrics["fall"]
                == float(terminated and termination_step < HORIZON)
            and valid_termination
        ):
          raise AssertionError(
              f"invalid restored rollout row "
              f"{command_name}/{reset_key}/{index}: {row}"
          )
        rows.append(row)
  expected_rows = len(selected_commands) * len(selected_resets) * len(CONDITIONS)
  if len(rows) != expected_rows:
    raise AssertionError(
        f"restored rollout produced {len(rows)} rows, expected {expected_rows}"
    )
  return rows


def _run(
    output: Path,
    identity: dict[str, Any],
    model: str,
    seed: int,
    num_timesteps: int,
    num_evals: int,
    evaluation_grid: str,
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
  if actor_parameters != MODEL_SPECS[model].actor_parameters:
    raise AssertionError("initialized actor differs from installed core identity")
  checkpoint_config = ppo_checkpoint.network_config(
      checkpoint_observation_size(base_env.observation_size),
      base_env.action_size,
      effective["normalize_observations"],
      network_factory,
  )
  serialized_kwargs = _serialized_factory_kwargs(checkpoint_config, model)

  run_record = {
      "experiment": "weak-actuator-ppo",
      **identity,
      "model": model,
      "model_tier": MODEL_TIERS[model],
      "actor_parameters": actor_parameters,
      "seed": seed,
      "environment": ENV_NAME,
      "environment_impl": "jax",
      "protocol_source_commit": ORIGINAL_WEAK_ACTUATOR_COMMIT,
      "requested_steps": num_timesteps,
      "actual_steps": actual_steps,
      "num_evals": effective["num_evals"],
      "evaluation_grid": evaluation_grid,
      "terminal_checkpoint": f"checkpoints/{actual_steps:012d}",
      "curriculum_sha256": CURRICULUM_SHA256,
      "official_ppo_config_sha256": config_sha256(official),
      "effective_ppo_config_sha256": config_sha256(effective),
      "serialized_factory_kwargs": serialized_kwargs,
      "dependencies": dependencies,
  }
  write_json(output / "run.json", run_record)
  write_json(output / "curriculum.json", {
      **CURRICULUM, "sha256": CURRICULUM_SHA256,
  })
  write_json(output / "ppo_config.json", {
      "official": official,
      "official_sha256": config_sha256(official),
      "effective": effective,
      "effective_sha256": config_sha256(effective),
      "step_accounting": accounting,
  })

  callbacks = TerminalCheckpointCallbacks(
      output, checkpoint_config, actual_steps,
  )
  training_params.update({
      "seed": seed,
      "network_factory": network_factory,
      "wrap_env_fn": wrapper.wrap_for_brax_training,
      "randomization_fn": weak_actuator_randomize,
      "save_checkpoint_path": None,
      "policy_params_fn": callbacks.policy_params_fn,
      "progress_fn": callbacks.progress_fn,
  })
  started = time.monotonic()
  try:
    _, params, final_metrics = ppo.train(
        environment=base_env, **training_params,
    )
  finally:
    callbacks.close()
  wall_time = time.monotonic() - started
  callbacks.assert_complete()
  assert_finite_tree(params, "terminal_parameters")
  assert_numeric_finite(final_metrics, "final_metrics")
  assert_exact_parameter_tree(callbacks.terminal_params, params)

  changed_actor_leaves = changed_leaf_count(
      callbacks.initial_params[1], params[1],
  )
  if changed_actor_leaves == 0:
    raise AssertionError("no actor tensors changed during training")
  initial_logits, initial_actions = _policy_outputs(
      networks, callbacks.initial_params,
  )
  terminal_logits, terminal_actions = _policy_outputs(networks, params)
  if np.array_equal(initial_logits, terminal_logits):
    raise AssertionError("fixed-probe logits did not change during training")
  if np.array_equal(initial_actions, terminal_actions):
    raise AssertionError("fixed-probe actions did not change during training")

  checkpoint = output / "checkpoints" / f"{actual_steps:012d}"
  restored_networks, restored_params, restored_config = load_checkpoint(
      checkpoint, expected_model=model,
  )
  assert_finite_tree(restored_params, "restored_parameters")
  assert_exact_parameter_tree(params, restored_params)
  if parameter_count(restored_params[1]) != actor_parameters:
    raise AssertionError("restored actor parameter count changed")
  _serialized_factory_kwargs(restored_config, model)
  restored_logits, restored_actions = _policy_outputs(
      restored_networks, restored_params,
  )
  if not np.array_equal(terminal_logits, restored_logits):
    raise AssertionError("restored logits differ from terminal logits")
  if not np.array_equal(terminal_actions, restored_actions):
    raise AssertionError("restored actions differ from terminal actions")
  if not (
      np.isfinite(restored_logits).all()
      and np.isfinite(restored_actions).all()
      and np.all(np.abs(restored_actions) <= 1.0)
  ):
    raise AssertionError("restored policy outputs are non-finite or unbounded")

  rows = _rollout_rows(
      restored_networks, restored_params, base_env, model, actual_steps,
      evaluation_grid,
  )
  write_csv(output / "condition_metrics.csv", rows)
  write_json(output / "evaluation.json", {
      "model": model,
      "checkpoint_step": actual_steps,
      "grid": evaluation_grid,
      "commands": list(COMMANDS) if evaluation_grid == "full" else ["stand"],
      "reset_keys": list(RESET_KEYS) if evaluation_grid == "full" else [0],
      "horizon": HORIZON,
      "condition_count": len(CONDITIONS),
      "row_count": len(rows),
      "ordered_conditions_valid": True,
      "continuous_metrics_finite": True,
  })

  initial_reward = float(callbacks.initial_metrics["eval/episode_reward"])
  terminal_reward = float(callbacks.terminal_metrics["eval/episode_reward"])
  initial_length = float(
      callbacks.initial_metrics["eval/avg_episode_length"]
  )
  terminal_length = float(
      callbacks.terminal_metrics["eval/avg_episode_length"]
  )
  result = {
      "status": "COMPLETE",
      **identity,
      "model": model,
      "seed": seed,
      "requested_steps": num_timesteps,
      "actual_steps": actual_steps,
      "num_evals": num_evals,
      "evaluation_grid": evaluation_grid,
      "runtime": runtime_information(),
      "jax_compatibility": compatibility,
      "wall_time_seconds": wall_time,
      "environment_steps_per_second": actual_steps / wall_time,
      "initial_metrics": callbacks.initial_metrics,
      "terminal_metrics": callbacks.terminal_metrics,
      "learning": {
          "reward_delta": terminal_reward - initial_reward,
          "average_episode_length_delta": terminal_length - initial_length,
      },
      "actor_parameters": actor_parameters,
      "changed_actor_leaves": changed_actor_leaves,
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
      "restored_rollout_rows": len(rows),
  }
  write_json(output / "result.json", result)
  return result


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Train and evaluate a weak-actuator Go1 policy.",
  )
  parser.add_argument(
      "--output-root", default=os.environ.get("GO1_EXPERIMENT_OUTPUT_ROOT"),
  )
  parser.add_argument(
      "--model", choices=CANONICAL_MODELS, default=DEFAULT_MODEL,
  )
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
  parser.add_argument(
      "--num-timesteps", type=int, default=DEFAULT_NUM_TIMESTEPS,
  )
  parser.add_argument("--num-evals", type=int, default=DEFAULT_NUM_EVALS)
  parser.add_argument(
      "--evaluation-grid", choices=("smoke", "full"),
      default=DEFAULT_EVALUATION_GRID,
  )
  parser.add_argument("--require-gpu", action="store_true")
  return parser


def main() -> None:
  parser = make_parser()
  args = parser.parse_args()
  if args.output_root is None:
    parser.error(
        "--output-root or GO1_EXPERIMENT_OUTPUT_ROOT is required"
    )
  if args.seed < 0:
    parser.error("--seed must be nonnegative")
  if args.num_timesteps <= 0:
    parser.error("--num-timesteps must be positive")
  if args.num_evals < 2:
    parser.error("--num-evals must be at least 2")

  identity = experiment_identity()
  output = validate_output_path(
      args.output_root, identity["experiment_commit"], args.model, args.seed,
      args.num_timesteps, args.num_evals, args.evaluation_grid,
      repository=identity["repository"],
  )
  output.mkdir(parents=True, exist_ok=False)
  try:
    _run(
        output,
        identity,
        args.model,
        args.seed,
        args.num_timesteps,
        args.num_evals,
        args.evaluation_grid,
        args.require_gpu,
    )
  except Exception as error:
    failure = output / "failure.json"
    if not failure.exists():
      write_json(failure, {
          "status": "FAILED",
          **identity,
          "model": args.model,
          "seed": args.seed,
          "requested_steps": args.num_timesteps,
          "num_evals": args.num_evals,
          "evaluation_grid": args.evaluation_grid,
          "error_type": type(error).__name__,
          "error": str(error),
          "traceback": traceback.format_exc(),
          "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
      })
    raise


if __name__ == "__main__":
  main()
