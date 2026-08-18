"""Healthy nominal evaluation for cell A or C of the frozen 2x2 study."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback
from typing import Any

from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground import registry, wrapper

from go1_core import load_checkpoint
from go1_robot_experiments.constants import COMMANDS, ENV_NAME, HORIZON, RESET_KEYS
from go1_robot_experiments.dependencies import verify_runtime_dependencies
from go1_robot_experiments.protocol import (
    assert_environment_contract,
    patch_batched_command,
)
from go1_robot_experiments.runtime import experiment_identity, runtime_information
from go1_robot_experiments.util import write_csv, write_json
from experiments.weak_actuator.train import _serialized_factory_kwargs
from experiments.double_actuator.evidence import (
    ACTUAL_STEPS,
    file_sha256,
    validate_training,
)
from experiments.double_actuator.protocol import MODELS
from experiments.double_actuator.rollout import make_double_actuator_rollout

NOMINAL_METRICS = (
    "velocity_rmse",
    "yaw_rmse",
    "absolute_mechanical_power",
    "torque_rms",
    "action_delta_rms",
    "survival",
    "fall",
    "undiscounted_return",
    "actuator_force_utilization",
    "saturation_fraction",
)
CELL_REGIMES = {"A": "healthy_only", "C": "damage_curriculum"}


def output_path(
    root: str | Path, evaluator_commit: str, cell: str, model: str, seed: int,
) -> Path:
  root = Path(root).resolve()
  home = Path.home().resolve()
  if root == home or root.is_relative_to(home):
    raise ValueError("evaluation outputs must not use home quota")
  if not root.is_relative_to(Path("/scratch/network/dy0130")):
    raise ValueError("evaluation root must be under /scratch/network/dy0130")
  return (
      root / evaluator_commit / "double-actuator-2x2" / f"cell-{cell}"
      / model / f"seed-{seed}" / "steps-400000000" / "evals-19"
      / "nominal"
  )


def _reset_keys(reset_index: int) -> jax.Array:
  key = jax.random.PRNGKey(reset_index)
  return jnp.broadcast_to(key, (1,) + key.shape)


def evaluate_nominal(
    training: Path,
    output_root: Path,
    model: str,
    seed: int,
    cell: str,
    training_commit: str,
    require_gpu: bool,
) -> dict[str, Any]:
  if cell not in CELL_REGIMES:
    raise ValueError(f"nominal evaluation cell must be A or C, got {cell}")
  if require_gpu and jax.default_backend() != "gpu":
    raise RuntimeError(f"GPU required, got {jax.default_backend()!r}")
  identity = experiment_identity()
  dependencies = verify_runtime_dependencies()
  regime = CELL_REGIMES[cell]
  source = validate_training(
      training, model, seed, training_commit, regime, identity,
  )
  output = output_path(
      output_root, identity["experiment_commit"], cell, model, seed,
  )
  if output.exists():
    raise FileExistsError(f"immutable nominal evaluation exists: {output}")
  output.mkdir(parents=True)
  try:
    base_env = registry.load(
        ENV_NAME,
        config=registry.get_default_config(ENV_NAME),
        config_overrides={"impl": "jax"},
    )
    assert_environment_contract(base_env)
    checkpoint = training / "checkpoints" / f"{ACTUAL_STEPS:012d}"
    networks, params, config = load_checkpoint(checkpoint, expected_model=model)
    _serialized_factory_kwargs(config, model)
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
    condition_ranges = base_env.mjx_model.actuator_forcerange[None, ...]

    def fixed_randomize(mjx_model):
      randomized = mjx_model.tree_replace({
          "actuator_forcerange": condition_ranges,
      })
      in_axes = jax.tree_util.tree_map(lambda unused: None, mjx_model)
      return randomized, in_axes.tree_replace({"actuator_forcerange": 0})

    vectorized_env = wrapper.BraxDomainRandomizationVmapWrapper(
        base_env, fixed_randomize,
    )
    reset = jax.jit(vectorized_env.reset)
    force_limits = jnp.max(jnp.abs(condition_ranges), axis=-1)
    diagnostic_indices = jnp.asarray([[0, 1]], dtype=jnp.int32)
    rollout = jax.jit(make_double_actuator_rollout(
        base_env, vectorized_env, policy, force_limits, diagnostic_indices,
    ))
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for command_name, command_value in COMMANDS.items():
      command = jnp.asarray(command_value, dtype=jnp.float32)
      for reset_key in RESET_KEYS:
        state = patch_batched_command(
            reset(_reset_keys(reset_key)), command,
        )
        result = jax.device_get(rollout(state, command))
        metrics = {name: float(result[name][0]) for name in NOMINAL_METRICS}
        if not all(np.isfinite(value) for value in metrics.values()):
          raise FloatingPointError(
              f"non-finite nominal row {command_name}/{reset_key}"
          )
        terminated = bool(result["was_terminated"][0])
        termination_step = int(result["termination_step"][0])
        active_steps = int(result["active_steps"][0])
        valid_termination = (
            terminated and termination_step == active_steps
            and 1 <= active_steps <= HORIZON
        ) or (
            not terminated and termination_step == -1
            and active_steps == HORIZON
        )
        if not valid_termination:
          raise AssertionError("invalid nominal termination accounting")
        rows.append({
            "cell": cell,
            "training_regime": regime,
            "model": model,
            "seed": seed,
            "checkpoint_step": ACTUAL_STEPS,
            "condition": "healthy",
            "command": command_name,
            "command_value": list(command_value),
            "reset_key": reset_key,
            **metrics,
            "was_terminated": terminated,
            "termination_step": termination_step if terminated else None,
            "active_steps": active_steps,
        })
    wall_time = time.monotonic() - started
    expected = len(COMMANDS) * len(RESET_KEYS)
    if len(rows) != expected:
      raise AssertionError(f"nominal rows {len(rows)} != {expected}")
    csv_path = output / "condition_metrics.csv"
    write_csv(csv_path, rows)
    record = {
        "status": "COMPLETE",
        **identity,
        "cell": cell,
        "training_regime": regime,
        "model": model,
        "seed": seed,
        "training_experiment_commit": training_commit,
        "checkpoint_step": ACTUAL_STEPS,
        "condition": "healthy",
        "row_count": len(rows),
        "commands": list(COMMANDS),
        "reset_keys": list(RESET_KEYS),
        "horizon": HORIZON,
        "row_order": ["command", "reset_key"],
        "continuous_metrics_finite": True,
        "wall_time_seconds": wall_time,
        "runtime": runtime_information(),
        "dependencies": dependencies,
        "training_evidence": source["evidence"],
        "condition_metrics_sha256": file_sha256(csv_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(output / "evaluation.json", record)
    return record
  except Exception as error:
    if not (output / "failure.json").exists():
      write_json(output / "failure.json", {
          "status": "FAILED",
          **identity,
          "cell": cell,
          "training_regime": regime,
          "model": model,
          "seed": seed,
          "training_experiment_commit": training_commit,
          "error_type": type(error).__name__,
          "error": str(error),
          "traceback": traceback.format_exc(),
          "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
      })
    raise


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--training-root", required=True)
  parser.add_argument("--output-root", required=True)
  parser.add_argument("--model", choices=MODELS, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--cell", choices=tuple(CELL_REGIMES), required=True)
  parser.add_argument("--training-commit", required=True)
  parser.add_argument("--require-gpu", action="store_true")
  args = parser.parse_args()
  evaluate_nominal(
      Path(args.training_root).resolve(),
      Path(args.output_root).resolve(),
      args.model,
      args.seed,
      args.cell,
      args.training_commit,
      args.require_gpu,
  )


if __name__ == "__main__":
  main()
