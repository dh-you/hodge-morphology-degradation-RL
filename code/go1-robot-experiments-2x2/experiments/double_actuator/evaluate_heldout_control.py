"""Held-out asymmetric paired-damage evaluation for a certified checkpoint."""

from __future__ import annotations

import argparse
import functools
import json
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
from go1_robot_experiments.protocol import assert_environment_contract, patch_batched_command
from go1_robot_experiments.runtime import experiment_identity, runtime_information
from go1_robot_experiments.util import write_csv, write_json
from experiments.weak_actuator.train import _serialized_factory_kwargs
from experiments.double_actuator.evidence import (
    ACTUAL_STEPS,
    file_sha256,
    validate_training,
)
from experiments.double_actuator.protocol import (
    MODELS,
    PAIR_MANIFEST_SHA256,
    STRENGTHS,
    evaluation_conditions,
    held_out_shard,
)
from experiments.double_actuator.rollout import make_double_actuator_rollout

METRICS = (
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
    "damaged_a_force_utilization",
    "damaged_b_force_utilization",
    "damaged_pair_force_utilization",
    "damaged_a_tracking_rmse",
    "damaged_b_tracking_rmse",
    "damaged_pair_tracking_rmse",
)


def _read_json(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def _force_ranges(base: jax.Array, conditions: tuple[dict[str, Any], ...]) -> jax.Array:
  count = len(conditions)
  ranges = jnp.broadcast_to(base, (count,) + base.shape)
  rows = jnp.arange(count)
  actuator_a = jnp.asarray([row["actuator_a"] for row in conditions], dtype=jnp.int32)
  actuator_b = jnp.asarray([row["actuator_b"] for row in conditions], dtype=jnp.int32)
  strength_a = jnp.asarray([row["strength_a"] for row in conditions], dtype=jnp.float32)
  strength_b = jnp.asarray([row["strength_b"] for row in conditions], dtype=jnp.float32)
  ranges = ranges.at[rows, actuator_a, :].multiply(strength_a[:, None])
  return ranges.at[rows, actuator_b, :].multiply(strength_b[:, None])


def _reset_keys(reset_index: int, count: int) -> jax.Array:
  key = jax.random.PRNGKey(reset_index)
  return jnp.broadcast_to(key, (count,) + key.shape)


def output_path(
    root: str | Path, evaluator_commit: str, model: str, seed: int, shard: int,
) -> Path:
  root = Path(root).resolve()
  home = Path.home().resolve()
  if root == home or root.is_relative_to(home):
    raise ValueError("evaluation outputs must not use home quota")
  if not root.is_relative_to(Path("/scratch/network/dy0130")):
    raise ValueError("evaluation root must be under /scratch/network/dy0130")
  return (
      root / evaluator_commit / "double-actuator-2x2" / "cell-B"
      / model / f"seed-{seed}" / "steps-400000000" / "evals-19"
      / "heldout" / f"shard-{shard}"
  )


def evaluate(
    training: Path,
    output_root: Path,
    model: str,
    seed: int,
    shard: int,
    training_commit: str,
    require_gpu: bool,
) -> dict[str, Any]:
  if require_gpu and jax.default_backend() != "gpu":
    raise RuntimeError(f"GPU required, got {jax.default_backend()!r}")
  identity = experiment_identity()
  dependencies = verify_runtime_dependencies()
  source = validate_training(
      training, model, seed, training_commit, "healthy_only", identity,
  )
  output = output_path(
      output_root, identity["experiment_commit"], model, seed, shard,
  )
  if output.exists():
    raise FileExistsError(f"immutable held-out evaluation exists: {output}")
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
    conditions = evaluation_conditions(shard)
    if len(conditions) != 11 * 81:
      raise AssertionError("held-out shard condition count changed")
    condition_ranges = _force_ranges(base_env.mjx_model.actuator_forcerange, conditions)

    def fixed_randomize(mjx_model):
      randomized = mjx_model.tree_replace({"actuator_forcerange": condition_ranges})
      in_axes = jax.tree_util.tree_map(lambda unused: None, mjx_model)
      return randomized, in_axes.tree_replace({"actuator_forcerange": 0})

    vectorized_env = wrapper.BraxDomainRandomizationVmapWrapper(base_env, fixed_randomize)
    reset = jax.jit(vectorized_env.reset)
    force_limits = jnp.max(jnp.abs(condition_ranges), axis=-1)
    damage_indices = jnp.asarray(
        [(row["actuator_a"], row["actuator_b"]) for row in conditions],
        dtype=jnp.int32,
    )
    rollout = jax.jit(make_double_actuator_rollout(
        base_env, vectorized_env, policy, force_limits, damage_indices,
    ))
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for command_name, command_value in COMMANDS.items():
      command = jnp.asarray(command_value, dtype=jnp.float32)
      for reset_key in RESET_KEYS:
        state = patch_batched_command(
            reset(_reset_keys(reset_key, len(conditions))), command,
        )
        result = jax.device_get(rollout(state, command))
        for index, condition in enumerate(conditions):
          metrics = {name: float(result[name][index]) for name in METRICS}
          if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(
                f"non-finite row {command_name}/{reset_key}/{index}"
            )
          terminated = bool(result["was_terminated"][index])
          termination_step = int(result["termination_step"][index])
          active_steps = int(result["active_steps"][index])
          valid_termination = (
              terminated and termination_step == active_steps
              and 1 <= active_steps <= HORIZON
          ) or (
              not terminated and termination_step == -1 and active_steps == HORIZON
          )
          if not valid_termination:
            raise AssertionError("invalid termination accounting")
          rows.append({
              "cell": "B",
              "training_regime": "healthy_only",
              "model": model,
              "seed": seed,
              "checkpoint_step": ACTUAL_STEPS,
              **condition,
              "command": command_name,
              "command_value": list(command_value),
              "reset_key": reset_key,
              **metrics,
              "was_terminated": terminated,
              "termination_step": termination_step if terminated else None,
              "active_steps": active_steps,
          })
    wall_time = time.monotonic() - started
    expected = len(COMMANDS) * len(RESET_KEYS) * 11 * len(STRENGTHS) ** 2
    if len(rows) != expected:
      raise AssertionError(f"held-out rows {len(rows)} != {expected}")
    csv_path = output / "condition_metrics.csv"
    write_csv(csv_path, rows)
    record = {
        "status": "COMPLETE",
        **identity,
        "cell": "B",
        "training_regime": "healthy_only",
        "model": model,
        "seed": seed,
        "shard": shard,
        "training_experiment_commit": training_commit,
        "checkpoint_step": ACTUAL_STEPS,
        "split": "held_out",
        "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
        "held_out_pair_ids": [pair.pair_id for pair in held_out_shard(shard)],
        "pair_count": len(held_out_shard(shard)),
        "strengths": list(STRENGTHS),
        "condition_count": len(conditions),
        "row_count": len(rows),
        "commands": list(COMMANDS),
        "reset_keys": list(RESET_KEYS),
        "horizon": HORIZON,
        "row_order": ["command", "reset_key", "pair_id", "strength_a", "strength_b"],
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
          "cell": "B",
          "training_regime": "healthy_only",
          "model": model,
          "seed": seed,
          "shard": shard,
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
  parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
  parser.add_argument("--training-commit", required=True)
  parser.add_argument("--require-gpu", action="store_true")
  args = parser.parse_args()
  evaluate(
      Path(args.training_root).resolve(), Path(args.output_root).resolve(),
      args.model, args.seed, args.shard, args.training_commit, args.require_gpu,
  )


if __name__ == "__main__":
  main()
