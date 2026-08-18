"""Deterministically export submission-critical results from frozen D evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from go1_robot_experiments.constants import COMMANDS, RESET_KEYS
from go1_robot_experiments.runtime import experiment_identity
from go1_robot_experiments.util import write_csv, write_json
from experiments.double_actuator.protocol import (
    HELD_OUT_SPECS,
    MODEL_LABELS,
    MODELS,
    PAIR_MANIFEST_SHA256,
    SEEDS,
    STRENGTHS,
)

DAMAGE_TRAINING_COMMIT = "9a125fd245874b1d3be50238f3cdddc510832ce0"
DAMAGE_EVALUATOR_COMMIT = "6b0b315743a9b10042d8c9fe5993f7f18011c3fd"
ACTUAL_STEPS = 412_876_800
EXPECTED_ROWS = 22 * 81 * len(COMMANDS) * len(RESET_KEYS)
CELL_ROWS = len(HELD_OUT_SPECS) * len(COMMANDS) * len(RESET_KEYS)
SURFACE_METRICS = {
    "undiscounted_return": "full_surface_return_auc",
    "survival": "full_surface_survival_auc",
    "fall": "full_surface_fall_rate_auc",
    "velocity_rmse": "full_surface_velocity_rmse_auc",
    "absolute_mechanical_power": "full_surface_mechanical_power_auc",
    "action_delta_rms": "full_surface_action_change_rms_auc",
}


def _read_json(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _damage_root(
    main_root: Path, retry_root: Path, model: str, seed: int,
) -> Path:
  campaign_root = (
      retry_root if model == "multirank_hodge_full_w150" else main_root
  )
  return (
      campaign_root / DAMAGE_TRAINING_COMMIT / "double-actuator" / model
      / f"seed-{seed}" / "steps-400000000" / "evals-19"
  )


def _row_key(row: dict[str, str]) -> tuple[int, int, int, int, int]:
  return (
      tuple(COMMANDS).index(row["command"]),
      RESET_KEYS.index(int(row["reset_key"])),
      int(row["pair_id"]),
      STRENGTHS.index(float(row["strength_a"])),
      STRENGTHS.index(float(row["strength_b"])),
  )


def _surface_auc(surface: np.ndarray) -> float:
  strengths = np.asarray(sorted(STRENGTHS), dtype=np.float64)
  first = np.trapezoid(surface, x=strengths, axis=0)
  return float(np.trapezoid(first, x=strengths) / 0.25)


def _read_progress(path: Path) -> list[dict[str, Any]]:
  records = []
  with path.open() as handle:
    for line in handle:
      record = json.loads(line)
      step = int(record["step"])
      metrics = record["metrics"]
      reward = float(metrics["eval/episode_reward"])
      episode_length = float(metrics["eval/avg_episode_length"])
      if not all(math.isfinite(value) for value in (reward, episode_length)):
        raise ValueError(f"non-finite progress metric: {path}")
      records.append({
          "environment_steps": step,
          "episode_reward": reward,
          "average_episode_length": episode_length,
      })
  if len(records) != 19:
    raise ValueError(f"progress evaluations {len(records)} != 19: {path}")
  steps = [row["environment_steps"] for row in records]
  if steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != ACTUAL_STEPS:
    raise ValueError(f"invalid progress step schedule: {path}")
  return records


def _read_damage_surface(
    path: Path, model: str, seed: int,
) -> tuple[dict[str, np.ndarray], int]:
  shape = (len(STRENGTHS), len(STRENGTHS))
  sums = {
      metric: np.zeros(shape, dtype=np.float64) for metric in SURFACE_METRICS
  }
  counts = np.zeros(shape, dtype=np.int64)
  previous = None
  row_count = 0
  held_out_ids = {pair.pair_id for pair in HELD_OUT_SPECS}
  with path.open(newline="") as handle:
    for row in csv.DictReader(handle):
      if row["model"] != model or int(row["seed"]) != seed:
        raise ValueError(f"D row identity mismatch: {path}")
      if int(row["checkpoint_step"]) != ACTUAL_STEPS:
        raise ValueError(f"D checkpoint step mismatch: {path}")
      if int(row["pair_id"]) not in held_out_ids:
        raise ValueError(f"D row contains a non-held-out pair: {path}")
      key = _row_key(row)
      if previous is not None and key <= previous:
        raise ValueError(f"D rows are not canonical and unique: {path}")
      previous = key
      strength_a = float(row["strength_a"])
      strength_b = float(row["strength_b"])
      index = (STRENGTHS.index(strength_a), STRENGTHS.index(strength_b))
      for metric in SURFACE_METRICS:
        value = float(row[metric])
        if not math.isfinite(value):
          raise ValueError(f"non-finite D metric {metric}: {path}")
        sums[metric][index] += value
      counts[index] += 1
      row_count += 1
  if row_count != EXPECTED_ROWS:
    raise ValueError(f"D row count {row_count} != {EXPECTED_ROWS}: {path}")
  if not np.all(counts == CELL_ROWS):
    raise ValueError(f"D surface cells have incorrect macro counts: {path}")
  surfaces = {metric: values / counts for metric, values in sums.items()}
  return surfaces, row_count


def _summary_rows(
    seed_values: dict[tuple[str, int], dict[str, float]],
) -> list[dict[str, Any]]:
  rows = []
  metric_names = tuple(next(iter(seed_values.values())))
  for model in MODELS:
    for metric in metric_names:
      values = [seed_values[(model, seed)][metric] for seed in SEEDS]
      aggregate = {
          "across_seed_mean": mean(values),
          "across_seed_sample_sd": stdev(values),
          "across_seed_minimum": min(values),
          "across_seed_maximum": max(values),
      }
      for seed, value in zip(SEEDS, values, strict=True):
        rows.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "seed": seed,
            "metric": metric,
            "value": value,
            **aggregate,
        })
  return rows


def export_damage(
    main_root: Path, retry_root: Path, output: Path,
) -> dict[str, Any]:
  identity = experiment_identity()
  if output.exists():
    raise FileExistsError(f"immutable analysis output exists: {output}")
  output.mkdir(parents=True)
  progress: dict[tuple[str, int], list[dict[str, Any]]] = {}
  surfaces: dict[tuple[str, int], dict[str, np.ndarray]] = {}
  seed_values: dict[tuple[str, int], dict[str, float]] = {}
  evidence = []
  common_steps = None
  for model in MODELS:
    for seed in SEEDS:
      root = _damage_root(main_root, retry_root, model, seed)
      run_path = root / "run.json"
      result_path = root / "result.json"
      progress_path = root / "progress.jsonl"
      evaluation_path = root / "heldout" / "evaluation.json"
      metrics_path = root / "heldout" / "condition_metrics.csv"
      run = _read_json(run_path)
      result = _read_json(result_path)
      evaluation = _read_json(evaluation_path)
      checks = {
          "training_status": result.get("status") == "COMPLETE",
          "training_commit": run.get("experiment_commit") == DAMAGE_TRAINING_COMMIT,
          "evaluation_status": evaluation.get("status") == "COMPLETE",
          "evaluation_commit": (
              evaluation.get("experiment_commit") == DAMAGE_EVALUATOR_COMMIT
          ),
          "training_link": (
              evaluation.get("training_experiment_commit")
              == DAMAGE_TRAINING_COMMIT
          ),
          "model": run.get("model") == model == evaluation.get("model"),
          "seed": run.get("seed") == seed == evaluation.get("seed"),
          "steps": (
              run.get("actual_steps") == ACTUAL_STEPS
              == evaluation.get("checkpoint_step")
          ),
          "pair_manifest": (
              run.get("pair_manifest_sha256") == PAIR_MANIFEST_SHA256
              == evaluation.get("pair_manifest_sha256")
          ),
          "rows": evaluation.get("row_count") == EXPECTED_ROWS,
      }
      if not all(checks.values()):
        raise ValueError(f"frozen D evidence failed validation: {checks}")
      progress[(model, seed)] = _read_progress(progress_path)
      steps = [row["environment_steps"] for row in progress[(model, seed)]]
      if common_steps is None:
        common_steps = steps
      elif steps != common_steps:
        raise ValueError("D progress checkpoint schedules are not identical")
      surface, rows = _read_damage_surface(metrics_path, model, seed)
      surfaces[(model, seed)] = surface
      values = {
          output_name: _surface_auc(surface[input_name])
          for input_name, output_name in SURFACE_METRICS.items()
      }
      zero_index = STRENGTHS.index(0.0)
      values["dual_dead_return"] = float(
          surface["undiscounted_return"][zero_index, zero_index]
      )
      seed_values[(model, seed)] = values
      evidence.append({
          "cell": "D",
          "training_regime": "damage_curriculum",
          "model": model,
          "seed": seed,
          "artifact_root": str(root),
          "retry_root": model == "multirank_hodge_full_w150",
          "row_count": rows,
          "run_json_sha256": _sha256(run_path),
          "result_json_sha256": _sha256(result_path),
          "progress_jsonl_sha256": _sha256(progress_path),
          "evaluation_json_sha256": _sha256(evaluation_path),
          "condition_metrics_sha256": _sha256(metrics_path),
          "checks": checks,
      })

  learning_rows = []
  for model in MODELS:
    for eval_index, step in enumerate(common_steps or []):
      rewards = [
          progress[(model, seed)][eval_index]["episode_reward"] for seed in SEEDS
      ]
      lengths = [
          progress[(model, seed)][eval_index]["average_episode_length"]
          for seed in SEEDS
      ]
      for seed, reward, episode_length in zip(
          SEEDS, rewards, lengths, strict=True,
      ):
        learning_rows.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "seed": seed,
            "evaluation_index": eval_index,
            "environment_steps": step,
            "episode_reward": reward,
            "average_episode_length": episode_length,
            "across_seed_mean_reward": mean(rewards),
            "across_seed_sample_sd_reward": stdev(rewards),
            "across_seed_mean_episode_length": mean(lengths),
            "across_seed_sample_sd_episode_length": stdev(lengths),
        })

  diagonal_rows = []
  for model in MODELS:
    for strength in STRENGTHS:
      strength_index = STRENGTHS.index(strength)
      values = [
          float(surfaces[(model, seed)]["undiscounted_return"][
              strength_index, strength_index
          ])
          for seed in SEEDS
      ]
      for seed, value in zip(SEEDS, values, strict=True):
        diagonal_rows.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "seed": seed,
            "residual_strength_a": strength,
            "residual_strength_b": strength,
            "heldout_pairs": len(HELD_OUT_SPECS),
            "commands": len(COMMANDS),
            "reset_keys": len(RESET_KEYS),
            "within_seed_return": value,
            "across_seed_mean_return": mean(values),
            "across_seed_sample_sd_return": stdev(values),
            "across_seed_minimum_return": min(values),
            "across_seed_maximum_return": max(values),
        })

  learning_path = output / "learning_curves_damage.csv"
  diagonal_path = output / "damage_diagonal.csv"
  summary_path = output / "damage_summary.csv"
  write_csv(learning_path, learning_rows)
  write_csv(diagonal_path, diagonal_rows)
  write_csv(summary_path, _summary_rows(seed_values))
  generated = {
      path.name: _sha256(path)
      for path in (learning_path, diagonal_path, summary_path)
  }
  write_json(output / "damage_source_manifest.json", {
      "status": "COMPLETE",
      **identity,
      "cell": "D",
      "role": "frozen_primary_read_only",
      "damage_training_commit": DAMAGE_TRAINING_COMMIT,
      "damage_evaluator_commit": DAMAGE_EVALUATOR_COMMIT,
      "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
      "models": list(MODELS),
      "seeds": list(SEEDS),
      "policy_count": len(evidence),
      "row_count": len(evidence) * EXPECTED_ROWS,
      "progress_evaluations_per_policy": 19,
      "inference": "none_preliminary_descriptive_three_seed",
      "evidence": evidence,
      "generated_files": generated,
  })
  return {
      "policies": len(evidence),
      "evaluation_rows": len(evidence) * EXPECTED_ROWS,
      "generated_files": generated,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--main-root", required=True)
  parser.add_argument("--hodge-full-retry-root", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  result = export_damage(
      Path(args.main_root).resolve(),
      Path(args.hodge_full_retry_root).resolve(),
      Path(args.output).resolve(),
  )
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
