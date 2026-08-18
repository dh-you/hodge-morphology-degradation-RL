"""Export deterministic A--D controls after all A--C artifacts complete."""

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
from go1_robot_experiments.util import write_csv, write_json
from experiments.double_actuator.export_2x2 import (
    ACTUAL_STEPS,
    DAMAGE_EVALUATOR_COMMIT,
    DAMAGE_TRAINING_COMMIT,
    SURFACE_METRICS,
    _damage_root,
    _read_damage_surface,
    _read_progress,
    _sha256,
    _surface_auc,
)
from experiments.double_actuator.healthy_protocol import HEALTHY_PROTOCOL_SHA256
from experiments.double_actuator.protocol import (
    MODEL_LABELS,
    MODELS,
    PAIR_MANIFEST_SHA256,
    SEEDS,
    STRENGTHS,
)

NOMINAL_METRICS = (
    "undiscounted_return",
    "velocity_rmse",
    "yaw_rmse",
    "survival",
    "fall",
    "absolute_mechanical_power",
    "action_delta_rms",
)
SURFACE_OUTPUTS = tuple(SURFACE_METRICS.values()) + ("dual_dead_return",)
EXPECTED_NOMINAL_ROWS = len(COMMANDS) * len(RESET_KEYS)
EXPECTED_DAMAGE_ROWS = 22 * 81 * len(COMMANDS) * len(RESET_KEYS)


def _read_json(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def _tree_sha256(root: Path) -> tuple[str, int]:
  files = sorted(path for path in root.rglob("*") if path.is_file())
  if not files:
    raise ValueError(f"empty checkpoint tree: {root}")
  digest = hashlib.sha256()
  for path in files:
    relative = path.relative_to(root).as_posix()
    digest.update(relative.encode())
    digest.update(b"\0")
    digest.update(str(path.stat().st_size).encode())
    digest.update(b"\0")
    digest.update(bytes.fromhex(_sha256(path)))
    digest.update(b"\n")
  return digest.hexdigest(), len(files)


def _healthy_training_root(root: Path, commit: str, model: str, seed: int) -> Path:
  return (
      root / commit / "double-actuator-healthy" / model / f"seed-{seed}"
      / "steps-400000000" / "evals-19"
  )


def _cell_root(root: Path, commit: str, cell: str, model: str, seed: int) -> Path:
  suffix = "nominal" if cell in {"A", "C"} else "heldout"
  return (
      root / commit / "double-actuator-2x2" / f"cell-{cell}" / model
      / f"seed-{seed}" / "steps-400000000" / "evals-19" / suffix
  )


def _read_nominal(path: Path, cell: str, model: str, seed: int) -> dict[str, float]:
  values = {metric: [] for metric in NOMINAL_METRICS}
  previous = None
  with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
  if len(rows) != EXPECTED_NOMINAL_ROWS:
    raise ValueError(f"nominal row count {len(rows)} != 90: {path}")
  for row in rows:
    if row["cell"] != cell or row["model"] != model or int(row["seed"]) != seed:
      raise ValueError(f"nominal identity mismatch: {path}")
    key = (tuple(COMMANDS).index(row["command"]), RESET_KEYS.index(int(row["reset_key"])))
    if previous is not None and key <= previous:
      raise ValueError(f"nominal rows are not canonical and unique: {path}")
    previous = key
    for metric in NOMINAL_METRICS:
      value = float(row[metric])
      if not math.isfinite(value):
        raise ValueError(f"non-finite nominal {metric}: {path}")
      values[metric].append(value)
  return {metric: mean(metric_values) for metric, metric_values in values.items()}


def _surface_values(surface: dict[str, np.ndarray]) -> dict[str, float]:
  values = {
      output: _surface_auc(surface[source])
      for source, output in SURFACE_METRICS.items()
  }
  zero = STRENGTHS.index(0.0)
  values["dual_dead_return"] = float(surface["undiscounted_return"][zero, zero])
  return values


def _aggregate_seed_rows(
    cell: str, values: dict[tuple[str, int], dict[str, float]],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  metrics = tuple(next(iter(values.values())))
  for model in MODELS:
    for metric in metrics:
      per_seed = [values[(model, seed)][metric] for seed in SEEDS]
      aggregate = {
          "across_seed_mean": mean(per_seed),
          "across_seed_sample_sd": stdev(per_seed),
          "across_seed_minimum": min(per_seed),
          "across_seed_maximum": max(per_seed),
      }
      for seed, value in zip(SEEDS, per_seed, strict=True):
        rows.append({
            "cell": cell,
            "model": model,
            "model_label": MODEL_LABELS[model],
            "seed": seed,
            "metric": metric,
            "value": value,
            **aggregate,
        })
  return rows


def _restricted_auc(surface: np.ndarray, maximum: float) -> float:
  strengths = np.asarray(sorted(value for value in STRENGTHS if value <= maximum))
  indices = [STRENGTHS.index(float(value)) for value in strengths]
  restricted = surface[np.ix_(indices, indices)]
  return float(
      np.trapezoid(np.trapezoid(restricted, x=strengths, axis=0), x=strengths)
      / maximum**2
  )


def _log_weak_auc(surface: np.ndarray) -> float:
  strengths = np.asarray(
      sorted(value for value in STRENGTHS if 0.05 <= value <= 0.2)
  )
  indices = [STRENGTHS.index(float(value)) for value in strengths]
  restricted = surface[np.ix_(indices, indices)]
  coordinates = np.log(strengths)
  area = (coordinates[-1] - coordinates[0]) ** 2
  return float(
      np.trapezoid(
          np.trapezoid(restricted, x=coordinates, axis=0), x=coordinates
      ) / area
  )


def export_controls(
    artifact_root: Path,
    implementation_commit: str,
    d_main_root: Path,
    d_retry_root: Path,
    output: Path,
) -> dict[str, Any]:
  if output.exists():
    raise FileExistsError(f"immutable control output exists: {output}")
  output.mkdir(parents=True)
  progress: dict[tuple[str, int], list[dict[str, Any]]] = {}
  nominal: dict[str, dict[tuple[str, int], dict[str, float]]] = {
      "A": {}, "C": {},
  }
  surfaces: dict[str, dict[tuple[str, int], dict[str, np.ndarray]]] = {
      "B": {}, "D": {},
  }
  surface_values: dict[str, dict[tuple[str, int], dict[str, float]]] = {
      "B": {}, "D": {},
  }
  policy_manifest = []
  evaluation_manifest = []

  for model in MODELS:
    for seed in SEEDS:
      healthy_root = _healthy_training_root(
          artifact_root, implementation_commit, model, seed,
      )
      damage_root = _damage_root(d_main_root, d_retry_root, model, seed)
      for regime, root, commit in (
          ("healthy_only", healthy_root, implementation_commit),
          ("damage_curriculum", damage_root, DAMAGE_TRAINING_COMMIT),
      ):
        run_path = root / "run.json"
        result_path = root / "result.json"
        run = _read_json(run_path)
        result = _read_json(result_path)
        checkpoint = root / "checkpoints" / f"{ACTUAL_STEPS:012d}"
        tree_hash, tree_files = _tree_sha256(checkpoint)
        checks = {
            "status": result.get("status") == "COMPLETE",
            "commit": run.get("experiment_commit") == commit == result.get("experiment_commit"),
            "model": run.get("model") == model == result.get("model"),
            "seed": run.get("seed") == seed == result.get("seed"),
            "steps": run.get("actual_steps") == ACTUAL_STEPS == result.get("actual_steps"),
            "checkpoint_roundtrip": result.get("checkpoint") == {
                "path": f"checkpoints/{ACTUAL_STEPS:012d}",
                "exact_parameter_roundtrip": True,
                "exact_logits_roundtrip": True,
                "exact_actions_roundtrip": True,
            },
        }
        if regime == "healthy_only":
          checks["protocol"] = (
              run.get("healthy_protocol_sha256") == HEALTHY_PROTOCOL_SHA256
          )
          checks["cross_regime"] = (
              _read_json(root / "cross_regime_randomization.json").get("status")
              == "EXACT"
          )
        else:
          checks["pair_manifest"] = (
              run.get("pair_manifest_sha256") == PAIR_MANIFEST_SHA256
          )
        if not all(checks.values()):
          raise ValueError(f"policy manifest validation failed: {checks}")
        policy_manifest.append({
            "training_regime": regime,
            "model": model,
            "seed": seed,
            "training_commit": commit,
            "artifact_root": str(root),
            "checkpoint_tree_sha256": tree_hash,
            "checkpoint_file_count": tree_files,
            "run_json_sha256": _sha256(run_path),
            "result_json_sha256": _sha256(result_path),
            "checks": checks,
        })
      progress[(model, seed)] = _read_progress(healthy_root / "progress.jsonl")

      for cell in ("A", "C"):
        root = _cell_root(artifact_root, implementation_commit, cell, model, seed)
        record_path = root / "evaluation.json"
        metrics_path = root / "condition_metrics.csv"
        record = _read_json(record_path)
        checks = {
            "status": record.get("status") == "COMPLETE",
            "cell": record.get("cell") == cell,
            "model": record.get("model") == model,
            "seed": record.get("seed") == seed,
            "rows": record.get("row_count") == EXPECTED_NOMINAL_ROWS,
            "metrics_hash": record.get("condition_metrics_sha256") == _sha256(metrics_path),
        }
        if not all(checks.values()):
          raise ValueError(f"nominal evaluation validation failed: {checks}")
        nominal[cell][(model, seed)] = _read_nominal(metrics_path, cell, model, seed)
        evaluation_manifest.append({
            "cell": cell, "model": model, "seed": seed,
            "artifact_root": str(root), "row_count": EXPECTED_NOMINAL_ROWS,
            "evaluation_json_sha256": _sha256(record_path),
            "condition_metrics_sha256": _sha256(metrics_path), "checks": checks,
        })

      for cell in ("B", "D"):
        root = (
            _cell_root(artifact_root, implementation_commit, "B", model, seed)
            if cell == "B" else damage_root / "heldout"
        )
        record_path = root / "evaluation.json"
        metrics_path = root / "condition_metrics.csv"
        record = _read_json(record_path)
        expected_eval_commit = (
            implementation_commit if cell == "B" else DAMAGE_EVALUATOR_COMMIT
        )
        checks = {
            "status": record.get("status") == "COMPLETE",
            "commit": record.get("experiment_commit") == expected_eval_commit,
            "model": record.get("model") == model,
            "seed": record.get("seed") == seed,
            "rows": record.get("row_count") == EXPECTED_DAMAGE_ROWS,
            "pair_manifest": record.get("pair_manifest_sha256") == PAIR_MANIFEST_SHA256,
            "metrics_hash": record.get("condition_metrics_sha256") == _sha256(metrics_path),
        }
        if not all(checks.values()):
          raise ValueError(f"surface evaluation validation failed: {checks}")
        surface, unused_count = _read_damage_surface(metrics_path, model, seed)
        surfaces[cell][(model, seed)] = surface
        surface_values[cell][(model, seed)] = _surface_values(surface)
        evaluation_manifest.append({
            "cell": cell, "model": model, "seed": seed,
            "artifact_root": str(root), "row_count": EXPECTED_DAMAGE_ROWS,
            "evaluation_json_sha256": _sha256(record_path),
            "condition_metrics_sha256": _sha256(metrics_path), "checks": checks,
        })

  learning_rows = []
  common_steps = None
  for model in MODELS:
    model_steps = [row["environment_steps"] for row in progress[(model, SEEDS[0])]]
    if common_steps is None:
      common_steps = model_steps
    elif model_steps != common_steps:
      raise ValueError("healthy learning schedules differ")
    for index, step in enumerate(model_steps):
      rewards = [progress[(model, seed)][index]["episode_reward"] for seed in SEEDS]
      lengths = [progress[(model, seed)][index]["average_episode_length"] for seed in SEEDS]
      for seed, reward, length in zip(SEEDS, rewards, lengths, strict=True):
        learning_rows.append({
            "model": model, "model_label": MODEL_LABELS[model], "seed": seed,
            "evaluation_index": index, "environment_steps": step,
            "episode_reward": reward, "average_episode_length": length,
            "across_seed_mean_reward": mean(rewards),
            "across_seed_sample_sd_reward": stdev(rewards),
            "across_seed_mean_episode_length": mean(lengths),
            "across_seed_sample_sd_episode_length": stdev(lengths),
        })

  healthy_rows = _aggregate_seed_rows("A", nominal["A"])
  ad_rows = healthy_rows + _aggregate_seed_rows("B", surface_values["B"])
  ad_rows += _aggregate_seed_rows("C", nominal["C"])
  ad_rows += _aggregate_seed_rows("D", surface_values["D"])
  surface_rows = []
  for cell in ("B", "D"):
    for model in MODELS:
      for seed in SEEDS:
        for ia, strength_a in enumerate(STRENGTHS):
          for ib, strength_b in enumerate(STRENGTHS):
            surface_rows.append({
                "cell": cell, "model": model, "model_label": MODEL_LABELS[model],
                "seed": seed, "strength_a": strength_a,
                "strength_b": strength_b,
                **{
                    output: float(surfaces[cell][(model, seed)][source][ia, ib])
                    for source, output in SURFACE_METRICS.items()
                },
            })

  contrasts = []
  for seed in SEEDS:
    for comparison, other in (
        ("Hodge-F_minus_Native", "native_mlp_w480_240_120"),
        ("Hodge-F_minus_Hodge-L", "multirank_hodge_lower_w177"),
    ):
      hodge = "multirank_hodge_full_w150"
      contrasts.append({
          "seed": seed, "comparison": comparison,
          "A_return_difference": (
              nominal["A"][(hodge, seed)]["undiscounted_return"]
              - nominal["A"][(other, seed)]["undiscounted_return"]
          ),
          "D_full_surface_return_auc_difference": (
              surface_values["D"][(hodge, seed)]["full_surface_return_auc"]
              - surface_values["D"][(other, seed)]["full_surface_return_auc"]
          ),
      })
    for model in MODELS:
      contrasts.append({
          "seed": seed, "comparison": f"{model}:regime_effects",
          "C_minus_A_nominal_return": (
              nominal["C"][(model, seed)]["undiscounted_return"]
              - nominal["A"][(model, seed)]["undiscounted_return"]
          ),
          "A_minus_C_nominal_robustness_cost": (
              nominal["A"][(model, seed)]["undiscounted_return"]
              - nominal["C"][(model, seed)]["undiscounted_return"]
          ),
          "D_minus_B_damage_training_benefit": (
              surface_values["D"][(model, seed)]["full_surface_return_auc"]
              - surface_values["B"][(model, seed)]["full_surface_return_auc"]
          ),
      })

  exploratory = []
  for regime, nominal_cell, damage_cell in (
      ("healthy_only", "A", "B"), ("damage_curriculum", "C", "D"),
  ):
    for model in MODELS:
      for seed in SEEDS:
        surface = surfaces[damage_cell][(model, seed)]["undiscounted_return"]
        dead = float(surface[STRENGTHS.index(0.0), STRENGTHS.index(0.0)])
        weak = _log_weak_auc(surface)
        healthy = nominal[nominal_cell][(model, seed)]["undiscounted_return"]
        exploratory.append({
            "status": "post-D exploratory",
            "training_regime": regime, "model": model, "seed": seed,
            "uniform_0_0.2_surface_return_auc": _restricted_auc(surface, 0.2),
            "target_mixture_return_score": 0.3 * healthy + 0.5 * weak + 0.2 * dead,
            "target_mixture_healthy_component": healthy,
            "target_mixture_loguniform_weak_component": weak,
            "target_mixture_dual_dead_component": dead,
        })

  files = {
      "learning_curves_healthy.csv": learning_rows,
      "healthy_summary.csv": healthy_rows,
      "ad_summary.csv": ad_rows,
      "surface_summary.csv": surface_rows,
      "contrasts_interactions.csv": contrasts,
      "exploratory_summary.csv": exploratory,
  }
  hashes = {}
  for name, rows in files.items():
    path = output / name
    write_csv(path, rows)
    hashes[name] = _sha256(path)
  write_json(output / "policy_manifest_30.json", {
      "status": "COMPLETE", "policy_count": len(policy_manifest),
      "records": policy_manifest,
  })
  write_json(output / "evaluation_manifest_60.json", {
      "status": "COMPLETE", "cell_count": len(evaluation_manifest),
      "records": evaluation_manifest,
  })
  for name in ("policy_manifest_30.json", "evaluation_manifest_60.json"):
    hashes[name] = _sha256(output / name)
  write_json(output / "generated_manifest.json", {
      "status": "COMPLETE", "implementation_commit": implementation_commit,
      "damage_training_commit": DAMAGE_TRAINING_COMMIT,
      "damage_evaluator_commit": DAMAGE_EVALUATOR_COMMIT,
      "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
      "healthy_protocol_sha256": HEALTHY_PROTOCOL_SHA256,
      "inference": "none_preliminary_descriptive_three_seed",
      "exploratory_label": "post-D exploratory",
      "generated_files": hashes,
  })
  return {"policies": len(policy_manifest), "cells": len(evaluation_manifest), "files": hashes}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--artifact-root", required=True)
  parser.add_argument("--implementation-commit", required=True)
  parser.add_argument("--d-main-root", required=True)
  parser.add_argument("--d-hodge-full-retry-root", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  result = export_controls(
      Path(args.artifact_root).resolve(), args.implementation_commit,
      Path(args.d_main_root).resolve(),
      Path(args.d_hodge_full_retry_root).resolve(),
      Path(args.output).resolve(),
  )
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
