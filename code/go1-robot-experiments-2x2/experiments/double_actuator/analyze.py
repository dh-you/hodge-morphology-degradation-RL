"""Descriptive three-seed analysis for held-out compound damage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from go1_robot_experiments.constants import COMMANDS, RESET_KEYS
from go1_robot_experiments.util import write_csv, write_json
from experiments.double_actuator.protocol import (
    MODEL_LABELS,
    MODELS,
    PAIR_MANIFEST_SHA256,
    SEEDS,
    STRENGTHS,
)

EXPECTED_ROWS = 22 * 81 * len(COMMANDS) * len(RESET_KEYS)
PAPER_WORDING = (
    "We report preliminary results across three independent training seeds "
    "and show all seed-level outcomes; given the small sample size, "
    "comparisons are descriptive."
)


def _read_json(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def _surface(path: Path, model: str, seed: int) -> tuple[np.ndarray, int]:
  strengths = tuple(sorted(STRENGTHS))
  index = {value: position for position, value in enumerate(strengths)}
  totals = np.zeros((len(strengths), len(strengths)), dtype=np.float64)
  counts = np.zeros_like(totals, dtype=np.int64)
  previous = None
  rows = 0
  with path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      if row["model"] != model or int(row["seed"]) != seed:
        raise ValueError(f"row identity mismatch: {path}")
      key = (
          tuple(COMMANDS).index(row["command"]),
          int(row["reset_key"]),
          int(row["pair_id"]),
          STRENGTHS.index(float(row["strength_a"])),
          STRENGTHS.index(float(row["strength_b"])),
      )
      if previous is not None and key <= previous:
        raise ValueError(f"noncanonical or duplicate row order: {path}")
      previous = key
      a = index[float(row["strength_a"])]
      b = index[float(row["strength_b"])]
      value = float(row["undiscounted_return"])
      if not math.isfinite(value):
        raise ValueError(f"nonfinite return: {path}")
      totals[a, b] += value
      counts[a, b] += 1
      rows += 1
  if rows != EXPECTED_ROWS:
    raise ValueError(f"row count {rows} != {EXPECTED_ROWS}: {path}")
  expected_cell = 22 * len(COMMANDS) * len(RESET_KEYS)
  if not np.all(counts == expected_cell):
    raise ValueError(f"unbalanced surface cells: {path}")
  return totals / counts, rows


def surface_auc(surface: np.ndarray) -> float:
  strengths = np.asarray(sorted(STRENGTHS), dtype=np.float64)
  integral_b = np.trapezoid(surface, x=strengths, axis=1)
  return float(np.trapezoid(integral_b, x=strengths) / 0.25)


def analyze(campaign_root: Path, output: Path) -> dict[str, Any]:
  if output.exists():
    raise FileExistsError(f"immutable analysis exists: {output}")
  seed_rows = []
  surface_rows = []
  evidence = []
  surfaces: dict[tuple[str, int], np.ndarray] = {}
  sorted_strengths = tuple(sorted(STRENGTHS))
  for model in MODELS:
    for seed in SEEDS:
      training = campaign_root / model / f"seed-{seed}" / "steps-400000000" / "evals-19"
      run = _read_json(training / "run.json")
      result = _read_json(training / "result.json")
      evaluation = _read_json(training / "heldout" / "evaluation.json")
      metrics = training / "heldout" / "condition_metrics.csv"
      checks = {
          "training": result.get("status") == "COMPLETE",
          "evaluation": evaluation.get("status") == "COMPLETE",
          "model": run.get("model") == model == evaluation.get("model"),
          "seed": run.get("seed") == seed == evaluation.get("seed"),
          "steps": run.get("actual_steps") == 412_876_800,
          "rows": evaluation.get("row_count") == EXPECTED_ROWS,
          "manifest": evaluation.get("pair_manifest_sha256") == PAIR_MANIFEST_SHA256,
      }
      if not all(checks.values()):
        raise ValueError(f"artifact validation failed {model}/{seed}: {checks}")
      surface, row_count = _surface(metrics, model, seed)
      surfaces[(model, seed)] = surface
      auc = surface_auc(surface)
      seed_rows.append({
          "model": model,
          "model_label": MODEL_LABELS[model],
          "seed": seed,
          "heldout_absolute_return_surface_auc": auc,
      })
      for ai, strength_a in enumerate(sorted_strengths):
        for bi, strength_b in enumerate(sorted_strengths):
          surface_rows.append({
              "model": model,
              "seed": seed,
              "strength_a": strength_a,
              "strength_b": strength_b,
              "mean_undiscounted_return": float(surface[ai, bi]),
          })
      evidence.append({
          "model": model,
          "seed": seed,
          "row_count": row_count,
          "metrics_sha256": _sha256(metrics),
          "checkpoint": result["checkpoint"]["path"],
          "slurm_job_id": run.get("slurm", {}).get("job_id"),
          "evaluation_slurm_job_id": evaluation.get("slurm_job_id"),
      })

  summary_rows = []
  by_identity = {(row["model"], row["seed"]): row for row in seed_rows}
  for model in MODELS:
    values = [
        by_identity[(model, seed)]["heldout_absolute_return_surface_auc"]
        for seed in SEEDS
    ]
    summary_rows.append({
        "model": model,
        "model_label": MODEL_LABELS[model],
        "n_seeds": len(values),
        "mean": mean(values),
        "sample_sd": stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    })
  paired_rows = []
  full = "multirank_hodge_full_w150"
  for comparator, comparison in (
      ("native_mlp_w480_240_120", "Hodge-F minus Native MLP"),
      ("multirank_hodge_lower_w177", "Hodge-F minus Hodge-L"),
  ):
    for seed in SEEDS:
      full_value = by_identity[(full, seed)]["heldout_absolute_return_surface_auc"]
      comparator_value = by_identity[(comparator, seed)]["heldout_absolute_return_surface_auc"]
      paired_rows.append({
          "comparison": comparison,
          "seed": seed,
          "hodge_f": full_value,
          "comparator": comparator_value,
          "difference": full_value - comparator_value,
      })

  output.mkdir(parents=True)
  write_csv(output / "seed_metrics.csv", seed_rows)
  write_csv(output / "model_summary.csv", summary_rows)
  write_csv(output / "paired_differences.csv", paired_rows)
  write_csv(output / "severity_surfaces.csv", surface_rows)
  write_json(output / "manifest.json", {
      "status": "COMPLETE",
      "analysis": "preliminary_descriptive_three_seed",
      "paper_wording": PAPER_WORDING,
      "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
      "models": list(MODELS),
      "seeds": list(SEEDS),
      "policy_count": len(seed_rows),
      "total_evaluation_rows": len(seed_rows) * EXPECTED_ROWS,
      "primary_endpoint": "heldout_compound_damage_absolute_return_surface_auc",
      "inference": "none",
      "evidence": evidence,
  })

  figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
  display_models = tuple(reversed(MODELS))
  for x, model in enumerate(display_models):
    values = [
        by_identity[(model, seed)]["heldout_absolute_return_surface_auc"]
        for seed in SEEDS
    ]
    axes[0].scatter([x] * len(values), values, s=42, zorder=3)
    axes[0].plot([x - 0.18, x + 0.18], [mean(values)] * 2, color="black", lw=2)
  axes[0].set_xticks(range(len(display_models)), [MODEL_LABELS[m] for m in display_models], rotation=25, ha="right")
  axes[0].set_ylabel("Held-out return surface AUC")
  axes[0].set_title("All three training seeds")
  mean_surface = np.mean([surfaces[(full, seed)] for seed in SEEDS], axis=0)
  image = axes[1].imshow(
      mean_surface,
      origin="lower",
      extent=[min(sorted_strengths), max(sorted_strengths), min(sorted_strengths), max(sorted_strengths)],
      aspect="equal",
  )
  axes[1].set_xlabel("Actuator B residual strength")
  axes[1].set_ylabel("Actuator A residual strength")
  axes[1].set_title("Hodge-F mean held-out surface")
  figure.colorbar(image, ax=axes[1], label="Mean undiscounted return")
  metadata = {"Creator": "go1-robot-experiments", "CreationDate": None, "ModDate": None}
  figure.savefig(output / "primary_result.png", dpi=220)
  figure.savefig(output / "primary_result.pdf", metadata=metadata)
  plt.close(figure)

  report = [
      "# Preliminary Double-Actuator Workshop Results",
      "",
      PAPER_WORDING,
      "",
      "| Model | Seeds | Mean AUC | Sample SD | Range |",
      "|---|---:|---:|---:|---:|",
  ]
  for row in summary_rows:
    report.append(
        f"| {row['model_label']} | 3 | {row['mean']:.6g} | "
        f"{row['sample_sd']:.6g} | {row['minimum']:.6g}–{row['maximum']:.6g} |"
    )
  report.extend([
      "",
      "Comparisons are descriptive. No significance tests or claims of confirmed superiority are made.",
      "",
  ])
  (output / "REPORT.md").write_text("\n".join(report))
  return {"policies": len(seed_rows), "rows": len(seed_rows) * EXPECTED_ROWS}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--campaign-root", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  analyze(Path(args.campaign_root).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
  main()
