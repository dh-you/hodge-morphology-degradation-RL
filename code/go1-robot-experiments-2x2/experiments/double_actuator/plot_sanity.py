"""Create clearly labeled sanity-check plots from certified CSV exports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.double_actuator.protocol import MODEL_LABELS, MODELS


def _rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    return list(csv.DictReader(handle))


def _learning_plot(path: Path, output: Path, title: str) -> None:
  rows = _rows(path)
  figure, axis = plt.subplots(figsize=(7.0, 4.0))
  for model in MODELS:
    model_rows = [row for row in rows if row["model"] == model and int(row["seed"]) == 11]
    model_rows.sort(key=lambda row: int(row["environment_steps"]))
    axis.plot(
        [int(row["environment_steps"]) / 1e6 for row in model_rows],
        [float(row["across_seed_mean_reward"]) for row in model_rows],
        label=MODEL_LABELS[model],
    )
  axis.set(xlabel="Environment steps (millions)", ylabel="Evaluation reward", title=title)
  axis.legend(fontsize=7)
  figure.tight_layout()
  figure.savefig(output, dpi=160, metadata={"Title": title})
  plt.close(figure)


def _diagonal_plot(path: Path, output: Path) -> None:
  rows = _rows(path)
  figure, axis = plt.subplots(figsize=(7.0, 4.0))
  for model in MODELS:
    model_rows = [row for row in rows if row["model"] == model and int(row["seed"]) == 11]
    model_rows.sort(key=lambda row: float(row["residual_strength_a"]))
    axis.plot(
        [float(row["residual_strength_a"]) for row in model_rows],
        [float(row["across_seed_mean_return"]) for row in model_rows],
        marker="o", label=MODEL_LABELS[model],
    )
  axis.set(
      xlabel="Equal residual strength",
      ylabel="Held-out absolute return",
      title="SANITY CHECK — frozen D equal-severity diagonal",
  )
  axis.legend(fontsize=7)
  figure.tight_layout()
  figure.savefig(output, dpi=160, metadata={"Title": "sanity check D diagonal"})
  plt.close(figure)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--damage-export", required=True)
  parser.add_argument("--control-export", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  damage = Path(args.damage_export).resolve()
  controls = Path(args.control_export).resolve()
  output = Path(args.output).resolve()
  output.mkdir(parents=True, exist_ok=False)
  _learning_plot(
      controls / "learning_curves_healthy.csv",
      output / "sanity_learning_curves_healthy.png",
      "SANITY CHECK — healthy-only learning curves",
  )
  _learning_plot(
      damage / "learning_curves_damage.csv",
      output / "sanity_learning_curves_damage.png",
      "SANITY CHECK — frozen D damage-curriculum learning curves",
  )
  _diagonal_plot(
      damage / "damage_diagonal.csv",
      output / "sanity_damage_diagonal.png",
  )


if __name__ == "__main__":
  main()
