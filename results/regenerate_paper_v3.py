"""Regenerate 2x2 paper deliverables for v3.

Produces:
  - 2x2 plain-average metrics (Return, Survival, Vel RMSE, Yaw RMSE, Power)
    for cells A, B, C, D across MLP-C, MLP-D, GCN, Hodge-L, Hodge-F
    (CSV + tex + md)
  - healthy and damage learning curves (CSV + PNG + PDF)
  - cell-D return-along-actuator-strength (diagonal) chart (CSV + PNG + PDF)

Run with the analysis venv:
  /scratch/network/dy0130/go1-robot-experiments/.venv/bin/python regenerate_paper_v3.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path("/scratch/network/dy0130/go1-robot-experiments-2x2-artifacts")
EXP = ARTIFACTS / "06a4ce8c1c8cc26ff8d304eaf395b4f0c2a8bb8a"
OUT = ARTIFACTS / "double-actuator-2x2-paper-v3"

D_ORIGINAL = Path(
    "/scratch/network/dy0130/go1-robot-experiments-artifacts"
    "/9a125fd245874b1d3be50238f3cdddc510832ce0/double-actuator"
)
D_RETRY = Path(
    "/scratch/network/dy0130/go1-robot-experiments-artifacts-retries/3348141-timeout"
    "/9a125fd245874b1d3be50238f3cdddc510832ce0/double-actuator"
)

MODELS = {
    "native_mlp_w480_240_120": "MLP-C",
    "pointwise_w252": "MLP-D",
    "gcn_w1107": "GCN",
    "multirank_hodge_lower_w177": "Hodge-L",
    "multirank_hodge_full_w150": "Hodge-F",
}
SEEDS = (11, 12, 13)

METRICS = [
    ("undiscounted_return", "Return"),
    ("survival", "Survival"),
    ("velocity_rmse", "Vel RMSE"),
    ("yaw_rmse", "Yaw RMSE"),
    ("absolute_mechanical_power", "Power"),
]
COLUMNS = [metric for metric, _ in METRICS]

CELLS = {
    "A": ("healthy_only", "nominal"),
    "B": ("healthy_only", "heldout"),
    "C": ("damage_curriculum", "nominal"),
    "D": ("damage_curriculum", "heldout"),
}

STRENGTHS = (0.50, 0.25, 0.20, 0.15, 0.10, 0.075, 0.05, 0.025, 0.0)

LINE_STYLES = {
    "MLP-C": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "MLP-D": {"color": "#ff7f0e", "linestyle": "--", "marker": "v"},
    "GCN": {"color": "#d62728", "linestyle": "--", "marker": "s"},
    "Hodge-L": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    "Hodge-F": {"color": "#9467bd", "linestyle": "-", "marker": "D"},
}
MODEL_ORDER = ("MLP-C", "MLP-D", "GCN", "Hodge-L", "Hodge-F")


def cell_d_heldout(model: str, seed: int) -> Path:
  root = D_RETRY if model == "multirank_hodge_full_w150" else D_ORIGINAL
  return (
      root / model / f"seed-{seed}" / "steps-400000000" / "evals-19"
      / "heldout" / "condition_metrics.csv"
  )


def cell_metrics_csv(cell: str, model: str, seed: int) -> Path:
  if cell in ("A", "C"):
    sub = "nominal"
    root = EXP / f"double-actuator-2x2/cell-{cell}"
    return (
        root / model / f"seed-{seed}" / "steps-400000000" / "evals-19"
        / sub / "condition_metrics.csv"
    )
  if cell == "B":
    root = EXP / "double-actuator-2x2/cell-B"
    return (
        root / model / f"seed-{seed}" / "steps-400000000" / "evals-19"
        / "heldout" / "condition_metrics.csv"
    )
  return cell_d_heldout(model, seed)


def read_metrics(path: Path) -> list[dict[str, float]]:
  rows = []
  with path.open(newline="") as handle:
    for row in csv.DictReader(handle):
      rows.append({col: float(row[col]) for col in COLUMNS})
  return rows


def plain_average(rows: list[dict[str, float]], column: str) -> float:
  return sum(row[column] for row in rows) / len(rows)


def aggregate(values: list[float]) -> dict[str, float]:
  return {
      "mean": statistics.fmean(values),
      "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
      "min": min(values),
      "max": max(values),
  }


def progress_rewards(progress_path: Path) -> list[tuple[int, float]]:
  points = []
  with progress_path.open() as handle:
    for line in handle:
      record = json.loads(line)
      step = int(record["step"])
      reward = float(record["metrics"]["eval/episode_reward"])
      points.append((step, reward))
  return points


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(header)
    writer.writerows(rows)


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)

  # ------------------------------------------------------------------ metrics
  seed_rows = []
  summaries = {cell: {label: {} for label in MODEL_ORDER} for cell in CELLS}
  for cell in CELLS:
    for model, label in MODELS.items():
      per_metric = {metric: [] for metric, _ in METRICS}
      for seed in SEEDS:
        path = cell_metrics_csv(cell, model, seed)
        if not path.exists():
          raise FileNotFoundError(f"missing metrics: {path}")
        rows = read_metrics(path)
        if cell in ("A", "C") and len(rows) != 90:
          raise ValueError(f"nominal row count {len(rows)} != 90: {path}")
        if cell in ("B", "D") and len(rows) != 160380:
          raise ValueError(f"heldout row count {len(rows)} != 160380: {path}")
        values = {metric: plain_average(rows, metric) for metric, _ in METRICS}
        for metric, _ in METRICS:
          per_metric[metric].append(values[metric])
        seed_rows.append([cell, model, label, seed] + [values[m] for m, _ in METRICS])
      summaries[cell][label] = {
          metric: aggregate(values) for metric, values in per_metric.items()
      }

  header = ["cell", "model", "model_label", "seed"] + [m for m, _ in METRICS]
  write_csv(OUT / "2x2_metrics.csv", header, seed_rows)

  summary_rows = []
  for cell in CELLS:
    for label in MODEL_ORDER:
      row = [cell, label]
      for metric, _ in METRICS:
        agg = summaries[cell][label][metric]
        row += [f"{agg['mean']:.6g}", f"{agg['sd']:.6g}", f"{agg['min']:.6g}", f"{agg['max']:.6g}"]
      summary_rows.append(row)
  summary_header = ["cell", "model"]
  for metric, _ in METRICS:
    summary_header += [f"{metric}_mean", f"{metric}_sd", f"{metric}_min", f"{metric}_max"]
  write_csv(OUT / "2x2_metrics_summary.csv", summary_header, summary_rows)

  # LaTeX tables: one per cell, model rows, metric columns.
  tex_parts = []
  md_parts = []
  for cell in CELLS:
    tex_rows = []
    md_rows = []
    for label in MODEL_ORDER:
      cells = []
      for metric, _ in METRICS:
        agg = summaries[cell][label][metric]
        cells.append(f"${agg['mean']:.2f} \\pm {agg['sd']:.2f}$")
      tex_rows.append(f"{label} & " + " & ".join(cells) + r" \\")
      md_rows.append(f"| {label} | " + " | ".join(f"{agg['mean']:.2f} \u00b1 {agg['sd']:.2f}" for agg in (summaries[cell][label][m] for m, _ in METRICS)) + " |")
    metric_header = " & ".join(m for _, m in METRICS)
    tex_parts.append(
        "\\begin{tabular}{l" + "r" * len(METRICS) + "}\n\\toprule\n"
        + f"Model & {metric_header} \\\\\n\\midrule\n"
        + "\n".join(tex_rows)
        + "\n\\bottomrule\n\\end{tabular}"
    )
    md_parts.append(
        f"## Cell {cell}\n\n| Model | "
        + " | ".join(m for _, m in METRICS)
        + " |\n|---|---" + "---|" * len(METRICS)
        + "\n" + "\n".join(md_rows)
    )
  (OUT / "2x2_metrics_table.tex").write_text(
      "\n\n".join(f"% Cell {cell}\n{part}" for cell, part in zip(CELLS, tex_parts))
  )
  (OUT / "2x2_metrics_table.md").write_text("\n\n".join(md_parts))

  # ----------------------------------------------------------- learning curves
  def learning_curve_csv(progress_root_fn, label, out_name):
    rows = []
    for model, mlabel in MODELS.items():
      for seed in SEEDS:
        path = progress_root_fn(model, seed)
        if not path.exists():
          raise FileNotFoundError(f"missing progress: {path}")
        for step, reward in progress_rewards(path):
          rows.append([model, mlabel, seed, step, reward])
    write_csv(OUT / out_name, ["model", "model_label", "seed", "environment_steps", "eval_episode_reward"], rows)

  def _load_learning_data(csv_path):
    data = {}
    with csv_path.open(newline="") as handle:
      for row in csv.DictReader(handle):
        per_seed = data.setdefault(row["model_label"], {})
        per_seed.setdefault(int(row["seed"]), []).append(
            (int(row["environment_steps"]), float(row["eval_episode_reward"])))
    return data

  def _panel_curves(axis, data):
    max_step = 0
    for label in MODEL_ORDER:
      seeds = data[label]
      steps = sorted({step for seed in seeds for step, _ in seeds[seed]})
      means = [statistics.fmean(dict(seeds[seed])[step] for seed in seeds) for step in steps]
      sds = [statistics.stdev(dict(seeds[seed])[step] for seed in seeds) if len(seeds) > 1 else 0.0 for step in steps]
      style = LINE_STYLES[label]
      axis.plot(steps, means, **style, label=label, lw=1.8, ms=4)
      axis.fill_between(steps, np.asarray(means) - np.asarray(sds), np.asarray(means) + np.asarray(sds),
                        color=style["color"], alpha=0.15)
      max_step = max(max_step, max(steps))
    return max_step

  def plot_learning_curve(csv_path, out_name, title, ylabel):
    data = _load_learning_data(csv_path)
    with plt.rc_context({
        "font.size": 28,
        "axes.labelsize": 30,
        "axes.titlesize": 32,
        "legend.fontsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
    }):
      figure, axis = plt.subplots(figsize=(11.2, 7.0), constrained_layout=True)
      max_step = _panel_curves(axis, data)
      axis.set_xlabel("Environment steps")
      axis.set_ylabel(ylabel)
      axis.set_title(title)
      axis.legend(frameon=False)
      axis.set_xlim(0, max_step)
      axis.margins(x=0.02)
      figure.savefig(OUT / f"{out_name}.png", dpi=220)
      figure.savefig(OUT / f"{out_name}.pdf")
      plt.close(figure)

  def plot_learning_curve_combined(csv_paths, titles, out_name, ylabel):
    panels = [_load_learning_data(path) for path in csv_paths]
    with plt.rc_context({
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "legend.fontsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    }):
      figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True, sharex=True)
      for axis, data, title in zip(axes, panels, titles):
        max_step = _panel_curves(axis, data)
        axis.set_title(title)
        axis.set_xlabel("Environment steps")
        axis.set_xlim(0, max_step)
        axis.margins(x=0.02)
        axis.ticklabel_format(style="sci", axis="x", scilimits=(8, 8))
      axes[0].set_ylabel(ylabel)
      axes[1].legend(frameon=False, loc="lower right")
      figure.savefig(OUT / f"{out_name}.png", dpi=300)
      figure.savefig(OUT / f"{out_name}.pdf")
      plt.close(figure)

  healthy_progress = lambda model, seed: (
      EXP / "double-actuator-healthy" / model / f"seed-{seed}"
      / "steps-400000000" / "evals-19" / "progress.jsonl"
  )
  damage_progress = lambda model, seed: (
      (D_RETRY if model == "multirank_hodge_full_w150" else D_ORIGINAL)
      / model / f"seed-{seed}" / "steps-400000000" / "evals-19" / "progress.jsonl"
  )

  learning_curve_csv(healthy_progress, "healthy", "learning_curves_healthy.csv")
  plot_learning_curve(OUT / "learning_curves_healthy.csv", "learning_curves_healthy",
                      "Healthy regime", "Eval episode reward")

  learning_curve_csv(damage_progress, "damage", "learning_curves_damage.csv")
  plot_learning_curve(OUT / "learning_curves_damage.csv", "learning_curves_damage",
                      "Degraded regime", "Eval episode reward")

  plot_learning_curve_combined(
      [OUT / "learning_curves_healthy.csv", OUT / "learning_curves_damage.csv"],
      ["Healthy regime", "Degraded regime"],
      "learning_curves_combined",
      "Eval episode reward",
  )

  def cell_b_heldout(model: str, seed: int) -> Path:
    return (
        EXP / "double-actuator-2x2/cell-B" / model / f"seed-{seed}"
        / "steps-400000000" / "evals-19" / "heldout" / "condition_metrics.csv"
    )

  def compute_diagonal(path_fn, name, title, csv_name, png_name):
    diagonal = {label: [] for label in MODEL_ORDER}
    diagonal_rows = []
    for model, label in MODELS.items():
      per_strength = {strength: [] for strength in STRENGTHS}
      for seed in SEEDS:
        path = path_fn(model, seed)
        if not path.exists():
          raise FileNotFoundError(f"missing diagonal source: {path}")
        with path.open(newline="") as handle:
          for row in csv.DictReader(handle):
            a = float(row["strength_a"])
            b = float(row["strength_b"])
            if a == b:
              per_strength[a].append(float(row["undiscounted_return"]))
      for strength in STRENGTHS:
        values = per_strength[strength]
        if len(values) != 22 * 9 * 10 * 3:
          raise ValueError(f"diagonal count {len(values)} unexpected for {label} @ {strength}")
        agg = aggregate(values)
        diagonal[label].append((strength, agg["mean"], agg["sd"]))
        diagonal_rows.append([model, label, strength, f"{agg['mean']:.6g}", f"{agg['sd']:.6g}"])
    write_csv(OUT / csv_name,
              ["model", "model_label", "strength", "mean_return", "sd_return"],
              diagonal_rows)

    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for label in MODEL_ORDER:
      strengths = [point[0] for point in diagonal[label]]
      means = [point[1] for point in diagonal[label]]
      sds = [point[2] for point in diagonal[label]]
      style = LINE_STYLES[label]
      axis.plot(strengths, means, **style, label=label, lw=1.8, ms=4)
      axis.fill_between(strengths, np.asarray(means) - np.asarray(sds), np.asarray(means) + np.asarray(sds),
                        color=style["color"], alpha=0.15)
    axis.set_xlabel("Residual actuator strength (both actuators)")
    axis.set_ylabel("Mean undiscounted return")
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.invert_xaxis()
    axis.set_xlim(0.5, 0.0)
    figure.savefig(OUT / f"{png_name}.png", dpi=220)
    figure.savefig(OUT / f"{png_name}.pdf")
    plt.close(figure)
    return diagonal

  # Cell D diagonal
  diagonal_d = compute_diagonal(
      cell_d_heldout,
      "D",
      "Cell D: damage-trained, held-out damage severity sweep",
      "damage_diagonal.csv",
      "damage_diagonal",
  )

  # Cell B diagonal (H -> D)
  diagonal_b = compute_diagonal(
      cell_b_heldout,
      "B",
      "Cell B: healthy-trained, held-out damage severity sweep (H→D)",
      "h_to_d_diagonal.csv",
      "h_to_d_diagonal",
  )

  # Combined B vs D comparison figure
  figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
  for ax, diagonal, title in zip(axes, [diagonal_b, diagonal_d], ["H → D (Cell B)", "D → D (Cell D)"]):
    for label in MODEL_ORDER:
      strengths = [point[0] for point in diagonal[label]]
      means = [point[1] for point in diagonal[label]]
      sds = [point[2] for point in diagonal[label]]
      style = LINE_STYLES[label]
      ax.plot(strengths, means, **style, label=label, lw=1.8, ms=4)
      ax.fill_between(strengths, np.asarray(means) - np.asarray(sds), np.asarray(means) + np.asarray(sds),
                      color=style["color"], alpha=0.15)
    ax.set_xlabel("Residual actuator strength (both actuators)")
    ax.set_ylabel("Mean undiscounted return")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.set_xlim(0.5, 0.0)
  axes[1].legend(frameon=False, loc="lower left")
  figure.savefig(OUT / "h_to_d_vs_d_to_d_diagonal.png", dpi=220)
  figure.savefig(OUT / "h_to_d_vs_d_to_d_diagonal.pdf")
  plt.close(figure)

  print(f"outputs written to {OUT}")


if __name__ == "__main__":
  main()
