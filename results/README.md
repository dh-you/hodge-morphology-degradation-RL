# Double-actuator 2×2 paper package (v3)

Generated 2026-08-16 from the certified artifact tree at
`/scratch/network/dy0130/go1-robot-experiments-2x2-artifacts/06a4ce8c1c8cc26ff8d304eaf395b4f0c2a8bb8a`.

## Design

The 2×2 study crosses training regime with evaluation domain:

| | Healthy evaluation | Held-out compound-damage evaluation |
|---|---|---|
| Healthy-only training | **Cell A** | **Cell B** (H → D) |
| Damage curriculum | **Cell C** | **Cell D** (D → D) |

Cells A, B, and C are descriptive post-D controls. Cell D is the frozen primary
experiment, trained at `9a125fd…` and evaluated at `6b0b315…`. For Cell D,
Hodge-F uses the officially registered timeout-retry lineage
(`artifacts-retries/3348141-timeout`) as documented in the frozen
`damage_manifest.json`.

## Models

Five capacity-matched baselines are included:

| Label | Canonical name | Family | Params |
|---|---|---|---|
| MLP-C | `native_mlp_w480_240_120` | centralized MLP | 170,784 |
| MLP-D | `pointwise_w252` | decentralized (joint) MLP | 171,478 |
| GCN | `gcn_w1107` | graph convolution | 171,361 |
| Hodge-L | `multirank_hodge_lower_w177` | rank-1 Hodge | 171,705 |
| Hodge-F | `multirank_hodge_full_w150` | rank-2 Hodge | 171,296 |

All models are within 0.5% of the 171,296 target capacity.

## Metrics

All reported numbers are **plain arithmetic averages**, not AUCs or normalized
integrals. Columns in `condition_metrics.csv` used:

- **Return**: `undiscounted_return`
- **Survival**: `survival` — fraction of the 1,000-step horizon that the policy
  remained standing (not 0/1)
- **Vel RMSE**: `velocity_rmse`
- **Yaw RMSE**: `yaw_rmse`
- **Power**: `absolute_mechanical_power`

Cell A/C averages are over 90 healthy rows (9 commands × 10 reset keys).
Cell B/D averages are over 160,380 held-out rows
(22 pairs × 9 × 9 strengths × 9 commands × 10 reset keys).

Because survival is averaged as a fraction rather than integrated via the
frozen normalized-survival-AUC, these survival numbers differ from the earlier
paper table (e.g. Cell-D Hodge-F: 0.94 plain mean vs. 0.98 normalized AUC).

## Figures

Figures are rendered at 10 pt in a Times-compatible serif font
(`Nimbus Roman`). True Times New Roman is not installed in this headless Linux
environment; Nimbus Roman is the standard metric-compatible substitute and is
indistinguishable in print.

## Files

- `2x2_metrics.csv` — per-seed plain averages
- `2x2_metrics_summary.csv` — mean ± sd (across 3 seeds) per cell/model
- `2x2_metrics_table.md` — markdown summary tables
- `2x2_metrics_table.tex` — LaTeX summary tables
- `learning_curves_healthy.csv` + `.png` + `.pdf` — healthy-only training curves
- `learning_curves_damage.csv` + `.png` + `.pdf` — damage-curriculum training curves
- `damage_diagonal.csv` + `.png` + `.pdf` — Cell-D return along the equal-severity diagonal
- `h_to_d_diagonal.csv` + `.png` + `.pdf` — Cell-B (H→D) return along the equal-severity diagonal
- `h_to_d_vs_d_to_d_diagonal.png` + `.pdf` — side-by-side B-vs-D severity sweep
- `regenerate_paper_v3.py` — the script that produced this package

## Method notes

- Three independent seeds: 11, 12, 13.
- Learning curves show mean eval episode reward across seeds with ±1 sample SD
  shaded bands.
- Degradation charts show mean return (±1 sample SD) along `strength_a == strength_b`.
- Cell-B diagonal is the zero-shot robustness sweep (healthy-trained policy,
  held-out compound damage).
- Cell-D diagonal is the learned-damage-generalization sweep (damage-trained
  policy, held-out compound damage).
