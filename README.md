# Go1 Double-Actuator 2×2 Study — Research Repository

Companion repository for the paper on morphology-structured policies for
compound actuator damage generalization on the Unitree Go1 quadruped.

## Overview

This repository contains the experiment code, the frozen model library, and the
paper's result tables and figures for the **double-actuator 2×2 study**:

| | Healthy evaluation | Held-out compound-damage evaluation |
|---|---|---|
| Healthy-only training | **Cell A** (nominal capacity) | **Cell B** (zero-shot robustness, H → D) |
| Damage curriculum | **Cell C** (damage-trained nominal) | **Cell D** (learned damage generalization, D → D) |

Cell D is the frozen primary experiment. Cells A–C are descriptive post-D
controls.

## Repository layout

```
.
├── README.md
├── code/
│   ├── go1-morphology-core/        # frozen model library (v1.0.0)
│   │   ├── src/go1_core/           # architectures, topology, PPO networks
│   │   ├── tests/
│   │   └── pyproject.toml
│   └── go1-robot-experiments-2x2/  # experiment code
│       ├── experiments/double_actuator/  # protocol, training, evaluation, analysis
│       ├── jobs/                   # SLURM job scripts
│       ├── src/go1_robot_experiments/    # shared rollout/protocol code
│       └── pyproject.toml
└── results/
    ├── 2x2_metrics.csv             # per-seed plain averages
    ├── 2x2_metrics_summary.csv     # mean ± SD across 3 seeds
    ├── 2x2_metrics_table.{md,tex}  # paper-ready tables
    ├── learning_curves_*.{csv,png,pdf}
    ├── damage_diagonal.{csv,png,pdf}      # D → D severity sweep
    ├── h_to_d_diagonal.{csv,png,pdf}      # B → D severity sweep
    ├── h_to_d_vs_d_to_d_diagonal.{png,pdf}
    ├── README.md                   # results-specific methodology
    └── regenerate_paper_v3.py      # script that produced results/
```

## Models

Five capacity-matched actors (all within 0.5% of 171,296 parameters):

| Label | Canonical name | Family | Params |
|---|---|---|---:|
| MLP-C | `native_mlp_w480_240_120` | centralized MLP | 170,784 |
| MLP-D | `pointwise_w252` | decentralized (joint) MLP | 171,478 |
| GCN | `gcn_w1107` | graph convolution | 171,361 |
| Hodge-L | `multirank_hodge_lower_w177` | rank-1 Hodge | 171,705 |
| Hodge-F | `multirank_hodge_full_w150` | rank-2 Hodge | 171,296 |

Structured actors (MLP-D, GCN, Hodge-L, Hodge-F) share a frozen
simplicial-complex topology of the Go1's 12 joints (16 edges, 5 faces) with
incidence matrices `B1` and `B2`. Hodge models propagate over rank-0/1 (L) and
rank-0/1/2 (F) cochains.

## Metrics

All results are **plain arithmetic averages** (not AUCs):

- **Return**: undiscounted episode return
- **Survival**: fraction of the 1,000-step horizon standing
- **Vel RMSE**: velocity tracking error
- **Yaw RMSE**: yaw-rate tracking error
- **Power**: absolute mechanical power

Cells A/C average 90 healthy rows (9 commands × 10 reset keys).
Cells B/D average 160,380 held-out rows (22 pairs × 9×9 strengths × 9 commands
× 10 reset keys). Three independent training seeds: 11, 12, 13.

## Reproducing

### Setup

Python 3.12 and `uv` are required.

```bash
cd code/go1-robot-experiments-2x2
uv sync --frozen
```

Install the model library (frozen at v1.0.0):

```bash
pip install ../go1-morphology-core
# or from the frozen tag:
# pip install "go1-morphology-core @ git+ssh://git@github.com/dh-you/go1-morphology-core.git@v1.0.0"
```

Link the pinned MuJoCo Menagerie assets:

```bash
uv run python scripts/link_pinned_menagerie.py \
  --source /path/to/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie \
  --replace-installed-copy
```

### Training

Training requires a clean committed checkout; outputs are bound to the exact
experiment commit. Example for Cell D (damage curriculum):

```bash
export GO1_EXPERIMENT_OUTPUT_ROOT=/scratch/network/$USER/go1-results
python -m experiments.double_actuator.train \
  --output-root "$GO1_EXPERIMENT_OUTPUT_ROOT" \
  --model multirank_hodge_full_w150 \
  --seed 11 \
  --num-timesteps 400000000 \
  --num-evals 19 \
  --evaluation-grid full \
  --require-gpu
```

### Evaluation and analysis

Held-out damage evaluation (per shard):

```bash
python -m experiments.double_actuator.evaluate \
  --training-root "$GO1_EXPERIMENT_OUTPUT_ROOT/double-actuator" \
  --model multirank_hodge_full_w150 --seed 11 --shard 0 \
  --training-commit 9a125fd245874b1d3be50238f3cdddc510832ce0 --require-gpu
```

The `results/` folder was produced from the certified artifact tree using
`regenerate_paper_v3.py`. See `results/README.md` for methodology details and
the artifact-root paths.

## Key results (Cell D, plain means across 3 seeds)

| Model | Return ↑ | Survival ↑ | Vel RMSE ↓ | Yaw RMSE ↓ | Power |
|---|---|---|---|---|---|
| MLP-C | 19.98 | 0.88 | 0.32 | 0.30 | 13.34 |
| MLP-D | 18.15 | 0.82 | 0.32 | 0.31 | 14.52 |
| GCN | 15.35 | 0.73 | 0.34 | 0.51 | 7.41 |
| Hodge-L | 18.61 | 0.83 | 0.32 | 0.41 | 9.79 |
| Hodge-F | **22.40** | **0.94** | **0.25** | **0.16** | 31.22 |

Hodge-F (rank-2 Hodge) dominates every robustness metric at the cost of ~3×
the mechanical power of the most efficient baseline (GCN).

## Provenance

- Experiment code commit: `06a4ce8…` (`Add certified double-actuator 2x2 controls`)
- Cell D training commit: `9a125fd…`
- Cell D evaluation commit: `6b0b315…`
- Model library: `go1-morphology-core` v1.0.0 (commit `5c02a404…`)
- Certified artifact root:
  `/scratch/network/dy0130/go1-robot-experiments-2x2-artifacts/06a4ce8c1c8cc26ff8d304eaf395b4f0c2a8bb8a/`

## License

See `code/go1-morphology-core/LICENSE` and `code/go1-robot-experiments-2x2/LICENSE`
(if present) for code licensing. Figures and tables in `results/` are available
under the paper's terms.
