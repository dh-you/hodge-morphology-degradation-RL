# Higher-Order Morphology Priors for Quadruped RL Under Actuator Degradation

[Derek You](https://dh-you.github.io/)<sup>1</sup>, [Zafir Shamsi](https://www.zshamsi.com/)<sup>2</sup>, [Keqin Wang](https://keqinwang.github.io/)<sup>3</sup>, [Christine Allen-Blanchette](https://www.christineallenblanchette.com/)<sup>3</sup>

<sup>1</sup>Edison Academy Magnet School &nbsp;&middot;&nbsp; <sup>2</sup>South Brunswick High School &nbsp;&middot;&nbsp; <sup>3</sup>Princeton University

**IROS Workshop on Biologically Inspired Locomotion, Planning, and Control (BLPC) 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2610.10934-b31b1b)](https://arxiv.org/abs/2610.10934)
[![Project Page](https://img.shields.io/badge/Project-Page-4488ff)](https://dh-you.github.io/hodge-morphology-degradation-RL/)
[![Code](https://img.shields.io/badge/Code-GitHub-333)](https://github.com/dh-you/hodge-morphology-degradation-RL)

## Abstract

Actuator degradation turns quadruped locomotion into a coordination problem requiring joints to compensate for lost actuation. Prior work suggests that morphology-aware graph policies improve learning and generalization under body perturbations. We ask whether these benefits can be strengthened by explicitly modeling higher-order mechanical structure. We represent the Unitree Go1 as a cell complex with limb- and body-level rank-2 cells and apply Hodge-based message passing. Under degradation training, the node-edge-face Hodge actor achieves the highest return on unseen actuator degradations, with higher survival and lower velocity-tracking error. These results support higher-order morphology as a useful inductive bias for whole-body compensation under actuator degradation.

## Quick Start

Training requires Python 3.12 and `uv`:

```bash
cd code/go1-robot-experiments-2x2
uv sync --frozen
pip install ../go1-morphology-core
```

Link pinned MuJoCo Menagerie assets:

```bash
uv run python scripts/link_pinned_menagerie.py \
  --source /path/to/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie \
  --replace-installed-copy
```

Full training and evaluation instructions are in the [README archive](#repository-layout).

## Repository Layout

| Path | Purpose |
|---|---|
| `code/go1-morphology-core/` | Frozen model library (v1.0.0) — architectures, topology, PPO networks |
| `code/go1-robot-experiments-2x2/` | Experiment code — training, evaluation, analysis |
| `results/` | Paper-ready tables, figures, learning curves, severity sweeps |

See `results/README.md` for methodology and `code/go1-robot-experiments-2x2/README.md` for detailed reproduce steps.

## Key Results (Cell D)

| Model | Return | Survival | Vel RMSE | Yaw RMSE | Power |
|---|---|---|---|---|---|
| MLP-C | 19.98 | 0.88 | 0.32 | 0.30 | 13.34 |
| MLP-D | 18.15 | 0.82 | 0.32 | 0.31 | 14.52 |
| GCN | 15.35 | 0.73 | 0.34 | 0.51 | 7.41 |
| Hodge-L | 18.61 | 0.83 | 0.32 | 0.41 | 9.79 |
| **Hodge-F** | **22.40** | **0.94** | **0.25** | **0.16** | 31.22 |

Hodge-F (rank-2 Hodge) dominates every robustness metric at ~3x the power of GCN.

## Citation

```bibtex
@article{you2026higherorder,
  title={Higher-Order Morphology Priors for Quadruped Reinforcement Learning Under Actuator Degradation},
  author={You, Derek and Shamsi, Zafir and Wang, Keqin and Allen-Blanchette, Christine},
  journal={arXiv preprint arXiv:2610.10934},
  year={2026},
  note={Accepted to IROS Workshop BLPC 2026}
}
```

## License

See `code/go1-morphology-core/LICENSE` and `code/go1-robot-experiments-2x2/LICENSE` (if present) for code licensing. Figures and tables in `results/` are available under the paper's terms.
