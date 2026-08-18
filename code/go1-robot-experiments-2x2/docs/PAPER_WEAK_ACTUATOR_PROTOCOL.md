# Paper-ready weak-actuator protocol

This document freezes the design before the new paper campaign is submitted.
It is additive to the scientific protocol at commit
`771c3d73b1fb2a97a417eb58a945266c1770cd65`; the training runner,
weak-actuator curriculum, evaluator, pinned dependencies, and core models are
not changed.

## Studies and roster

The main architecture study contains six models and seeds 1 through 5:

| Paper label | Core model ID | Actor parameters | Scientific role |
|---|---|---:|---|
| Native MLP | `native_mlp` | 192,408 | over-capacity centralized reference |
| Central MLP | `mlp_match_lower` | 93,162 | matched unstructured baseline |
| Joint MLP | `pointwise_match_lower` | 93,244 | matched joint-factorized baseline |
| GCN | `gcn_match_lower` | 92,987 | matched pairwise baseline |
| Hodge-L | `multirank_hodge_lower` | 92,962 | lower-order ablation |
| Hodge-F | `multirank_hodge_full` | 126,570 | proposed rank-2 model |

The extended Hodge study contains Hodge-L and Hodge-F at paired seeds 1
through 10. Hodge seeds 1 through 5 are reused, byte-for-byte, from commit
`771c3d73b1fb2a97a417eb58a945266c1770cd65`. New work consists of the four
non-Hodge baselines at seeds 1 through 5 and both Hodge models at seeds 6
through 10: 30 new runs and 40 unique policies overall.

Native MLP is an over-capacity reference, not a parameter-matched control for
the Hodge-F face pathway. A result may support the bounded statement that
higher generic capacity alone is insufficient, but cannot establish that the
additional Hodge-F parameters are irrelevant.

## Frozen execution

Every new run uses the existing combined train, terminal-checkpoint restore,
and full-evaluation runner with:

- 400,000,000 requested and 412,876,800 realized environment steps;
- 19 evaluations and terminal checkpoint `000412876800`;
- 109 conditions, 9 commands, 10 reset keys, and a 1,000-step horizon;
- the exact core, Playground, Menagerie, PPO configuration, and curriculum
  identities already guarded by the runner;
- one A100 `3g.20gb`, 11 CPUs, 128 GB memory, and a four-hour Slurm limit.

Seed is the experimental unit. The 9,810 evaluation rows within a seed are
repeated measurements. A valid low-performing seed is never rerun. Technical
failures may be retried only under a separate artifact root recorded in the
final manifest; the canonical failed artifact is retained.

## Endpoints and inference

The frozen primary endpoint is seed-level degraded-strength absolute-return
AUC. For each seed and strength, the return is macro-averaged over 12
actuators, 9 commands, and 10 reset keys. Normalized trapezoidal integration
uses strengths `0, .025, .05, .075, .1, .15, .2, .25, .5`; healthy is
excluded.

The six-model study is descriptive: every seed is shown, with the mean,
sample standard deviation, and 95% t interval. It has no omnibus or pairwise
hypothesis tests and makes no universal-winner claim.

The Hodge study reports Hodge-F minus Hodge-L paired differences over seeds 1
through 10, a 95% paired-t confidence interval, Cohen's d-z, and an exact
two-sided sign-flip test over all 1,024 assignments. The primary test uses
alpha 0.05. Five secondary AUC tests form one Holm-corrected family: survival,
return retention, fall rate, velocity RMSE, and yaw RMSE. Collapse threshold
and physical-effort outcomes are descriptive.

The collapse threshold is the lowest degraded strength in the descending
sequence from 0.5 to 0.0 for which that strength and every less-degraded
strength satisfy return retention at least 0.90, survival at least 0.95, and
fall rate at most 0.05. It is absent if strength 0.5 fails and is 0.0 if every
degraded strength passes.

**The Hodge seed study tests the pre-existing hypothesis that Hodge-L and
Hodge-F differ in seed-level actuator-robust locomotion outcomes.
Degraded-strength absolute-return AUC was already the frozen primary
robustness endpoint, so formal paired inference uses all seeds 1-10. The
observed plateau-versus-high-compensation behavior motivates a separate
mechanistic analysis of locomotion discovery; any categorical breakthrough
criterion defined after inspecting seeds 1-5 is labeled exploratory unless
frozen before evaluating seeds 6-10.**

No categorical breakthrough rule is prespecified here. Any such analysis
created from the observed data is therefore exploratory.

## Acceptance and outputs

Aggregation is all-or-nothing. It rejects missing or duplicate runs, failed
or incomplete records, wrong commits or run identities, non-finite values,
incorrect step or row counts, malformed ordering, failed checkpoint
roundtrips, changed scientific files, and unregistered retry paths.

The final paper pack contains input manifests and hashes, run/seed/model
tables, severity/command/joint summaries, paired Hodge inference, Markdown
and LaTeX tables, and deterministic PDF/220-dpi PNG figures. Individual seed
points remain visible. Machine-readable outputs must reproduce byte-for-byte
when generated twice from the same manifest.
