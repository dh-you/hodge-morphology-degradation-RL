# Full 2x2 Double-Actuator Study

## Status and precedence

Cell D is the frozen primary experiment. It was trained at commit
\`9a125fd245874b1d3be50238f3cdddc510832ce0\` and evaluated at commit
\`6b0b315743a9b10042d8c9fe5993f7f18011c3fd\`. Its original artifact and
retry roots are immutable. A, B, and C are post-D descriptive controls and
must not be represented as preregistered confirmatory evidence.

| Training regime | Healthy evaluation | Held-out compound-damage evaluation |
|---|---|---|
| Healthy only | A: nominal capacity | B: zero-shot robustness |
| Damage curriculum | C: damage-trained nominal performance | D: learned damage generalization |

The study uses the five frozen capacity-matched models and seeds 11, 12, and
13. With three seeds, every report shows individual seeds, the mean, sample
standard deviation, minimum, and maximum. No confirmatory significance claim
is made.

## Healthy-only protocol

Healthy training uses the same environment, reward, PPO configuration,
checkpoint schedule, domain-randomization implementation, seeds, model/reset
keys, observation and action contracts, and model configurations as D. The
actuator force-range multipliers remain one. There is no injury category,
actuator-pair, or residual-strength sampling.

Healthy training has its own semantic record and SHA-256. It never writes or
claims D's curriculum hash. For each model and seed, the runner loads D's
saved domain-randomization keys, reconstructs the official randomized model
before D's actuator modification, and requires exact domain-key, model-tree,
and force-range hashes to match the healthy pre-actuator randomized model.

## Evaluation contracts

A and C each contain exactly 90 healthy rows in command/reset order. B uses
the exact frozen D held-out pair manifest, strengths, commands, reset keys,
rollout, and logical row-key order. Each B policy is evaluated in two
immutable 11-pair shards of 80,190 rows. The merger sorts by the frozen
logical key and proves equality to the D row contract; it never concatenates
files.

D is exported read-only. The submission-critical D exports are the 19-point
damage learning curves, equal-severity diagonal, full-surface/dual-dead
summary, and a source evidence manifest with hashes.

## Priorities

P0 comprises C, all healthy training, A, B, the two learning-curve exports,
the D diagonal, A and D summary tables, and reproducible sanity plots. P1
contains the full A-D control tables, severity surfaces, paired descriptive
contrasts, and interaction quantities. P2 contains the [0,0.2]^2 AUC,
target-mixture score, and moderation analyses; all P2 products are explicitly
post-D exploratory. P1 and P2 never delay P0.

Valid low-performing policies remain results. Only documented technical
failures may be retried. The public \`go1_core\` API, canonical models, D
protocol, pair split, PPO semantics, and historical roots are unchanged.
