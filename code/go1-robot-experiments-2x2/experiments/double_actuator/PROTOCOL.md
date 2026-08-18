# Double-Actuator Capacity-Matched Workshop Protocol

Status: frozen before compound-damage results.

This preliminary study trains five noncanonical capacity controls at approximately
171,296 actor parameters using seeds 11, 12, and 13. It does not modify the
single-actuator campaign, canonical core identities, or historical artifacts.

## Models

| Model ID | Actor parameters |
|---|---:|
| `native_mlp_w480_240_120` | 170,784 |
| `pointwise_w252` | 171,478 |
| `gcn_w1107` | 171,361 |
| `multirank_hodge_lower_w177` | 171,705 |
| `multirank_hodge_full_w150` | 171,296 |

Only actor widths change. Depth, topology, operators, features, decoder,
critic, PPO, reward, observation/action contract, and environment remain
unchanged. The core is pinned to experimental commit
`5c02a404f5b059be9ce074b896ea4997ddcf4818`; no release is created.

## Training

All policies use 400,000,000 requested and 412,876,800 actual environment
steps, 19 progress evaluations, 8,192 environments, and terminal checkpoint
`000412876800`. Category probabilities are 30% healthy, 50% dual weak, and
20% dual dead. Dual-weak strengths are sampled independently from
`LogUniform[0.05,0.20)`. Damage pairs are sampled uniformly from the frozen
44-pair training set.

Streams are separated as PPO/model/reset seed `s`, domain-randomization seed
`100000+s`, and pair/injury seed `200000+s`. Initial assignments are identical
across architectures within a seed; policy-dependent trajectories may diverge.
Training uses A100 `3g.20gb` MIG only and is separate from evaluation.

## Pair split and evaluation

The canonical manifest contains all 66 unordered actuator pairs. The frozen
22-pair held-out set has 4 same-limb, 2 body-face, and 16 cross-limb pairs,
with every actuator appearing 3–4 times; the 44-pair complement is used for
training.

The deadline-first evaluator covers only held-out pairs over the complete
ordered 9-by-9 residual-strength surface
`[.50,.25,.20,.15,.10,.075,.05,.025,0]`, nine commands, and ten resets. It
produces 160,380 degraded rows per policy and 2,405,700 rows overall.

Each policy is evaluated in two immutable 11-pair shards. A shard contains
891 pair/strength conditions and 80,190 rows. The two GPU shard jobs use the
same certified terminal checkpoint and evaluator commit. A dependent CPU job
verifies disjoint and complete pair coverage, exact row identities and order,
finite metrics, and shard hashes before writing the canonical 160,380-row
merged artifact. Neither shard may be selected, omitted, or rerun based on its
scientific values.

The primary endpoint is the held-out absolute-return surface AUC: macro-average
return over held-out pairs, commands, and resets at each cell; tensor-product
trapezoidal integration over `[0,.5]^2`; divide by `.25`; exclude healthy
trials.

## Workshop interpretation

> We report preliminary results across three independent training seeds and
> show all seed-level outcomes; given the small sample size, comparisons are
> descriptive.

Report all seed points, mean, sample SD or range, Hodge-F minus Native paired
differences, Hodge-F minus Hodge-L paired differences, and severity surfaces.
Do not perform significance tests or claim confirmed superiority. Seeds 11–13
remain valid if the unchanged study is later extended with seeds 14–17.

Technical seed-10 smokes are excluded from analysis and may test only runtime,
finite outputs, deterministic assignments, checkpointing, and restoration.
Valid low-performing scientific runs are retained. Only documented technical
failures may be retried under a separate provenance identity.
