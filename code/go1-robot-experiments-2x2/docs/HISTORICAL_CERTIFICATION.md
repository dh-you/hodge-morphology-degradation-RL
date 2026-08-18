# Job 3 Provenance

Status: Job 3 is formally blocked by historical GPU non-repeatability.
Job 3A completed with the predeclared classification `inconclusive`.

Scientific interpretation: Job 3A did not establish statistical equivalence
under its intentionally conservative finite-envelope criterion, but it found
no evidence of a systematic difference between the extracted consumer and
frozen evaluator. All deterministic implementation boundaries matched
exactly; unresolved differences arise only inside a GPU execution regime that
is itself non-repeatable. The formal classification remains `inconclusive`,
and the original Job-3 blocker and fixed tolerances remain unchanged.

## Frozen identities

| Component | Identity |
|---|---|
| Historical campaign | `clean-model-weak-actuator-scratch-400m` |
| Historical source commit | `1659809766f6aba43a1074cc332922317e9bec5a` |
| Training array | `3343705` |
| Evaluation oracle array | `3343855` |
| Terminal step | `412876800` |
| Go1 core commit | `652a05e3d7f446d2cd770a8361a5b3dac79e37e8` |
| MuJoCo Playground | `220abb9d6a41eaa258047dacf9ca70a92e008b78` |
| MuJoCo Menagerie | `1b86ece576591213e2b666ebf59508454200ca97` |

Job 5 distribution update: the installed core is now pinned to private GitHub
release commit `debc05563fdc0d8dd1befde4426781ce611c7386`, package version
`1.0.0`. The certified implementation identity above remains
`652a05e3d7f446d2cd770a8361a5b3dac79e37e8`; the release commit changes only
core documentation and packaging metadata. Historical Job 3, Job 3A, and Job
4 records below retain the exact core identity used during their execution.

The packaged core is installed from the exact private GitHub release commit.
Playground is installed from its exact Git commit with CUDA dependencies,
while the installed Menagerie asset directory is a symlink to the clean
pinned local checkout.
Execution verifies `direct_url.json`, source HEADs, worktree cleanliness, and
the installed asset link before loading an environment.

## Source-to-consumer mapping

| Frozen source symbol | Consumer owner |
|---|---|
| `claim1.protocol.COMMANDS`, `RESET_KEYS` | `constants.py` |
| `claim1.evaluation.patch_command` | `protocol.py` |
| `condition_force_ranges`, `fixed_condition_randomize` | `protocol.py` |
| `make_weak_actuator_rollout` and physical diagnostics | `rollout.py` |
| condition/command/joint/strength summaries and AUC | `summaries.py` |
| 400M terminal artifact validation and model routing | `artifacts.py`, through public `go1_core` only |
| terminal evaluation | `evaluation.py` |
| historical comparison and certification | `compare.py` |

No actor, feature builder, topology operator, training loop, or checkpoint
saver is present in the packaged consumer. The consumer imports the public
`go1_core` API for contract, registry, and checkpoint behavior. Job 4's one-run
training and checkpoint-save harness is non-packaged certification machinery
under `certification/job4`.

## Reference fixtures

`src/go1_robot_experiments/reference_manifest.json` records hashes for each
canonical terminal `run.json`, `result.json`, checkpoint network config,
historical `evaluation.json`, and `condition_metrics.csv`. It also records
reset, first-policy, and first-step array shapes, dtypes, and raw-byte hashes.
Those diagnostic hashes are generated once using the frozen worktree and the
canonical `mlp_match_lower` terminal checkpoint; no diagnostic arrays or
historical rollout rows are committed.

CPU diagnostics are exact raw-byte fixtures. Two frozen-source A100 probes
(`3345007` and `3345011`) showed that GPU observation and first-step raw hashes
can vary between otherwise identical executions, while reset keys, command,
force assignment, reset qpos/qvel, policy logits/actions, reward, and done stay
exact. GPU certification therefore gates those stable boundaries exactly and
records shape, dtype, and raw-hash match status for the backend-sensitive
arrays. It does not weaken the rollout metric tolerance or hide the raw drift.

The reset diagnostic covers broadcast reset keys, the stand command, all 109
condition force ranges, qpos/qvel, and both observations. Policy diagnostics
cover the 24 logits and 12 deterministic actions. First-step diagnostics cover
qpos/qvel, both observations, actuator force, reward, and done.

## Certification policy

Discrete row identity, ordering, null masks, termination, active-step, fall,
and survival fields must match exactly. All continuous metrics, summaries, and
degraded-strength AUCs use immutable `rtol=1e-6`, `atol=1e-6`. A mismatch is
localized in this order: reset, first policy output, first simulation step,
then gradual rollout drift. Tolerances are never changed automatically.

The final certification table will be written only after the 109-row smoke and
all six 9,810-row terminal evaluations pass.

## Job 3A differential certification

Status: **inconclusive**. Job 3 remains **blocked**. This result does not alter
the historical CSV oracle or the fixed `rtol=atol=1e-6` Job-3 rule.

Job 3A executed the certification-only harness from consumer implementation
commit `d789ba12a7cdef7d99fce924b660a7722e9b23b2`. The model implementation
came from core commit `652a05e3d7f446d2cd770a8361a5b3dac79e37e8`;
the frozen evaluator came from source commit
`1659809766f6aba43a1074cc332922317e9bec5a`. The terminal checkpoint was
`mlp_match_lower` step `412876800` from training array `3343705`.

| Guarded execution identity | Frozen value |
|---|---|
| Environment | `Go1JoystickFlatTerrain`, `impl="jax"` |
| Checkpoint | `train/array-3343705/mlp_match_lower/seed-0/checkpoints/000412876800` |
| Checkpoint config SHA-256 | `7a5abcbc15eeade04139598e9f92a03c9b08b2744f0d333826714cc3fcc1c99c` |
| Training `run.json` SHA-256 | `5b856060ea4e9296ed71db9980567aafcb7355ac9157f4a918eb2bd53b424e22` |
| Training `result.json` SHA-256 | `98052f368da0734ae654f6c0d59db9f023309f82c0e9b71dd7375370b3d0f158` |
| Playground commit | `220abb9d6a41eaa258047dacf9ca70a92e008b78` |
| Menagerie commit | `1b86ece576591213e2b666ebf59508454200ca97` |
| JAX / JAXlib | `0.11.0` / `0.11.0` |
| CUDA toolkit | `12.8` |
| GPU / driver | NVIDIA A100 80GB PCIe / `610.43.02` |
| Slurm job / node | `3345034` / `adroit-h11g1` |

The consumer, core, frozen source, Playground, and Menagerie worktrees all
passed exact-HEAD and clean-worktree guards inside the GPU process. Checkpoint
validation also required the frozen hashes above, serialized model identity,
historical factory kwargs, actor count, completion record, and exact restore
flags before either evaluator was constructed.

The CPU boundary was exact for all pre-MJX gates, two frozen resets, the
consumer reset, two frozen first steps, the consumer first step, and both
horizon-1 evaluator outputs. Slurm job `3345034` then completed 36 balanced
triplets on `adroit-h11g1`, an NVIDIA A100 80GB PCIe, in 6 minutes 38
seconds. Each of the six F1/F2/N execution orders occurred exactly six times
and every call was synchronized.

| Fixed decision item | Result |
|---|---|
| CPU deterministic boundary | exact pass |
| GPU pre-MJX gates | exact pass |
| Completed triplets | 36 / 36 |
| Predeclared primary fields | 28 / 28 valid |
| Holm-corrected rejections at FWER 0.01 | 0 |
| Primary fields entirely inside global F/F envelopes | 17 / 28 |
| Primary fields with at least one F/N envelope exceedance | 11 / 28 |
| Consumer-only non-finite primary outputs | none |
| Frozen non-finite primary outputs | none |
| Classification | **inconclusive** |

The following table is the complete envelope-failure set. `max F/N` is the
maximum of the 72 individual F1/N and F2/N distances. The envelope is globally
defined by `max F/F` over all 36 blocks; block-local or call-position
envelopes did not affect classification.

| Primary field | max F/F | max F/N | F/N outside | raw p | Holm p |
|---|---:|---:|---:|---:|---:|
| `first_step.qpos` | 0.0154419 | 0.0154831 | 5 | 0.517017 | 1.0 |
| `first_step.qvel` | 3.77542 | 3.78570 | 5 | 0.501132 | 1.0 |
| `first_step.state` | 3.77542 | 3.78570 | 5 | 0.501132 | 1.0 |
| `first_step.actuator_force` | 0.319527 | 0.332222 | 2 | 0.539762 | 1.0 |
| `rollout.velocity_rmse` | 0.272760 | 0.274007 | 1 | 0.877256 | 1.0 |
| `rollout.yaw_rmse` | 1.04795 | 1.05345 | 1 | 0.887831 | 1.0 |
| `rollout.absolute_mechanical_power` | 41.3962 | 43.1237 | 2 | 0.856286 | 1.0 |
| `rollout.survival` | 0.932000 | 0.933000 | 1 | 0.818181 | 1.0 |
| `rollout.saturation_fraction` | 0.00700022 | 0.00766690 | 2 | 0.374683 | 1.0 |
| `rollout.damaged_actuator_force_utilization` | 0.0551379 | 0.0584947 | 2 | 0.0930545 | 1.0 |
| `rollout.damaged_joint_tracking_rmse` | 0.421334 | 0.424271 | 1 | 0.728311 | 1.0 |

The earliest primary divergence for both frozen/frozen and frozen/new was the
GPU-materialized reset `state`; both maxima were one float32-scale increment,
`5.960464477539063e-08`. Reset `qpos`, `qvel`, and `info_rng` were
exact. No pre-MJX exact gate failed, no primary hypothesis rejected, and the
consumer did not introduce non-finite primary values. The predeclared global
envelope rule nevertheless forbids an `indistinguishable` classification
because of the 11 fields above. Failure to reject is not interpreted as
equality.

### Immutable evidence

Artifact root:

```text
/scratch/network/dy0130/go1-robot-experiments-artifacts/
  d789ba12a7cdef7d99fce924b660a7722e9b23b2/job3a/
```

| Evidence file | SHA-256 |
|---|---|
| `cpu/cpu_boundary.json` | `ff6f5bcee93d110fa1979810e0ff8e9737d924124258e996694ffff2ffbb5b92` |
| `cpu/materialized_state/manifest.json` | `4b829be60d68d877ed65c74f9bb702bf2e23f5ce673e8dcc6854b15444bc15cb` |
| `gpu/job3a_gpu.json` | `bb5b5e522b737903eadf784561ce7ce3006c7f80b059fe74a992c900593c3244` |
| `gpu/materialized_state/manifest.json` | `d1555d506f8211f78be7684171e2c032a392d9c48e257c301ca457a4151ac0bc` |
| `gpu/representative_states/reset/F1/manifest.json` | `11cfaad943e930092b28dcb0b78501625d7469dc6d68ceaca37bd5c033eaa7d4` |
| `report/job3a_report.json` | `91342751ef5e3b0171a84d3d339cadd4921bf0ce7eaa4a1c1d732fe49406998b` |

The original Job-3 blocker remains at implementation commit `2a52eb7293...`;
its external `certification/BLOCKER.json` SHA-256 is
`6027ae258a08ddabfaa0f196b31ad6c5a88e0d7b34aee6fe547ca88df95144f6`.
No Job-3 tolerance, evaluator semantic, checkpoint, core file, experiment,
training path, or full-array job was changed or run.

## Job 4 standalone PPO training certification

Status: **pass**. This is a bounded training-integration certification, not a
claim of historical learning-trajectory parity. Job 3 remains formally
blocked and Job 3A remains `inconclusive` under their unchanged criteria.

The non-packaged Job 4 harness ran from clean consumer implementation commit
`213897ce9ca63cfe57278155966eaad428b34fc9` against installed core commit
`652a05e3d7f446d2cd770a8361a5b3dac79e37e8`. It trained fresh
`multirank_hodge_lower` actor, critic, observation-normalizer, and optimizer
state with seed 0 for exactly `22,937,600` environment steps. The run used the
frozen weak-actuator curriculum and saved only terminal checkpoint
`checkpoints/000022937600`.

| Frozen item | Certified value |
|---|---|
| Environment | `Go1JoystickFlatTerrain`, `impl="jax"` |
| Actor parameters | `92,962` |
| Evaluations / resets per evaluation | `2` / `10` |
| Rollout batches / optimizer steps | `140` / `17,920` |
| Official PPO config SHA-256 | `7df61b65d671c7f2df21ecf075143473a976a632b85c041503ad1aa52e15c407` |
| Effective PPO config SHA-256 | `40498cb2522be7bcd1bd41665aff952e06feb98d962bd4b652a3a2515d0a5222` |
| Curriculum SHA-256 | `d2e14a8b3eb31dc9d0728008c891ac71c69d46a739047c0c8087cb38af38898e` |
| Playground / Menagerie | `220abb9d6a41eaa258047dacf9ca70a92e008b78` / `1b86ece576591213e2b666ebf59508454200ca97` |
| JAX / JAXlib | `0.11.0` / `0.11.0` |
| GPU / driver / CUDA | NVIDIA A100 PCIe 40GB, `3g.20gb` MIG / `610.43.02` / `12.8` |
| Slurm job / node | `3345137` / `adroit-h11g2` |
| Slurm elapsed / harness wall time | `00:11:07` / `618.969 s` |

The initially requested full-A100 allocation, Slurm job `3345134`, was
cancelled while pending with zero runtime and before the output path existed.
Job `3345137` was the first and only Job 4 execution; changing to the
historically used `3g.20gb` A100 MIG resource did not change the immutable
implementation commit, protocol, seed, model, budget, or acceptance gates.

| Learning gate | Initial | Terminal | Required delta | Observed delta | Result |
|---|---:|---:|---:|---:|---|
| Evaluation reward | `0.0000257086` | `6.0445089340` | `1.0` | `6.0444832254` | PASS |
| Average episode length | `26.9140625` | `518.453125` | `100` | `491.5390625` | PASS |

All 51 changed actor leaves differed from step 0. Fixed-probe logits and
deterministic actions changed during training, while the terminal values
matched their restored values byte-for-byte. All training metrics, parameter
leaves, logits, and actions were finite; restored actions were within
`[-1, 1]`.

| Certification gate | Result |
|---|---|
| Frozen configuration and exact step accounting | PASS |
| Actor count and changed actor tensors | PASS |
| Reward and episode-length learning margins | PASS |
| Exact checkpoint parameter-tree/value roundtrip | PASS |
| Exact logits roundtrip | PASS |
| Exact deterministic-action roundtrip | PASS |
| Historical serialized factory kwargs | PASS |
| Finite and bounded restored outputs | PASS |
| Restored-policy 109-condition rollout | PASS |
| Historical trajectory parity claimed | NO |

The restored rollout used `stand`, reset key 0, all 109 ordered healthy/damage
conditions, and a 1,000-step horizon. It produced exactly 109 rows with finite
continuous metrics, valid survival/termination fields, and no runtime failure.
Those rows were not compared with historical CSV values.

### Immutable Job 4 evidence

Artifact root:

```text
/scratch/network/dy0130/go1-robot-experiments-artifacts/
  213897ce9ca63cfe57278155966eaad428b34fc9/
  job4/multirank_hodge_lower/seed-0/
```

| Evidence | SHA-256 |
|---|---|
| `certification_report.json` | `02a5a38eb046102a368bccf8b4c9e4437c6abedc8504906e731463aa0863559c` |
| `run.json` | `c21ae4f2f1eaa0220912be60a17bde22964e90cbf5561f51726461798a1f707e` |
| `curriculum.json` | `b447b55343e58704e8bc738b70d14cd9f53ed28b5b5283263ee392594db2eb81` |
| `ppo_config.json` | `fee33cf409d42fa659873e2dd54f94bbe950477ed83ca70c5bad327dfb47557d` |
| `progress.jsonl` | `f8403cd12e148c32c2807ee00d772424f4a2033a4e242476429b1332e216dff3` |
| `checkpoints/000022937600` tree | `1f5fb80e64e6b5dbebf83df01a05f6b3d88ad4ebf11da2adf1b012c78b5964c2` |
| `restored_rollout_rows.csv` | `1bdb51abbff57739abda124e242781af27f51a67d245d5928301e22818927c59` |

Job 4 stops here. It did not add a public training API, retry another
seed/model, extend the budget, run a full historical reproduction, modify
`go1_core`, or create a release tag.
