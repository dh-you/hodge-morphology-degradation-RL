# Weak-actuator protocol continuity

The active experiment inherits its scientific mechanics from the frozen
pre-split repository at commit
`1659809766f6aba43a1074cc332922317e9bec5a`.

## Source mapping

| Frozen source | Active implementation | Preserved behavior |
|---|---|---|
| `claim1.protocol` | `constants.py`, `experiments/weak_actuator/protocol.py` | Environment identity, commands, reset keys, official PPO configuration, batch accounting |
| `clean_weak_actuator_scratch_400m.py` | `experiments/weak_actuator/protocol.py` | PRNG splitting, healthy/weak/dead sampling, actuator selection, log-uniform weak strength, force-range mutation |
| `clean_weak_actuator_evaluation.py` | `constants.py`, `protocol.py`, `rollout.py` | 109-condition order, force ranges, fixed randomization, paired reset keys, command patching, terminal masking, physical metrics |
| historical model factories | exact-pinned `go1_core` | Six canonical model identities, network construction, actor counts, checkpoint reconstruction |

The curriculum SHA-256 remains
`d2e14a8b3eb31dc9d0728008c891ac71c69d46a739047c0c8087cb38af38898e`.
The official PPO configuration SHA-256 remains
`7df61b65d671c7f2df21ecf075143473a976a632b85c041503ad1aa52e15c407`.
Both are runtime assertions.

## Exact scientific invariants

- Environment: `Go1JoystickFlatTerrain`, `impl="jax"`.
- Curriculum probabilities: healthy `0.30`, weak `0.50`, dead `0.20`.
- Weak strength: log-uniform `[0.05, 0.20)`.
- One uniform actuator selection over the frozen 12-actuator Go1 order.
- Fixed assignment per vectorized environment member; no reset resampling.
- PPO network widths: policy and value `(512, 256, 128)`.
- Actor observation: 48D `state`; critic observation: 123D
  `privileged_state`; distribution: 24D; action: 12D.
- Complete evaluator: 109 conditions × 9 commands × 10 reset keys × 1,000
  steps, producing 9,810 ordered rows.
- Terminal lanes count their first terminal transition, then freeze and mask
  subsequent metrics.
- Checkpoint reconstruction, terminal parameters, probe logits, and
  deterministic actions must round-trip exactly.

## Intentional research interfaces

The old campaign hard-coded one model/seed/budget and stored several campaign
checkpoints. The active runner instead:

- accepts any canonical model, nonnegative seed, positive step budget, and an
  evaluation count of at least two;
- exposes the original 400M/19-evaluation accounting as ordinary inputs;
- saves only the terminal checkpoint;
- offers a 109-row smoke evaluation in addition to the original 9,810-row
  full evaluation;
- emits raw current-run rows without the old cross-model, cross-checkpoint
  campaign aggregator;
- writes results under an immutable path containing every run-defining input.

These interfaces change scheduling and artifact selection, not the actuator
curriculum, model semantics, environment contract, rollout equations, or full
evaluation grid.

## Verification

The cleanup audit used the frozen commit itself, extracted into a temporary
read-only tree:

- 11 original pure weak-actuator protocol tests passed;
- normalized AST bodies matched exactly for PRNG splitting, force mutation,
  fixed evaluation randomization, condition arrays, paired reset keys,
  physical diagnostics, and tree masking;
- the renamed training randomizer matched exactly;
- the rollout matched after alpha-normalizing one local variable name;
- fixed-seed 4,096-lane raw-byte hashes matched for category, actuator index,
  sampled weak strength, and effective strength.

The four sampler hashes are permanent assertions in
`tests/test_weak_actuator_experiment.py`.
