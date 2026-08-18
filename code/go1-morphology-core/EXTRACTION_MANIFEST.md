# Frozen Go1 Extraction Manifest

This document is the inventory handoff for a future task-independent
`go1_core` package. It records the implementation that produced the completed
seed-0 six-model weak-actuator campaign. It does not define a new API and does
not authorize cleanup, redesign, or changes to the scientific behavior.

Inventory status: **complete, with no unresolved blockers**. All dynamic probes
were executed on CPU against the clean reference worktree at the authoritative
commit. No file in the source repository or reference worktree was modified.

## 1. Authority and provenance

| Field | Frozen value |
|---|---|
| Source repository | `morphology-playground-marl` |
| Source remote | `git@github.com:dh-you/morphology-playground-marl.git` |
| Source branch | `robustness/clean-model-weak-actuator-scratch-400m` |
| Authoritative source/campaign commit | `1659809766f6aba43a1074cc332922317e9bec5a` |
| Campaign base commit | `dc1aeca2271c4ba52914d458da47ad9a73fa3e60` |
| MuJoCo Playground commit | `220abb9d6a41eaa258047dacf9ca70a92e008b78` |
| MuJoCo Menagerie commit | `1b86ece576591213e2b666ebf59508454200ca97` |
| Environment registry name | `Go1JoystickFlatTerrain` |
| Campaign | `clean-model-weak-actuator-scratch-400m` |
| Seed | `0` |
| Requested/actual training steps | `400000000` / `412876800` |
| Canonical training array | `3343705` |
| Canonical terminal step | `412876800` |

The local source/artifact checkout used for this inventory is
`/scratch/network/dy0130/morphology-playground-structured-ppo`. The read-only
runtime probe used the clean worktree
`/scratch/network/dy0130/worktrees/clean-model-unified-rhmp-fix`; its HEAD was
verified as the authoritative commit above immediately before probing. The
worktree name is historical and is not evidence of RHMP inclusion.

### Pinned runtime

`pyproject.toml` requires Python `3.12.*` and pins MuJoCo Playground by Git
commit. The actual local locked environment used for the evidence pass was:

| Component | Version |
|---|---:|
| Python | `3.12.13` |
| JAX | `0.11.0` |
| Flax | `0.12.8` |
| Brax | `0.14.2` |
| MuJoCo | `3.11.0` |
| MuJoCo MJX | `3.11.0` |
| MuJoCo Playground distribution (`playground`) | `0.2.0` at the pinned commit |

The dynamic evidence below was collected with `JAX_PLATFORMS=cpu` and
`jax.default_backend() == "cpu"`.

### Frozen source blobs

The following are Git blob object IDs at the authoritative commit. Future
extraction work must read these versions, not similarly named files from a
newer branch.

| Git blob | Source path | Relevant responsibility |
|---|---|---|
| `a116d87cec5cb5a36ebe2d4854aa199f665796d6` | `pyproject.toml` | Runtime requirement and Playground pin |
| `f9787f81cfa35cacefa5cab283d2d7dfcdd54a20` | `uv.lock` | Exact dependency resolution |
| `34d240af80752c1655d5b2c413661a37f5937177` | `claim1/topology.py` | Frozen cell complex and operator hashes |
| `afa85939027a57214f48a25f9a274897b9a6ed31` | `claim1/networks.py` | Feature builder, native policy, pointwise encoder/decoder |
| `e31ee78977c8b22cc38d1d861448c9785979a4be` | `claim1/protocol.py` | Environment identity and official PPO configuration |
| `6fd1a33703be489b4e2c37dee7c17c1ee208f509` | `clean_models/metrics.py` | Identity/learned metric primitives |
| `372ab743455d61004ea900e52f5bbd3106a87a07` | `clean_models/multirank.py` | Persistent multi-rank encoder |
| `0900b609c909f2de464420dac6fbecdcebe48bfa` | `clean_models/networks.py` | Clean model configs, GCN, multi-rank actor/PPO integration |
| `8b669443afee395b6881972a92778a79450086af` | `train_clean_models.py` | Canonical counts and shared training checks |
| `1f0afcc98a0e1d0f8e2ce87357b088d7e4d6a142` | `train_clean_model_capacity.py` | Matched MLP, pointwise, and GCN definitions |
| `19a73b557227a54311b59f644c7ea33f0d350026` | `unified_clean_models.py` | Catalog routing and non-native checkpoint restore |
| `cf1c5c83df43265a83bfb09ffe247cac12e8edeb` | `robustness/clean_weak_actuator_scratch_400m.py` | Six-model entrypoint, PPO call, save/restore path |
| `5889346a610b46f5c053554b817431995330d39f` | `robustness/clean_weak_actuator_evaluation.py` | Frozen actuator order and task evaluator boundary |

## 2. Inventory scope

The only v1 model identities inventoried here are:

1. `native_mlp`
2. `mlp_match_lower`
3. `pointwise_match_lower`
4. `gcn_match_lower`
5. `multirank_hodge_lower`
6. `multirank_hodge_full`

Candidate core responsibilities are the Go1 contract, topology, joint feature
construction, these six actors, their shared critic/action distribution,
model registry, Brax PPO network integration, checkpoint compatibility, and
task-neutral inference/evaluation helpers.

Explicitly outside v1 are `rhmp`, `rhmp_near_identity`, the older Claim 1
matched models, terrain generation, weak/dead-actuator sampling, curricula,
domain randomizers, task reward/environment implementations, campaign
evaluators and aggregators, plots, launchers, Slurm scripts, and all training
or evaluation artifacts. Later Hodge reruns are evidence outside the canonical
campaign selection and are not substituted for the checkpoints below.

## 3. Proposed `Go1ContractV1` facts

This section documents the established contract; it does not implement the
future dataclass.

### Observation keys and ordered slices

Both observations are float32. The actor consumes only `state`; the critic
consumes `privileged_state`. The checkpoint config records shapes `(48,)` and
`(123,)` respectively.

Actor `state` is the following ordered concatenation from the pinned upstream
`Joystick._get_obs`:

| Slice | Width | Meaning |
|---|---:|---|
| `[0:3]` | 3 | Noisy base linear velocity in the local/body frame |
| `[3:6]` | 3 | Noisy gyroscope/angular velocity |
| `[6:9]` | 3 | Noisy projected gravity |
| `[9:21]` | 12 | Noisy joint angles minus the default pose, in joint order |
| `[21:33]` | 12 | Noisy joint velocities, in joint order |
| `[33:45]` | 12 | Previous action, in action order |
| `[45:48]` | 3 | Command |

`privileged_state` begins with the complete 48D `state` and appends:

| Slice | Width | Meaning |
|---|---:|---|
| `[48:51]` | 3 | Noise-free gyroscope |
| `[51:54]` | 3 | Accelerometer |
| `[54:57]` | 3 | Noise-free projected gravity |
| `[57:60]` | 3 | Noise-free local linear velocity |
| `[60:63]` | 3 | Global angular velocity |
| `[63:75]` | 12 | Noise-free joint angles minus default pose |
| `[75:87]` | 12 | Noise-free joint velocities |
| `[87:99]` | 12 | Actuator force |
| `[99:103]` | 4 | Previous foot-contact indicators |
| `[103:115]` | 12 | Four foot linear velocities, flattened |
| `[115:119]` | 4 | Foot air times |
| `[119:122]` | 3 | Applied torso force, translational components |
| `[122:123]` | 1 | Perturbation timing flag |

The pinned environment reports exactly
`{"state": (48,), "privileged_state": (123,)}` and action size `12`.
Changing a slice, frame, noise placement, key, shape, or ordering is a contract
change rather than a refactor.

### Normalization and joint features

The PPO checkpoint contains a keyed Welford running-statistics tree. Actor
preprocessing selects the `state` normalizer and normalizes the complete 48D
state before `build_joint_features` runs. The critic independently selects and
normalizes `privileged_state`. `normalize_observations` is `true`, mode is
`welford`, and `normalize_observations_std_eps` is `0.0`.

For each batch prefix, `build_joint_features(normalized_state)` produces
`(..., 12, 20)` with this per-joint channel order:

| Joint-feature slice | Width | Source |
|---|---:|---|
| `[0:9]` | 9 | Actor state `[0:9]`, broadcast to all joints |
| `[9:12]` | 3 | Command `[45:48]`, broadcast to all joints |
| `[12:15]` | 3 | This joint's angle offset, velocity, and previous action from the three ordered 12D blocks |
| `[15:20]` | 5 | Frozen static morphology features |

The exact float32 static feature rows, in joint order, are:

```text
FR_hip    [ 1, -1, 1, 0, 0]
FR_thigh  [ 1, -1, 0, 1, 0]
FR_calf   [ 1, -1, 0, 0, 1]
FL_hip    [ 1,  1, 1, 0, 0]
FL_thigh  [ 1,  1, 0, 1, 0]
FL_calf   [ 1,  1, 0, 0, 1]
RR_hip    [-1, -1, 1, 0, 0]
RR_thigh  [-1, -1, 0, 1, 0]
RR_calf   [-1, -1, 0, 0, 1]
RL_hip    [-1,  1, 1, 0, 0]
RL_thigh  [-1,  1, 0, 1, 0]
RL_calf   [-1,  1, 0, 0, 1]
```

Its C-order raw-byte SHA-256 is
`5f61c716b7a912ce1534b675061cc68c56a451f5b7ab7e38b1f056088c5786d5`.

### Joint and action contract

Joint order and action order are identical:

```text
0  FR_hip       3  FL_hip       6  RR_hip       9  RL_hip
1  FR_thigh     4  FL_thigh     7  RR_thigh    10  RL_thigh
2  FR_calf      5  FL_calf      8  RR_calf     11  RL_calf
```

The pinned default pose, in that order, is:

```text
[0.1, 0.9, -1.8, -0.1, 0.9, -1.8,
 0.1, 0.9, -1.8, -0.1, 0.9, -1.8]
```

The environment applies the 12D policy action as:

```text
motor_targets = default_pose + action * 0.5
```

Brax uses `NormalTanhDistribution(event_size=12)`, whose parameter size is 24.
The structured policies decode two values per joint and concatenate all twelve
channel-0 values followed by all twelve channel-1 values. The distribution
turns those 24 logits into twelve tanh-postprocessed actions. This 24D layout
and the final twelve-action order are frozen.

## 4. Frozen topology

Topology identity is `go1_joint_planar_closure`, schema version `1`. NumPy
operators are constructed as float64 and hashed as float64 C-order bytes. The
exported JAX `B1`, `B2`, and normalized Hodge operators are float32.

### Oriented edges

`B1[source, edge] = -1` and `B1[target, edge] = +1`.

| Edge | Source -> target | Relation |
|---:|---|---|
| 0 | `FR_hip -> FR_thigh` | physical serial limb |
| 1 | `FR_thigh -> FR_calf` | physical serial limb |
| 2 | `FL_hip -> FL_thigh` | physical serial limb |
| 3 | `FL_thigh -> FL_calf` | physical serial limb |
| 4 | `RR_hip -> RR_thigh` | physical serial limb |
| 5 | `RR_thigh -> RR_calf` | physical serial limb |
| 6 | `RL_hip -> RL_thigh` | physical serial limb |
| 7 | `RL_thigh -> RL_calf` | physical serial limb |
| 8 | `FL_hip -> FR_hip` | four-HAA body loop |
| 9 | `FR_hip -> RR_hip` | four-HAA body loop |
| 10 | `RR_hip -> RL_hip` | four-HAA body loop |
| 11 | `RL_hip -> FL_hip` | four-HAA body loop |
| 12 | `FR_hip -> FR_calf` | virtual planar limb closure |
| 13 | `FL_hip -> FL_calf` | virtual planar limb closure |
| 14 | `RR_hip -> RR_calf` | virtual planar limb closure |
| 15 | `RL_hip -> RL_calf` | virtual planar limb closure |

### Oriented faces

Face columns in `B2` are ordered `FR_limb`, `FL_limb`, `RR_limb`, `RL_limb`,
`four_HAA_body_loop`:

```text
FR_limb:              +e0 +e1 -e12
FL_limb:              +e2 +e3 -e13
RR_limb:              +e4 +e5 -e14
RL_limb:              +e6 +e7 -e15
four_HAA_body_loop:   +e8 +e9 +e10 +e11
```

### Exact incidence matrices

`B1` has shape `(12, 16)`:

```text
[-1, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0,-1, 0, 0, 0]
[ 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
[ 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 1, 0,-1, 0, 0]
[ 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
[ 0, 0, 0, 0,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0,-1, 0]
[ 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
[ 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 1,-1, 0, 0, 0,-1]
[ 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
```

`B2` has shape `(16, 5)`:

```text
[ 1, 0, 0, 0, 0]
[ 1, 0, 0, 0, 0]
[ 0, 1, 0, 0, 0]
[ 0, 1, 0, 0, 0]
[ 0, 0, 1, 0, 0]
[ 0, 0, 1, 0, 0]
[ 0, 0, 0, 1, 0]
[ 0, 0, 0, 1, 0]
[ 0, 0, 0, 0, 1]
[ 0, 0, 0, 0, 1]
[ 0, 0, 0, 0, 1]
[ 0, 0, 0, 0, 1]
[-1, 0, 0, 0, 0]
[ 0,-1, 0, 0, 0]
[ 0, 0,-1, 0, 0]
[ 0, 0, 0,-1, 0]
```

### Derived operators and fingerprints

`A` is the undirected node adjacency from the 16 edges. The Hodge terms are
`L_LOWER = B1.T @ B1` and `L_UPPER = B2 @ B2.T`. Frozen normalization uses:

```text
A_HAT       = A / 2.9999999999999996
L_LOWER_HAT = L_LOWER / 6.3722813232690125
L_UPPER_HAT = L_UPPER / 4.000000000000002
```

| Float64 array | Shape | SHA-256 of C-order bytes |
|---|---:|---|
| `A` | `(12, 12)` | `5136e58e34990eb9a6711d74978cb000f09d86b8e567a45f8827f5ae6e8bc15d` |
| `B1` | `(12, 16)` | `15249a6102b39eda0c7fc77853ac91007c937acbb39b667f77ca7982f9539e8f` |
| `B2` | `(16, 5)` | `5247ce6fa57bc1de8887c3044e6bda647ac0382dc9ff04846494af1f99bf4187` |
| `L_LOWER` | `(16, 16)` | `d6efe200280819fb7e4e79ee9b038df5e80e9ece174d51a54fdc78dec1a368af` |
| `L_UPPER` | `(16, 16)` | `848cae4a9d2f2dc32709bd70cf6be3bd1c6e3323f64d72ab08962aa1fc753edc` |
| `A_HAT` | `(12, 12)` | `b2250bb44c9132cf97642a538df1c231f9e1483936504be44463a1a4ebe1e554` |
| `L_LOWER_HAT` | `(16, 16)` | `77d52234e99104e6966d6d671b81b59d4a74cd137aee50e5451c5d9e077be0cc` |
| `L_UPPER_HAT` | `(16, 16)` | `6a555e9323cf505338232d12cf63c82162afed51b0d52b8b9b7b7b5fa51cf444` |

The GCN does not use `A_HAT`. It derives a node adjacency from shared unsigned
incidence, removes then re-adds the diagonal as self-loops, and applies
`D^-1/2 A D^-1/2`. The float32 C-order hashes are:

| GCN operator | Shape | SHA-256 |
|---|---:|---|
| Adjacency with self-loops | `(12, 12)` | `35c5745d515c4dc1bd010acbe9aaa75c0ff9a9bd1fe6bdbf402dfa4dd5ec9a3b` |
| Normalized adjacency | `(12, 12)` | `8b8a9c5214898160f687eb801d6cff1dfcb6964d908404a771e77b5bdf415e3a` |

Frozen validation additionally requires `rank(B1) == 11`, `rank(B2) == 5`,
`B1 @ B2 == 0` exactly, connected symmetric adjacency, unique non-self edges,
and symmetric positive-semidefinite lower/upper Hodge operators.

## 5. Model routes and architecture identities

### Common route

Training resolves a model name through:

```text
run_training
  -> make_campaign_networks(model_name=...)
     -> native_mlp: make_claim1_networks
     -> other five: make_unified_model_networks
  -> PPONetworks(policy_network, value_network, NormalTanhDistribution)
  -> brax.training.agents.ppo.train
```

All six enforce actor key/shape `state: (48,)`, critic key/shape
`privileged_state: (123,)`, and action size `12`. All use the same Brax value
network with hidden widths `(512, 256, 128)`. Only actor construction differs.

### `native_mlp`

- Route: `make_campaign_networks` -> `claim1.networks.make_claim1_networks` ->
  `brax.training.agents.ppo.networks.make_ppo_networks`.
- Actor: normalized 48D state -> MLP widths `(512, 256, 128)` -> 24 logits.
- Initialization follows Brax's pinned `make_policy_network` defaults, including
  LeCun-uniform policy kernels.
- This path does not call `build_joint_features` and does not consume topology.

### `mlp_match_lower`

- Route: `make_campaign_networks` -> `make_unified_model_networks` (capacity) ->
  `make_capacity_control_networks` -> `make_clean_model_networks("mlp")` ->
  frozen `make_claim1_networks("native_mlp")`.
- Actor: normalized 48D state -> MLP widths `(344, 172, 86)` -> 24 logits.
- Initialization is the same pinned Brax MLP path as `native_mlp`.
- The public identity remains `mlp_match_lower` in serialized checkpoint kwargs
  even though the inner clean factory deliberately reuses the native MLP path.

### `pointwise_match_lower`

- Route: `make_campaign_networks` -> `make_unified_model_networks` (capacity) ->
  `make_capacity_control_networks` -> `CapacityStructuredPolicy`.
- Actor: normalized 48D state -> `build_joint_features` (`12 x 20`) ->
  `PointwiseEncoder(width=174)` -> 128D per-joint embedding ->
  `NodeDecoder` (`Dense(48)`, swish, `Dense(2)`) -> 24 logits.
- The pointwise encoder is `Dense(174)` plus two self-only `Dense(174)` layers,
  all with swish, followed by `Dense(128)`.
- `claim1.networks._dense` fixes LeCun-uniform kernels and zero biases.
  No topology operator is consumed after feature construction.

### `gcn_match_lower`

- Route: `make_campaign_networks` -> `make_unified_model_networks` (capacity) ->
  `make_capacity_control_networks` -> `CapacityStructuredPolicy`.
- Actor: normalized 48D state -> `12 x 20` joint features -> normalized GCN
  propagation -> `Dense(581)` + ReLU -> normalized propagation -> `Dense(128)`
  -> the same 48/2 `NodeDecoder` -> 24 logits.
- `GCNEncoder` uses Flax `nn.Dense` defaults; `NodeDecoder` uses the explicit
  Claim 1 LeCun-uniform/zero-bias helper.

### `multirank_hodge_lower`

- Route: `make_campaign_networks` -> `make_unified_model_networks` (canonical)
  -> `make_clean_model_networks` -> `CleanStructuredPolicy` ->
  `MultiRankEncoder(max_rank=1, metric_mode="identity")`.
- Actor: `12 x 20` joint features -> 128D node lift -> 128D edge lift -> four
  synchronous `MultiRankLayer`s over node/edge cochains -> final node cochain ->
  per-joint `Dense(48)` + swish + `Dense(2)` -> 24 logits.
- The lower model constructs no face cochain. Identity metrics have no learned
  metric leaves; each layer's `raw_alpha_0` and `raw_alpha_1` initialize to zero.
- Dense layers not routed through Claim 1 `_dense` use the pinned Flax defaults.

### `multirank_hodge_full`

- Route is the same canonical clean route, with
  `MultiRankEncoder(max_rank=2, metric_mode="identity")`.
- It adds a 128D face lift, maintains node/edge/face cochains, and includes the
  `B2` upper paths in each of four synchronous layers. Decoder/output layout is
  identical to the lower model.
- Identity metrics again add no learned metric parameters; each layer has
  zero-initialized mixing scalars for ranks 0, 1, and 2.

The old `claim1.networks.HodgeEncoder` is not the implementation of either
canonical multi-rank model. The latter are defined by
`clean_models.multirank.MultiRankEncoder` and its lifts/layers.

## 6. PPO and checkpoint path

The official PPO configuration is reconstructed from pinned Brax defaults plus
MuJoCo Playground's `brax_ppo_config(Go1JoystickFlatTerrain, "jax")`. Its
frozen SHA-256 is
`7df61b65d671c7f2df21ecf075143473a976a632b85c041503ad1aa52e15c407`.
The full campaign changes only `num_timesteps` and `num_evals`; the resulting
effective configuration hash is
`c4435ca47c1bb9fcdd17ecff236840e24fe602832c23dfd73e77fbf570935350`.

| PPO field | Effective value |
|---|---:|
| Seed | `0` |
| Environments / evaluation environments | `8192` / `128` |
| Requested timesteps / evaluations | `400000000` / `19` |
| Batch size / minibatches | `256` / `32` |
| Unroll length / updates per batch | `20` / `4` |
| Learning rate | `0.0003` |
| Discount / GAE lambda | `0.97` / `0.95` |
| PPO clipping epsilon | `0.3` |
| Entropy cost / value loss coefficient | `0.01` / `0.5` |
| Max gradient norm | `1.0` |
| Episode length / action repeat | `1000` / `1` |
| Observation normalization | keyed Welford, enabled |
| Advantage normalization | enabled |
| Bootstrap on timeout | disabled |
| Distributional critic | disabled |
| Deterministic training evaluation | disabled |

The weak-actuator `randomization_fn` is injected only when invoking
`ppo.train`; it is task-specific and is not a model or PPO-core dependency.
The wrapper is MuJoCo Playground's `wrap_for_brax_training`.

Checkpoint configuration is created with `ppo_checkpoint.network_config` from
float32 observation specs, action size 12, normalization enabled, and a partial
`make_campaign_networks`. Serialized `network_factory_kwargs` contain only:

```text
model_name
policy_hidden_layer_sizes = [512, 256, 128]
value_hidden_layer_sizes  = [512, 256, 128]
policy_obs_key             = "state"
value_obs_key              = "privileged_state"
```

`ValidatedTrainingCallbacks` pairs a finite device-copied parameter tree with
the matching progress step before calling `ppo_checkpoint.save`. Frozen save
steps are:

```text
22937600, 68812800, 114688000, 206438400, 298188800, 412876800
```

The restored PPO parameter tuple is indexed as `[0]` observation processor,
`[1]` actor, and `[2]` critic. Restore first reads the serialized config and
model identity. `native_mlp` rebuilds through `make_claim1_networks`; all other
identities rebuild through `load_unified_checkpoint` and
`make_unified_model_networks`. Brax then loads the Orbax parameter tree.
Historical training already required exact tree structure, exact parameter
values, and exact logits/actions on a real environment batch after restore.

Inference uses `ppo_networks.make_inference_fn(networks)`. It applies processor
and actor params to produce 24 logits, then samples or takes the mode of the
shared `NormalTanhDistribution`. Deterministic certification uses the mode.

## 7. Canonical checkpoint index

Artifact root, relative to the source checkout:

```text
artifacts/clean-model-weak-actuator-scratch-400m/
  1659809766f6aba43a1074cc332922317e9bec5a/
```

Absolute local root:

```text
/scratch/network/dy0130/morphology-playground-structured-ppo/artifacts/
  clean-model-weak-actuator-scratch-400m/
  1659809766f6aba43a1074cc332922317e9bec5a/
```

For each model below, the canonical run directory is
`train/array-3343705/<model>/seed-0`, and the checkpoint is
`checkpoints/000412876800`.

| Model | `run.json` SHA-256 | `result.json` SHA-256 | `ppo_network_config.json` SHA-256 |
|---|---|---|---|
| `native_mlp` | `7baddfccd09ec8089b61a37c11c1378b565d8c99e832dc0c30f9089746f21014` | `590184cf8b5d94ab9150f9075071ab0645313634dab363a1be43cf3e4bebec71` | `e9fe577427b1b2e2cbb4efcda8cf3ed02b9841d9e849e6eadc6b58273313b801` |
| `mlp_match_lower` | `5b856060ea4e9296ed71db9980567aafcb7355ac9157f4a918eb2bd53b424e22` | `98052f368da0734ae654f6c0d59db9f023309f82c0e9b71dd7375370b3d0f158` | `7a5abcbc15eeade04139598e9f92a03c9b08b2744f0d333826714cc3fcc1c99c` |
| `pointwise_match_lower` | `7ab58295be7153357f4b1d7bfa0d161795473ba56ded75686cf080797bf7a355` | `68bd44af38345c30daa089a4fec42b59b41fb62b896579b38e7db57af489f193` | `83e073adb31ffcc2ea51cd392022428b8928c2515fd5f970436d4794a23d6afe` |
| `gcn_match_lower` | `01f1a6c1ae2e0bda61299c7aef88cd4ca69e143da8cffe6f1cda25c5dd5527bc` | `c4288096c23f6fefff7c3f4f9483fcee16c2f79cf9ca5aed2d5c75b85a6f2332` | `b13a331a233d3bcb54212973692830018b6b7aa9adef2e3809390d9e765deb8c` |
| `multirank_hodge_lower` | `8ad9a8bb689001d4f55f34cca2751c3dd744e98f4c9d595149404907c72b285b` | `c3954a801916fbec1619c99cd06a5585c2e28cfd1e5f0c1505c9ffa6d18173f2` | `958f886a2a5d91df02af863cfe482ea6eb9eda160c3d1b76fdd668e7886fa931` |
| `multirank_hodge_full` | `322ff2a9d36d8a7f2a52ed26558875021855e3fa11f52736bf383276dc028709` | `e91820b13fec01b2b5e838130f0c6b318b4f70e09090b36367b736d4394a0915` | `f0f74df44317a4d852f1bfd039398f2ea8c6eb8c9ec4ca093419de785dffc653` |

The aggregate's 36 selected evaluations are under evaluation array `3343855`.
Each selected terminal `evaluation.json` points back to the corresponding
training array `3343705` run above. Training array `3343754` and evaluation
array `3343924` contain later Hodge reruns and are explicitly non-canonical for
this inventory. The incomplete/repaired RHMP artifact tree is also excluded.

## 8. Dynamic evidence and reproducible fingerprints

### Tree-signature method

For the full PPO tuple and actor subtree separately:

1. Enumerate `jax.tree_util.tree_flatten_with_path(tree)`.
2. Convert each leaf to an entry with `jax.tree_util.keystr(path)`, integer
   `shape`, and NumPy `dtype` string.
3. Sort entries lexicographically by `path`.
4. Encode the entry list with JSON `sort_keys=True` and separators `(',', ':')`.
5. SHA-256 the encoded bytes.

The signature intentionally hashes structure, shapes, and dtypes, not tensor
values. Functional tensor identity is covered by checkpoint restore checks and
the policy fingerprints.

### Fingerprint input and method

The actor batch is float32 with shape `(3, 48)`:

```python
jnp.stack((
    jnp.zeros((48,), dtype=jnp.float32),
    jnp.linspace(-1.0, 1.0, 48, dtype=jnp.float32),
    jnp.sin(jnp.arange(48, dtype=jnp.float32)),
))
```

Its little-endian float32 C-order byte hash is
`4dbe125f73f5977e9c6110b6ab11281425286a091380a2173911b5eaba7165d9`.
The keyed observation also supplies a zero float32 privileged batch of shape
`(3, 123)`; policy evaluation ignores it.

Normalized logits are produced by
`networks.policy_network.apply(params[0], params[1], observation)`.
Deterministic actions are produced by the pinned Brax inference factory with
`deterministic=True` and `jax.random.PRNGKey(0)`. Each result is converted to
little-endian float32 C-order bytes before SHA-256. Both computations were
repeated in-process and matched exactly.

| Model | Actor params | Actor tree: leaves / SHA-256 | Full tree: leaves / scalars / SHA-256 | Logits `(3,24)` SHA-256 | Actions `(3,12)` SHA-256 |
|---|---:|---|---|---|---|
| `native_mlp` | 192408 | 8 / `09f36e3da7227b35f65b62a8c051016cb07d3ea70382bc3bb2783d617c13eba6` | 25 / 420765 / `870c0a71c0b8dbcc27edd36174e5fc89baca19f9b8bb77d3d98f59777fb95681` | `795befd2feba6bc1e0e623c77e4c3e30c3f2cb4552028b10ce2a74ba6dfaaf54` | `3ea66a2265d76b58e22065a20f9725c537561b56fd5117d25a5d5c4ae7b6f22f` |
| `mlp_match_lower` | 93162 | 8 / `c0ab68491e8b10aa90b2d486e0d6dab17b6d3493ccd95e53732b405963f2d4ec` | 25 / 321519 / `522d4ce2f0e74844a5b7cf9b0038fe34bd2a0a168da7e0ccfce1d90f7222e117` | `1ead175496f19fe0714a20aad600db60d2a7c429c02c173849c81024d17dcfe3` | `02fed7bf8226fe8169adcf44091caef8fc07f2875782cddeaaa7ecbe97e1948e` |
| `pointwise_match_lower` | 93244 | 12 / `2a6d48b8c46e55893b02977636253032d5828ad04a3e42828cd60d1033fc8ace` | 29 / 321601 / `6ca9dc01b92c4aab0360a2efd2548140e8fffaa572411f2efd6443a707cfb0c7` | `38b13d631db6e39988ef2958f5aa4c3f7cd43c570bbeaa4dc4bd80d28a934d4d` | `b9f25afdfec69fea69778ef68c49feec38d99c2861ac5359317703c8df3dc55b` |
| `gcn_match_lower` | 92987 | 8 / `4f9ee45b773dd67d5815eefc55b5b3257b4ecadf0d014b6b9c17d08f7a2bb536` | 25 / 321344 / `3a377023c01b8cd0d311eaafd777ee977407098f566f6f070ecdd4b07df6f7ff` | `b3655d22abe8c31200814681c3c5a28fd663a1e80947c574fb908b7030bb9f38` | `6a62a63efbca5f3389b5c4f8f978b74eb9bbeca209c537d3c4ec64039cf27791` |
| `multirank_hodge_lower` | 92962 | 56 / `9bf4e874136e4fad66509706a0e838736a9e928b8bbad71608606358bea4450d` | 73 / 321319 / `dd3c985fd3d55b80a51b8c41880d25121470a7061ad0ef50dc38cd012dfe21be` | `10a246a2d3386d3d264d15abe387e43dc0f00511aa5268fee37d392f47a73baf` | `7a4cce65ca72a186db24d63c4007e11ebea3f8ce088ebe8dba16dcd797bf03ed` |
| `multirank_hodge_full` | 126570 | 80 / `375f9ee6132449eaebe213f8798551184eb186a656dad1716b9c266fb3aabe26` | 97 / 354927 / `cd6a4f737660c8ad5aa632ed330e51f4146c6058b3afa1148a0452ca7ad4b4e0` | `ee82903f88aeac4283d2b74b2aec128bff2b021762cc741d42537104f1a7d864` | `2ae3ac36ea0f0f196b658bfa9033d663d59c06c4a4bd2da1e5d40e5556f12adc` |

For every model, the loaded actor scalar count equals both the campaign record
and the expected model identity; all checkpoint leaves, logits, and actions are
finite. Every `result.json` records `completed: true`, the complete six-step
save schedule, terminal checkpoint `checkpoints/000412876800`, exact restored
parameters/logits/actions, action range validity, and a four-observation real
batch restore check. Both identity-metric Hodge checkpoints record zero learned
metric parameter leaves, as intended.

## 9. Extraction boundary and coupling hazards

The dependency rule for the future package is: tasks may depend on `go1_core`;
`go1_core` may not depend on a task or campaign.

Current reusable implementation candidates are split across research-oriented
locations:

- Topology and joint feature construction live under `claim1`, although the
  six-model campaign treats them as shared Go1 machinery.
- `clean_models.networks` imports both `claim1.networks` and
  `claim1.topology`.
- `unified_clean_models` imports model configs/counts/factories from
  `train_clean_models.py` and `train_clean_model_capacity.py`; a loader therefore
  depends on experiment entrypoints.
- `CapacityStructuredPolicy` and the three matched model identities are
  implemented inside `train_clean_model_capacity.py` rather than a model
  package.
- The campaign file combines reusable PPO step accounting, checkpoint
  callbacks, and restore routing with weak-actuator source guards,
  randomization, evaluation, aggregation, and artifact policy.
- The observation/action contract is asserted piecemeal in model factories
  and inherited from pinned upstream environment code rather than represented
  by one versioned object.
- `MODEL_CONFIGS` and the unified catalog contain non-v1 identities. Copying a
  whole registry would silently expand scope beyond these six models.

During extraction, move behavior rather than redesigning it. Preserve serialized
model names, module/parameter naming where checkpoint compatibility requires it,
normalization selection, topology orientations, feature order, initializers,
decoder concatenation, critic construction, action distribution, and checkpoint
kwargs. Weak-actuator and terrain consumers must import these facilities; no
task-specific environment, curriculum, evaluator, or artifact dependency may
flow back into the core.

## 10. Inventory acceptance record

- Reference worktree HEAD: PASS (`1659809766f6aba43a1074cc332922317e9bec5a`).
- Reference tracked worktree clean: PASS.
- Six canonical terminal checkpoints present and loadable on CPU: PASS.
- Serialized model identity, 48D/123D float32 observations, and 12D actions:
  PASS for all six.
- Actor counts and finite full parameter trees: PASS for all six.
- Reproducible full/actor tree signatures: PASS for all six.
- Existing exact restore parameters/logits/actions: PASS for all six.
- Repeated deterministic fingerprint hashes: PASS for all six.
- Aggregate selection traces to training array `3343705`: PASS for all six.
- Later Hodge reruns and RHMP variants excluded: PASS.
- Source repositories changed by inventory: no.

This is the stopping point for the inventory job. No core source tree,
`pyproject.toml`, tests, migration shim, evaluator, remote, or release tag is
created by this task.
