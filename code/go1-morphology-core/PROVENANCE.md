# Go1 Core 0.1.0 Provenance and Parity Record

This package is a mechanical extraction of the six canonical Go1 actors from
`morphology-playground-marl` commit
`1659809766f6aba43a1074cc332922317e9bec5a`. It is task-independent and does
not contain campaign, environment, terrain, evaluator, launcher, curriculum,
or training code.

## Frozen authority

| Field | Value |
|---|---|
| Source repository | `morphology-playground-marl` |
| Source commit | `1659809766f6aba43a1074cc332922317e9bec5a` |
| Inventory commit | `419b7118ba378e7881cb09ad36675e70bc9339c4` |
| Inventory file | `EXTRACTION_MANIFEST.md` |
| Inventory SHA-256 | `62aff4b4abf8332ba2ab1e2b490761b417c967e5d16e51bcc663c2c842041a82` |
| Training array | `3343705` |
| Terminal step | `412876800` |

The authoritative source worktree was clean at the frozen commit before and
after extraction. The active source checkout was also left unchanged.

## Source-to-core mapping

| Frozen source | Extracted responsibility |
|---|---|
| `claim1/topology.py` | Joint/edge/face order and exact `B1`/`B2` incidence |
| `claim1/networks.py` | Static joint features, 48D-to-`12x20` features, pointwise encoder/decoder, native PPO route |
| `clean_models/metrics.py` | Private identity/learned metric implementation shared by MultiRank |
| `clean_models/multirank.py` | Private incidence-based persistent MultiRank encoder |
| `clean_models/networks.py` | Canonical GCN operator, GCN actor, identity-Hodge actors, shared critic/distribution route |
| `train_clean_model_capacity.py` | Matched MLP, pointwise, and GCN capacity configurations and modules |
| `unified_clean_models.py` | Serialized-name routing and checkpoint reconstruction behavior |
| `robustness/clean_weak_actuator_scratch_400m.py` | Six-model campaign selection and native/unified loader split |

The runtime closure contains joint/action order, static features, edges/faces,
`B1`, `B2`, and the GCN adjacency/normalization. Legacy Claim-1 normalized
Hodge Laplacians, spectral-radius constants, `HodgeEncoder`, and
`PairwiseEncoder` remain historical context in `EXTRACTION_MANIFEST.md`; they
are not runtime code.

Learned metric classes remain private because identity MultiRank shares their
implementation path. No RHMP identity, fixture, registry entry, or public
export is present.

## Checkpoint routing contract

Historical `ppo_network_config.json` files are authoritative. All six store:

```text
policy_hidden_layer_sizes = [512, 256, 128]
value_hidden_layer_sizes  = [512, 256, 128]
policy_obs_key             = "state"
value_obs_key              = "privileged_state"
```

Those outer factory widths are preserved verbatim. Capacity actors select
their frozen architecture internally: notably `mlp_match_lower` constructs
`(344, 172, 86)` while still accepting the serialized outer
`[512, 256, 128]`. The loader returns `(networks, params, config)` in the
historical order.

## Fixture generation

Fixtures were generated once on CPU from the clean frozen worktree and the
canonical terminal checkpoints under:

```text
artifacts/clean-model-weak-actuator-scratch-400m/
  1659809766f6aba43a1074cc332922317e9bec5a/
  train/array-3343705/<model>/seed-0/checkpoints/000412876800
```

`tests/fixtures/metadata.json` records every array's shape, dtype, raw
little-endian float32 SHA-256, `.npy` SHA-256, source paths, and generation
expressions. The committed set contains the documented three-row input,
its frozen joint features, and logits/actions for all six checkpoints.

Reference tests locate the external campaign root through
`GO1_REFERENCE_ARTIFACT_ROOT`. They skip clearly when it is absent; Job-2
acceptance ran them with the variable set.

## Verified dependency and wheel boundary

The locked runtime is Python 3.12 with JAX/JAXlib 0.11.0, Flax 0.12.8, Brax
0.14.2, NumPy 2.5.1, and Orbax Checkpoint 0.12.1. Pytest 9.1.1 is the locked
development dependency. MuJoCo Playground and CUDA extras are absent.

The built wheel was
`dist/go1_morphology_core-0.1.0-py3-none-any.whl`. Its only importable project
package was `go1_core`; the other entries were standard `.dist-info`
metadata. Static and clean-subprocess checks found no imports of `claim1`,
`clean_models`, experiment/training modules, `robustness`, or
`mujoco_playground`.

## Acceptance results

Commands were run with `JAX_PLATFORMS=cpu`:

```text
pytest -m 'not reference' -q  -> 20 passed, 7 deselected
pytest -m reference -q        -> 7 passed, 20 deselected
pytest -q                     -> 27 passed, 0 skipped
```

| Model | Params | Actor tree | Restore | Logits | Actions |
|---|---|---|---|---|---|
| Native MLP | PASS | PASS | PASS | PASS | PASS |
| Matched MLP | PASS | PASS | PASS | PASS | PASS |
| Pointwise | PASS | PASS | PASS | PASS | PASS |
