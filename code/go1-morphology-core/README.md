# Go1 Morphology Core

By Derek You.

`go1-morphology-core` is the frozen, task-independent policy core extracted
from the canonical Go1 weak-actuator campaign. Version `1.0.0` preserves the
six certified actor architectures, the Go1 policy contract, PPO network
construction, and historical Brax checkpoint loading.

The package intentionally contains no environment, terrain, task reward,
curriculum, evaluator, launcher, campaign, or training implementation.

## Installation

Python `3.12.*` is required. The private Git release can be installed by tag:

```bash
pip install "go1-morphology-core @ git+ssh://git@github.com/dh-you/go1-morphology-core.git@v1.0.0"
```

Consumers that require an immutable dependency should pin the exact commit
identified by the `v1.0.0` tag instead of the tag name.

Check the installed distribution version through package metadata:

```python
from importlib.metadata import version

assert version("go1-morphology-core") == "1.0.0"
```

`go1_core.__version__` is deliberately not part of the public interface.

## Frozen Go1 contract

| Interface | Frozen value |
|---|---|
| Actor observation key / shape | `state` / `(48,)` float32 |
| Critic observation key / shape | `privileged_state` / `(123,)` float32 |
| Distribution parameters | `24` |
| Actions / joints | `12` in identical Go1 joint order |
| Action application | `default_pose + action * 0.5` |

The actor state consists of local velocity, angular velocity, projected
gravity, joint-angle offsets, joint velocities, previous actions, and command
in the order recorded by `GO1_CONTRACT_V1`. Structured actors transform the
normalized 48D state into twelve 20D joint features.

## Canonical models

| Model ID | Actor parameters |
|---|---:|
| `native_mlp` | 192,408 |
| `mlp_match_lower` | 93,162 |
| `pointwise_match_lower` | 93,244 |
| `gcn_match_lower` | 92,987 |
| `multirank_hodge_lower` | 92,962 |
| `multirank_hodge_full` | 126,570 |

No other model identity is registered in v1.0.0.

## Validate a consumer interface

```python
import jax
import jax.numpy as jnp

from go1_core import assert_compatible_environment

observation_size = {
    "state": jax.ShapeDtypeStruct((48,), jnp.float32),
    "privileged_state": jax.ShapeDtypeStruct((123,), jnp.float32),
}
assert_compatible_environment(observation_size, action_size=12)
```

For an environment object, pass its `observation_size` and `action_size`. The
core does not import or require MuJoCo Playground.

## Construct PPO networks

```python
from go1_core import make_go1_ppo_networks

networks = make_go1_ppo_networks(
    observation_size,
    action_size=12,
    model_name="multirank_hodge_lower",
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",
    value_obs_key="privileged_state",
)
```

The serialized widths `(512, 256, 128)` are part of the historical checkpoint
contract for every model. Capacity-controlled actors route those outer kwargs
to their frozen internal architecture; for example, `mlp_match_lower` still
constructs `(344, 172, 86)` internally.

## Restore and run a policy

```python
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jnp

from go1_core import load_checkpoint

networks, params, config = load_checkpoint(
    "/path/to/checkpoints/000412876800",
    expected_model="multirank_hodge_lower",
)
policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
observation = {
    "state": jnp.zeros((1, 48), dtype=jnp.float32),
    "privileged_state": jnp.zeros((1, 123), dtype=jnp.float32),
}
actions, extras = policy(observation, jax.random.PRNGKey(0))
assert actions.shape == (1, 12)
```

`load_checkpoint` returns `(networks, params, config)` in the historical order.
See `examples/inference.py` for a complete task-neutral executable example.

## Certification and limits

The six historical terminal checkpoints passed exact parameter-tree,
checkpoint restore, logits, and deterministic-action parity on CPU. A fresh
`multirank_hodge_lower` policy also completed a bounded 22,937,600-step PPO
training certification, exact save/restore roundtrip, and a 109-condition
rollout through the external consumer.

Historical long-horizon GPU CSV parity is not claimed. The frozen evaluator is
itself non-repeatable on the available GPU runtime; the exact Job 3 and Job 3A
status is recorded in `RELEASE_PROVENANCE.md`.

The public API is limited to the exports in `go1_core.__all__`. Scientific
provenance and frozen fingerprints are recorded in `EXTRACTION_MANIFEST.md`,
`PROVENANCE.md`, and `RELEASE_PROVENANCE.md`.
