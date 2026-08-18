"""Task-neutral checkpoint restore and deterministic Go1 inference example."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jnp
import numpy as np

from go1_core import (
    CANONICAL_MODELS,
    GO1_CONTRACT_V1,
    MODEL_SPECS,
    assert_compatible_environment,
    load_checkpoint,
    make_go1_ppo_networks,
)


def observation_spec() -> dict[str, jax.ShapeDtypeStruct]:
  """Returns the frozen keyed observation interface without a task package."""
  return {
      GO1_CONTRACT_V1.policy_obs_key: jax.ShapeDtypeStruct(
          GO1_CONTRACT_V1.policy_observation_shape, jnp.float32,
      ),
      GO1_CONTRACT_V1.value_obs_key: jax.ShapeDtypeStruct(
          GO1_CONTRACT_V1.value_observation_shape, jnp.float32,
      ),
  }


def probe_observation() -> dict[str, jax.Array]:
  """Returns the documented three-row deterministic policy probe."""
  state = jnp.stack((
      jnp.zeros((48,), dtype=jnp.float32),
      jnp.linspace(-1.0, 1.0, 48, dtype=jnp.float32),
      jnp.sin(jnp.arange(48, dtype=jnp.float32)),
  ))
  return {
      GO1_CONTRACT_V1.policy_obs_key: state,
      GO1_CONTRACT_V1.value_obs_key: jnp.zeros(
          (3, 123), dtype=jnp.float32,
      ),
  }


def run(checkpoint: Path, model: str) -> dict[str, object]:
  """Validates the contract, restores one checkpoint, and runs inference."""
  if version("go1-morphology-core") != "1.0.0":
    raise RuntimeError("this example certifies go1-morphology-core 1.0.0")

  spec = observation_spec()
  assert_compatible_environment(spec, GO1_CONTRACT_V1.action_size)
  make_go1_ppo_networks(
      spec,
      GO1_CONTRACT_V1.action_size,
      model_name=model,
      policy_hidden_layer_sizes=(512, 256, 128),
      value_hidden_layer_sizes=(512, 256, 128),
      policy_obs_key=GO1_CONTRACT_V1.policy_obs_key,
      value_obs_key=GO1_CONTRACT_V1.value_obs_key,
  )

  networks, params, config = load_checkpoint(
      checkpoint, expected_model=model,
  )
  actor_parameters = sum(
      int(np.asarray(leaf).size)
      for leaf in jax.tree_util.tree_leaves(params[1])
  )
  if actor_parameters != MODEL_SPECS[model].actor_parameters:
    raise RuntimeError("restored actor parameter count changed")

  observation = probe_observation()
  logits = networks.policy_network.apply(params[0], params[1], observation)
  policy = ppo_networks.make_inference_fn(networks)(
      params, deterministic=True,
  )
  actions, unused_extras = policy(observation, jax.random.PRNGKey(0))
  logits_array = np.asarray(logits)
  actions_array = np.asarray(actions)
  if logits_array.shape != (3, 24) or actions_array.shape != (3, 12):
    raise RuntimeError("policy output shapes changed")
  if not np.isfinite(logits_array).all() or not np.isfinite(actions_array).all():
    raise RuntimeError("policy outputs contain non-finite values")
  if np.any(actions_array < -1.0) or np.any(actions_array > 1.0):
    raise RuntimeError("deterministic actions exceed [-1, 1]")

  return {
      "action_shape": list(actions_array.shape),
      "actor_parameters": actor_parameters,
      "checkpoint": str(checkpoint.resolve()),
      "distribution_parameter_shape": list(logits_array.shape),
      "model": model,
      "package_version": version("go1-morphology-core"),
      "serialized_model": config.network_factory_kwargs.to_dict()["model_name"],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument(
      "--model", choices=CANONICAL_MODELS,
      default="multirank_hodge_lower",
  )
  args = parser.parse_args()
  print(json.dumps(run(args.checkpoint, args.model), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
