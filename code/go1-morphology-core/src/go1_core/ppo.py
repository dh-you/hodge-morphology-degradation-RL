"""Brax PPO network construction for canonical Go1 checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks

from go1_core._models import make_structured_policy_network
from go1_core.contracts import assert_compatible_environment
from go1_core.registry import get_model_spec

HISTORICAL_POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)
HISTORICAL_VALUE_HIDDEN_LAYER_SIZES = (512, 256, 128)


def _shape(value: Any) -> tuple[int, ...]:
  if isinstance(value, Mapping) and "shape" in value:
    return tuple(value["shape"])
  if hasattr(value, "shape"):
    return tuple(value.shape)
  if isinstance(value, int):
    return (value,)
  return tuple(value)


def _canonical_observation_size(observation_size: Mapping[str, Any]):
  return {key: _shape(value) for key, value in observation_size.items()}


def make_go1_ppo_networks(
    observation_size,
    action_size,
    preprocess_observations_fn=types.identity_observation_preprocessor,
    model_name="native_mlp",
    policy_hidden_layer_sizes=HISTORICAL_POLICY_HIDDEN_LAYER_SIZES,
    value_hidden_layer_sizes=HISTORICAL_VALUE_HIDDEN_LAYER_SIZES,
    policy_obs_key="state",
    value_obs_key="privileged_state",
) -> ppo_networks.PPONetworks:
  """Builds a frozen canonical actor and the historical shared PPO critic."""
  spec = get_model_spec(model_name)
  if policy_obs_key != "state" or value_obs_key != "privileged_state":
    raise ValueError(
        "Go1 requires policy_obs_key='state' and "
        "value_obs_key='privileged_state'"
    )
  if tuple(policy_hidden_layer_sizes) != HISTORICAL_POLICY_HIDDEN_LAYER_SIZES:
    raise ValueError(
        "historical serialized policy_hidden_layer_sizes must be "
        f"{HISTORICAL_POLICY_HIDDEN_LAYER_SIZES}"
    )
  if tuple(value_hidden_layer_sizes) != HISTORICAL_VALUE_HIDDEN_LAYER_SIZES:
    raise ValueError(
        "historical serialized value_hidden_layer_sizes must be "
        f"{HISTORICAL_VALUE_HIDDEN_LAYER_SIZES}"
    )
  assert_compatible_environment(observation_size, action_size)
  observation_size = _canonical_observation_size(observation_size)

  if spec.family in ("native_mlp", "matched_mlp"):
    actor_widths = spec.hidden_layer_sizes
    return ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=actor_widths,
        value_hidden_layer_sizes=HISTORICAL_VALUE_HIDDEN_LAYER_SIZES,
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
    )

  base = ppo_networks.make_ppo_networks(
      observation_size=observation_size,
      action_size=action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      policy_hidden_layer_sizes=HISTORICAL_POLICY_HIDDEN_LAYER_SIZES,
      value_hidden_layer_sizes=HISTORICAL_VALUE_HIDDEN_LAYER_SIZES,
      policy_obs_key=policy_obs_key,
      value_obs_key=value_obs_key,
  )
  return ppo_networks.PPONetworks(
      policy_network=make_structured_policy_network(
          observation_size=observation_size,
          preprocess_observations_fn=preprocess_observations_fn,
          policy_obs_key=policy_obs_key,
          model_name=model_name,
      ),
      value_network=base.value_network,
      parametric_action_distribution=base.parametric_action_distribution,
  )
