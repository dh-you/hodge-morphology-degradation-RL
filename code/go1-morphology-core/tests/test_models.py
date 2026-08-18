from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest

from go1_core import (
    CANONICAL_MODELS,
    MODEL_SPECS,
    make_go1_ppo_networks,
)
from tests.helpers import tree_signature

OBSERVATION_SIZE = {"state": (48,), "privileged_state": (123,)}
EXPECTED_ACTOR_TREES = {
    "native_mlp": (8, "09f36e3da7227b35f65b62a8c051016cb07d3ea70382bc3bb2783d617c13eba6"),
    "mlp_match_lower": (8, "c0ab68491e8b10aa90b2d486e0d6dab17b6d3493ccd95e53732b405963f2d4ec"),
    "pointwise_match_lower": (12, "2a6d48b8c46e55893b02977636253032d5828ad04a3e42828cd60d1033fc8ace"),
    "gcn_match_lower": (8, "4f9ee45b773dd67d5815eefc55b5b3257b4ecadf0d014b6b9c17d08f7a2bb536"),
    "multirank_hodge_lower": (56, "9bf4e874136e4fad66509706a0e838736a9e928b8bbad71608606358bea4450d"),
    "multirank_hodge_full": (80, "375f9ee6132449eaebe213f8798551184eb186a656dad1716b9c266fb3aabe26"),
}

EXPERIMENTAL_ACTOR_COUNTS = {
    "native_mlp_w480_240_120": 170_784,
    "pointwise_w252": 171_478,
    "gcn_w1107": 171_361,
    "multirank_hodge_lower_w177": 171_705,
    "multirank_hodge_full_w150": 171_296,
}


@pytest.mark.parametrize("model", CANONICAL_MODELS)
def test_actor_parameter_count_and_tree_signature(model):
  networks = make_go1_ppo_networks(
      OBSERVATION_SIZE, 12, model_name=model,
  )
  params = networks.policy_network.init(jax.random.PRNGKey(0))
  leaves, scalars, digest = tree_signature(params)
  expected_leaves, expected_digest = EXPECTED_ACTOR_TREES[model]
  assert scalars == MODEL_SPECS[model].actor_parameters
  assert leaves == expected_leaves
  assert digest == expected_digest
  assert networks.parametric_action_distribution.param_size == 24
  actions = networks.parametric_action_distribution.mode(
      jnp.zeros((1, 24), dtype=jnp.float32)
  )
  assert actions.shape == (1, 12)


@pytest.mark.parametrize("model, expected", EXPERIMENTAL_ACTOR_COUNTS.items())
def test_experimental_actor_parameter_count(model, expected):
  networks = make_go1_ppo_networks(
      OBSERVATION_SIZE, 12, model_name=model,
  )
  params = networks.policy_network.init(jax.random.PRNGKey(0))
  assert tree_signature(params)[1] == expected
  assert MODEL_SPECS[model].actor_parameters == expected
  assert model not in CANONICAL_MODELS


def test_historical_factory_widths_are_authoritative_outer_kwargs():
  signature = inspect.signature(make_go1_ppo_networks)
  assert signature.parameters["policy_hidden_layer_sizes"].default == (512, 256, 128)
  assert signature.parameters["value_hidden_layer_sizes"].default == (512, 256, 128)
  matched = make_go1_ppo_networks(
      OBSERVATION_SIZE,
      12,
      model_name="mlp_match_lower",
      policy_hidden_layer_sizes=[512, 256, 128],
      value_hidden_layer_sizes=[512, 256, 128],
  )
  params = matched.policy_network.init(jax.random.PRNGKey(0))
  assert tree_signature(params)[1] == 93_162
  with pytest.raises(ValueError, match="historical serialized policy"):
    make_go1_ppo_networks(
        OBSERVATION_SIZE,
        12,
        model_name="mlp_match_lower",
        policy_hidden_layer_sizes=(344, 172, 86),
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"policy_obs_key": "actor"},
        {"value_obs_key": "critic"},
        {"value_hidden_layer_sizes": (256, 128)},
        {"model_name": "rhmp"},
    ),
)
def test_factory_rejects_noncanonical_interfaces(kwargs):
  with pytest.raises(ValueError):
    make_go1_ppo_networks(OBSERVATION_SIZE, 12, **kwargs)
