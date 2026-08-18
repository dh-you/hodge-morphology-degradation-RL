"""Checkpoint-compatible actor modules for the six canonical models."""

from __future__ import annotations

from collections.abc import Mapping

from brax.training import networks as brax_networks
from flax import linen as nn
import jax
import jax.numpy as jnp

from go1_core._multirank import MultiRankEncoder
from go1_core.features import NodeDecoder, PointwiseEncoder, build_joint_features
from go1_core.registry import get_model_spec
from go1_core.topology import GCN_NORMALIZED_ADJACENCY


class GCNEncoder(nn.Module):
  """Two-layer Kipf-Welling-style normalized graph convolution."""

  width: int = 128

  def setup(self):
    self.convolution_0 = nn.Dense(self.width, name="convolution_0")
    self.convolution_1 = nn.Dense(128, name="convolution_1")

  def first_layer(
      self,
      nodes: jax.Array,
      normalized_adjacency: jax.Array = GCN_NORMALIZED_ADJACENCY,
  ) -> jax.Array:
    nodes = jnp.einsum("ij,...jc->...ic", normalized_adjacency, nodes)
    return nn.relu(self.convolution_0(nodes))

  def __call__(
      self,
      nodes: jax.Array,
      normalized_adjacency: jax.Array = GCN_NORMALIZED_ADJACENCY,
  ) -> jax.Array:
    nodes = self.first_layer(nodes, normalized_adjacency)
    nodes = jnp.einsum("ij,...jc->...ic", normalized_adjacency, nodes)
    return self.convolution_1(nodes)


class CapacityStructuredPolicy(nn.Module):
  """Historical pointwise and matched-GCN actor module."""

  model_name: str

  def setup(self):
    config = get_model_spec(self.model_name)
    if config.family == "pointwise":
      self.encoder = PointwiseEncoder(config.width, name="encoder")
    elif config.family == "gcn":
      self.encoder = GCNEncoder(config.width, name="encoder")
    else:
      raise ValueError(
          f"structured capacity actor cannot build {self.model_name!r}"
      )
    self.decoder = NodeDecoder(name="decoder")

  def encode(self, node_features: jax.Array) -> jax.Array:
    if isinstance(self.encoder, GCNEncoder):
      return self.encoder(node_features, GCN_NORMALIZED_ADJACENCY)
    return self.encoder(node_features)

  def per_joint_parameters(self, node_features: jax.Array) -> jax.Array:
    return self.decoder(self.encode(node_features))

  def __call__(self, node_features: jax.Array) -> jax.Array:
    per_joint = self.per_joint_parameters(node_features)
    return jnp.concatenate(
        (per_joint[..., 0], per_joint[..., 1]), axis=-1,
    )


class CleanStructuredPolicy(nn.Module):
  """Historical persistent multi-rank actor module."""

  model_name: str

  def setup(self):
    config = get_model_spec(self.model_name)
    if config.family != "multirank":
      raise ValueError(
          f"clean structured actor cannot build {self.model_name!r}"
      )
    self.encoder = MultiRankEncoder(
        max_rank=config.max_rank,
        metric_mode=config.metric_mode,
        width=config.width,
        num_layers=config.num_layers,
        name="encoder",
    )
    self.decoder_hidden = nn.Dense(48, name="decoder_hidden")
    self.decoder_output = nn.Dense(2, name="decoder_output")

  def encode(self, node_features: jax.Array) -> jax.Array:
    return self.encoder(node_features).x0

  def per_joint_parameters(self, node_features: jax.Array) -> jax.Array:
    hidden = nn.swish(self.decoder_hidden(self.encode(node_features)))
    return self.decoder_output(hidden)

  def __call__(self, node_features: jax.Array) -> jax.Array:
    per_joint = self.per_joint_parameters(node_features)
    return jnp.concatenate(
        (per_joint[..., 0], per_joint[..., 1]), axis=-1,
    )


def make_structured_policy_network(
    *, observation_size, preprocess_observations_fn, policy_obs_key: str,
    model_name: str,
):
  config = get_model_spec(model_name)
  if config.family in ("pointwise", "gcn"):
    module = CapacityStructuredPolicy(model_name)
  elif config.family == "multirank":
    module = CleanStructuredPolicy(model_name)
  else:
    raise ValueError(f"model has no structured actor: {model_name!r}")
  dummy_nodes = jnp.zeros((1, 12, 20), dtype=jnp.float32)

  def apply(processor_params, policy_params, observation):
    if not isinstance(observation, Mapping):
      raise TypeError("Go1 structured policies require keyed observations")
    normalized_state = preprocess_observations_fn(
        observation[policy_obs_key],
        brax_networks.normalizer_select(processor_params, policy_obs_key),
    )
    return module.apply(policy_params, build_joint_features(normalized_state))

  return brax_networks.FeedForwardNetwork(
      init=lambda key: module.init(key, dummy_nodes),
      apply=apply,
  )
