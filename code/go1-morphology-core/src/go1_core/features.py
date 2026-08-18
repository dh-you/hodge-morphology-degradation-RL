"""Frozen 48D-state to per-joint feature construction."""

from __future__ import annotations

from flax import linen as nn
import jax
import jax.numpy as jnp

from go1_core.topology import STATIC_FEATURES

KERNEL_INIT = jax.nn.initializers.lecun_uniform()
BIAS_INIT = jax.nn.initializers.zeros


def _dense(features: int, name: str, bias: bool = True) -> nn.Dense:
  return nn.Dense(
      features,
      use_bias=bias,
      kernel_init=KERNEL_INIT,
      bias_init=BIAS_INIT,
      name=name,
  )


def build_joint_features(normalized_state: jax.Array) -> jax.Array:
  """Maps normalized official 48D state observations to twelve 20D nodes."""
  if normalized_state.shape[-1] != 48:
    raise ValueError(
        f"expected final state dimension 48, got {normalized_state.shape}"
    )
  shared = jnp.broadcast_to(
      normalized_state[..., None, :9],
      normalized_state.shape[:-1] + (12, 9),
  )
  command = jnp.broadcast_to(
      normalized_state[..., None, 45:48],
      normalized_state.shape[:-1] + (12, 3),
  )
  local = jnp.stack(
      (
          normalized_state[..., 9:21],
          normalized_state[..., 21:33],
          normalized_state[..., 33:45],
      ),
      axis=-1,
  )
  static = jnp.broadcast_to(
      STATIC_FEATURES,
      normalized_state.shape[:-1] + STATIC_FEATURES.shape,
  )
  return jnp.concatenate((shared, command, local, static), axis=-1)


class PointwiseEncoder(nn.Module):
  width: int

  @nn.compact
  def __call__(self, node_features: jax.Array) -> jax.Array:
    nodes = nn.swish(_dense(self.width, "input")(node_features))
    for layer in range(2):
      nodes = nn.swish(_dense(self.width, f"self_{layer}")(nodes))
    return _dense(128, "projection")(nodes)


class NodeDecoder(nn.Module):

  @nn.compact
  def __call__(self, embeddings: jax.Array) -> jax.Array:
    hidden = nn.swish(_dense(48, "hidden")(embeddings))
    return _dense(2, "output")(hidden)
