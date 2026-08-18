"""Private cell-space metrics shared by the frozen MultiRank implementation."""

from __future__ import annotations

import math

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class IdentityMetric:
  """Parameter-free identity metric."""


@struct.dataclass
class SPDMetric:
  """A diagonal-plus-low-rank symmetric positive-definite metric."""

  diagonal: jax.Array
  factor: jax.Array


def apply_metric(
    metric: IdentityMetric | SPDMetric, values: jax.Array,
) -> jax.Array:
  if isinstance(metric, IdentityMetric):
    return values
  if not isinstance(metric, SPDMetric):
    raise TypeError(f"unsupported metric type: {type(metric)!r}")
  projected = jnp.einsum("...ir,...ic->...rc", metric.factor, values)
  low_rank = jnp.einsum("...ir,...rc->...ic", metric.factor, projected)
  return low_rank + metric.diagonal[..., :, None] * values


class MetricPredictor(nn.Module):
  """Private learned-metric path retained because MultiRank shares it."""

  hidden_width: int = 16
  metric_rank: int = 8
  epsilon: float = 1e-6
  output_initialization: str = "standard"

  @nn.compact
  def __call__(
      self, cells: jax.Array, unsigned_neighborhood: jax.Array,
  ) -> SPDMetric:
    if cells.shape[-2] != unsigned_neighborhood.shape[0]:
      raise ValueError(
          "cell count and neighborhood size disagree: "
          f"{cells.shape[-2]} != {unsigned_neighborhood.shape[0]}"
      )
    if unsigned_neighborhood.shape[0] != unsigned_neighborhood.shape[1]:
      raise ValueError("unsigned_neighborhood must be square")

    degree = jnp.sum(unsigned_neighborhood, axis=-1)
    neighbor_average = unsigned_neighborhood / jnp.maximum(degree, 1.0)[:, None]
    neighbor_cells = jnp.einsum("ij,...jc->...ic", neighbor_average, cells)
    cell_energy = jnp.mean(jnp.square(cells), axis=-1, keepdims=True)
    neighbor_energy = jnp.einsum(
        "ij,...jf->...if", neighbor_average, cell_energy,
    )
    neighbor_inner_product = jnp.sum(
        cells * neighbor_cells, axis=-1, keepdims=True,
    )
    broadcast_degree = jnp.broadcast_to(
        degree, cells.shape[:-2] + (cells.shape[-2],),
    )[..., None]
    statistics = jnp.concatenate(
        (
            cell_energy,
            neighbor_inner_product,
            neighbor_energy,
            broadcast_degree,
        ),
        axis=-1,
    )

    hidden = nn.swish(nn.Dense(self.hidden_width, name="hidden")(statistics))
    if self.output_initialization == "standard":
      raw = nn.Dense(1 + self.metric_rank, name="output")(hidden)
      diagonal = jax.nn.softplus(raw[..., 0]) + self.epsilon
      return SPDMetric(diagonal=diagonal, factor=raw[..., 1:])
    if self.output_initialization != "near_identity":
      raise ValueError(
          f"unknown metric output initialization: {self.output_initialization!r}"
      )

    identity_bias = math.log(math.expm1(1.0 - self.epsilon))
    raw_diagonal = nn.Dense(
        1,
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.constant(identity_bias),
        name="diagonal_head",
    )(hidden)
    factor = nn.Dense(
        self.metric_rank,
        kernel_init=nn.initializers.normal(stddev=1e-3),
        bias_init=nn.initializers.zeros,
        name="factor_head",
    )(hidden)
    diagonal = jax.nn.softplus(raw_diagonal[..., 0]) + self.epsilon
    return SPDMetric(diagonal=diagonal, factor=factor)
