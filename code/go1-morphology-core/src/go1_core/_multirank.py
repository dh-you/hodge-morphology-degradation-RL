"""Private frozen persistent multi-rank incidence propagation."""

from __future__ import annotations

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from go1_core._metrics import (
    IdentityMetric,
    MetricPredictor,
    SPDMetric,
    apply_metric,
)
from go1_core.topology import (
    B1,
    B1_JAX,
    B2,
    B2_JAX,
    EDGES,
    EDGE_RELATIONS,
    FACES,
    FACE_NAMES,
)

_RELATION_NAMES = (
    "physical_serial_limb",
    "four_haa_body_loop",
    "virtual_planar_limb_closure",
)
_RELATION_INDEX = {name: index for index, name in enumerate(_RELATION_NAMES)}
_EDGE_SOURCES = jnp.asarray([edge[0] for edge in EDGES], dtype=jnp.int32)
_EDGE_TARGETS = jnp.asarray([edge[1] for edge in EDGES], dtype=jnp.int32)
_EDGE_RELATION_ONE_HOT = jax.nn.one_hot(
    jnp.asarray([_RELATION_INDEX[name] for name in EDGE_RELATIONS]), 3,
)
_NODE_DEGREE = jnp.asarray(np.abs(B1).sum(axis=1), dtype=jnp.float32)
_FACE_BOUNDARY_SIZE = jnp.asarray(
    [len(boundary) for boundary in FACES], dtype=jnp.float32,
)
_FACE_BOUNDARY_SIZE = _FACE_BOUNDARY_SIZE / jnp.max(_FACE_BOUNDARY_SIZE)
_FACE_TYPE_ONE_HOT = jax.nn.one_hot(
    jnp.asarray([0 if name.endswith("_limb") else 1 for name in FACE_NAMES]), 2,
)


def _unsigned_neighborhood(incidence: np.ndarray) -> np.ndarray:
  shared = np.abs(incidence) @ np.abs(incidence).T
  neighborhood = (shared > 0).astype(np.float32)
  np.fill_diagonal(neighborhood, 0.0)
  return neighborhood


_RANK0_NEIGHBORHOOD = jnp.asarray(
    _unsigned_neighborhood(B1), dtype=jnp.float32,
)
_RANK1_LOWER_NEIGHBORHOOD = jnp.asarray(
    _unsigned_neighborhood(B1.T), dtype=jnp.float32,
)
_RANK1_UPPER_NEIGHBORHOOD = _unsigned_neighborhood(B2)
_RANK1_FULL_NEIGHBORHOOD = jnp.asarray(
    np.maximum(
        np.asarray(_RANK1_LOWER_NEIGHBORHOOD),
        _RANK1_UPPER_NEIGHBORHOOD,
    ),
    dtype=jnp.float32,
)
_RANK2_NEIGHBORHOOD = jnp.asarray(
    _unsigned_neighborhood(B2.T), dtype=jnp.float32,
)


def _propagate(operator: jax.Array, values: jax.Array) -> jax.Array:
  return jnp.einsum("ij,...jc->...ic", operator, values)


@struct.dataclass
class CochainState:
  x0: jax.Array
  x1: jax.Array
  x2: jax.Array | None = None


class NodeLift(nn.Module):
  width: int = 128

  @nn.compact
  def __call__(self, node_features: jax.Array) -> jax.Array:
    nodes = nn.swish(nn.Dense(self.width, name="hidden")(node_features))
    return nn.Dense(self.width, name="output")(nodes)


class EdgeLift(nn.Module):
  width: int = 128

  @nn.compact
  def __call__(self, nodes: jax.Array) -> jax.Array:
    sources = jnp.take(nodes, _EDGE_SOURCES, axis=-2)
    targets = jnp.take(nodes, _EDGE_TARGETS, axis=-2)
    prefix = nodes.shape[:-2]
    source_degree = jnp.broadcast_to(
        _NODE_DEGREE[_EDGE_SOURCES], prefix + (len(EDGES),),
    )[..., None]
    target_degree = jnp.broadcast_to(
        _NODE_DEGREE[_EDGE_TARGETS], prefix + (len(EDGES),),
    )[..., None]
    relations = jnp.broadcast_to(
        _EDGE_RELATION_ONE_HOT, prefix + _EDGE_RELATION_ONE_HOT.shape,
    )
    symmetric_input = jnp.concatenate(
        (sources + targets, source_degree + target_degree, relations), axis=-1,
    )
    antisymmetric_input = jnp.concatenate(
        (sources - targets, source_degree - target_degree, relations), axis=-1,
    )

    symmetric = nn.swish(
        nn.Dense(self.width, name="symmetric_hidden")(symmetric_input)
    )
    symmetric = nn.Dense(self.width, name="symmetric_output")(symmetric)
    antisymmetric = nn.swish(
        nn.Dense(self.width, name="antisymmetric_hidden")(antisymmetric_input)
    )
    antisymmetric = nn.Dense(
        self.width, name="antisymmetric_output",
    )(antisymmetric)
    return symmetric + antisymmetric


class FaceLift(nn.Module):
  width: int = 128

  @nn.compact
  def __call__(self, edges: jax.Array, b2: jax.Array = B2_JAX) -> jax.Array:
    boundary_signal = _propagate(b2.T, edges)
    prefix = edges.shape[:-2]
    boundary_size = jnp.broadcast_to(
        _FACE_BOUNDARY_SIZE, prefix + (_FACE_BOUNDARY_SIZE.shape[0],),
    )[..., None]
    face_type = jnp.broadcast_to(
        _FACE_TYPE_ONE_HOT, prefix + _FACE_TYPE_ONE_HOT.shape,
    )
    features = jnp.concatenate(
        (boundary_signal, boundary_size, face_type), axis=-1,
    )
    hidden = nn.swish(nn.Dense(self.width, name="hidden")(features))
    return nn.Dense(self.width, name="output")(hidden)


class NormGate(nn.Module):
  hidden_width: int = 16
  epsilon: float = 1e-6

  @nn.compact
  def __call__(self, messages: jax.Array) -> jax.Array:
    rms = jnp.sqrt(
        jnp.mean(jnp.square(messages), axis=-1, keepdims=True) + self.epsilon
    )
    hidden = nn.swish(nn.Dense(self.hidden_width, name="hidden")(rms))
    multiplier = nn.sigmoid(nn.Dense(1, name="output")(hidden))
    return messages * multiplier


class RMSNormNoAffine(nn.Module):
  epsilon: float = 1e-6

  @nn.compact
  def __call__(self, values: jax.Array) -> jax.Array:
    return values * jax.lax.rsqrt(
        jnp.mean(jnp.square(values), axis=-1, keepdims=True) + self.epsilon
    )


class MultiRankLayer(nn.Module):
  max_rank: int
  metric_mode: str
  metric_hidden_width: int = 16
  metric_rank: int = 8
  epsilon: float = 1e-6
  metric_initialization: str = "standard"

  def _metric(
      self,
      cells: jax.Array,
      neighborhood: jax.Array,
      name: str,
  ) -> IdentityMetric | SPDMetric:
    if self.metric_mode == "identity":
      return IdentityMetric()
    if self.metric_mode != "learned_spd":
      raise ValueError(f"unknown metric mode: {self.metric_mode!r}")
    return MetricPredictor(
        hidden_width=self.metric_hidden_width,
        metric_rank=self.metric_rank,
        epsilon=self.epsilon,
        output_initialization=self.metric_initialization,
        name=name,
    )(cells, neighborhood)

  @nn.compact
  def __call__(
      self,
      state: CochainState,
      b1: jax.Array = B1_JAX,
      b2: jax.Array = B2_JAX,
  ) -> CochainState:
    if self.max_rank not in (1, 2):
      raise ValueError(f"max_rank must be 1 or 2, got {self.max_rank}")
    if self.max_rank == 2 and state.x2 is None:
      raise ValueError("rank-two propagation requires a face cochain")

    rank1_neighborhood = (
        _RANK1_LOWER_NEIGHBORHOOD
        if self.max_rank == 1 else _RANK1_FULL_NEIGHBORHOOD
    )
    h1_upper = self._metric(
        state.x1, rank1_neighborhood, "metric_rank1_upper",
    )
    h1_cross = self._metric(
        state.x1, rank1_neighborhood, "metric_rank1_cross",
    )
    h0_lower = self._metric(
        state.x0, _RANK0_NEIGHBORHOOD, "metric_rank0_lower",
    )
    h0_cross = self._metric(
        state.x0, _RANK0_NEIGHBORHOOD, "metric_rank0_cross",
    )

    b1_transpose_x0 = _propagate(b1.T, state.x0)
    b1_x1 = _propagate(b1, state.x1)
    m0_self = _propagate(b1, apply_metric(h1_upper, b1_transpose_x0))
    m0_cross = _propagate(b1, apply_metric(h1_cross, state.x1))
    m1_self = _propagate(b1.T, apply_metric(h0_lower, b1_x1))
    m1_cross = _propagate(b1.T, apply_metric(h0_cross, state.x0))

    m2_self = None
    m2_cross = None
    if self.max_rank == 2:
      h2_upper = self._metric(
          state.x2, _RANK2_NEIGHBORHOOD, "metric_rank2_upper",
      )
      h2_cross = self._metric(
          state.x2, _RANK2_NEIGHBORHOOD, "metric_rank2_cross",
      )
      h1_lower = self._metric(
          state.x1, rank1_neighborhood, "metric_rank1_lower",
      )
      b2_transpose_x1 = _propagate(b2.T, state.x1)
      b2_x2 = _propagate(b2, state.x2)
      m1_self = m1_self + _propagate(
          b2, apply_metric(h2_upper, b2_transpose_x1),
      )
      m1_cross = m1_cross + _propagate(
          b2, apply_metric(h2_cross, state.x2),
      )
      m2_self = _propagate(b2.T, apply_metric(h1_lower, b2_x2))
      m2_cross = _propagate(b2.T, apply_metric(h1_cross, state.x1))

    updated = []
    messages = ((m0_self, m0_cross), (m1_self, m1_cross))
    if self.max_rank == 2:
      messages += ((m2_self, m2_cross),)
    previous = (state.x0, state.x1, state.x2)
    for rank, (same_rank, cross_rank) in enumerate(messages):
      raw_alpha = self.param(
          f"raw_alpha_{rank}", nn.initializers.zeros, (),
      )
      mixed = nn.sigmoid(raw_alpha) * same_rank
      mixed += (1.0 - nn.sigmoid(raw_alpha)) * cross_rank
      gated = NormGate(
          hidden_width=16,
          epsilon=self.epsilon,
          name=f"norm_gate_{rank}",
      )(mixed)
      normalized = RMSNormNoAffine(
          epsilon=self.epsilon, name=f"rms_norm_{rank}",
      )(gated)
      updated.append(previous[rank] + normalized)

    return CochainState(
        x0=updated[0],
        x1=updated[1],
        x2=None if self.max_rank == 1 else updated[2],
    )


class MultiRankEncoder(nn.Module):
  max_rank: int
  metric_mode: str
  width: int = 128
  num_layers: int = 4
  metric_hidden_width: int = 16
  metric_rank: int = 8
  epsilon: float = 1e-6
  metric_initialization: str = "standard"

  def setup(self):
    if self.max_rank not in (1, 2):
      raise ValueError(f"max_rank must be 1 or 2, got {self.max_rank}")
    if self.metric_mode not in ("identity", "learned_spd"):
      raise ValueError(f"unknown metric mode: {self.metric_mode!r}")
    if self.metric_initialization not in ("standard", "near_identity"):
      raise ValueError(
          f"unknown metric initialization: {self.metric_initialization!r}"
      )
    self.node_lift = NodeLift(self.width, name="node_lift")
    self.edge_lift = EdgeLift(self.width, name="edge_lift")
    if self.max_rank == 2:
      self.face_lift = FaceLift(self.width, name="face_lift")
    self.layers = tuple(
        MultiRankLayer(
            max_rank=self.max_rank,
            metric_mode=self.metric_mode,
            metric_hidden_width=self.metric_hidden_width,
            metric_rank=self.metric_rank,
            epsilon=self.epsilon,
            metric_initialization=self.metric_initialization,
            name=f"layer_{layer}",
        )
        for layer in range(self.num_layers)
    )

  def _initial_state(
      self,
      node_features: jax.Array,
      b2: jax.Array,
  ) -> CochainState:
    x0 = self.node_lift(node_features)
    x1 = self.edge_lift(x0)
    x2 = self.face_lift(x1, b2=b2) if self.max_rank == 2 else None
    return CochainState(x0=x0, x1=x1, x2=x2)

  def encode_with_history(
      self,
      node_features: jax.Array,
      b1: jax.Array = B1_JAX,
      b2: jax.Array = B2_JAX,
  ) -> tuple[CochainState, ...]:
    state = self._initial_state(node_features, b2)
    history = [state]
    for layer in self.layers:
      state = layer(state, b1=b1, b2=b2)
      history.append(state)
    return tuple(history)

  def propagate(
      self,
      state: CochainState,
      b1: jax.Array = B1_JAX,
      b2: jax.Array = B2_JAX,
  ) -> CochainState:
    for layer in self.layers:
      state = layer(state, b1=b1, b2=b2)
    return state

  def __call__(
      self,
      node_features: jax.Array,
      b1: jax.Array = B1_JAX,
      b2: jax.Array = B2_JAX,
  ) -> CochainState:
    return self.encode_with_history(node_features, b1=b1, b2=b2)[-1]
