"""Runtime topology closure for the six frozen Go1 models."""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np

TOPOLOGY_NAME = "go1_joint_planar_closure"
TOPOLOGY_SCHEMA_VERSION = 1

JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)
EDGES = (
    (0, 1), (1, 2), (3, 4), (4, 5),
    (6, 7), (7, 8), (9, 10), (10, 11),
    (3, 0), (0, 6), (6, 9), (9, 3),
    (0, 2), (3, 5), (6, 8), (9, 11),
)
EDGE_RELATIONS = (
    *(("physical_serial_limb",) * 8),
    *(("four_haa_body_loop",) * 4),
    *(("virtual_planar_limb_closure",) * 4),
)
FACE_NAMES = ("FR_limb", "FL_limb", "RR_limb", "RL_limb", "four_HAA_body_loop")
FACES = (
    ((0, 1.0), (1, 1.0), (12, -1.0)),
    ((2, 1.0), (3, 1.0), (13, -1.0)),
    ((4, 1.0), (5, 1.0), (14, -1.0)),
    ((6, 1.0), (7, 1.0), (15, -1.0)),
    ((8, 1.0), (9, 1.0), (10, 1.0), (11, 1.0)),
)


def _incidence_matrices() -> tuple[np.ndarray, np.ndarray]:
  b1 = np.zeros((12, 16), dtype=np.float64)
  for edge, (source, target) in enumerate(EDGES):
    b1[source, edge], b1[target, edge] = -1.0, 1.0
  b2 = np.zeros((16, 5), dtype=np.float64)
  for face, terms in enumerate(FACES):
    for edge, coefficient in terms:
      b2[edge, face] = coefficient
  return b1, b2


def _gcn_operators(b1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  unsigned_incidence = np.abs(b1)
  shared_edges = unsigned_incidence @ unsigned_incidence.T
  adjacency = (shared_edges > 0).astype(np.float32)
  np.fill_diagonal(adjacency, 0.0)
  adjacency_with_self_loops = (
      adjacency + np.eye(adjacency.shape[0], dtype=np.float32)
  )
  degree = adjacency_with_self_loops.sum(axis=-1)
  inverse_sqrt_degree = np.diag(np.power(degree, -0.5))
  normalized = (
      inverse_sqrt_degree @ adjacency_with_self_loops @ inverse_sqrt_degree
  )
  return adjacency_with_self_loops, normalized.astype(np.float32)


B1, B2 = _incidence_matrices()
B1_JAX = jnp.asarray(B1, dtype=jnp.float32)
B2_JAX = jnp.asarray(B2, dtype=jnp.float32)
GCN_ADJACENCY_WITH_SELF_LOOPS, _GCN_NORMALIZED_ADJACENCY = _gcn_operators(B1)
GCN_NORMALIZED_ADJACENCY = jnp.asarray(
    _GCN_NORMALIZED_ADJACENCY, dtype=jnp.float32,
)

STATIC_FEATURES = jnp.asarray([
    (1, -1, 1, 0, 0), (1, -1, 0, 1, 0), (1, -1, 0, 0, 1),
    (1, 1, 1, 0, 0), (1, 1, 0, 1, 0), (1, 1, 0, 0, 1),
    (-1, -1, 1, 0, 0), (-1, -1, 0, 1, 0), (-1, -1, 0, 0, 1),
    (-1, 1, 1, 0, 0), (-1, 1, 0, 1, 0), (-1, 1, 0, 0, 1),
], dtype=jnp.float32)

ARRAY_SHA256 = {
    "B1": "15249a6102b39eda0c7fc77853ac91007c937acbb39b667f77ca7982f9539e8f",
    "B2": "5247ce6fa57bc1de8887c3044e6bda647ac0382dc9ff04846494af1f99bf4187",
    "GCN_ADJACENCY_WITH_SELF_LOOPS":
        "35c5745d515c4dc1bd010acbe9aaa75c0ff9a9bd1fe6bdbf402dfa4dd5ec9a3b",
    "GCN_NORMALIZED_ADJACENCY":
        "8b8a9c5214898160f687eb801d6cff1dfcb6964d908404a771e77b5bdf415e3a",
    "STATIC_FEATURES":
        "5f61c716b7a912ce1534b675061cc68c56a451f5b7ab7e38b1f056088c5786d5",
}


def array_sha256(value: np.ndarray | jax.Array) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate() -> None:
  assert B1.shape == (12, 16) and B1.dtype == np.float64
  assert B2.shape == (16, 5) and B2.dtype == np.float64
  assert np.linalg.matrix_rank(B1) == 11
  assert np.linalg.matrix_rank(B2) == 5
  assert np.array_equal(B1 @ B2, np.zeros((12, 5)))
  assert len({tuple(sorted(edge)) for edge in EDGES}) == len(EDGES)
  assert all(source != target for source, target in EDGES)
  arrays = {
      "B1": B1,
      "B2": B2,
      "GCN_ADJACENCY_WITH_SELF_LOOPS": GCN_ADJACENCY_WITH_SELF_LOOPS,
      "GCN_NORMALIZED_ADJACENCY": _GCN_NORMALIZED_ADJACENCY,
      "STATIC_FEATURES": STATIC_FEATURES,
  }
  assert all(array_sha256(value) == ARRAY_SHA256[name]
             for name, value in arrays.items())


validate()
