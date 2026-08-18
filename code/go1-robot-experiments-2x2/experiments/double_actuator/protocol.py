"""Frozen pair split, curriculum, and deterministic assignment protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from go1_core import GO1_CONTRACT_V1

MODELS = (
    "multirank_hodge_full_w150",
    "multirank_hodge_lower_w177",
    "gcn_w1107",
    "pointwise_w252",
    "native_mlp_w480_240_120",
)
MODEL_PARAMETERS = {
    "native_mlp_w480_240_120": 170_784,
    "pointwise_w252": 171_478,
    "gcn_w1107": 171_361,
    "multirank_hodge_lower_w177": 171_705,
    "multirank_hodge_full_w150": 171_296,
}
MODEL_LABELS = {
    "native_mlp_w480_240_120": "Native MLP",
    "pointwise_w252": "Joint MLP",
    "gcn_w1107": "GCN",
    "multirank_hodge_lower_w177": "Hodge-L",
    "multirank_hodge_full_w150": "Hodge-F",
}
TARGET_PARAMETERS = 171_296
SEEDS = (11, 12, 13)
SMOKE_SEED = 10
STRENGTHS = (0.50, 0.25, 0.20, 0.15, 0.10, 0.075, 0.05, 0.025, 0.0)
HEALTHY_PROBABILITY = 0.30
WEAK_PROBABILITY = 0.50
DEAD_PROBABILITY = 0.20
MIN_WEAK_STRENGTH = 0.05
MAX_WEAK_STRENGTH = 0.20
CATEGORY_HEALTHY = 0
CATEGORY_WEAK = 1
CATEGORY_DEAD = 2
DEFAULT_BATCH_SIZE = 8_192


@dataclass(frozen=True)
class PairSpec:
  pair_id: int
  actuator_a: int
  actuator_b: int
  name_a: str
  name_b: str
  split: str
  topology: str


_SAME_LIMB_HELD_OUT = (
    ("FR_thigh", "FR_calf"),
    ("FL_hip", "FL_calf"),
    ("RR_thigh", "RR_calf"),
    ("RL_hip", "RL_thigh"),
)
_BODY_FACE_HELD_OUT = (
    ("FR_hip", "RL_hip"),
    ("FL_hip", "RR_hip"),
)
_CROSS_LIMB_HELD_OUT = (
    ("FL_hip", "RR_thigh"),
    ("FL_thigh", "RL_calf"),
    ("FL_thigh", "RL_hip"),
    ("FL_thigh", "RR_calf"),
    ("FL_thigh", "RR_hip"),
    ("FR_calf", "FL_calf"),
    ("FR_calf", "RL_thigh"),
    ("FR_calf", "RR_thigh"),
    ("FR_hip", "FL_calf"),
    ("FR_hip", "RL_calf"),
    ("FR_hip", "RL_thigh"),
    ("FR_thigh", "RL_calf"),
    ("FR_thigh", "RL_thigh"),
    ("FR_thigh", "RR_hip"),
    ("RR_calf", "RL_hip"),
    ("RR_thigh", "RL_calf"),
)
HELD_OUT_PAIRS = (
    *_SAME_LIMB_HELD_OUT,
    *_BODY_FACE_HELD_OUT,
    *_CROSS_LIMB_HELD_OUT,
)


def _unordered(pair: tuple[str, str]) -> frozenset[str]:
  return frozenset(pair)


def _topology(name_a: str, name_b: str) -> str:
  pair = _unordered((name_a, name_b))
  if pair in {_unordered(item) for item in _SAME_LIMB_HELD_OUT}:
    return "same_limb"
  if pair in {_unordered(item) for item in _BODY_FACE_HELD_OUT}:
    return "body_face"
  return "cross_limb"


def enumerate_pairs() -> tuple[PairSpec, ...]:
  names = tuple(GO1_CONTRACT_V1.action_names)
  held_out = {_unordered(pair) for pair in HELD_OUT_PAIRS}
  return tuple(
      PairSpec(
          pair_id=pair_id,
          actuator_a=index_a,
          actuator_b=index_b,
          name_a=names[index_a],
          name_b=names[index_b],
          split=("held_out" if _unordered((names[index_a], names[index_b])) in held_out else "train"),
          topology=_topology(names[index_a], names[index_b]),
      )
      for pair_id, (index_a, index_b) in enumerate(
          itertools.combinations(range(len(names)), 2)
      )
  )


PAIRS = enumerate_pairs()
TRAIN_PAIRS = tuple(pair for pair in PAIRS if pair.split == "train")
HELD_OUT_SPECS = tuple(pair for pair in PAIRS if pair.split == "held_out")


def canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


def pair_manifest() -> dict[str, Any]:
  rows = [
      {
          "pair_id": pair.pair_id,
          "actuator_a": pair.actuator_a,
          "actuator_b": pair.actuator_b,
          "name_a": pair.name_a,
          "name_b": pair.name_b,
          "split": pair.split,
          "topology": pair.topology,
      }
      for pair in PAIRS
  ]
  digest = hashlib.sha256(canonical_json(rows).encode()).hexdigest()
  return {"pairs": rows, "sha256": digest}


PAIR_MANIFEST = pair_manifest()
PAIR_MANIFEST_SHA256 = PAIR_MANIFEST["sha256"]
CURRICULUM = {
    "assignment_scope": "fixed_vectorized_environment_member",
    "reset_resampling": False,
    "injuries_per_nonhealthy_environment": 2,
    "category_probabilities": {
        "healthy": HEALTHY_PROBABILITY,
        "dual_weak": WEAK_PROBABILITY,
        "dual_dead": DEAD_PROBABILITY,
    },
    "weak_strength_distribution": {
        "family": "independent_log_uniform",
        "minimum": MIN_WEAK_STRENGTH,
        "maximum_exclusive": MAX_WEAK_STRENGTH,
    },
    "pair_sampling": "uniform_over_44_training_pairs",
    "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
    "random_streams": {
        "ppo_model_reset": "seed",
        "domain_randomization": "100000 + seed",
        "pair_injury": "200000 + seed",
    },
}
CURRICULUM_SHA256 = hashlib.sha256(canonical_json(CURRICULUM).encode()).hexdigest()


def validate_protocol() -> None:
  if len(PAIRS) != 66 or len(TRAIN_PAIRS) != 44 or len(HELD_OUT_SPECS) != 22:
    raise AssertionError("pair split must be 66 total, 44 train, and 22 held out")
  identities = {(pair.actuator_a, pair.actuator_b) for pair in PAIRS}
  if len(identities) != 66 or any(a >= b for a, b in identities):
    raise AssertionError("pairs must be unique, unordered canonical pairs")
  held_counts = {name: 0 for name in GO1_CONTRACT_V1.action_names}
  for pair in HELD_OUT_SPECS:
    held_counts[pair.name_a] += 1
    held_counts[pair.name_b] += 1
  if set(held_counts.values()) - {3, 4}:
    raise AssertionError(f"held-out actuator incidence changed: {held_counts}")
  strata = {name: 0 for name in ("same_limb", "body_face", "cross_limb")}
  for pair in HELD_OUT_SPECS:
    strata[pair.topology] += 1
  if strata != {"same_limb": 4, "body_face": 2, "cross_limb": 16}:
    raise AssertionError(f"held-out topology balance changed: {strata}")
  for model, count in MODEL_PARAMETERS.items():
    if abs(count - TARGET_PARAMETERS) / TARGET_PARAMETERS > 0.005:
      raise AssertionError(f"model outside 0.5% capacity band: {model}")


validate_protocol()


def _array_hash(value: Any) -> str:
  array = np.asarray(jax.device_get(value))
  canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
  payload = canonical.dtype.str.encode() + str(canonical.shape).encode() + canonical.tobytes()
  return hashlib.sha256(payload).hexdigest()


def _tree_hash(value: Any) -> str:
  digest = hashlib.sha256()
  for leaf in jax.tree_util.tree_leaves(jax.device_get(value)):
    array = np.asarray(leaf)
    canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    digest.update(canonical.dtype.str.encode())
    digest.update(str(canonical.shape).encode())
    digest.update(canonical.tobytes())
  return digest.hexdigest()


def assignment_arrays(seed: int, batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, jax.Array]:
  injury_keys = jax.random.split(jax.random.PRNGKey(200_000 + seed), 4)
  category_u = jax.random.uniform(injury_keys[0], (batch_size,))
  category = jnp.where(
      category_u < HEALTHY_PROBABILITY,
      CATEGORY_HEALTHY,
      jnp.where(
          category_u < HEALTHY_PROBABILITY + WEAK_PROBABILITY,
          CATEGORY_WEAK,
          CATEGORY_DEAD,
      ),
  ).astype(jnp.int32)
  pair_slot = jax.random.randint(
      injury_keys[1], (batch_size,), 0, len(TRAIN_PAIRS), dtype=jnp.int32,
  )
  pair_indices = jnp.asarray(
      [(pair.actuator_a, pair.actuator_b) for pair in TRAIN_PAIRS],
      dtype=jnp.int32,
  )[pair_slot]
  log_min = jnp.log(jnp.float32(MIN_WEAK_STRENGTH))
  log_max = jnp.log(jnp.float32(MAX_WEAK_STRENGTH))
  weak_a = jnp.exp(log_min + jax.random.uniform(injury_keys[2], (batch_size,)) * (log_max - log_min))
  weak_b = jnp.exp(log_min + jax.random.uniform(injury_keys[3], (batch_size,)) * (log_max - log_min))
  effective_a = jnp.where(category == CATEGORY_HEALTHY, 1.0, jnp.where(category == CATEGORY_WEAK, weak_a, 0.0)).astype(jnp.float32)
  effective_b = jnp.where(category == CATEGORY_HEALTHY, 1.0, jnp.where(category == CATEGORY_WEAK, weak_b, 0.0)).astype(jnp.float32)
  recorded_pair = jnp.where(category == CATEGORY_HEALTHY, -1, pair_slot)
  return {
      "category": category,
      "training_pair_slot": recorded_pair,
      "actuator_indices": pair_indices,
      "sampled_weak_strength_a": weak_a,
      "sampled_weak_strength_b": weak_b,
      "effective_strength_a": effective_a,
      "effective_strength_b": effective_b,
      "domain_keys": jax.random.split(jax.random.PRNGKey(100_000 + seed), batch_size),
  }


def apply_pair_injuries(
    force_ranges: jax.Array, assignments: dict[str, jax.Array],
) -> jax.Array:
  batch_size = assignments["category"].shape[0]
  if force_ranges.ndim == 2:
    force_ranges = jnp.broadcast_to(
        force_ranges, (batch_size,) + force_ranges.shape,
    )
  if force_ranges.ndim != 3 or force_ranges.shape[0] != batch_size:
    raise ValueError("forcerange must be unbatched or match assignment batch")
  indices = assignments["actuator_indices"]
  rows = jnp.arange(batch_size)
  force_ranges = force_ranges.at[rows, indices[:, 0], :].multiply(
      assignments["effective_strength_a"][:, None]
  )
  return force_ranges.at[rows, indices[:, 1], :].multiply(
      assignments["effective_strength_b"][:, None]
  )


def prepare_training_randomization(
    model: Any,
    seed: int,
    official_randomizer: Callable[[Any, jax.Array], tuple[Any, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[Any, Any, dict[str, jax.Array], dict[str, Any]]:
  assignments = assignment_arrays(seed, batch_size)
  randomized, in_axes = official_randomizer(model, assignments["domain_keys"])
  force_ranges = apply_pair_injuries(
      randomized.actuator_forcerange, assignments,
  )
  randomized = randomized.tree_replace({"actuator_forcerange": force_ranges})
  hashes = {
      name: _array_hash(value) for name, value in assignments.items()
  }
  hashes["randomized_model_tree"] = _tree_hash(randomized)
  return (
      randomized,
      in_axes.tree_replace({"actuator_forcerange": 0}),
      assignments,
      hashes,
  )


def held_out_shard(shard: int) -> tuple[PairSpec, ...]:
  if shard not in (0, 1):
    raise ValueError("held-out shard must be 0 or 1")
  return HELD_OUT_SPECS[shard * 11:(shard + 1) * 11]


def evaluation_conditions(shard: int) -> tuple[dict[str, Any], ...]:
  return tuple(
      {
          "pair_id": pair.pair_id,
          "actuator_a": pair.actuator_a,
          "actuator_b": pair.actuator_b,
          "name_a": pair.name_a,
          "name_b": pair.name_b,
          "topology": pair.topology,
          "strength_a": strength_a,
          "strength_b": strength_b,
      }
      for pair in held_out_shard(shard)
      for strength_a in STRENGTHS
      for strength_b in STRENGTHS
  )
