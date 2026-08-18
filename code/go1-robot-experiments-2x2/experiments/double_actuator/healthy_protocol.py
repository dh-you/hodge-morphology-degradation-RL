"""Frozen healthy-only control protocol for the double-actuator 2x2 study."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import jax

from experiments.double_actuator.protocol import (
    CURRICULUM_SHA256,
    DEFAULT_BATCH_SIZE,
    _array_hash,
    _tree_hash,
    canonical_json,
)

HEALTHY_PROTOCOL = {
    "training_regime": "healthy_only",
    "assignment_scope": "fixed_vectorized_environment_member",
    "actuator_damage": {
        "multipliers": "all_one",
        "category_sampling": False,
        "pair_sampling": False,
        "strength_sampling": False,
    },
    "domain_randomization": {
        "implementation": "official_mujoco_playground",
        "random_stream": "100000 + seed",
        "unchanged_from_damage_campaign": True,
    },
    "random_streams": {
        "ppo_model_reset": "seed",
        "domain_randomization": "100000 + seed",
        "pair_injury": "disabled",
    },
    "reference_damage_training_commit": "9a125fd245874b1d3be50238f3cdddc510832ce0",
    "reference_damage_curriculum_sha256": CURRICULUM_SHA256,
}
HEALTHY_PROTOCOL_SHA256 = hashlib.sha256(
    canonical_json(HEALTHY_PROTOCOL).encode()
).hexdigest()


def healthy_assignment_arrays(
    seed: int, batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, jax.Array]:
  """Returns the only sampled healthy-control assignment: official DR keys."""
  return {
      "domain_keys": jax.random.split(
          jax.random.PRNGKey(100_000 + seed), batch_size,
      ),
  }


def prepare_healthy_randomization(
    model: Any,
    seed: int,
    official_randomizer: Callable[[Any, jax.Array], tuple[Any, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[Any, Any, dict[str, jax.Array], dict[str, str]]:
  """Applies official DR without sampling or applying actuator injuries."""
  assignments = healthy_assignment_arrays(seed, batch_size)
  randomized, in_axes = official_randomizer(model, assignments["domain_keys"])
  hashes = {
      "domain_keys": _array_hash(assignments["domain_keys"]),
      "pre_actuator_randomized_model_tree": _tree_hash(randomized),
      "pre_actuator_forcerange": _array_hash(randomized.actuator_forcerange),
  }
  return randomized, in_axes, assignments, hashes
