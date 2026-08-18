from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from experiments.double_actuator.evaluate_nominal import (
    CELL_REGIMES,
    NOMINAL_METRICS,
)
from experiments.double_actuator.healthy_protocol import (
    HEALTHY_PROTOCOL,
    HEALTHY_PROTOCOL_SHA256,
    healthy_assignment_arrays,
    prepare_healthy_randomization,
)
from experiments.double_actuator.merge_heldout_control import (
    EXPECTED_ROWS,
    EXPECTED_SHARD_ROWS,
    _row_key as control_row_key,
)
from experiments.double_actuator.merge_shards import _row_key as damage_row_key
from experiments.double_actuator.protocol import (
    CURRICULUM_SHA256,
    DEFAULT_BATCH_SIZE,
    HELD_OUT_SPECS,
    STRENGTHS,
    assignment_arrays,
)
from experiments.double_actuator.train_healthy import output_path


class FakeModel(NamedTuple):
  actuator_forcerange: jax.Array
  body_mass: jax.Array


def fake_randomizer(model, keys):
  batch_size = keys.shape[0]
  force_ranges = jnp.broadcast_to(
      model.actuator_forcerange, (batch_size,) + model.actuator_forcerange.shape,
  )
  mass_offset = jax.random.uniform(keys[0], (batch_size, 1))
  body_mass = jnp.broadcast_to(model.body_mass, (batch_size, 1)) + mass_offset
  return (
      FakeModel(force_ranges, body_mass),
      FakeModel(0, 0),
  )


def test_healthy_protocol_has_distinct_frozen_identity():
  assert HEALTHY_PROTOCOL_SHA256 != CURRICULUM_SHA256
  assert HEALTHY_PROTOCOL["training_regime"] == "healthy_only"
  assert HEALTHY_PROTOCOL["actuator_damage"] == {
      "multipliers": "all_one",
      "category_sampling": False,
      "pair_sampling": False,
      "strength_sampling": False,
  }
  assert HEALTHY_PROTOCOL["random_streams"]["pair_injury"] == "disabled"


def test_healthy_domain_keys_match_damage_preinjury_stream():
  healthy = healthy_assignment_arrays(11, DEFAULT_BATCH_SIZE)
  damage = assignment_arrays(11, DEFAULT_BATCH_SIZE)
  assert set(healthy) == {"domain_keys"}
  np.testing.assert_array_equal(healthy["domain_keys"], damage["domain_keys"])


def test_healthy_randomizer_preserves_official_actuator_ranges():
  base = FakeModel(
      actuator_forcerange=jnp.arange(24, dtype=jnp.float32).reshape(12, 2),
      body_mass=jnp.asarray([10.0], dtype=jnp.float32),
  )
  randomized, axes, assignments, hashes = prepare_healthy_randomization(
      base, 11, fake_randomizer, 8,
  )
  expected = np.broadcast_to(np.asarray(base.actuator_forcerange), (8, 12, 2))
  np.testing.assert_array_equal(randomized.actuator_forcerange, expected)
  assert axes.actuator_forcerange == 0
  assert set(assignments) == {"domain_keys"}
  assert set(hashes) == {
      "domain_keys",
      "pre_actuator_randomized_model_tree",
      "pre_actuator_forcerange",
  }


def test_healthy_output_path_is_separate_from_damage_campaign():
  path = output_path(
      "/scratch/network/dy0130/results", "a" * 40,
      "native_mlp_w480_240_120", 11, 400_000_000, 19,
  )
  assert "double-actuator-healthy" in path.parts
  assert "double-actuator" not in path.parts


def test_nominal_cells_and_metrics_are_explicit():
  assert CELL_REGIMES == {"A": "healthy_only", "C": "damage_curriculum"}
  assert {
      "undiscounted_return",
      "velocity_rmse",
      "yaw_rmse",
      "survival",
      "fall",
      "absolute_mechanical_power",
      "action_delta_rms",
  } <= set(NOMINAL_METRICS)


def test_control_merge_uses_frozen_d_row_key_contract():
  row = {
      "command": "stand",
      "reset_key": "0",
      "pair_id": str(HELD_OUT_SPECS[0].pair_id),
      "strength_a": str(STRENGTHS[0]),
      "strength_b": str(STRENGTHS[1]),
  }
  assert control_row_key(row) == damage_row_key(row)
  assert EXPECTED_SHARD_ROWS == 80_190
  assert EXPECTED_ROWS == 160_380
