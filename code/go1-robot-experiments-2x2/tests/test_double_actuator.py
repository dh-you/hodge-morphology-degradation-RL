from __future__ import annotations

import itertools

import jax
import numpy as np
import pytest

from go1_core import CANONICAL_MODELS, MODEL_SPECS, make_go1_ppo_networks
from go1_robot_experiments.constants import COMMANDS
from experiments.double_actuator.analyze import PAPER_WORDING, surface_auc
from experiments.double_actuator.merge_shards import (
    EXPECTED_ROWS,
    EXPECTED_SHARD_ROWS,
    _row_key,
)
from experiments.double_actuator.protocol import (
    CATEGORY_DEAD,
    CATEGORY_HEALTHY,
    CATEGORY_WEAK,
    DEFAULT_BATCH_SIZE,
    HELD_OUT_SPECS,
    MODEL_PARAMETERS,
    MODELS,
    PAIRS,
    SEEDS,
    STRENGTHS,
    TRAIN_PAIRS,
    apply_pair_injuries,
    assignment_arrays,
    evaluation_conditions,
    held_out_shard,
)
from experiments.double_actuator.train import _randomization_batch, output_path
from experiments.weak_actuator.protocol import parameter_count

OBSERVATION_SIZE = {"state": (48,), "privileged_state": (123,)}


def test_pair_split_is_exact_unique_and_balanced():
  assert len(PAIRS) == 66
  assert len(TRAIN_PAIRS) == 44
  assert len(HELD_OUT_SPECS) == 22
  assert {(p.actuator_a, p.actuator_b) for p in PAIRS} == set(itertools.combinations(range(12), 2))
  assert not ({p.pair_id for p in TRAIN_PAIRS} & {p.pair_id for p in HELD_OUT_SPECS})
  strata = {name: 0 for name in ("same_limb", "body_face", "cross_limb")}
  incidence = np.zeros(12, dtype=int)
  for pair in HELD_OUT_SPECS:
    strata[pair.topology] += 1
    incidence[pair.actuator_a] += 1
    incidence[pair.actuator_b] += 1
  assert strata == {"same_limb": 4, "body_face": 2, "cross_limb": 16}
  assert set(incidence) <= {3, 4}


def test_assignment_streams_are_deterministic_independent_and_train_only():
  left = assignment_arrays(11, 8192)
  right = assignment_arrays(11, 8192)
  other = assignment_arrays(12, 8192)
  for key in left:
    np.testing.assert_array_equal(left[key], right[key])
  assert not np.array_equal(left["category"], other["category"])
  category = np.asarray(left["category"])
  pair_slot = np.asarray(left["training_pair_slot"])
  assert set(np.unique(category)) == {CATEGORY_HEALTHY, CATEGORY_WEAK, CATEGORY_DEAD}
  assert np.all(pair_slot[category == CATEGORY_HEALTHY] == -1)
  assert np.all((pair_slot[category != CATEGORY_HEALTHY] >= 0) & (pair_slot[category != CATEGORY_HEALTHY] < 44))
  weak = category == CATEGORY_WEAK
  strength_a = np.asarray(left["effective_strength_a"])[weak]
  strength_b = np.asarray(left["effective_strength_b"])[weak]
  assert np.all((strength_a >= 0.05) & (strength_a < 0.20))
  assert np.all((strength_b >= 0.05) & (strength_b < 0.20))
  assert not np.array_equal(strength_a, strength_b)
  dead = category == CATEGORY_DEAD
  assert np.all(np.asarray(left["effective_strength_a"])[dead] == 0)
  assert np.all(np.asarray(left["effective_strength_b"])[dead] == 0)


def test_pair_injury_modifies_exactly_two_distinct_actuators():
  assignments = {
      "category": np.asarray([CATEGORY_WEAK], dtype=np.int32),
      "actuator_indices": np.asarray([[2, 9]], dtype=np.int32),
      "effective_strength_a": np.asarray([0.1], dtype=np.float32),
      "effective_strength_b": np.asarray([0.2], dtype=np.float32),
  }
  base = np.ones((12, 2), dtype=np.float32)
  damaged = np.asarray(apply_pair_injuries(base, assignments))[0]
  changed = np.flatnonzero(np.any(damaged != base, axis=1))
  np.testing.assert_array_equal(changed, [2, 9])
  np.testing.assert_allclose(damaged[2], 0.1)
  np.testing.assert_allclose(damaged[9], 0.2)


def test_asymmetric_heldout_grid_and_shards_are_exact():
  assert len(held_out_shard(0)) == len(held_out_shard(1)) == 11
  for shard in (0, 1):
    rows = evaluation_conditions(shard)
    assert len(rows) == 891
    identities = {
        (row["pair_id"], row["strength_a"], row["strength_b"])
        for row in rows
    }
    assert len(identities) == 891
    first_pair = held_out_shard(shard)[0].pair_id
    assert (first_pair, 0.05, 0.20) in identities
    assert (first_pair, 0.20, 0.05) in identities


@pytest.mark.parametrize("model", MODELS)
def test_frozen_experimental_parameter_counts(model):
  networks = make_go1_ppo_networks(OBSERVATION_SIZE, 12, model_name=model)
  params = networks.policy_network.init(jax.random.PRNGKey(0))
  assert parameter_count(params) == MODEL_PARAMETERS[model]
  assert MODEL_SPECS[model].actor_parameters == MODEL_PARAMETERS[model]
  assert model not in CANONICAL_MODELS


def test_surface_auc_constant_linear_and_bilinear():
  strengths = np.asarray(sorted(STRENGTHS), dtype=np.float64)
  a, b = np.meshgrid(strengths, strengths, indexing="ij")
  assert surface_auc(np.full_like(a, 7.0)) == pytest.approx(7.0)
  assert surface_auc(a + 2.0 * b) == pytest.approx(0.75)
  assert surface_auc(a * b) == pytest.approx(0.0625)


def test_workshop_analysis_is_explicitly_descriptive():
  assert SEEDS == (11, 12, 13)
  assert "preliminary" in PAPER_WORDING
  assert "descriptive" in PAPER_WORDING


def test_campaign_output_rejects_home():
  with pytest.raises(ValueError, match="home quota"):
    output_path("/home/dy0130/results", "a" * 40, MODELS[0], 11, 400_000_000, 19)


def test_frozen_randomization_is_sliced_to_the_requested_wrapper_batch():
  randomized = {
      "batched": np.arange(24).reshape(8, 3),
      "fixed": np.asarray([7, 8]),
  }
  axes = {"batched": 0, "fixed": None}
  selected = _randomization_batch(randomized, axes, 3)
  np.testing.assert_array_equal(selected["batched"], randomized["batched"][:3])
  np.testing.assert_array_equal(selected["fixed"], randomized["fixed"])
  with pytest.raises(ValueError, match="invalid randomization batch size"):
    _randomization_batch(randomized, axes, DEFAULT_BATCH_SIZE + 1)


def test_heldout_shard_and_merge_cardinalities_are_frozen():
  assert EXPECTED_SHARD_ROWS == 80_190
  assert EXPECTED_ROWS == 160_380


def test_merged_row_key_preserves_protocol_order():
  first = {
      "command": next(iter(COMMANDS)),
      "reset_key": "0",
      "pair_id": str(HELD_OUT_SPECS[0].pair_id),
      "strength_a": str(STRENGTHS[0]),
      "strength_b": str(STRENGTHS[0]),
  }
  asymmetric_next = {**first, "strength_b": str(STRENGTHS[1])}
  assert _row_key(first) < _row_key(asymmetric_next)
