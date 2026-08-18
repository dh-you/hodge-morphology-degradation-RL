from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from go1_core import (
    CANONICAL_MODELS,
    GO1_CONTRACT_V1,
    MODEL_SPECS,
    assert_compatible_environment,
    get_model_spec,
)
from go1_core.features import build_joint_features
from go1_core import topology

FIXTURES = Path(__file__).parent / "fixtures"


def test_contract_v1_is_exact():
  contract = GO1_CONTRACT_V1
  assert contract.name == "go1_contract"
  assert contract.version == 1
  assert contract.policy_obs_key == "state"
  assert contract.value_obs_key == "privileged_state"
  assert contract.policy_observation_shape == (48,)
  assert contract.value_observation_shape == (123,)
  assert contract.observation_dtype == "float32"
  assert contract.action_size == 12
  assert contract.distribution_parameter_size == 24
  assert contract.joint_names == (
      "FR_hip", "FR_thigh", "FR_calf",
      "FL_hip", "FL_thigh", "FL_calf",
      "RR_hip", "RR_thigh", "RR_calf",
      "RL_hip", "RL_thigh", "RL_calf",
  )
  assert contract.action_names == contract.joint_names
  assert contract.default_pose == (
      0.1, 0.9, -1.8, -0.1, 0.9, -1.8,
      0.1, 0.9, -1.8, -0.1, 0.9, -1.8,
  )
  assert contract.action_scale == 0.5
  assert tuple((start, stop) for _, start, stop in contract.actor_slices) == (
      (0, 3), (3, 6), (6, 9), (9, 21), (21, 33), (33, 45), (45, 48),
  )
  assert tuple((start, stop) for _, start, stop in contract.privileged_slices) == (
      (0, 48), (48, 51), (51, 54), (54, 57), (57, 60),
      (60, 63), (63, 75), (75, 87), (87, 99), (99, 103),
      (103, 115), (115, 119), (119, 122), (122, 123),
  )


def test_environment_contract_accepts_checkpoint_specs_and_rejects_drift():
  valid = {
      "state": {
          "shape": [48], "dtype": "<class 'jax.numpy.float32'>",
      },
      "privileged_state": {
          "shape": [123], "dtype": "<class 'jax.numpy.float32'>",
      },
  }
  assert_compatible_environment(valid, 12)
  assert_compatible_environment({"state": (48,), "privileged_state": (123,)}, 12)
  with pytest.raises(ValueError, match="observation keys"):
    assert_compatible_environment({**valid, "extra": (1,)}, 12)
  with pytest.raises(ValueError, match="state shape"):
    assert_compatible_environment({**valid, "state": (47,)}, 12)
  with pytest.raises(ValueError, match="dtype float32"):
    assert_compatible_environment({**valid, "state": np.zeros(48, np.float64)}, 12)
  with pytest.raises(ValueError, match="action size"):
    assert_compatible_environment(valid, 11)


def test_registry_is_exact_ordered_and_immutable():
  assert CANONICAL_MODELS == (
      "native_mlp",
      "mlp_match_lower",
      "pointwise_match_lower",
      "gcn_match_lower",
      "multirank_hodge_lower",
      "multirank_hodge_full",
  )
  assert {name: MODEL_SPECS[name].actor_parameters for name in CANONICAL_MODELS} == {
      "native_mlp": 192_408,
      "mlp_match_lower": 93_162,
      "pointwise_match_lower": 93_244,
      "gcn_match_lower": 92_987,
      "multirank_hodge_lower": 92_962,
      "multirank_hodge_full": 126_570,
  }
  assert get_model_spec("mlp_match_lower").hidden_layer_sizes == (344, 172, 86)
  with pytest.raises(TypeError):
    MODEL_SPECS["other"] = MODEL_SPECS["native_mlp"]
  with pytest.raises(ValueError, match="unknown canonical"):
    get_model_spec("rhmp")


def test_runtime_topology_is_exact():
  assert topology.JOINT_NAMES == GO1_CONTRACT_V1.joint_names
  assert topology.EDGES == (
      (0, 1), (1, 2), (3, 4), (4, 5),
      (6, 7), (7, 8), (9, 10), (10, 11),
      (3, 0), (0, 6), (6, 9), (9, 3),
      (0, 2), (3, 5), (6, 8), (9, 11),
  )
  assert topology.FACES == (
      ((0, 1.0), (1, 1.0), (12, -1.0)),
      ((2, 1.0), (3, 1.0), (13, -1.0)),
      ((4, 1.0), (5, 1.0), (14, -1.0)),
      ((6, 1.0), (7, 1.0), (15, -1.0)),
      ((8, 1.0), (9, 1.0), (10, 1.0), (11, 1.0)),
  )
  assert topology.B1.shape == (12, 16)
  assert topology.B2.shape == (16, 5)
  assert topology.B1.dtype == np.float64
  assert topology.B2.dtype == np.float64
  assert np.array_equal(topology.B1 @ topology.B2, np.zeros((12, 5)))
  assert np.linalg.matrix_rank(topology.B1) == 11
  assert np.linalg.matrix_rank(topology.B2) == 5
  assert topology.array_sha256(topology.B1) == topology.ARRAY_SHA256["B1"]
  assert topology.array_sha256(topology.B2) == topology.ARRAY_SHA256["B2"]
  assert (
      topology.array_sha256(topology.GCN_ADJACENCY_WITH_SELF_LOOPS)
      == topology.ARRAY_SHA256["GCN_ADJACENCY_WITH_SELF_LOOPS"]
  )
  assert (
      topology.array_sha256(topology.GCN_NORMALIZED_ADJACENCY)
      == topology.ARRAY_SHA256["GCN_NORMALIZED_ADJACENCY"]
  )


def test_joint_features_match_frozen_fixture_exactly():
  probe = np.load(FIXTURES / "policy_probe_input.npy", allow_pickle=False)
  expected = np.load(FIXTURES / "policy_probe_features.npy", allow_pickle=False)
  actual = np.asarray(build_joint_features(jnp.asarray(probe)))
  assert actual.shape == (3, 12, 20)
  assert actual.dtype == np.float32
  assert np.array_equal(actual, expected)
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_fixture_metadata_and_file_hashes_are_self_consistent():
  metadata = json.loads((FIXTURES / "metadata.json").read_text())
  assert metadata["source_commit"] == "1659809766f6aba43a1074cc332922317e9bec5a"
  assert metadata["training_array"] == "3343705"
  assert metadata["checkpoint_step"] == 412_876_800
  for filename, record in metadata["arrays"].items():
    path = FIXTURES / filename
    array = np.load(path, allow_pickle=False)
    assert list(array.shape) == record["shape"]
    assert str(array.dtype) == record["dtype"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["npy_sha256"]
    assert (
        hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        == record["raw_sha256"]
    )
