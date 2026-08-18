from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from go1_core import GO1_CONTRACT_V1
from go1_robot_experiments.constants import CONDITIONS, WEAK_STRENGTHS
from go1_robot_experiments.protocol import (
    condition_force_ranges,
    condition_indices,
    condition_strengths,
    paired_reset_keys,
    patch_command,
    tree_where,
)
from go1_robot_experiments.rollout import compute_physical_step_diagnostics


def test_exact_condition_grid_and_contract_order():
  assert len(CONDITIONS) == 109
  assert CONDITIONS[0].condition == "healthy"
  assert [row.strength for row in CONDITIONS[1:13]] == [0.5] * 12
  assert tuple(row.strength for row in CONDITIONS[1::12]) == WEAK_STRENGTHS
  assert tuple(row.actuator_name for row in CONDITIONS[1:13]) == (
      GO1_CONTRACT_V1.action_names
  )


def test_condition_arrays_and_force_scaling():
  ranges = jnp.asarray([[-float(i + 1), float(i + 1)] for i in range(12)])
  result = np.asarray(condition_force_ranges(ranges))
  assert result.shape == (109, 12, 2)
  np.testing.assert_array_equal(result[0], np.asarray(ranges))
  np.testing.assert_allclose(result[1, 0], [-0.5, 0.5])
  np.testing.assert_allclose(result[12, 11], [-6.0, 6.0])
  assert condition_indices().dtype == jnp.int32
  assert condition_strengths().dtype == jnp.float32


def test_paired_reset_keys_are_broadcast_without_splitting():
  keys = np.asarray(paired_reset_keys(7))
  assert keys.shape == (109, 2)
  expected = np.asarray(jax.random.PRNGKey(7))
  np.testing.assert_array_equal(keys, np.broadcast_to(expected, keys.shape))


class FakeState(NamedTuple):
  info: dict
  obs: dict

  def replace(self, **kwargs):
    return self._replace(**kwargs)


def test_command_patch_preserves_other_observation_channels():
  state = FakeState(
      info={"rng": jnp.asarray([3, 4]), "command": jnp.ones(3)},
      obs={
          "state": jnp.arange(48, dtype=jnp.float32),
          "privileged_state": jnp.arange(123, dtype=jnp.float32),
      },
  )
  command = jnp.asarray([0.75, 0.0, -0.6], dtype=jnp.float32)
  patched = patch_command(state, command)
  np.testing.assert_array_equal(patched.info["rng"], state.info["rng"])
  np.testing.assert_array_equal(patched.obs["state"][:45], state.obs["state"][:45])
  np.testing.assert_array_equal(patched.obs["state"][45:48], command)
  np.testing.assert_array_equal(patched.obs["privileged_state"][45:48], command)
  assert int(patched.info["steps_until_next_cmd"]) == 1001


def test_tree_where_freezes_inactive_batched_lanes():
  mask = jnp.asarray([True, False])
  true = {"x": jnp.asarray([[1, 2], [3, 4]]), "static": jnp.asarray(5)}
  false = {"x": jnp.asarray([[9, 9], [8, 8]]), "static": jnp.asarray(6)}
  selected = tree_where(mask, true, false)
  np.testing.assert_array_equal(selected["x"], [[1, 2], [8, 8]])
  assert int(selected["static"]) == 5


def test_physical_diagnostic_formulas_include_dead_actuator_masking():
  torque = jnp.asarray([[2.0, 1.0], [0.0, 3.0]])
  position = jnp.asarray([[1.0, 1.0], [2.0, 2.0]])
  action = jnp.zeros((2, 2))
  limits = jnp.asarray([[4.0, 2.0], [2.0, 0.0]])
  result = compute_physical_step_diagnostics(
      torque, position, action, limits,
      jnp.asarray([0, 1]), jnp.asarray([False, True]),
      jnp.asarray([0.5, 0.5]), 0.5,
  )
  np.testing.assert_allclose(result["actuator_force_utilization"], [0.5, 0.0])
  np.testing.assert_allclose(result["damaged_actuator_force_utilization"], [0, 0])
  np.testing.assert_allclose(result["damaged_joint_error"], [0, 1.5])
