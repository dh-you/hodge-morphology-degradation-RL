"""Frozen weak-actuator condition grid and environment-side operations."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from go1_core import GO1_CONTRACT_V1, assert_compatible_environment

from go1_robot_experiments.constants import CONDITIONS, HORIZON


def condition_indices() -> jax.Array:
  return jnp.asarray([
      0 if condition.actuator_index is None else condition.actuator_index
      for condition in CONDITIONS
  ], dtype=jnp.int32)


def condition_strengths() -> jax.Array:
  return jnp.asarray(
      [condition.strength for condition in CONDITIONS], dtype=jnp.float32,
  )


def condition_force_ranges(force_ranges: jax.Array) -> jax.Array:
  ranges = jnp.broadcast_to(
      force_ranges, (len(CONDITIONS),) + force_ranges.shape,
  )
  return ranges.at[
      jnp.arange(len(CONDITIONS)), condition_indices(), :
  ].multiply(condition_strengths()[:, None])


def fixed_condition_randomize(model):
  randomized = model.tree_replace({
      "actuator_forcerange": condition_force_ranges(model.actuator_forcerange),
  })
  in_axes = jax.tree_util.tree_map(lambda unused: None, model)
  return randomized, in_axes.tree_replace({"actuator_forcerange": 0})


def paired_reset_keys(reset_index: int) -> jax.Array:
  key = jax.random.PRNGKey(reset_index)
  return jnp.broadcast_to(key, (len(CONDITIONS),) + key.shape)


def patch_command(state, command: jax.Array, horizon: int = HORIZON):
  """Patches a raw reset state without consuming observation-noise RNG."""
  info = dict(state.info)
  info["command"] = jnp.asarray(command, dtype=jnp.float32)
  info["steps_until_next_cmd"] = jnp.asarray(horizon + 1, dtype=jnp.int32)
  obs = dict(state.obs)
  obs["state"] = obs["state"].at[45:48].set(command)
  obs["privileged_state"] = obs["privileged_state"].at[45:48].set(command)
  return state.replace(info=info, obs=obs)


def patch_batched_command(state, command: jax.Array):
  return jax.vmap(patch_command, in_axes=(0, None))(state, command)


def tree_where(mask: jax.Array, true_tree, false_tree):
  """Selects batched leaves while preserving static and scalar leaves."""
  batch_size = mask.shape[0]

  def select(true_value, false_value):
    if not hasattr(true_value, "shape"):
      return true_value
    if true_value.ndim == 0 or true_value.shape[0] != batch_size:
      return true_value
    expanded = mask.reshape((batch_size,) + (1,) * (true_value.ndim - 1))
    return jnp.where(expanded, true_value, false_value)

  return jax.tree.map(select, true_tree, false_tree)


def assert_environment_contract(base_env: Any) -> None:
  assert_compatible_environment(base_env.observation_size, base_env.action_size)
  contract = GO1_CONTRACT_V1
  actuator_names = tuple(
      base_env.mj_model.actuator(index).name
      for index in range(base_env.mj_model.nu)
  )
  if actuator_names != contract.action_names:
    raise ValueError("Playground actuator order differs from Go1ContractV1")
  if not np.array_equal(
      np.asarray(base_env._default_pose, dtype=np.float32),
      np.asarray(contract.default_pose, dtype=np.float32),
  ):
    raise ValueError("Playground default pose differs from Go1ContractV1")
  if float(base_env._config.action_scale) != contract.action_scale:
    raise ValueError("Playground action scale differs from Go1ContractV1")
