"""Frozen 109-lane weak-actuator rollout and divergence diagnostics."""

from __future__ import annotations

from collections.abc import Callable
import jax
import jax.numpy as jnp

from go1_robot_experiments.constants import (
    CONDITIONS,
    HORIZON,
    SATURATION_THRESHOLD,
)
from go1_robot_experiments.protocol import condition_indices, tree_where


def compute_physical_step_diagnostics(
    torque: jax.Array,
    joint_position: jax.Array,
    action: jax.Array,
    force_limits: jax.Array,
    damage_indices: jax.Array,
    has_damage: jax.Array,
    default_pose: jax.Array,
    action_scale: float,
) -> dict[str, jax.Array]:
  enabled = force_limits > 0.0
  denominators = jnp.where(enabled, force_limits, 1.0)
  per_actuator_utilization = jnp.where(
      enabled, jnp.abs(torque) / denominators, 0.0,
  )
  enabled_count = jnp.maximum(jnp.sum(enabled, axis=-1), 1)
  utilization = jnp.sum(per_actuator_utilization, axis=-1) / enabled_count
  saturation = jnp.sum(
      enabled & (jnp.abs(torque) >= SATURATION_THRESHOLD * force_limits),
      axis=-1,
  ) / enabled_count
  row = jnp.arange(torque.shape[0])
  damaged_utilization = jnp.where(
      has_damage, per_actuator_utilization[row, damage_indices], 0.0,
  )
  targets = default_pose[None, :] + action * action_scale
  joint_error = joint_position - targets
  damaged_error = jnp.where(
      has_damage, joint_error[row, damage_indices], 0.0,
  )
  return {
      "actuator_force_utilization": utilization,
      "saturation_fraction": saturation,
      "damaged_actuator_force_utilization": damaged_utilization,
      "damaged_joint_error": damaged_error,
  }


def make_weak_actuator_rollout(
    base_env,
    vectorized_env,
    policy: Callable,
    force_limits: jax.Array,
    horizon: int = HORIZON,
):
  """Counts the first terminal transition, then freezes and masks its lane."""
  damage_indices = condition_indices()
  has_damage = jnp.asarray([
      condition.actuator_index is not None for condition in CONDITIONS
  ], dtype=bool)
  default_pose = jnp.asarray(base_env._default_pose)
  action_scale = float(base_env._config.action_scale)

  def rollout(initial_state, command):
    batch_size = initial_state.done.shape[0]
    zeros = jnp.zeros((batch_size,), dtype=jnp.float32)
    carry = (
        initial_state,
        jnp.zeros((batch_size, 12), dtype=jnp.float32),
        jnp.ones((batch_size,), dtype=bool),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.full((batch_size,), -1, dtype=jnp.int32),
        zeros, zeros, zeros, zeros, zeros, zeros,
        zeros, zeros, zeros, zeros,
    )

    def scan_step(current, unused):
      del unused
      (
          state, previous_action, active, active_steps, termination_step,
          velocity_sq_sum, yaw_sq_sum, power_sum, torque_sq_sum,
          action_delta_sq_sum, reward_sum, utilization_sum, saturation_sum,
          damaged_utilization_sum, damaged_tracking_sq_sum,
      ) = current
      was_active = active
      action, _ = policy(state.obs, jax.random.PRNGKey(0))
      next_state = vectorized_env.step(state, action)
      local_velocity = jax.vmap(base_env.get_local_linvel)(next_state.data)
      gyro = jax.vmap(base_env.get_gyro)(next_state.data)
      velocity_error = local_velocity[:, :2] - command[:2]
      yaw_error = gyro[:, 2] - command[2]
      torque = next_state.data.actuator_force
      joint_velocity = next_state.data.qvel[:, 6:]
      joint_position = next_state.data.qpos[:, 7:]
      action_delta = action - previous_action
      physical = compute_physical_step_diagnostics(
          torque, joint_position, action, force_limits, damage_indices,
          has_damage, default_pose, action_scale,
      )
      weight = was_active.astype(jnp.float32)
      velocity_sq_sum += weight * jnp.sum(velocity_error ** 2, axis=-1)
      yaw_sq_sum += weight * yaw_error ** 2
      power_sum += weight * jnp.sum(
          jnp.abs(torque * joint_velocity), axis=-1,
      )
      torque_sq_sum += weight * jnp.sum(torque ** 2, axis=-1)
      action_delta_sq_sum += weight * jnp.sum(action_delta ** 2, axis=-1)
      reward_sum += jnp.where(was_active, next_state.reward, 0.0)
      utilization_sum += weight * physical["actuator_force_utilization"]
      saturation_sum += weight * physical["saturation_fraction"]
      damaged_utilization_sum += (
          weight * physical["damaged_actuator_force_utilization"]
      )
      damaged_tracking_sq_sum += (
          weight * physical["damaged_joint_error"] ** 2
      )
      next_active_steps = active_steps + was_active.astype(jnp.int32)
      newly_done = was_active & next_state.done.astype(bool)
      termination_step = jnp.where(
          newly_done, next_active_steps, termination_step,
      )
      state = tree_where(was_active, next_state, state)
      previous_action = jnp.where(
          was_active[:, None], action, previous_action,
      )
      active = was_active & ~newly_done
      return (
          state, previous_action, active, next_active_steps, termination_step,
          velocity_sq_sum, yaw_sq_sum, power_sum, torque_sq_sum,
          action_delta_sq_sum, reward_sum, utilization_sum, saturation_sum,
          damaged_utilization_sum, damaged_tracking_sq_sum,
      ), None

    final, _ = jax.lax.scan(scan_step, carry, xs=None, length=horizon)
    (
        unused_state, unused_action, unused_active, active_steps,
        termination_step, velocity_sq_sum, yaw_sq_sum, power_sum,
        torque_sq_sum, action_delta_sq_sum, reward_sum, utilization_sum,
        saturation_sum, damaged_utilization_sum, damaged_tracking_sq_sum,
    ) = final
    del unused_state, unused_action, unused_active
    step_count = jnp.maximum(active_steps.astype(jnp.float32), 1.0)
    was_terminated = termination_step >= 0
    return {
        "velocity_rmse": jnp.sqrt(velocity_sq_sum / (step_count * 2.0)),
        "yaw_rmse": jnp.sqrt(yaw_sq_sum / step_count),
        "absolute_mechanical_power": power_sum / step_count,
        "torque_rms": jnp.sqrt(torque_sq_sum / (step_count * 12.0)),
        "action_delta_rms": jnp.sqrt(
            action_delta_sq_sum / (step_count * 12.0),
        ),
        "survival": active_steps.astype(jnp.float32) / float(horizon),
        "fall": (
            was_terminated & (termination_step < horizon)
        ).astype(jnp.float32),
        "undiscounted_return": reward_sum,
        "actuator_force_utilization": utilization_sum / step_count,
        "saturation_fraction": saturation_sum / step_count,
        "damaged_actuator_force_utilization": (
            damaged_utilization_sum / step_count
        ),
        "damaged_joint_tracking_rmse": jnp.sqrt(
            damaged_tracking_sq_sum / step_count,
        ),
        "was_terminated": was_terminated,
        "termination_step": termination_step,
        "active_steps": active_steps,
    }

  return rollout
