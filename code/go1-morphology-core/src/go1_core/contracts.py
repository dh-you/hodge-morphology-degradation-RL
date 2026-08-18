"""Versioned observation and action contract for the frozen Go1 actors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Go1ContractV1:
  """Immutable description of the policy-facing Go1 interface."""

  name: str
  version: int
  policy_obs_key: str
  value_obs_key: str
  policy_observation_shape: tuple[int, ...]
  value_observation_shape: tuple[int, ...]
  observation_dtype: str
  action_size: int
  distribution_parameter_size: int
  joint_names: tuple[str, ...]
  action_names: tuple[str, ...]
  default_pose: tuple[float, ...]
  action_scale: float
  actor_slices: tuple[tuple[str, int, int], ...]
  privileged_slices: tuple[tuple[str, int, int], ...]


_JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)

GO1_CONTRACT_V1 = Go1ContractV1(
    name="go1_contract",
    version=1,
    policy_obs_key="state",
    value_obs_key="privileged_state",
    policy_observation_shape=(48,),
    value_observation_shape=(123,),
    observation_dtype="float32",
    action_size=12,
    distribution_parameter_size=24,
    joint_names=_JOINT_NAMES,
    action_names=_JOINT_NAMES,
    default_pose=(
        0.1, 0.9, -1.8, -0.1, 0.9, -1.8,
        0.1, 0.9, -1.8, -0.1, 0.9, -1.8,
    ),
    action_scale=0.5,
    actor_slices=(
        ("noisy_local_linear_velocity", 0, 3),
        ("noisy_angular_velocity", 3, 6),
        ("noisy_projected_gravity", 6, 9),
        ("noisy_joint_angle_offset", 9, 21),
        ("noisy_joint_velocity", 21, 33),
        ("previous_action", 33, 45),
        ("command", 45, 48),
    ),
    privileged_slices=(
        ("state", 0, 48),
        ("angular_velocity", 48, 51),
        ("accelerometer", 51, 54),
        ("projected_gravity", 54, 57),
        ("local_linear_velocity", 57, 60),
        ("global_angular_velocity", 60, 63),
        ("joint_angle_offset", 63, 75),
        ("joint_velocity", 75, 87),
        ("actuator_force", 87, 99),
        ("previous_foot_contact", 99, 103),
        ("foot_linear_velocity", 103, 115),
        ("foot_air_time", 115, 119),
        ("torso_force", 119, 122),
        ("perturbation_timing", 122, 123),
    ),
)


def _shape_and_dtype(value: Any) -> tuple[tuple[int, ...], Any | None]:
  if isinstance(value, Mapping) and "shape" in value:
    return tuple(value["shape"]), value.get("dtype")
  if hasattr(value, "shape"):
    return tuple(value.shape), getattr(value, "dtype", None)
  if isinstance(value, int):
    return (value,), None
  return tuple(value), None


def _is_float32(dtype: Any) -> bool:
  if dtype is None:
    return True
  try:
    return np.dtype(dtype) == np.dtype(np.float32)
  except TypeError:
    text = str(dtype)
    return text in {
        "float32",
        "<class 'numpy.float32'>",
        "<class 'jax.numpy.float32'>",
    }


def assert_compatible_environment(
    observation_size: Mapping[str, Any], action_size: int,
) -> None:
  """Raises when a Brax-style interface differs from ``Go1ContractV1``."""
  if not isinstance(observation_size, Mapping):
    raise TypeError("Go1 requires keyed observations")
  expected_keys = {
      GO1_CONTRACT_V1.policy_obs_key,
      GO1_CONTRACT_V1.value_obs_key,
  }
  if set(observation_size) != expected_keys:
    raise ValueError(
        f"Go1 requires observation keys {sorted(expected_keys)}, "
        f"got {sorted(observation_size)}"
    )
  expected_shapes = {
      GO1_CONTRACT_V1.policy_obs_key:
          GO1_CONTRACT_V1.policy_observation_shape,
      GO1_CONTRACT_V1.value_obs_key:
          GO1_CONTRACT_V1.value_observation_shape,
  }
  for key, expected_shape in expected_shapes.items():
    shape, dtype = _shape_and_dtype(observation_size[key])
    if shape != expected_shape:
      raise ValueError(f"Go1 requires {key} shape {expected_shape}, got {shape}")
    if not _is_float32(dtype):
      raise ValueError(f"Go1 requires {key} dtype float32, got {dtype}")
  if action_size != GO1_CONTRACT_V1.action_size:
    raise ValueError(
        f"Go1 requires action size {GO1_CONTRACT_V1.action_size}, "
        f"got {action_size}"
    )
