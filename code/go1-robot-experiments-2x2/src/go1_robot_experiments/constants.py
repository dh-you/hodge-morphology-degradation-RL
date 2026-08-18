"""Pinned dependencies and weak-actuator scientific protocol constants."""

from __future__ import annotations

from dataclasses import dataclass

from go1_core import CANONICAL_MODELS, GO1_CONTRACT_V1


CORE_COMMIT = "5c02a404f5b059be9ce074b896ea4997ddcf4818"
CORE_REPOSITORY_URL = (
    "ssh://git@github.com/dh-you/go1-morphology-core.git"
)
ORIGINAL_WEAK_ACTUATOR_COMMIT = "1659809766f6aba43a1074cc332922317e9bec5a"
PLAYGROUND_COMMIT = "220abb9d6a41eaa258047dacf9ca70a92e008b78"
MENAGERIE_COMMIT = "1b86ece576591213e2b666ebf59508454200ca97"
ENV_NAME = "Go1JoystickFlatTerrain"
HORIZON = 1_000
RESET_KEYS = tuple(range(10))
COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.75, 0.0, 0.0),
    "backward": (-0.75, 0.0, 0.0),
    "left_lateral": (0.0, 0.4, 0.0),
    "right_lateral": (0.0, -0.4, 0.0),
    "left_yaw": (0.0, 0.0, 0.6),
    "right_yaw": (0.0, 0.0, -0.6),
    "forward_left_yaw": (0.75, 0.0, 0.6),
    "forward_right_yaw": (0.75, 0.0, -0.6),
}
WEAK_STRENGTHS = (0.50, 0.25, 0.20, 0.15, 0.10, 0.075, 0.05, 0.025, 0.0)
CONDITION_LABELS = {
    0.50: "weak_50",
    0.25: "weak_25",
    0.20: "weak_20",
    0.15: "weak_15",
    0.10: "weak_10",
    0.075: "weak_7p5",
    0.05: "weak_5",
    0.025: "weak_2p5",
    0.0: "dead",
}
MODEL_TIERS = {
    "native_mlp": "native_over_capacity",
    "mlp_match_lower": "capacity",
    "pointwise_match_lower": "capacity",
    "gcn_match_lower": "capacity",
    "multirank_hodge_lower": "canonical_hodge",
    "multirank_hodge_full": "canonical_hodge",
}
SATURATION_THRESHOLD = 0.95
BASE_METRICS = (
    "velocity_rmse",
    "yaw_rmse",
    "absolute_mechanical_power",
    "torque_rms",
    "action_delta_rms",
    "survival",
    "fall",
    "undiscounted_return",
)
EXTRA_METRICS = (
    "actuator_force_utilization",
    "saturation_fraction",
    "damaged_actuator_force_utilization",
    "damaged_joint_tracking_rmse",
)
SUMMARY_METRICS = (*BASE_METRICS, *EXTRA_METRICS)
@dataclass(frozen=True)
class WeakActuatorCondition:
  condition: str
  actuator_index: int | None
  actuator_name: str | None
  strength: float


CONDITIONS = (
    WeakActuatorCondition("healthy", None, None, 1.0),
    *(
        WeakActuatorCondition(CONDITION_LABELS[strength], index, name, strength)
        for strength in WEAK_STRENGTHS
        for index, name in enumerate(GO1_CONTRACT_V1.action_names)
    ),
)

if tuple(MODEL_TIERS) != CANONICAL_MODELS:
  raise AssertionError("experiment model order differs from the pinned core")
if len(CONDITIONS) != 109:
  raise AssertionError("weak-actuator grid must contain 109 conditions")
