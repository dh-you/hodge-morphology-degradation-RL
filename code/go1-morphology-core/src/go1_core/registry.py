"""Registry for frozen canonical and explicitly experimental Go1 models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ModelSpec:
  name: str
  family: str
  actor_parameters: int
  hidden_layer_sizes: tuple[int, ...] | None = None
  width: int | None = None
  num_layers: int | None = None
  max_rank: int | None = None
  metric_mode: str | None = None


_CANONICAL_SPECS = (
    ModelSpec(
        "native_mlp", "native_mlp", 192_408,
        hidden_layer_sizes=(512, 256, 128),
    ),
    ModelSpec(
        "mlp_match_lower", "matched_mlp", 93_162,
        hidden_layer_sizes=(344, 172, 86),
    ),
    ModelSpec("pointwise_match_lower", "pointwise", 93_244, width=174),
    ModelSpec("gcn_match_lower", "gcn", 92_987, width=581, num_layers=2),
    ModelSpec(
        "multirank_hodge_lower", "multirank", 92_962,
        width=128, num_layers=4, max_rank=1, metric_mode="identity",
    ),
    ModelSpec(
        "multirank_hodge_full", "multirank", 126_570,
        width=128, num_layers=4, max_rank=2, metric_mode="identity",
    ),
)

# Unreleased capacity controls for the double-actuator workshop study. They
# have distinct identities so no frozen canonical model changes meaning.
_EXPERIMENTAL_SPECS = (
    ModelSpec(
        "native_mlp_w480_240_120", "native_mlp", 170_784,
        hidden_layer_sizes=(480, 240, 120),
    ),
    ModelSpec("pointwise_w252", "pointwise", 171_478, width=252),
    ModelSpec("gcn_w1107", "gcn", 171_361, width=1_107, num_layers=2),
    ModelSpec(
        "multirank_hodge_lower_w177", "multirank", 171_705,
        width=177, num_layers=4, max_rank=1, metric_mode="identity",
    ),
    ModelSpec(
        "multirank_hodge_full_w150", "multirank", 171_296,
        width=150, num_layers=4, max_rank=2, metric_mode="identity",
    ),
)

CANONICAL_MODELS = tuple(spec.name for spec in _CANONICAL_SPECS)
MODEL_SPECS = MappingProxyType({
    spec.name: spec for spec in (*_CANONICAL_SPECS, *_EXPERIMENTAL_SPECS)
})


def get_model_spec(model_name: str) -> ModelSpec:
  try:
    return MODEL_SPECS[model_name]
  except KeyError as error:
    raise ValueError(f"unknown canonical Go1 model: {model_name!r}") from error
