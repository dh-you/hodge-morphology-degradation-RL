"""Frozen campaign, validation, and paper analysis for weak actuators."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np
from scipy import stats

from go1_robot_experiments.constants import (
    COMMANDS,
    CONDITIONS,
    CORE_COMMIT,
    HORIZON,
    MENAGERIE_COMMIT,
    MODEL_TIERS,
    ORIGINAL_WEAK_ACTUATOR_COMMIT,
    PLAYGROUND_COMMIT,
    RESET_KEYS,
    SUMMARY_METRICS,
    WEAK_STRENGTHS,
)
from experiments.weak_actuator.protocol import (
    CURRICULUM_SHA256,
    OFFICIAL_PPO_CONFIG_SHA256,
    ORIGINAL_FULL_ACTUAL_TIMESTEPS,
    ORIGINAL_FULL_NUM_EVALS,
    ORIGINAL_FULL_NUM_TIMESTEPS,
)


REUSED_HODGE_COMMIT = "771c3d73b1fb2a97a417eb58a945266c1770cd65"
REQUESTED_STEPS = ORIGINAL_FULL_NUM_TIMESTEPS
ACTUAL_STEPS = ORIGINAL_FULL_ACTUAL_TIMESTEPS
NUM_EVALS = ORIGINAL_FULL_NUM_EVALS
EVALUATION_GRID = "full"
ROW_COUNT = 9_810
PRIMARY_ENDPOINT = "absolute_return_auc"
SECONDARY_ENDPOINTS = (
    "survival_auc",
    "return_retention_auc",
    "fall_rate_auc",
    "velocity_rmse_auc",
    "yaw_rmse_auc",
)
PHYSICAL_ENDPOINTS = (
    "absolute_mechanical_power_auc",
    "torque_rms_auc",
    "action_delta_rms_auc",
    "actuator_force_utilization_auc",
    "saturation_fraction_auc",
    "damaged_actuator_force_utilization_auc",
    "damaged_joint_tracking_rmse_auc",
)
AUC_ENDPOINTS = (PRIMARY_ENDPOINT, *SECONDARY_ENDPOINTS, *PHYSICAL_ENDPOINTS)
HODGE_MODELS = ("multirank_hodge_lower", "multirank_hodge_full")
BASELINE_MODELS = (
    "native_mlp",
    "mlp_match_lower",
    "pointwise_match_lower",
    "gcn_match_lower",
)
MODEL_ORDER = (*BASELINE_MODELS, *HODGE_MODELS)
MODEL_LABELS = {
    "native_mlp": "Native MLP",
    "mlp_match_lower": "Central MLP",
    "pointwise_match_lower": "Joint MLP",
    "gcn_match_lower": "GCN",
    "multirank_hodge_lower": "Hodge-L",
    "multirank_hodge_full": "Hodge-F",
}
MODEL_PARAMETERS = {
    "native_mlp": 192_408,
    "mlp_match_lower": 93_162,
    "pointwise_match_lower": 93_244,
    "gcn_match_lower": 92_987,
    "multirank_hodge_lower": 92_962,
    "multirank_hodge_full": 126_570,
}
MODEL_ROLES = {
    "native_mlp": "over-capacity centralized reference",
    "mlp_match_lower": "matched unstructured baseline",
    "pointwise_match_lower": "matched joint-factorized baseline",
    "gcn_match_lower": "matched pairwise baseline",
    "multirank_hodge_lower": "lower-order ablation",
    "multirank_hodge_full": "proposed rank-2 model",
}
SCIENTIFIC_FILES = {
    "experiments/weak_actuator/train.py": (
        "2e93c09a076934577a517c0175eae4098b601111cdfd5af8ea96a91e751786ab"
    ),
    "experiments/weak_actuator/protocol.py": (
        "fc59601ddf24f8b24f1ca19de26e6102dee70b3a595fb5da97f4ac2e8437996b"
    ),
    "src/go1_robot_experiments/constants.py": (
        "e505cf1287a19050778157549d67643c992bce95e88e84bb80e812645aca76bb"
    ),
    "src/go1_robot_experiments/dependencies.py": (
        "ffcf242edb52ba682dbf33b537c682eed0e87298cf73eaafaa5237f9777a2ae1"
    ),
    "src/go1_robot_experiments/protocol.py": (
        "1e361c643548bbc6664c4517e81e4f80394842e0e9113443083363947f22bc8c"
    ),
    "src/go1_robot_experiments/rollout.py": (
        "cfd35c27f305d4094a2ae7a2b4089571138812759087506f4bb862decd10de2a"
    ),
    "src/go1_robot_experiments/runtime.py": (
        "dcda24eb8be803642cf76142f88d0057b6065e4ef80b121c2eadaaee00210cbe"
    ),
}
INTERPRETATION = (
    "The Hodge seed study tests the pre-existing hypothesis that Hodge-L "
    "and Hodge-F differ in seed-level actuator-robust locomotion outcomes. "
    "Degraded-strength absolute-return AUC was already the frozen primary "
    "robustness endpoint, so formal paired inference uses all seeds 1-10. "
    "The observed plateau-versus-high-compensation behavior motivates a "
    "separate mechanistic analysis of locomotion discovery; any categorical "
    "breakthrough criterion defined after inspecting seeds 1-5 is labeled "
    "exploratory unless frozen before evaluating seeds 6-10."
)
METRIC_COLUMNS = tuple(SUMMARY_METRICS)
NULLABLE_METRICS = {
    "damaged_actuator_force_utilization",
    "damaged_joint_tracking_rmse",
}
STRENGTHS_ASCENDING = tuple(sorted(float(value) for value in WEAK_STRENGTHS))
STRENGTHS_DESCENDING = tuple(sorted(STRENGTHS_ASCENDING, reverse=True))


@dataclass(frozen=True)
class Task:
  campaign: str
  task_id: int
  model: str
  seed: int


@dataclass(frozen=True)
class ExpectedRun:
  experiment_commit: str
  model: str
  seed: int
  studies: tuple[str, ...]


def baseline_tasks() -> tuple[Task, ...]:
  return tuple(
      Task("weak-actuator-baselines", index, model, seed)
      for index, (model, seed) in enumerate(
          (model, seed) for model in BASELINE_MODELS for seed in range(1, 6)
      )
  )


def hodge_extension_tasks() -> tuple[Task, ...]:
  return tuple(
      Task("weak-actuator-hodge-extension", index, model, seed)
      for index, (model, seed) in enumerate(
          (model, seed) for model in HODGE_MODELS for seed in range(6, 11)
      )
  )


def expected_runs(campaign_commit: str) -> tuple[ExpectedRun, ...]:
  _validate_commit(campaign_commit)
  membership: dict[tuple[str, int], set[str]] = defaultdict(set)
  commits: dict[tuple[str, int], str] = {}
  for model in MODEL_ORDER:
    for seed in range(1, 6):
      key = (model, seed)
      membership[key].add("main")
      commits[key] = (
          REUSED_HODGE_COMMIT if model in HODGE_MODELS else campaign_commit
      )
  for model in HODGE_MODELS:
    for seed in range(1, 11):
      key = (model, seed)
      membership[key].add("hodge")
      commits[key] = REUSED_HODGE_COMMIT if seed <= 5 else campaign_commit
  return tuple(
      ExpectedRun(commits[key], key[0], key[1], tuple(sorted(studies)))
      for key, studies in sorted(
          membership.items(), key=lambda item: (
              MODEL_ORDER.index(item[0][0]), item[0][1]
          )
      )
  )


def _validate_commit(commit: str) -> None:
  if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
    raise ValueError(f"not a full lowercase commit: {commit!r}")


def run_directory(
    artifact_root: str | Path, commit: str, model: str, seed: int,
) -> Path:
  return (
      Path(artifact_root).resolve()
      / commit
      / "weak-actuator"
      / model
      / f"seed-{seed}"
      / f"steps-{REQUESTED_STEPS}"
      / f"evals-{NUM_EVALS}"
      / f"grid-{EVALUATION_GRID}"
  )


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_tree(path: str | Path) -> tuple[str, int]:
  root = Path(path)
  files = sorted(child for child in root.rglob("*") if child.is_file())
  if not files:
    raise ValueError(f"empty checkpoint tree: {root}")
  digest = hashlib.sha256()
  for child in files:
    relative = child.relative_to(root).as_posix()
    digest.update(relative.encode())
    digest.update(b"\0")
    digest.update(str(child.stat().st_size).encode())
    digest.update(b"\0")
    digest.update(bytes.fromhex(sha256_file(child)))
    digest.update(b"\n")
  return digest.hexdigest(), len(files)


def _git(repository: Path, *args: str) -> str:
  return subprocess.check_output(
      ("git", "-C", str(repository), *args), text=True,
  ).strip()


def scientific_file_hashes(
    repository: str | Path, revision: str,
) -> dict[str, str]:
  root = Path(repository).resolve()
  _validate_commit(revision)
  observed = {}
  for path, expected_hash in SCIENTIFIC_FILES.items():
    data = subprocess.check_output(
        ("git", "-C", str(root), "show", f"{revision}:{path}"),
    )
    observed[path] = sha256_bytes(data)
    if observed[path] != expected_hash:
      raise ValueError(
          f"scientific file changed at {revision}: {path} "
          f"{observed[path]} != {expected_hash}"
      )
  return observed


def verify_source_identity(
    repository: str | Path, campaign_commit: str,
) -> dict[str, Any]:
  root = Path(repository).resolve()
  head = _git(root, "rev-parse", "HEAD")
  if head != campaign_commit:
    raise ValueError(f"repository HEAD {head} != campaign {campaign_commit}")
  dirty = _git(root, "status", "--porcelain")
  if dirty:
    raise ValueError(f"repository is dirty:\n{dirty}")
  reused = scientific_file_hashes(root, REUSED_HODGE_COMMIT)
  campaign = scientific_file_hashes(root, campaign_commit)
  if reused != campaign:
    raise AssertionError("campaign and reused Hodge scientific bytes differ")
  return {
      "repository": str(root),
      "reused_hodge_commit": REUSED_HODGE_COMMIT,
      "campaign_commit": campaign_commit,
      "byte_identical": True,
      "files": campaign,
  }


def _read_json(path: Path) -> Any:
  with path.open() as handle:
    value = json.load(handle)
  _assert_finite(value, path.name)
  return value


def _assert_finite(value: Any, label: str) -> None:
  if isinstance(value, Mapping):
    for key, child in value.items():
      _assert_finite(child, f"{label}.{key}")
  elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
    for index, child in enumerate(value):
      _assert_finite(child, f"{label}[{index}]")
  elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
    raise ValueError(f"non-finite value at {label}")


def _parse_optional_float(value: str) -> float | None:
  return None if value == "" else float(value)


def read_condition_rows(path: str | Path) -> list[dict[str, Any]]:
  with Path(path).open(newline="") as handle:
    raw_rows = list(csv.DictReader(handle))
  rows = []
  for raw in raw_rows:
    row = dict(raw)
    row["checkpoint_step"] = int(row["checkpoint_step"])
    row["actuator_index"] = (
        None if row["actuator_index"] == "" else int(row["actuator_index"])
    )
    row["actuator_name"] = row["actuator_name"] or None
    row["strength"] = float(row["strength"])
    row["command_value"] = json.loads(row["command_value"])
    row["reset_key"] = int(row["reset_key"])
    for metric in METRIC_COLUMNS:
      row[metric] = _parse_optional_float(row[metric])
    if row["was_terminated"] not in {"True", "False"}:
      raise ValueError(f"invalid termination boolean: {row['was_terminated']}")
    row["was_terminated"] = row["was_terminated"] == "True"
    row["termination_step"] = (
        None if row["termination_step"] == "" else int(row["termination_step"])
    )
    row["active_steps"] = int(row["active_steps"])
    rows.append(row)
  return rows


def expected_row_keys() -> tuple[tuple[Any, ...], ...]:
  return tuple(
      (
          condition.condition,
          condition.actuator_index,
          condition.actuator_name,
          float(condition.strength),
          command,
          reset_key,
      )
      for command in COMMANDS
      for reset_key in RESET_KEYS
      for condition in CONDITIONS
  )


def validate_condition_rows(rows: list[dict[str, Any]], model: str) -> None:
  expected = expected_row_keys()
  observed = tuple(
      (
          row["condition"], row["actuator_index"], row["actuator_name"],
          float(row["strength"]), row["command"], int(row["reset_key"]),
      )
      for row in rows
  )
  if len(rows) != ROW_COUNT or observed != expected:
    raise ValueError("condition rows are incomplete, duplicated, or misordered")
  if len(set(observed)) != ROW_COUNT:
    raise ValueError("condition rows contain duplicate scientific keys")
  for index, row in enumerate(rows):
    if row["model"] != model or row["model_tier"] != MODEL_TIERS[model]:
      raise ValueError(f"row {index} has wrong model identity")
    if row["checkpoint_step"] != ACTUAL_STEPS:
      raise ValueError(f"row {index} has wrong checkpoint step")
    if row["command_value"] != list(COMMANDS[row["command"]]):
      raise ValueError(f"row {index} has wrong command value")
    healthy = row["actuator_index"] is None
    for metric in METRIC_COLUMNS:
      value = row[metric]
      if healthy and metric in NULLABLE_METRICS:
        if value is not None:
          raise ValueError(f"healthy row {index} has {metric}")
      elif value is None or not math.isfinite(value):
        raise ValueError(f"row {index} has invalid {metric}")
    terminated = row["was_terminated"]
    active_steps = row["active_steps"]
    termination_step = row["termination_step"]
    if terminated:
      if termination_step != active_steps or not 1 <= active_steps <= HORIZON:
        raise ValueError(f"row {index} has invalid terminal audit")
    elif termination_step is not None or active_steps != HORIZON:
      raise ValueError(f"row {index} has invalid survival audit")
    expected_fall = float(terminated and termination_step < HORIZON)
    if row["fall"] != expected_fall or not 0.0 <= row["survival"] <= 1.0:
      raise ValueError(f"row {index} has invalid fall/survival fields")


def _mean(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
  values = [float(row[metric]) for row in rows if row.get(metric) is not None]
  if not values:
    raise ValueError(f"no values for {metric}")
  result = float(np.mean(values))
  if not math.isfinite(result):
    raise ValueError(f"non-finite mean for {metric}")
  return result


def _group_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
  output = {}
  for metric in METRIC_COLUMNS:
    values = [row[metric] for row in rows if row.get(metric) is not None]
    if not values and metric in NULLABLE_METRICS:
      output[metric] = None
    else:
      output[metric] = _mean(rows, metric)
  return output


def normalized_auc(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
  selected = {float(row["strength"]): float(row[metric]) for row in rows}
  if set(selected) != set(STRENGTHS_ASCENDING):
    raise ValueError(f"AUC grid mismatch for {metric}: {sorted(selected)}")
  x = np.asarray(STRENGTHS_ASCENDING, dtype=np.float64)
  y = np.asarray([selected[value] for value in x], dtype=np.float64)
  result = float(np.trapezoid(y, x) / (x[-1] - x[0]))
  if not math.isfinite(result):
    raise ValueError(f"non-finite AUC for {metric}")
  return result


def collapse_threshold(severity_rows: Sequence[Mapping[str, Any]]) -> float | None:
  by_strength = {float(row["strength"]): row for row in severity_rows}
  if set(by_strength) != set(STRENGTHS_DESCENDING):
    raise ValueError("collapse grid mismatch")
  lowest = None
  for strength in STRENGTHS_DESCENDING:
    row = by_strength[strength]
    passes = (
        float(row["return_retention"]) >= 0.90
        and float(row["survival"]) >= 0.95
        and float(row["fall"]) <= 0.05
    )
    if not passes:
      break
    lowest = strength
  return lowest


def t_interval(values: Sequence[float], confidence: float = 0.95) -> dict[str, float | int]:
  array = np.asarray(values, dtype=np.float64)
  if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
    raise ValueError("t interval requires at least two finite values")
  mean = float(np.mean(array))
  sample_sd = float(np.std(array, ddof=1))
  critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(array) - 1))
  margin = critical * sample_sd / math.sqrt(len(array))
  return {
      "n": len(array),
      "mean": mean,
      "sample_sd": sample_sd,
      "confidence": confidence,
      "ci_low": mean - margin,
      "ci_high": mean + margin,
  }


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
  values = np.asarray(differences, dtype=np.float64)
  if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
    raise ValueError("sign-flip test requires finite paired differences")
  observed = abs(float(np.mean(values)))
  count = 0
  for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
    statistic = abs(float(np.mean(values * np.asarray(signs))))
    if statistic >= observed - 1e-15:
      count += 1
  return count / (2 ** len(values))


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
  ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
  adjusted = {}
  running = 0.0
  total = len(ordered)
  for index, (name, pvalue) in enumerate(ordered):
    if not 0.0 <= pvalue <= 1.0:
      raise ValueError(f"invalid p-value for {name}")
    running = max(running, (total - index) * pvalue)
    adjusted[name] = min(1.0, running)
  return adjusted


def _run_key(model: str, seed: int) -> str:
  return f"{model}:seed-{seed}"


def _load_retry_registry(
    path: str | Path | None, expected: Sequence[ExpectedRun],
) -> tuple[dict[str, Path], dict[str, Any] | None]:
  if path is None:
    return {}, None
  registry_path = Path(path).resolve()
  value = _read_json(registry_path)
  if set(value) != {"runs"} or not isinstance(value["runs"], dict):
    raise ValueError("retry registry must contain only a runs mapping")
  allowed = {_run_key(item.model, item.seed) for item in expected}
  unknown = set(value["runs"]) - allowed
  if unknown:
    raise ValueError(f"retry registry has unexpected runs: {sorted(unknown)}")
  resolved = {}
  for key, raw_path in value["runs"].items():
    candidate = Path(raw_path)
    if not candidate.is_absolute():
      raise ValueError(f"retry path must be absolute: {raw_path}")
    resolved[key] = candidate.resolve()
  return resolved, {
      "path": str(registry_path),
      "sha256": sha256_file(registry_path),
      "runs": {key: str(value) for key, value in sorted(resolved.items())},
  }


def _validate_progress(path: Path) -> list[dict[str, Any]]:
  records = []
  with path.open() as handle:
    for line in handle:
      record = json.loads(line)
      _assert_finite(record, "progress")
      records.append(record)
  steps = [int(record["step"]) for record in records]
  if (
      len(records) != NUM_EVALS
      or steps[0] != 0
      or steps[-1] != ACTUAL_STEPS
      or steps != sorted(set(steps))
  ):
    raise ValueError(f"invalid progress sequence: {path}")
  return records


def validate_run(
    directory: Path, expected: ExpectedRun,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
  if not directory.is_dir():
    raise FileNotFoundError(f"missing run directory: {directory}")
  if (directory / "failure.json").exists():
    raise ValueError(f"run has a failure record: {directory}")
  required = (
      "run.json", "curriculum.json", "ppo_config.json", "progress.jsonl",
      "condition_metrics.csv", "evaluation.json", "result.json",
  )
  missing = [name for name in required if not (directory / name).is_file()]
  if missing:
    raise FileNotFoundError(f"run is incomplete {directory}: {missing}")
  run = _read_json(directory / "run.json")
  result = _read_json(directory / "result.json")
  evaluation = _read_json(directory / "evaluation.json")
  ppo_config = _read_json(directory / "ppo_config.json")
  curriculum = _read_json(directory / "curriculum.json")

  expected_values = {
      "experiment_commit": expected.experiment_commit,
      "model": expected.model,
      "seed": expected.seed,
      "requested_steps": REQUESTED_STEPS,
      "actual_steps": ACTUAL_STEPS,
      "num_evals": NUM_EVALS,
      "evaluation_grid": EVALUATION_GRID,
      "core_commit": CORE_COMMIT,
      "actor_parameters": MODEL_PARAMETERS[expected.model],
  }
  for name, value in expected_values.items():
    if run.get(name) != value or result.get(name) != value:
      raise ValueError(f"run identity mismatch for {name}: {directory}")
  if run.get("model_tier") != MODEL_TIERS[expected.model]:
    raise ValueError(f"wrong model tier: {directory}")
  if run.get("protocol_source_commit") != ORIGINAL_WEAK_ACTUATOR_COMMIT:
    raise ValueError(f"wrong source protocol: {directory}")
  if run.get("terminal_checkpoint") != "checkpoints/000412876800":
    raise ValueError(f"wrong terminal checkpoint declaration: {directory}")
  if run.get("curriculum_sha256") != CURRICULUM_SHA256:
    raise ValueError(f"wrong curriculum identity: {directory}")
  if run.get("official_ppo_config_sha256") != OFFICIAL_PPO_CONFIG_SHA256:
    raise ValueError(f"wrong PPO identity: {directory}")
  if ppo_config.get("official_sha256") != OFFICIAL_PPO_CONFIG_SHA256:
    raise ValueError(f"wrong serialized PPO identity: {directory}")
  if ppo_config.get("effective_sha256") != run.get("effective_ppo_config_sha256"):
    raise ValueError(f"effective PPO hash mismatch: {directory}")
  accounting = ppo_config.get("step_accounting", {})
  if (
      accounting.get("requested_environment_timesteps") != REQUESTED_STEPS
      or accounting.get("actual_environment_timesteps") != ACTUAL_STEPS
  ):
    raise ValueError(f"wrong step accounting: {directory}")
  if curriculum.get("sha256") != CURRICULUM_SHA256:
    raise ValueError(f"curriculum file mismatch: {directory}")
  factory = run.get("serialized_factory_kwargs")
  expected_factory = {
      "model_name": expected.model,
      "policy_hidden_layer_sizes": [512, 256, 128],
      "policy_obs_key": "state",
      "value_hidden_layer_sizes": [512, 256, 128],
      "value_obs_key": "privileged_state",
  }
  if factory != expected_factory:
    raise ValueError(f"checkpoint factory mismatch: {directory}")
  dependencies = run.get("dependencies", {})
  if (
      dependencies.get("core", {}).get("commit") != CORE_COMMIT
      or dependencies.get("playground", {}).get("commit") != PLAYGROUND_COMMIT
      or dependencies.get("menagerie", {}).get("commit") != MENAGERIE_COMMIT
      or not dependencies.get("playground", {}).get("clean")
      or not dependencies.get("menagerie", {}).get("clean")
  ):
    raise ValueError(f"dependency provenance mismatch: {directory}")
  if result.get("status") != "COMPLETE" or result.get("restored_rollout_rows") != ROW_COUNT:
    raise ValueError(f"run is not complete: {directory}")
  if result.get("runtime", {}).get("backend") != "gpu":
    raise ValueError(f"run did not execute on a GPU: {directory}")
  if result.get("changed_actor_leaves", 0) <= 0:
    raise ValueError(f"actor did not change during training: {directory}")
  checkpoint = result.get("checkpoint", {})
  if checkpoint != {
      "path": "checkpoints/000412876800",
      "exact_parameter_roundtrip": True,
      "exact_logits_roundtrip": True,
      "exact_actions_roundtrip": True,
  }:
    raise ValueError(f"checkpoint roundtrip failed: {directory}")
  probe = result.get("fixed_probe", {})
  if (
      probe.get("terminal_logits_sha256") != probe.get("restored_logits_sha256")
      or probe.get("terminal_actions_sha256") != probe.get("restored_actions_sha256")
  ):
    raise ValueError(f"fixed-probe roundtrip failed: {directory}")
  if (
      evaluation.get("row_count") != ROW_COUNT
      or evaluation.get("condition_count") != 109
      or evaluation.get("grid") != EVALUATION_GRID
      or evaluation.get("checkpoint_step") != ACTUAL_STEPS
      or evaluation.get("model") != expected.model
      or evaluation.get("ordered_conditions_valid") is not True
      or evaluation.get("continuous_metrics_finite") is not True
  ):
    raise ValueError(f"evaluation audit failed: {directory}")
  rows = read_condition_rows(directory / "condition_metrics.csv")
  validate_condition_rows(rows, expected.model)
  progress = _validate_progress(directory / "progress.jsonl")
  checkpoint_path = directory / "checkpoints" / "000412876800"
  checkpoint_sha256, checkpoint_files = sha256_tree(checkpoint_path)
  file_hashes = {
      name: sha256_file(directory / name) for name in required
  }
  catalog = {
      "model": expected.model,
      "label": MODEL_LABELS[expected.model],
      "role": MODEL_ROLES[expected.model],
      "actor_parameters": MODEL_PARAMETERS[expected.model],
      "seed": expected.seed,
      "experiment_commit": expected.experiment_commit,
      "studies": list(expected.studies),
      "artifact_path": str(directory),
      "status": "COMPLETE",
      "requested_steps": REQUESTED_STEPS,
      "actual_steps": ACTUAL_STEPS,
      "evaluation_rows": ROW_COUNT,
      "effective_ppo_config_sha256": run["effective_ppo_config_sha256"],
      "checkpoint_tree_sha256": checkpoint_sha256,
      "checkpoint_file_count": checkpoint_files,
      "wall_time_seconds": float(result["wall_time_seconds"]),
      "environment_steps_per_second": float(result["environment_steps_per_second"]),
      "artifact_hashes": file_hashes,
  }
  return catalog, rows, progress


def summarize_run(
    model: str,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]] | dict[str, Any]]:
  healthy_rows = [row for row in rows if row["condition"] == "healthy"]
  if len(healthy_rows) != len(COMMANDS) * len(RESET_KEYS):
    raise ValueError("healthy grid is incomplete")
  healthy = _group_metrics(healthy_rows)
  if healthy["undiscounted_return"] == 0.0:
    raise ValueError("healthy return is zero; retention is undefined")
  return_retention_reliable = (
      healthy["survival"] >= 0.95 and healthy["fall"] <= 0.05
  )
  severity = []
  for strength in STRENGTHS_ASCENDING:
    selected = [
        row for row in rows
        if row["actuator_index"] is not None and row["strength"] == strength
    ]
    if len(selected) != 12 * len(COMMANDS) * len(RESET_KEYS):
      raise ValueError(f"incomplete strength grid: {strength}")
    metrics = _group_metrics(selected)
    severity.append({
        "model": model,
        "label": MODEL_LABELS[model],
        "seed": seed,
        "strength": strength,
        **metrics,
        "return_retention": (
            metrics["undiscounted_return"] / healthy["undiscounted_return"]
        ),
        "return_retention_reliable": return_retention_reliable,
    })
  metric_to_endpoint = {
      "undiscounted_return": "absolute_return_auc",
      "survival": "survival_auc",
      "return_retention": "return_retention_auc",
      "fall": "fall_rate_auc",
      "velocity_rmse": "velocity_rmse_auc",
      "yaw_rmse": "yaw_rmse_auc",
      "absolute_mechanical_power": "absolute_mechanical_power_auc",
      "torque_rms": "torque_rms_auc",
      "action_delta_rms": "action_delta_rms_auc",
      "actuator_force_utilization": "actuator_force_utilization_auc",
      "saturation_fraction": "saturation_fraction_auc",
      "damaged_actuator_force_utilization": (
          "damaged_actuator_force_utilization_auc"
      ),
      "damaged_joint_tracking_rmse": "damaged_joint_tracking_rmse_auc",
  }
  aucs = {
      endpoint: normalized_auc(severity, metric)
      for metric, endpoint in metric_to_endpoint.items()
  }
  terminal_metrics = result["terminal_metrics"]
  seed_metrics = {
      "model": model,
      "label": MODEL_LABELS[model],
      "role": MODEL_ROLES[model],
      "actor_parameters": MODEL_PARAMETERS[model],
      "seed": seed,
      **aucs,
      "collapse_threshold": collapse_threshold(severity),
      "return_retention_reliable": return_retention_reliable,
      **{f"healthy_{name}": value for name, value in healthy.items()},
      "terminal_training_reward": float(terminal_metrics["eval/episode_reward"]),
      "terminal_training_episode_length": float(
          terminal_metrics["eval/avg_episode_length"]
      ),
  }
  command_rows = []
  for command in COMMANDS:
    command_healthy = [
        row for row in healthy_rows if row["command"] == command
    ]
    healthy_return = _mean(command_healthy, "undiscounted_return")
    if healthy_return == 0.0:
      raise ValueError(f"zero healthy return for command {command}")
    for strength in STRENGTHS_ASCENDING:
      selected = [
          row for row in rows
          if row["command"] == command
          and row["actuator_index"] is not None
          and row["strength"] == strength
      ]
      metrics = _group_metrics(selected)
      command_rows.append({
          "model": model, "label": MODEL_LABELS[model], "seed": seed,
          "command": command, "strength": strength, **metrics,
          "return_retention": metrics["undiscounted_return"] / healthy_return,
      })
  joint_rows = []
  for joint_type in ("hip", "thigh", "calf"):
    for strength in STRENGTHS_ASCENDING:
      selected = [
          row for row in rows
          if row["actuator_name"] is not None
          and row["actuator_name"].endswith(f"_{joint_type}")
          and row["strength"] == strength
      ]
      metrics = _group_metrics(selected)
      joint_rows.append({
          "model": model, "label": MODEL_LABELS[model], "seed": seed,
          "joint_type": joint_type, "strength": strength, **metrics,
          "return_retention": (
              metrics["undiscounted_return"] / healthy["undiscounted_return"]
          ),
      })
  actuator_rows = []
  actuator_names = [condition.actuator_name for condition in CONDITIONS[1:13]]
  for actuator_name in actuator_names:
    for strength in STRENGTHS_ASCENDING:
      selected = [
          row for row in rows
          if row["actuator_name"] == actuator_name
          and row["strength"] == strength
      ]
      metrics = _group_metrics(selected)
      actuator_rows.append({
          "model": model, "label": MODEL_LABELS[model], "seed": seed,
          "actuator_name": actuator_name, "strength": strength, **metrics,
          "return_retention": (
              metrics["undiscounted_return"] / healthy["undiscounted_return"]
          ),
      })
  return {
      "seed_metrics": seed_metrics,
      "severity": severity,
      "commands": command_rows,
      "joints": joint_rows,
      "actuators": actuator_rows,
  }


def _summary_rows(
    seed_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
  rows = []
  for model in MODEL_ORDER:
    selected = [row for row in seed_metrics if row["model"] == model and row["seed"] <= 5]
    if len(selected) != 5:
      raise ValueError(f"main study has wrong seed count for {model}")
    row: dict[str, Any] = {
        "model": model,
        "label": MODEL_LABELS[model],
        "role": MODEL_ROLES[model],
        "actor_parameters": MODEL_PARAMETERS[model],
        "n": 5,
    }
    for metric in AUC_ENDPOINTS:
      interval = t_interval([float(item[metric]) for item in selected])
      row.update({
          f"{metric}_mean": interval["mean"],
          f"{metric}_sample_sd": interval["sample_sd"],
          f"{metric}_ci_low": interval["ci_low"],
          f"{metric}_ci_high": interval["ci_high"],
      })
    rows.append(row)
  return rows


def _aggregate_curves(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
  groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
  for row in rows:
    groups[tuple(row[key] for key in keys)].append(row)
  output = []
  for group_key, selected in sorted(groups.items()):
    record = dict(zip(keys, group_key))
    record["label"] = MODEL_LABELS[record["model"]]
    record["n"] = len(selected)
    for metric in metrics:
      values = [float(row[metric]) for row in selected]
      record[f"{metric}_mean"] = float(np.mean(values))
      record[f"{metric}_sample_sd"] = (
          float(np.std(values, ddof=1)) if len(values) > 1 else None
      )
    output.append(record)
  return output


def _group_auc(
    rows: Sequence[Mapping[str, Any]], group_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  seed_groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
  for row in rows:
    seed_groups[(row["model"], int(row["seed"]), str(row[group_name]))].append(row)
  seed_auc = []
  for (model, seed, group), selected in sorted(seed_groups.items()):
    seed_auc.append({
        "model": model,
        "label": MODEL_LABELS[model],
        "seed": seed,
        group_name: group,
        "absolute_return_auc": normalized_auc(selected, "undiscounted_return"),
        "return_retention_auc": normalized_auc(selected, "return_retention"),
        "survival_auc": normalized_auc(selected, "survival"),
    })
  summary = _aggregate_curves(
      seed_auc, ("model", group_name),
      ("absolute_return_auc", "return_retention_auc", "survival_auc"),
  )
  return seed_auc, summary


def hodge_inference(
    seed_metrics: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  indexed = {
      (row["model"], int(row["seed"])): row
      for row in seed_metrics if row["model"] in HODGE_MODELS
  }
  expected = {(model, seed) for model in HODGE_MODELS for seed in range(1, 11)}
  if set(indexed) != expected:
    raise ValueError("Hodge study does not contain exactly paired seeds 1-10")
  paired = []
  inference = []
  raw_secondary_pvalues = {}
  for metric in AUC_ENDPOINTS:
    differences = []
    for seed in range(1, 11):
      lower = float(indexed[(HODGE_MODELS[0], seed)][metric])
      full = float(indexed[(HODGE_MODELS[1], seed)][metric])
      difference = full - lower
      differences.append(difference)
      paired.append({
          "endpoint": metric,
          "seed": seed,
          "hodge_lower": lower,
          "hodge_full": full,
          "full_minus_lower": difference,
      })
    interval = t_interval(differences)
    sample_sd = float(interval["sample_sd"])
    role = (
        "primary" if metric == PRIMARY_ENDPOINT
        else "secondary" if metric in SECONDARY_ENDPOINTS
        else "descriptive"
    )
    pvalue = exact_sign_flip_pvalue(differences) if role != "descriptive" else None
    if role == "secondary":
      raw_secondary_pvalues[metric] = float(pvalue)
    inference.append({
        "endpoint": metric,
        "role": role,
        "n_pairs": 10,
        "mean_full_minus_lower": interval["mean"],
        "sample_sd_difference": sample_sd,
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "cohen_dz": None if sample_sd == 0.0 else float(interval["mean"]) / sample_sd,
        "exact_two_sided_sign_flip_p": pvalue,
        "holm_adjusted_p": None,
        "alpha": 0.05 if role != "descriptive" else None,
        "reject": bool(pvalue < 0.05) if role == "primary" else None,
    })
  adjusted = holm_adjust(raw_secondary_pvalues)
  for row in inference:
    if row["role"] == "secondary":
      row["holm_adjusted_p"] = adjusted[row["endpoint"]]
      row["reject"] = adjusted[row["endpoint"]] < 0.05
  return paired, inference


def _write_json(path: Path, value: Any) -> None:
  with path.open("x") as handle:
    json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
  if not rows:
    raise ValueError(f"cannot write empty table: {path}")
  fieldnames = list(rows[0])
  if any(list(row) != fieldnames for row in rows):
    raise ValueError(f"table columns are inconsistent: {path}")
  with path.open("x", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _format_number(value: Any, digits: int = 4) -> str:
  if value is None:
    return "--"
  return f"{float(value):.{digits}g}"


def _write_report(
    output: Path,
    model_summary: Sequence[Mapping[str, Any]],
    inference: Sequence[Mapping[str, Any]],
) -> None:
  by_model = {row["model"]: row for row in model_summary}
  native = by_model["native_mlp"][f"{PRIMARY_ENDPOINT}_mean"]
  hodge_full = by_model["multirank_hodge_full"][f"{PRIMARY_ENDPOINT}_mean"]
  if hodge_full > native:
    capacity = (
        "Hodge-F's observed mean primary AUC exceeds the 192K Native MLP. "
        "This supports the bounded descriptive statement that higher generic "
        "capacity alone was insufficient in this campaign."
    )
  else:
    capacity = (
        "The 192K Native MLP matches or exceeds Hodge-F's observed mean "
        "primary AUC, so this campaign does not support the statement that "
        "higher generic capacity alone was insufficient."
    )
  lines = [
      "# Paper-ready weak-actuator results",
      "",
      "## Design",
      "",
      "The main architecture study is descriptive over seeds 1-5. The Hodge ",
      "study uses paired seeds 1-10. Seed is the experimental unit.",
      "",
      f"> **{INTERPRETATION}**",
      "",
      "No categorical breakthrough criterion was prespecified; any such ",
      "analysis is exploratory.",
      "",
      "## Main architecture study",
      "",
      "| Model | Params | Mean absolute-return AUC | SD | 95% t CI |",
      "|---|---:|---:|---:|---:|",
  ]
  for row in model_summary:
    lines.append(
        f"| {row['label']} | {row['actor_parameters']:,} | "
        f"{_format_number(row[f'{PRIMARY_ENDPOINT}_mean'])} | "
        f"{_format_number(row[f'{PRIMARY_ENDPOINT}_sample_sd'])} | "
        f"[{_format_number(row[f'{PRIMARY_ENDPOINT}_ci_low'])}, "
        f"{_format_number(row[f'{PRIMARY_ENDPOINT}_ci_high'])}] |"
    )
  lines.extend([
      "",
      capacity,
      "",
      "Native MLP is an over-capacity reference and does not isolate the "
      "parameter contribution of the Hodge-F face pathway.",
      "",
      "## Paired Hodge study",
      "",
      "| Endpoint | Role | Mean F-L | 95% t CI | Exact p | Holm p |",
      "|---|---|---:|---:|---:|---:|",
  ])
  for row in inference:
    if row["role"] == "descriptive":
      continue
    lines.append(
        f"| `{row['endpoint']}` | {row['role']} | "
        f"{_format_number(row['mean_full_minus_lower'])} | "
        f"[{_format_number(row['ci_low'])}, {_format_number(row['ci_high'])}] | "
        f"{_format_number(row['exact_two_sided_sign_flip_p'])} | "
        f"{_format_number(row['holm_adjusted_p'])} |"
    )
  lines.extend([
      "",
      "All six-model comparisons above are descriptive. The primary Hodge "
      "test is unadjusted at alpha 0.05; the five secondary tests are one "
      "Holm-corrected family.",
      "",
  ])
  (output / "REPORT.md").write_text("\n".join(lines))


def _latex_escape(value: str) -> str:
  return value.replace("_", "\\_").replace("%", "\\%")


def _write_latex_tables(
    output: Path,
    model_summary: Sequence[Mapping[str, Any]],
    inference: Sequence[Mapping[str, Any]],
) -> None:
  main = [
      "\\begin{tabular}{lrrrr}",
      "\\toprule",
      "Model & Params & Mean AUC & SD & 95\\% CI \\\\",
      "\\midrule",
  ]
  for row in model_summary:
    main.append(
        f"{_latex_escape(row['label'])} & {row['actor_parameters']:,} & "
        f"{_format_number(row[f'{PRIMARY_ENDPOINT}_mean'])} & "
        f"{_format_number(row[f'{PRIMARY_ENDPOINT}_sample_sd'])} & "
        f"[{_format_number(row[f'{PRIMARY_ENDPOINT}_ci_low'])}, "
        f"{_format_number(row[f'{PRIMARY_ENDPOINT}_ci_high'])}] \\\\"
    )
  main.extend(["\\bottomrule", "\\end{tabular}", ""])
  (output / "main_table.tex").write_text("\n".join(main))
  hodge = [
      "\\begin{tabular}{llrrr}",
      "\\toprule",
      "Endpoint & Role & Mean F-L & Exact $p$ & Holm $p$ \\\\",
      "\\midrule",
  ]
  for row in inference:
    if row["role"] == "descriptive":
      continue
    hodge.append(
        f"{_latex_escape(row['endpoint'])} & {row['role']} & "
        f"{_format_number(row['mean_full_minus_lower'])} & "
        f"{_format_number(row['exact_two_sided_sign_flip_p'])} & "
        f"{_format_number(row['holm_adjusted_p'])} \\\\"
    )
  hodge.extend(["\\bottomrule", "\\end{tabular}", ""])
  (output / "hodge_table.tex").write_text("\n".join(hodge))


COLORS = {
    "native_mlp": "#6c757d",
    "mlp_match_lower": "#0072b2",
    "pointwise_match_lower": "#56b4e9",
    "gcn_match_lower": "#009e73",
    "multirank_hodge_lower": "#e69f00",
    "multirank_hodge_full": "#d55e00",
}


def _plot_style() -> None:
  plt.rcParams.update({
      "font.size": 9,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "figure.dpi": 120,
      "savefig.bbox": "tight",
  })


def _save_figure(figure, figures: Path, name: str) -> None:
  figure.savefig(
      figures / f"{name}.png", dpi=220,
      metadata={"Software": "go1-robot-experiments"},
  )
  figure.savefig(
      figures / f"{name}.pdf",
      metadata={
          "Creator": "go1-robot-experiments",
          "Producer": "Matplotlib",
          "CreationDate": None,
          "ModDate": None,
      },
  )
  plt.close(figure)


def _plot_primary(
    seed_metrics: Sequence[Mapping[str, Any]],
    model_summary: Sequence[Mapping[str, Any]],
    figures: Path,
) -> None:
  figure, axis = plt.subplots(figsize=(7.2, 3.8))
  summary = {row["model"]: row for row in model_summary}
  for index, model in enumerate(MODEL_ORDER):
    selected = sorted(
        (row for row in seed_metrics if row["model"] == model and row["seed"] <= 5),
        key=lambda row: row["seed"],
    )
    for row in selected:
      jitter = (int(row["seed"]) - 3) * 0.045
      axis.scatter(
          index + jitter, row[PRIMARY_ENDPOINT], s=32,
          color=COLORS[model], alpha=0.9, zorder=3,
      )
    item = summary[model]
    mean = item[f"{PRIMARY_ENDPOINT}_mean"]
    axis.errorbar(
        index, mean,
        yerr=[
            [mean - item[f"{PRIMARY_ENDPOINT}_ci_low"]],
            [item[f"{PRIMARY_ENDPOINT}_ci_high"] - mean],
        ],
        fmt="D", color="black", markersize=4, capsize=4, zorder=4,
    )
  axis.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[model] for model in MODEL_ORDER])
  axis.set_ylabel("Degraded-strength absolute-return AUC")
  axis.set_title("Main architecture study: every seed and 95% t interval")
  axis.grid(axis="y", alpha=0.2)
  _save_figure(figure, figures, "primary_auc_seed_distribution")


def _plot_severity(
    severity: Sequence[Mapping[str, Any]], figures: Path,
) -> None:
  for metric, ylabel, name in (
      ("undiscounted_return", "Absolute return", "absolute_return_severity"),
      ("survival", "Survival fraction", "survival_severity"),
  ):
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for model in MODEL_ORDER:
      selected = [row for row in severity if row["model"] == model and row["seed"] <= 5]
      for seed in range(1, 6):
        seed_rows = sorted(
            (row for row in selected if row["seed"] == seed),
            key=lambda row: row["strength"],
        )
        axis.plot(
            [row["strength"] for row in seed_rows],
            [row[metric] for row in seed_rows],
            color=COLORS[model], alpha=0.16, linewidth=0.8,
        )
      means = [
          np.mean([
              row[metric] for row in selected if row["strength"] == strength
          ])
          for strength in STRENGTHS_ASCENDING
      ]
      axis.plot(
          STRENGTHS_ASCENDING, means, marker="o", markersize=3,
          color=COLORS[model], linewidth=1.8, label=MODEL_LABELS[model],
      )
    axis.set_xlabel("Damaged-actuator strength")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} across the severity sweep")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, frameon=False)
    _save_figure(figure, figures, name)


def _plot_falls_tracking(
    severity: Sequence[Mapping[str, Any]], figures: Path,
) -> None:
  figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True)
  for model in MODEL_ORDER:
    selected = [row for row in severity if row["model"] == model and row["seed"] <= 5]
    for axis, metric, title in zip(
        axes,
        ("fall", "velocity_rmse", "yaw_rmse"),
        ("Fall rate", "Velocity RMSE", "Yaw RMSE"),
    ):
      means = [
          np.mean([row[metric] for row in selected if row["strength"] == strength])
          for strength in STRENGTHS_ASCENDING
      ]
      axis.plot(
          STRENGTHS_ASCENDING, means, marker="o", markersize=2.5,
          color=COLORS[model], linewidth=1.5, label=MODEL_LABELS[model],
      )
      axis.set_title(title)
      axis.grid(alpha=0.2)
      axis.set_xlabel("Strength")
  axes[0].legend(ncol=2, frameon=False, fontsize=7)
  figure.suptitle("Failure and command-tracking outcomes")
  _save_figure(figure, figures, "falls_and_tracking")


def _plot_group_heatmaps(
    command_summary: Sequence[Mapping[str, Any]],
    joint_summary: Sequence[Mapping[str, Any]],
    figures: Path,
) -> None:
  figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
  for axis, rows, group_name, groups, title in (
      (axes[0], command_summary, "command", list(COMMANDS), "Command AUC"),
      (axes[1], joint_summary, "joint_type", ["hip", "thigh", "calf"], "Joint-type AUC"),
  ):
    indexed = {
        (row["model"], row[group_name]): row["absolute_return_auc_mean"]
        for row in rows
    }
    matrix = np.asarray([
        [indexed[(model, group)] for group in groups] for model in MODEL_ORDER
    ])
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(groups)), groups, rotation=45, ha="right")
    axis.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[model] for model in MODEL_ORDER])
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
  figure.suptitle("Absolute-return robustness by command and joint type")
  _save_figure(figure, figures, "command_and_joint_panels")


def _plot_learning(
    learning: Sequence[Mapping[str, Any]], figures: Path,
) -> None:
  figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
  for model in MODEL_ORDER:
    selected = [row for row in learning if row["model"] == model and row["seed"] <= 5]
    for seed in range(1, 6):
      seed_rows = sorted(
          (row for row in selected if row["seed"] == seed), key=lambda row: row["step"]
      )
      for axis, metric in zip(axes, ("evaluation_reward", "average_episode_length")):
        axis.plot(
            [row["step"] for row in seed_rows],
            [row[metric] for row in seed_rows],
            color=COLORS[model], alpha=0.18, linewidth=0.8,
        )
    steps = sorted({row["step"] for row in selected})
    for axis, metric in zip(axes, ("evaluation_reward", "average_episode_length")):
      means = [np.mean([row[metric] for row in selected if row["step"] == step]) for step in steps]
      axis.plot(steps, means, color=COLORS[model], linewidth=1.8, label=MODEL_LABELS[model])
  axes[0].set_title("Evaluation reward")
  axes[1].set_title("Average episode length")
  for axis in axes:
    axis.set_xlabel("Environment steps")
    axis.grid(alpha=0.2)
  axes[0].legend(ncol=2, frameon=False, fontsize=7)
  figure.suptitle("PPO learning curves; individual seeds are faint")
  _save_figure(figure, figures, "ppo_learning_curves")


def _plot_hodge_pairs(
    paired: Sequence[Mapping[str, Any]], figures: Path,
) -> None:
  selected = sorted(
      (row for row in paired if row["endpoint"] == PRIMARY_ENDPOINT),
      key=lambda row: row["seed"],
  )
  figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
  for row in selected:
    axes[0].plot(
        [0, 1], [row["hodge_lower"], row["hodge_full"]],
        marker="o", linewidth=1.0, alpha=0.75,
    )
  axes[0].set_xticks([0, 1], ["Hodge-L", "Hodge-F"])
  axes[0].set_ylabel("Absolute-return AUC")
  axes[0].set_title("Paired seed outcomes")
  axes[1].axhline(0.0, color="black", linewidth=0.8)
  axes[1].scatter(
      [row["seed"] for row in selected],
      [row["full_minus_lower"] for row in selected],
      color=COLORS["multirank_hodge_full"],
  )
  axes[1].set_xlabel("Seed")
  axes[1].set_ylabel("Hodge-F minus Hodge-L")
  axes[1].set_title("Paired primary differences")
  for axis in axes:
    axis.grid(alpha=0.2)
  _save_figure(figure, figures, "hodge_paired_primary")


def build_paper_pack(
    artifact_root: str | Path,
    repository: str | Path,
    campaign_commit: str,
    output: str | Path,
    retry_registry: str | Path | None = None,
) -> None:
  output_path = Path(output).resolve()
  if output_path.exists():
    raise FileExistsError(f"paper output already exists: {output_path}")
  source_identity = verify_source_identity(repository, campaign_commit)
  expected = expected_runs(campaign_commit)
  retry_paths, retry_record = _load_retry_registry(retry_registry, expected)
  catalogs = []
  seed_metrics = []
  severity = []
  commands = []
  joints = []
  actuators = []
  learning = []
  effective_hashes: dict[int, set[str]] = defaultdict(set)
  for item in expected:
    key = _run_key(item.model, item.seed)
    directory = retry_paths.get(
        key,
        run_directory(artifact_root, item.experiment_commit, item.model, item.seed),
    )
    catalog, rows, progress = validate_run(directory, item)
    result = _read_json(directory / "result.json")
    summaries = summarize_run(item.model, item.seed, rows, result)
    catalogs.append(catalog)
    seed_metrics.append(summaries["seed_metrics"])
    severity.extend(summaries["severity"])
    commands.extend(summaries["commands"])
    joints.extend(summaries["joints"])
    actuators.extend(summaries["actuators"])
    effective_hashes[item.seed].add(catalog["effective_ppo_config_sha256"])
    for record in progress:
      metrics = record["metrics"]
      learning.append({
          "model": item.model,
          "label": MODEL_LABELS[item.model],
          "seed": item.seed,
          "step": int(record["step"]),
          "evaluation_reward": float(metrics["eval/episode_reward"]),
          "average_episode_length": float(metrics["eval/avg_episode_length"]),
      })
  mismatched = {seed: values for seed, values in effective_hashes.items() if len(values) != 1}
  if mismatched:
    raise ValueError(f"models used different effective PPO configs: {mismatched}")
  if len(catalogs) != 40 or sum("main" in row["studies"] for row in catalogs) != 30:
    raise AssertionError("paper roster cardinality changed")
  if sum("hodge" in row["studies"] for row in catalogs) != 20:
    raise AssertionError("Hodge roster cardinality changed")
  model_summary = _summary_rows(seed_metrics)
  severity_summary = _aggregate_curves(
      [row for row in severity if row["seed"] <= 5],
      ("model", "strength"),
      (
          "undiscounted_return", "return_retention", "survival", "fall",
          "velocity_rmse", "yaw_rmse", "absolute_mechanical_power",
      ),
  )
  command_seed_auc, command_summary = _group_auc(
      [row for row in commands if row["seed"] <= 5], "command",
  )
  joint_seed_auc, joint_summary = _group_auc(
      [row for row in joints if row["seed"] <= 5], "joint_type",
  )
  paired, inference = hodge_inference(seed_metrics)
  manifest = {
      "schema_version": 1,
      "campaign": "weak-actuator-campaign",
      "source_identity": source_identity,
      "protocol": {
          "primary_endpoint": PRIMARY_ENDPOINT,
          "secondary_endpoints": list(SECONDARY_ENDPOINTS),
          "hodge_interpretation": INTERPRETATION,
          "categorical_breakthrough_analysis": "exploratory_not_prespecified",
          "requested_steps": REQUESTED_STEPS,
          "actual_steps": ACTUAL_STEPS,
          "num_evals": NUM_EVALS,
          "evaluation_grid": EVALUATION_GRID,
          "rows_per_run": ROW_COUNT,
          "seed_is_experimental_unit": True,
      },
      "cardinalities": {
          "main_policies": 30,
          "main_rows": 294_300,
          "hodge_policies": 20,
          "hodge_rows": 196_200,
          "new_policies": 30,
          "unique_policies": 40,
          "unique_rows": 392_400,
          "new_actual_environment_steps": 12_386_304_000,
      },
      "retry_registry": retry_record,
      "runs": catalogs,
  }
  output_path.mkdir(parents=True)
  figures = output_path / "figures"
  figures.mkdir()
  _write_json(output_path / "manifest.json", manifest)
  _write_json(output_path / "run_catalog.json", catalogs)
  _write_csv(output_path / "run_catalog.csv", [
      {key: value for key, value in row.items() if key != "artifact_hashes"}
      for row in catalogs
  ])
  _write_json(output_path / "seed_metrics.json", seed_metrics)
  _write_csv(output_path / "seed_metrics.csv", seed_metrics)
  _write_json(output_path / "model_summary.json", model_summary)
  _write_csv(output_path / "model_summary.csv", model_summary)
  _write_csv(output_path / "seed_severity.csv", severity)
  _write_csv(output_path / "severity_summary.csv", severity_summary)
  _write_csv(output_path / "seed_command_summary.csv", commands)
  _write_csv(output_path / "command_auc_by_seed.csv", command_seed_auc)
  _write_csv(output_path / "command_summary.csv", command_summary)
  _write_csv(output_path / "seed_joint_summary.csv", joints)
  _write_csv(output_path / "joint_auc_by_seed.csv", joint_seed_auc)
  _write_csv(output_path / "joint_summary.csv", joint_summary)
  _write_csv(output_path / "seed_actuator_summary.csv", actuators)
  _write_csv(output_path / "collapse_thresholds.csv", [
      {
          "model": row["model"], "label": row["label"], "seed": row["seed"],
          "collapse_threshold": row["collapse_threshold"],
      }
      for row in seed_metrics
  ])
  _write_json(output_path / "hodge_paired_inference.json", inference)
  _write_csv(output_path / "hodge_paired_deltas.csv", paired)
  _write_csv(output_path / "hodge_inference.csv", inference)
  _write_csv(output_path / "learning_curves.csv", learning)
  _write_report(output_path, model_summary, inference)
  _write_latex_tables(output_path, model_summary, inference)
  _plot_style()
  _plot_primary(seed_metrics, model_summary, figures)
  _plot_severity(severity, figures)
  _plot_falls_tracking(severity, figures)
  _plot_group_heatmaps(command_summary, joint_summary, figures)
  _plot_learning(learning, figures)
  _plot_hodge_pairs(paired, figures)
  hashes = []
  for path in sorted(child for child in output_path.rglob("*") if child.is_file()):
    relative = path.relative_to(output_path).as_posix()
    hashes.append(f"{sha256_file(path)}  {relative}")
  (output_path / "SHA256SUMS").write_text("\n".join(hashes) + "\n")


def _machine_readable_files(root: Path) -> dict[str, str]:
  suffixes = {".csv", ".json", ".md", ".tex"}
  return {
      path.relative_to(root).as_posix(): sha256_file(path)
      for path in sorted(root.rglob("*"))
      if path.is_file() and path.suffix in suffixes
  }


def build_reproducible_paper_pack(
    artifact_root: str | Path,
    repository: str | Path,
    campaign_commit: str,
    output: str | Path,
    retry_registry: str | Path | None = None,
) -> None:
  destination = Path(output).resolve()
  if destination.exists():
    raise FileExistsError(f"paper output already exists: {destination}")
  with tempfile.TemporaryDirectory(prefix="go1-weak-actuator-") as temporary:
    temporary_root = Path(temporary)
    first = temporary_root / "first"
    second = temporary_root / "second"
    build_paper_pack(artifact_root, repository, campaign_commit, first, retry_registry)
    build_paper_pack(artifact_root, repository, campaign_commit, second, retry_registry)
    if _machine_readable_files(first) != _machine_readable_files(second):
      raise AssertionError("machine-readable paper outputs are not reproducible")
    shutil.copytree(first, destination)


def protocol_record(repository: str | Path, campaign_commit: str) -> dict[str, Any]:
  return {
      "source_identity": verify_source_identity(repository, campaign_commit),
      "baseline_tasks": [asdict(task) for task in baseline_tasks()],
      "hodge_extension_tasks": [asdict(task) for task in hodge_extension_tasks()],
      "expected_runs": [asdict(run) for run in expected_runs(campaign_commit)],
      "cardinalities": {
          "new_tasks": 30,
          "unique_policies": 40,
          "main_policies": 30,
          "hodge_policies": 20,
          "new_actual_environment_steps": 12_386_304_000,
      },
  }


def make_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)
  protocol = subparsers.add_parser("protocol", help="print frozen campaign JSON")
  protocol.add_argument("--repository", default=".")
  protocol.add_argument("--campaign-commit", required=True)
  generate = subparsers.add_parser("generate", help="validate and build paper pack")
  generate.add_argument("--artifact-root", required=True)
  generate.add_argument("--repository", default=".")
  generate.add_argument("--campaign-commit", required=True)
  generate.add_argument("--output")
  generate.add_argument("--retry-registry")
  generate.add_argument("--verify-reproducible", action="store_true")
  return parser


def main() -> None:
  args = make_parser().parse_args()
  if args.command == "protocol":
    print(json.dumps(
        protocol_record(args.repository, args.campaign_commit),
        indent=2, sort_keys=True,
    ))
    return
  output = args.output or str(
      Path(args.artifact_root).resolve() / args.campaign_commit / "paper"
  )
  builder = build_reproducible_paper_pack if args.verify_reproducible else build_paper_pack
  builder(
      args.artifact_root,
      args.repository,
      args.campaign_commit,
      output,
      args.retry_registry,
  )


if __name__ == "__main__":
  main()
