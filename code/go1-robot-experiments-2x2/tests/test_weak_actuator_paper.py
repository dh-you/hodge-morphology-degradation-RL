from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from go1_robot_experiments.constants import (
    COMMANDS,
    CONDITIONS,
    HORIZON,
    MODEL_TIERS,
    RESET_KEYS,
    SUMMARY_METRICS,
)
from experiments.weak_actuator.paper import (
    ACTUAL_STEPS,
    AUC_ENDPOINTS,
    BASELINE_MODELS,
    HODGE_MODELS,
    INTERPRETATION,
    MODEL_LABELS,
    MODEL_ORDER,
    REUSED_HODGE_COMMIT,
    SECONDARY_ENDPOINTS,
    STRENGTHS_ASCENDING,
    baseline_tasks,
    collapse_threshold,
    exact_sign_flip_pvalue,
    expected_runs,
    hodge_extension_tasks,
    hodge_inference,
    holm_adjust,
    normalized_auc,
    scientific_file_hashes,
    summarize_run,
    t_interval,
    validate_condition_rows,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _synthetic_rows(model: str = "native_mlp") -> list[dict]:
  rows = []
  for command, command_value in COMMANDS.items():
    for reset_key in RESET_KEYS:
      for condition in CONDITIONS:
        strength = float(condition.strength)
        healthy = condition.actuator_index is None
        return_value = 20.0 if healthy else 10.0 + 10.0 * strength
        metrics = {
            "velocity_rmse": 1.0 - 0.2 * strength,
            "yaw_rmse": 0.8 - 0.1 * strength,
            "absolute_mechanical_power": 4.0,
            "torque_rms": 2.0,
            "action_delta_rms": 0.2,
            "survival": 1.0,
            "fall": 0.0,
            "undiscounted_return": return_value,
            "actuator_force_utilization": 0.3,
            "saturation_fraction": 0.0,
            "damaged_actuator_force_utilization": None if healthy else 0.2,
            "damaged_joint_tracking_rmse": None if healthy else 0.1,
        }
        assert set(metrics) == set(SUMMARY_METRICS)
        rows.append({
            "model": model,
            "model_tier": MODEL_TIERS[model],
            "checkpoint_step": ACTUAL_STEPS,
            "condition": condition.condition,
            "actuator_index": condition.actuator_index,
            "actuator_name": condition.actuator_name,
            "strength": strength,
            "command": command,
            "command_value": list(command_value),
            "reset_key": reset_key,
            **metrics,
            "was_terminated": False,
            "termination_step": None,
            "active_steps": HORIZON,
        })
  return rows


def test_campaign_roster_and_cardinalities_are_frozen():
  baselines = baseline_tasks()
  extension = hodge_extension_tasks()
  assert len(baselines) == 20
  assert [(task.task_id, task.model, task.seed) for task in baselines] == [
      (index, model, seed)
      for index, (model, seed) in enumerate(
          (model, seed) for model in BASELINE_MODELS for seed in range(1, 6)
      )
  ]
  assert len(extension) == 10
  assert [(task.task_id, task.model, task.seed) for task in extension] == [
      (index, model, seed)
      for index, (model, seed) in enumerate(
          (model, seed) for model in HODGE_MODELS for seed in range(6, 11)
      )
  ]
  runs = expected_runs("a" * 40)
  assert len(runs) == 40
  assert sum("main" in run.studies for run in runs) == 30
  assert sum("hodge" in run.studies for run in runs) == 20
  assert {
      run.experiment_commit
      for run in runs if run.model in HODGE_MODELS and run.seed <= 5
  } == {REUSED_HODGE_COMMIT}


@pytest.mark.parametrize(
    ("launcher", "tasks"),
    (
        ("jobs/weak_actuator_baselines.slurm", baseline_tasks()),
        ("jobs/weak_actuator_hodge_extension.slurm", hodge_extension_tasks()),
    ),
)
def test_slurm_dry_run_matches_frozen_task_mapping(launcher, tasks):
  for task in tasks:
    environment = {
        **os.environ,
        "DRY_RUN": "1",
        "SLURM_ARRAY_TASK_ID": str(task.task_id),
    }
    completed = subprocess.run(
        ("bash", str(REPOSITORY / launcher)),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    record = json.loads(completed.stdout)
    assert (record["campaign"], record["task_id"], record["model"], record["seed"]) == (
        task.campaign, task.task_id, task.model, task.seed,
    )
    assert record["requested_steps"] == 400_000_000
    assert record["actual_steps"] == 412_876_800
    assert record["num_evals"] == 19
    assert record["evaluation_grid"] == "full"


def test_frozen_scientific_hashes_match_reused_hodge_commit():
  hashes = scientific_file_hashes(REPOSITORY, REUSED_HODGE_COMMIT)
  assert len(hashes) == 7
  for relative, expected in hashes.items():
    assert scientific_file_hashes(REPOSITORY, REUSED_HODGE_COMMIT)[relative] == expected


def test_full_row_grid_validation_rejects_order_and_nonfinite_values():
  rows = _synthetic_rows()
  validate_condition_rows(rows, "native_mlp")
  swapped = list(rows)
  swapped[1], swapped[2] = swapped[2], swapped[1]
  with pytest.raises(ValueError, match="misordered"):
    validate_condition_rows(swapped, "native_mlp")
  nonfinite = _synthetic_rows()
  nonfinite[1]["velocity_rmse"] = float("nan")
  with pytest.raises(ValueError, match="invalid velocity_rmse"):
    validate_condition_rows(nonfinite, "native_mlp")


def test_macro_average_auc_and_collapse_rule_are_frozen():
  rows = _synthetic_rows()
  result = {
      "terminal_metrics": {
          "eval/episode_reward": 7.0,
          "eval/avg_episode_length": 800.0,
      },
  }
  summary = summarize_run("native_mlp", 1, rows, result)
  severity = summary["severity"]
  assert len(severity) == 9
  assert summary["seed_metrics"]["absolute_return_auc"] == pytest.approx(12.5)
  assert summary["seed_metrics"]["return_retention_auc"] == pytest.approx(0.625)
  assert summary["seed_metrics"]["survival_auc"] == 1.0
  assert summary["seed_metrics"]["collapse_threshold"] is None

  passing = [
      {
          "strength": strength,
          "return_retention": 0.95,
          "survival": 0.98,
          "fall": 0.02,
      }
      for strength in STRENGTHS_ASCENDING
  ]
  assert collapse_threshold(passing) == 0.0
  next(row for row in passing if row["strength"] == 0.1)["survival"] = 0.94
  assert collapse_threshold(passing) == 0.15


def test_auc_excludes_healthy_and_requires_the_exact_grid():
  rows = [
      {"strength": strength, "metric": 2.0 * strength + 1.0}
      for strength in STRENGTHS_ASCENDING
  ]
  assert normalized_auc(rows, "metric") == pytest.approx(1.5)
  with pytest.raises(ValueError, match="grid mismatch"):
    normalized_auc(rows[:-1], "metric")


def test_paired_statistics_and_holm_are_exact():
  interval = t_interval([1.0, 2.0, 3.0, 4.0])
  assert interval["mean"] == 2.5
  assert interval["sample_sd"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
  assert interval["ci_low"] < 2.5 < interval["ci_high"]
  assert exact_sign_flip_pvalue([1.0] * 10) == 2 / 1024
  assert holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04}) == {
      "a": 0.03, "b": 0.06, "c": 0.06,
  }


def test_hodge_inference_uses_all_ten_pairs_and_one_secondary_family():
  rows = []
  for model, shift in ((HODGE_MODELS[0], 0.0), (HODGE_MODELS[1], 1.0)):
    for seed in range(1, 11):
      rows.append({
          "model": model,
          "seed": seed,
          **{metric: float(seed) + shift for metric in AUC_ENDPOINTS},
      })
  paired, inference = hodge_inference(rows)
  assert len(paired) == 10 * len(AUC_ENDPOINTS)
  assert len([row for row in paired if row["endpoint"] == "absolute_return_auc"]) == 10
  primary = next(row for row in inference if row["role"] == "primary")
  assert primary["mean_full_minus_lower"] == 1.0
  assert primary["exact_two_sided_sign_flip_p"] == 2 / 1024
  secondaries = [row for row in inference if row["role"] == "secondary"]
  assert tuple(row["endpoint"] for row in secondaries) == SECONDARY_ENDPOINTS
  assert all(row["holm_adjusted_p"] is not None for row in secondaries)


def test_labels_and_statistical_interpretation_are_frozen():
  assert tuple(MODEL_LABELS) == MODEL_ORDER
  assert MODEL_LABELS["native_mlp"] == "Native MLP"
  assert "formal paired inference uses all seeds 1-10" in INTERPRETATION
  assert "categorical breakthrough criterion" in INTERPRETATION
  assert "exploratory" in INTERPRETATION
