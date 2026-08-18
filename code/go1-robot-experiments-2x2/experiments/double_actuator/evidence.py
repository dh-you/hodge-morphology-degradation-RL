"""Training-artifact validation shared by A, B, and C evaluators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.double_actuator.healthy_protocol import HEALTHY_PROTOCOL_SHA256
from experiments.double_actuator.protocol import (
    CURRICULUM_SHA256,
    PAIR_MANIFEST_SHA256,
)

ACTUAL_STEPS = 412_876_800
DAMAGE_TRAINING_COMMIT = "9a125fd245874b1d3be50238f3cdddc510832ce0"


def read_json(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def validate_training(
    training: Path,
    model: str,
    seed: int,
    training_commit: str,
    training_regime: str,
    evaluator_identity: dict[str, Any],
) -> dict[str, Any]:
  if training_regime not in {"healthy_only", "damage_curriculum"}:
    raise ValueError(f"invalid training regime: {training_regime}")
  run_path = training / "run.json"
  result_path = training / "result.json"
  run = read_json(run_path)
  result = read_json(result_path)
  expected_checkpoint = f"checkpoints/{ACTUAL_STEPS:012d}"
  checks = {
      "status": result.get("status") == "COMPLETE",
      "model": run.get("model") == model == result.get("model"),
      "seed": run.get("seed") == seed == result.get("seed"),
      "steps": run.get("actual_steps") == ACTUAL_STEPS == result.get("actual_steps"),
      "checkpoint": run.get("terminal_checkpoint") == expected_checkpoint,
      "checkpoint_exists": (training / expected_checkpoint).is_dir(),
      "training_commit": (
          run.get("experiment_commit") == training_commit
          == result.get("experiment_commit")
      ),
      "core_commit": run.get("core_commit") == evaluator_identity["core_commit"],
      "roundtrip": result.get("checkpoint") == {
          "path": expected_checkpoint,
          "exact_parameter_roundtrip": True,
          "exact_logits_roundtrip": True,
          "exact_actions_roundtrip": True,
      },
  }
  if training_regime == "damage_curriculum":
    checks.update({
        "frozen_damage_commit": training_commit == DAMAGE_TRAINING_COMMIT,
        "pair_manifest": (
            run.get("pair_manifest_sha256") == PAIR_MANIFEST_SHA256
        ),
        "curriculum": run.get("curriculum_sha256") == CURRICULUM_SHA256,
    })
    protocol_path = training / "curriculum.json"
  else:
    cross_path = training / "cross_regime_randomization.json"
    cross = read_json(cross_path) if cross_path.is_file() else {}
    checks.update({
        "regime": run.get("training_regime") == "healthy_only",
        "healthy_protocol": (
            run.get("healthy_protocol_sha256") == HEALTHY_PROTOCOL_SHA256
            == result.get("healthy_protocol_sha256")
        ),
        "no_damage_curriculum_file": not (training / "curriculum.json").exists(),
        "no_pair_manifest_file": not (training / "pair_manifest.json").exists(),
        "cross_regime_randomization": cross.get("status") == "EXACT",
        "reference_damage_commit": (
            cross.get("damage_training_commit") == DAMAGE_TRAINING_COMMIT
        ),
    })
    protocol_path = training / "healthy_protocol.json"
  checks["protocol_file"] = protocol_path.is_file()
  if not all(checks.values()):
    raise ValueError(f"training artifact validation failed: {checks}")
  evidence = {
      "training_root": str(training),
      "training_regime": training_regime,
      "training_commit": training_commit,
      "run_json_sha256": file_sha256(run_path),
      "result_json_sha256": file_sha256(result_path),
      "protocol_json_sha256": file_sha256(protocol_path),
      "checks": checks,
  }
  assignment_manifest = training / "assignment_manifest.json"
  assignments = training / "assignments.npz"
  if not assignment_manifest.is_file() or not assignments.is_file():
    raise FileNotFoundError("training assignment evidence is incomplete")
  evidence.update({
      "assignment_manifest_sha256": file_sha256(assignment_manifest),
      "assignments_npz_sha256": file_sha256(assignments),
  })
  return {"run": run, "result": result, "evidence": evidence}
