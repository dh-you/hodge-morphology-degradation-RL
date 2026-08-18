"""Validate all five bounded healthy-only GPU smoke artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.double_actuator.healthy_protocol import HEALTHY_PROTOCOL_SHA256
from experiments.double_actuator.protocol import MODELS

SMOKE_STEPS = 22_937_600


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--artifact-root", required=True)
  parser.add_argument("--implementation-commit", required=True)
  args = parser.parse_args()
  root = Path(args.artifact_root).resolve()
  failures = []
  for model in MODELS:
    run_root = (
        root / args.implementation_commit / "double-actuator-healthy" / model
        / "seed-10" / "steps-22937600" / "evals-2"
    )
    try:
      with (run_root / "run.json").open() as handle:
        run = json.load(handle)
      with (run_root / "result.json").open() as handle:
        result = json.load(handle)
      with (run_root / "cross_regime_randomization.json").open() as handle:
        cross = json.load(handle)
      checks = {
          "status": result.get("status") == "COMPLETE",
          "commit": run.get("experiment_commit") == args.implementation_commit,
          "model": run.get("model") == model == result.get("model"),
          "seed": run.get("seed") == 10 == result.get("seed"),
          "steps": run.get("actual_steps") == SMOKE_STEPS == result.get("actual_steps"),
          "protocol": run.get("healthy_protocol_sha256") == HEALTHY_PROTOCOL_SHA256,
          "cross_regime": cross.get("status") == "EXACT",
          "checkpoint": result.get("checkpoint") == {
              "path": f"checkpoints/{SMOKE_STEPS:012d}",
              "exact_parameter_roundtrip": True,
              "exact_logits_roundtrip": True,
              "exact_actions_roundtrip": True,
          },
      }
      if not all(checks.values()):
        failures.append({"model": model, "checks": checks})
    except Exception as error:
      failures.append({"model": model, "error": str(error)})
  if failures:
    raise SystemExit(json.dumps({"status": "FAILED", "failures": failures}, sort_keys=True))
  print(json.dumps({"status": "COMPLETE", "models": list(MODELS)}, sort_keys=True))


if __name__ == "__main__":
  main()
