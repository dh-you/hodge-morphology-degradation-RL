"""Merge and validate the two immutable held-out evaluation shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from go1_robot_experiments.constants import COMMANDS, RESET_KEYS
from go1_robot_experiments.runtime import experiment_identity
from go1_robot_experiments.util import write_csv, write_json
from experiments.double_actuator.evaluate import ACTUAL_STEPS, METRICS
from experiments.double_actuator.protocol import (
    HELD_OUT_SPECS,
    MODELS,
    PAIR_MANIFEST_SHA256,
    STRENGTHS,
    held_out_shard,
)

EXPECTED_SHARD_ROWS = 11 * len(STRENGTHS) ** 2 * len(COMMANDS) * len(RESET_KEYS)
EXPECTED_ROWS = 2 * EXPECTED_SHARD_ROWS


def _read_json(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def _row_key(row: dict[str, str]) -> tuple[int, int, int, int, int]:
  return (
      tuple(COMMANDS).index(row["command"]),
      RESET_KEYS.index(int(row["reset_key"])),
      int(row["pair_id"]),
      STRENGTHS.index(float(row["strength_a"])),
      STRENGTHS.index(float(row["strength_b"])),
  )


def _read_shard_rows(path: Path, model: str, seed: int, shard: int) -> list[dict[str, str]]:
  allowed_pairs = {pair.pair_id for pair in held_out_shard(shard)}
  rows: list[dict[str, str]] = []
  previous = None
  with path.open(newline="") as handle:
    for row in csv.DictReader(handle):
      if row["model"] != model or int(row["seed"]) != seed:
        raise ValueError(f"row identity mismatch: {path}")
      if int(row["pair_id"]) not in allowed_pairs:
        raise ValueError(f"row belongs to the wrong shard: {path}")
      key = _row_key(row)
      if previous is not None and key <= previous:
        raise ValueError(f"noncanonical or duplicate shard row order: {path}")
      previous = key
      for metric in METRICS:
        if not math.isfinite(float(row[metric])):
          raise ValueError(f"nonfinite {metric}: {path}")
      rows.append(row)
  if len(rows) != EXPECTED_SHARD_ROWS:
    raise ValueError(f"shard row count {len(rows)} != {EXPECTED_SHARD_ROWS}: {path}")
  return rows


def merge_shards(
    training: Path,
    model: str,
    seed: int,
    training_commit: str,
) -> dict[str, Any]:
  identity = experiment_identity()
  output = training / "heldout"
  metrics_output = output / "condition_metrics.csv"
  record_output = output / "evaluation.json"
  if metrics_output.exists() or record_output.exists():
    raise FileExistsError(f"immutable merged held-out evaluation exists: {output}")

  all_rows: list[dict[str, str]] = []
  evidence = []
  evaluator_commit = None
  seen_keys = set()
  for shard in (0, 1):
    shard_root = output / f"shard-{shard}"
    record_path = shard_root / "evaluation.json"
    metrics_path = shard_root / "condition_metrics.csv"
    record = _read_json(record_path)
    expected_pair_ids = [pair.pair_id for pair in held_out_shard(shard)]
    checks = {
        "status": record.get("status") == "COMPLETE",
        "model": record.get("model") == model,
        "seed": record.get("seed") == seed,
        "shard": record.get("shard") == shard,
        "training_commit": record.get("training_experiment_commit") == training_commit,
        "steps": record.get("checkpoint_step") == ACTUAL_STEPS,
        "manifest": record.get("pair_manifest_sha256") == PAIR_MANIFEST_SHA256,
        "pairs": record.get("held_out_pair_ids") == expected_pair_ids,
        "pair_count": record.get("pair_count") == 11,
        "conditions": record.get("condition_count") == 11 * len(STRENGTHS) ** 2,
        "rows": record.get("row_count") == EXPECTED_SHARD_ROWS,
    }
    if not all(checks.values()):
      raise ValueError(f"shard artifact validation failed {shard}: {checks}")
    if evaluator_commit is None:
      evaluator_commit = record.get("experiment_commit")
    elif record.get("experiment_commit") != evaluator_commit:
      raise ValueError("shards used different evaluator commits")
    rows = _read_shard_rows(metrics_path, model, seed, shard)
    for row in rows:
      key = _row_key(row)
      if key in seen_keys:
        raise ValueError(f"duplicate row across shards: {key}")
      seen_keys.add(key)
    all_rows.extend(rows)
    evidence.append({
        "shard": shard,
        "evaluation_slurm_job_id": record.get("slurm_job_id"),
        "evaluation_json_sha256": _sha256(record_path),
        "condition_metrics_sha256": _sha256(metrics_path),
        "row_count": len(rows),
    })

  if evaluator_commit != identity["experiment_commit"]:
    raise ValueError("merge and shard evaluator commits differ")
  expected_pairs = {pair.pair_id for pair in HELD_OUT_SPECS}
  observed_pairs = {int(row["pair_id"]) for row in all_rows}
  if observed_pairs != expected_pairs:
    raise ValueError("merged held-out pair set is incomplete or incorrect")
  if len(all_rows) != EXPECTED_ROWS or len(seen_keys) != EXPECTED_ROWS:
    raise ValueError("merged row cardinality is incomplete or duplicated")
  all_rows.sort(key=_row_key)
  write_csv(metrics_output, all_rows)
  record = {
      "status": "COMPLETE",
      **identity,
      "training_experiment_commit": training_commit,
      "model": model,
      "seed": seed,
      "checkpoint_step": ACTUAL_STEPS,
      "split": "held_out",
      "shards": [0, 1],
      "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
      "held_out_pair_ids": [pair.pair_id for pair in HELD_OUT_SPECS],
      "pair_count": len(HELD_OUT_SPECS),
      "strengths": list(STRENGTHS),
      "condition_count": len(HELD_OUT_SPECS) * len(STRENGTHS) ** 2,
      "row_count": len(all_rows),
      "commands": list(COMMANDS),
      "reset_keys": list(RESET_KEYS),
      "row_order": ["command", "reset_key", "pair_id", "strength_a", "strength_b"],
      "continuous_metrics_finite": True,
      "shard_evidence": evidence,
      "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
  }
  write_json(record_output, record)
  return record


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--training-root", required=True)
  parser.add_argument("--model", choices=MODELS, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--training-commit", required=True)
  args = parser.parse_args()
  merge_shards(
      Path(args.training_root).resolve(), args.model, args.seed,
      args.training_commit,
  )


if __name__ == "__main__":
  main()
