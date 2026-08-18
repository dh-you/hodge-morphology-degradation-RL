"""Small deterministic serialization and hashing helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np


def json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_safe(item) for item in value]
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, jax.Array):
    return np.asarray(value).tolist()
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  return repr(value)


def write_json(path: str | Path, value: Any) -> None:
  path = Path(path)
  with path.open("x") as handle:
    json.dump(json_safe(value), handle, indent=2, sort_keys=True)
    handle.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
  path = Path(path)
  if not rows:
    raise ValueError(f"cannot write empty CSV: {path}")
  with path.open("x", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
