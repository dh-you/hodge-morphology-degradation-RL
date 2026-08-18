from __future__ import annotations

import hashlib
import json

import jax
import numpy as np


def array_sha256(value) -> str:
  array = np.asarray(value, dtype="<f4")
  return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def tree_signature(tree) -> tuple[int, int, str]:
  entries = []
  scalar_count = 0
  for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
    array = np.asarray(value)
    scalar_count += int(array.size)
    entries.append({
        "path": jax.tree_util.keystr(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    })
  entries.sort(key=lambda entry: entry["path"])
  encoded = json.dumps(
      entries, sort_keys=True, separators=(",", ":"),
  ).encode()
  return len(entries), scalar_count, hashlib.sha256(encoded).hexdigest()
