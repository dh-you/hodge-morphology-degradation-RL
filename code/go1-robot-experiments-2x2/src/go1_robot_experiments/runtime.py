"""Runtime identity and device information for immutable experiment runs."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Any

import jax
import numpy as np

from go1_robot_experiments.constants import CORE_COMMIT


def _git(repository: Path, *args: str) -> str:
  return subprocess.check_output(
      ("git", "-C", str(repository), *args), text=True,
  ).strip()


def experiment_identity(repository: str | Path | None = None) -> dict[str, Any]:
  """Returns the clean repository and exact code identities for a run."""
  root = Path.cwd().resolve() if repository is None else Path(repository).resolve()
  try:
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
  except subprocess.CalledProcessError as error:
    raise RuntimeError("execution must originate in the experiment Git repo") from error
  head = _git(top, "rev-parse", "HEAD")
  if re.fullmatch(r"[0-9a-f]{40}", head) is None:
    raise RuntimeError("experiment HEAD is not a full lowercase SHA")
  dirty = _git(top, "status", "--porcelain")
  if dirty:
    raise RuntimeError(f"experiment worktree must be clean:\n{dirty}")
  return {
      "repository": str(top),
      "experiment_commit": head,
      "core_commit": CORE_COMMIT,
  }


def runtime_information() -> dict[str, Any]:
  device = jax.devices()[0]
  try:
    memory_stats = device.memory_stats() or {}
  except Exception:
    memory_stats = {}
  return {
      "backend": jax.default_backend(),
      "device": str(device),
      "memory_stats": {
          str(key): int(value) if isinstance(value, (int, np.integer)) else value
          for key, value in memory_stats.items()
      },
  }
