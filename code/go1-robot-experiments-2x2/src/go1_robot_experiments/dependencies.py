"""Runtime provenance guards for the pinned core and simulator sources."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import mujoco_playground

from go1_robot_experiments.constants import (
    CORE_COMMIT,
    CORE_REPOSITORY_URL,
    MENAGERIE_COMMIT,
    PLAYGROUND_COMMIT,
)


DEFAULT_PLAYGROUND_SOURCE = Path("/scratch/network/dy0130/upstream/mujoco_playground")
DEFAULT_MENAGERIE_SOURCE = (
    DEFAULT_PLAYGROUND_SOURCE
    / "mujoco_playground/external_deps/mujoco_menagerie"
)


def _git(repo: Path, *args: str) -> str:
  return subprocess.check_output(
      ("git", "-C", str(repo), *args), text=True,
  ).strip()


def _source_guard(path: Path, commit: str, name: str) -> dict[str, Any]:
  if not path.is_dir():
    raise RuntimeError(f"pinned {name} checkout is missing: {path}")
  head = _git(path, "rev-parse", "HEAD")
  if head != commit:
    raise RuntimeError(f"{name} checkout {head} != pinned {commit}")
  dirty = _git(path, "status", "--porcelain")
  if dirty:
    raise RuntimeError(f"pinned {name} checkout is dirty:\n{dirty}")
  return {"path": str(path), "commit": head, "clean": True}


def _direct_url(distribution: str) -> dict[str, Any]:
  dist = importlib.metadata.distribution(distribution)
  text = dist.read_text("direct_url.json")
  if text is None:
    raise RuntimeError(f"{distribution} has no direct_url.json")
  return json.loads(text)


def _commit_from_direct_url(value: dict[str, Any]) -> str | None:
  return value.get("vcs_info", {}).get("commit_id")


def verify_runtime_dependencies() -> dict[str, Any]:
  core_url = _direct_url("go1-morphology-core")
  playground_url = _direct_url("playground")
  if core_url.get("url") != CORE_REPOSITORY_URL:
    raise RuntimeError(
        "installed go1-morphology-core source is not the private release"
    )
  if _commit_from_direct_url(core_url) != CORE_COMMIT:
    raise RuntimeError(
        "installed go1-morphology-core commit differs from the campaign pin"
    )
  if _commit_from_direct_url(playground_url) != PLAYGROUND_COMMIT:
    raise RuntimeError("installed Playground commit differs from the task pin")
  playground_source = Path(
      os.environ.get("GO1_PLAYGROUND_SOURCE", str(DEFAULT_PLAYGROUND_SOURCE))
  ).resolve()
  menagerie_source = Path(
      os.environ.get("GO1_MENAGERIE_SOURCE", str(DEFAULT_MENAGERIE_SOURCE))
  ).resolve()
  playground = _source_guard(
      playground_source, PLAYGROUND_COMMIT, "MuJoCo Playground",
  )
  menagerie = _source_guard(
      menagerie_source, MENAGERIE_COMMIT, "MuJoCo Menagerie",
  )
  installed_menagerie = (
      Path(mujoco_playground.__file__).resolve().parent
      / "external_deps/mujoco_menagerie"
  )
  if not installed_menagerie.is_symlink():
    raise RuntimeError("installed Menagerie must be the pinned source symlink")
  if installed_menagerie.resolve() != menagerie_source:
    raise RuntimeError("installed Menagerie symlink targets the wrong checkout")
  return {
      "core": {
          "commit": CORE_COMMIT,
          "direct_url": core_url,
      },
      "playground": {**playground, "direct_url": playground_url},
      "menagerie": {**menagerie, "installed_link": str(installed_menagerie)},
  }
