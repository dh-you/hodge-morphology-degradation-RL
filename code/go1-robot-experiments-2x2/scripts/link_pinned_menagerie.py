#!/usr/bin/env python3
"""Link an otherwise absent installed Menagerie to the pinned checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import mujoco_playground


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument(
      "--replace-installed-copy",
      action="store_true",
      help="replace only the environment-local copied asset directory",
  )
  args = parser.parse_args()
  source = args.source.resolve()
  target = (
      Path(mujoco_playground.__file__).resolve().parent
      / "external_deps/mujoco_menagerie"
  )
  if target.is_symlink():
    if target.resolve() != source:
      raise RuntimeError(f"existing symlink targets {target.resolve()}, not {source}")
    return
  if target.exists():
    if not args.replace_installed_copy:
      raise RuntimeError(
          "installed Menagerie is a copied directory; rerun with "
          f"--replace-installed-copy to replace only {target}"
      )
    if target.name != "mujoco_menagerie" or target.parent.name != "external_deps":
      raise RuntimeError(f"refusing unexpected replacement target: {target}")
    shutil.rmtree(target)
  target.parent.mkdir(parents=True, exist_ok=True)
  target.symlink_to(source, target_is_directory=True)


if __name__ == "__main__":
  main()
