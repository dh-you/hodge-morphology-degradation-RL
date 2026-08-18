from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/go1_robot_experiments"


def test_package_imports_no_pre_split_research_modules():
  forbidden = {
      "claim1", "clean_models", "robustness", "unified_clean_models",
      "train_clean_models", "train_clean_model_capacity",
  }
  observed = set()
  for path in PACKAGE.glob("*.py"):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        observed.update(alias.name.split(".")[0] for alias in node.names)
      elif isinstance(node, ast.ImportFrom) and node.module:
        observed.add(node.module.split(".")[0])
  assert observed.isdisjoint(forbidden)


def test_task_package_contains_no_model_or_training_implementation():
  source = "\n".join(path.read_text() for path in PACKAGE.glob("*.py"))
  for forbidden in (
      "build_joint_features", "ppo.train", "checkpoint.save",
      "class HodgeEncoder", "class PairwiseEncoder", "class MultiRankEncoder",
  ):
    assert forbidden not in source
  assert "B1 =" not in source and "B2 =" not in source


def test_package_import_is_lazy():
  result = subprocess.run(
      [sys.executable, "-c", "import go1_robot_experiments,sys; print(sorted(m for m in sys.modules if m.startswith('go1_robot_experiments')))"],
      cwd="/tmp", text=True, capture_output=True, check=True,
  )
  assert result.stdout.strip() == "['go1_robot_experiments']"


def test_built_wheel_contains_only_scientific_support_package():
  wheels = list((ROOT / "dist").glob("*.whl"))
  if not wheels:
    return
  with zipfile.ZipFile(wheels[-1]) as archive:
    project_packages = {
        name.split("/", 1)[0]
        for name in archive.namelist()
        if "/" in name and not name.split("/", 1)[0].endswith(".dist-info")
    }
  assert project_packages == {"go1_robot_experiments"}
