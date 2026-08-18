from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import go1_core

PACKAGE = Path(go1_core.__file__).parent
FORBIDDEN_TOP_LEVEL = {
    "claim1",
    "clean_models",
    "robustness",
    "mujoco_playground",
    "train_clean_models",
    "train_clean_model_capacity",
    "unified_clean_models",
}
FORBIDDEN_RUNTIME_SYMBOLS = {
    "A_HAT",
    "L_LOWER_HAT",
    "L_UPPER_HAT",
    "RHO_A",
    "RHO_L_LOWER",
    "RHO_L_UPPER",
    "HodgeEncoder",
    "PairwiseEncoder",
}


def test_runtime_source_has_no_old_repository_imports_or_legacy_operators():
  for path in PACKAGE.rglob("*.py"):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        names = {alias.name.split(".")[0] for alias in node.names}
        assert names.isdisjoint(FORBIDDEN_TOP_LEVEL), (path, names)
      elif isinstance(node, ast.ImportFrom) and node.module:
        assert node.module.split(".")[0] not in FORBIDDEN_TOP_LEVEL, (
            path, node.module,
        )
    assert FORBIDDEN_RUNTIME_SYMBOLS.isdisjoint(set(source.split())), path
    for symbol in FORBIDDEN_RUNTIME_SYMBOLS:
      assert symbol not in source, (path, symbol)


def test_public_exports_are_exact():
  assert go1_core.__all__ == (
      "CANONICAL_MODELS",
      "GO1_CONTRACT_V1",
      "MODEL_SPECS",
      "Go1ContractV1",
      "ModelSpec",
      "assert_compatible_environment",
      "get_model_spec",
      "load_checkpoint",
      "make_go1_ppo_networks",
  )
  assert "create_model" not in dir(go1_core)
  assert "rhmp" not in dir(go1_core)


def test_clean_subprocess_import_does_not_load_old_packages(tmp_path):
  code = """
import json
import sys
import go1_core
forbidden = {
    'claim1', 'clean_models', 'robustness', 'mujoco_playground',
    'train_clean_models', 'train_clean_model_capacity', 'unified_clean_models',
}
loaded = sorted(forbidden.intersection({name.split('.')[0] for name in sys.modules}))
print(json.dumps(loaded))
raise SystemExit(bool(loaded))
"""
  env = os.environ.copy()
  env["PYTHONPATH"] = str(PACKAGE.parent)
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run(
      [sys.executable, "-c", code],
      cwd=tmp_path,
      env=env,
      text=True,
      capture_output=True,
      check=False,
  )
  assert result.returncode == 0, result.stderr
  lines = result.stdout.splitlines()
  assert lines, "subprocess emitted no JSON sentinel"
  assert json.loads(lines[-1]) == []
