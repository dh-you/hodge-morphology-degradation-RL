from __future__ import annotations

import importlib.metadata

from go1_robot_experiments.constants import (
    CORE_COMMIT,
    MENAGERIE_COMMIT,
    PLAYGROUND_COMMIT,
)
from go1_robot_experiments.dependencies import verify_runtime_dependencies


def test_exact_runtime_versions():
  assert {
      name: importlib.metadata.version(name)
      for name in (
          "jax", "jaxlib", "flax", "brax", "numpy",
          "orbax-checkpoint", "pytest",
      )
  } == {
      "jax": "0.11.0",
      "jaxlib": "0.11.0",
      "flax": "0.12.8",
      "brax": "0.14.2",
      "numpy": "2.5.1",
      "orbax-checkpoint": "0.12.1",
      "pytest": "9.1.1",
  }


def test_direct_urls_source_guards_and_menagerie_link():
  result = verify_runtime_dependencies()
  assert result["core"]["commit"] == CORE_COMMIT
  assert result["playground"]["commit"] == PLAYGROUND_COMMIT
  assert result["menagerie"]["commit"] == MENAGERIE_COMMIT
  assert result["playground"]["clean"] and result["menagerie"]["clean"]
