"""Historical Brax PPO checkpoint reconstruction for canonical Go1 models."""

from __future__ import annotations

from pathlib import Path

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from go1_core.ppo import make_go1_ppo_networks
from go1_core.registry import MODEL_SPECS


def load_checkpoint(
    path: str | Path, expected_model: str | None = None,
):
  """Returns ``(networks, params, config)`` for a historical checkpoint."""
  checkpoint_dir = Path(path).resolve()
  if not checkpoint_dir.is_dir():
    raise ValueError(f"checkpoint path does not exist: {checkpoint_dir}")
  config = ppo_checkpoint.load_config(checkpoint_dir)
  kwargs = config.network_factory_kwargs.to_dict()
  model = kwargs.get("model_name")
  if expected_model is not None and model != expected_model:
    raise ValueError(
        f"checkpoint model {model!r} != requested {expected_model!r}"
    )
  if model not in MODEL_SPECS:
    raise ValueError(f"checkpoint contains unknown Go1 model {model!r}")
  preprocess = (
      running_statistics.normalize
      if config.normalize_observations
      else types.identity_observation_preprocessor
  )
  networks = make_go1_ppo_networks(
      config.observation_size.to_dict(),
      config.action_size,
      preprocess_observations_fn=preprocess,
      **kwargs,
  )
  return networks, ppo_checkpoint.load(checkpoint_dir), config
