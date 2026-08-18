from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.agents.ppo import networks as ppo_networks

from go1_core import CANONICAL_MODELS, MODEL_SPECS, load_checkpoint
from tests.helpers import array_sha256, tree_signature

pytestmark = pytest.mark.reference
FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = {
    "native_mlp": {
        "actor": (8, 192_408, "09f36e3da7227b35f65b62a8c051016cb07d3ea70382bc3bb2783d617c13eba6"),
        "full": (25, 420_765, "870c0a71c0b8dbcc27edd36174e5fc89baca19f9b8bb77d3d98f59777fb95681"),
        "logits": "795befd2feba6bc1e0e623c77e4c3e30c3f2cb4552028b10ce2a74ba6dfaaf54",
        "actions": "3ea66a2265d76b58e22065a20f9725c537561b56fd5117d25a5d5c4ae7b6f22f",
    },
    "mlp_match_lower": {
        "actor": (8, 93_162, "c0ab68491e8b10aa90b2d486e0d6dab17b6d3493ccd95e53732b405963f2d4ec"),
        "full": (25, 321_519, "522d4ce2f0e74844a5b7cf9b0038fe34bd2a0a168da7e0ccfce1d90f7222e117"),
        "logits": "1ead175496f19fe0714a20aad600db60d2a7c429c02c173849c81024d17dcfe3",
        "actions": "02fed7bf8226fe8169adcf44091caef8fc07f2875782cddeaaa7ecbe97e1948e",
    },
    "pointwise_match_lower": {
        "actor": (12, 93_244, "2a6d48b8c46e55893b02977636253032d5828ad04a3e42828cd60d1033fc8ace"),
        "full": (29, 321_601, "6ca9dc01b92c4aab0360a2efd2548140e8fffaa572411f2efd6443a707cfb0c7"),
        "logits": "38b13d631db6e39988ef2958f5aa4c3f7cd43c570bbeaa4dc4bd80d28a934d4d",
        "actions": "b9f25afdfec69fea69778ef68c49feec38d99c2861ac5359317703c8df3dc55b",
    },
    "gcn_match_lower": {
        "actor": (8, 92_987, "4f9ee45b773dd67d5815eefc55b5b3257b4ecadf0d014b6b9c17d08f7a2bb536"),
        "full": (25, 321_344, "3a377023c01b8cd0d311eaafd777ee977407098f566f6f070ecdd4b07df6f7ff"),
        "logits": "b3655d22abe8c31200814681c3c5a28fd663a1e80947c574fb908b7030bb9f38",
        "actions": "6a62a63efbca5f3389b5c4f8f978b74eb9bbeca209c537d3c4ec64039cf27791",
    },
    "multirank_hodge_lower": {
        "actor": (56, 92_962, "9bf4e874136e4fad66509706a0e838736a9e928b8bbad71608606358bea4450d"),
        "full": (73, 321_319, "dd3c985fd3d55b80a51b8c41880d25121470a7061ad0ef50dc38cd012dfe21be"),
        "logits": "10a246a2d3386d3d264d15abe387e43dc0f00511aa5268fee37d392f47a73baf",
        "actions": "7a4cce65ca72a186db24d63c4007e11ebea3f8ce088ebe8dba16dcd797bf03ed",
    },
    "multirank_hodge_full": {
        "actor": (80, 126_570, "375f9ee6132449eaebe213f8798551184eb186a656dad1716b9c266fb3aabe26"),
        "full": (97, 354_927, "cd6a4f737660c8ad5aa632ed330e51f4146c6058b3afa1148a0452ca7ad4b4e0"),
        "logits": "ee82903f88aeac4283d2b74b2aec128bff2b021762cc741d42537104f1a7d864",
        "actions": "2ae3ac36ea0f0f196b658bfa9033d663d59c06c4a4bd2da1e5d40e5556f12adc",
    },
}


@pytest.fixture(scope="module")
def artifact_root() -> Path:
  configured = os.environ.get("GO1_REFERENCE_ARTIFACT_ROOT")
  if not configured:
    pytest.skip("GO1_REFERENCE_ARTIFACT_ROOT is not set")
  root = Path(configured).resolve()
  if not root.is_dir():
    pytest.fail(f"GO1_REFERENCE_ARTIFACT_ROOT does not exist: {root}")
  return root


def checkpoint_path(root: Path, model: str) -> Path:
  return (
      root / "train" / "array-3343705" / model / "seed-0"
      / "checkpoints" / "000412876800"
  )


@pytest.mark.parametrize("model", CANONICAL_MODELS)
def test_canonical_checkpoint_parity(artifact_root, model):
  networks, params, config = load_checkpoint(
      checkpoint_path(artifact_root, model), expected_model=model,
  )
  kwargs = config.network_factory_kwargs.to_dict()
  assert kwargs == {
      "model_name": model,
      "policy_hidden_layer_sizes": [512, 256, 128],
      "policy_obs_key": "state",
      "value_hidden_layer_sizes": [512, 256, 128],
      "value_obs_key": "privileged_state",
  }
  assert config.action_size == 12
  assert config.normalize_observations is True
  observation_specs = config.observation_size.to_dict()
  assert observation_specs["state"]["shape"] == [48]
  assert observation_specs["privileged_state"]["shape"] == [123]

  assert tree_signature(params[1]) == EXPECTED[model]["actor"]
  assert tree_signature(params) == EXPECTED[model]["full"]
  assert tree_signature(params[1])[1] == MODEL_SPECS[model].actor_parameters
  assert all(np.isfinite(np.asarray(value)).all()
             for value in jax.tree_util.tree_leaves(params))

  probe = np.load(FIXTURES / "policy_probe_input.npy", allow_pickle=False)
  observation = {
      "state": jnp.asarray(probe),
      "privileged_state": jnp.zeros((3, 123), dtype=jnp.float32),
  }
  logits = np.asarray(
      networks.policy_network.apply(params[0], params[1], observation),
      dtype="<f4",
  )
  inference = ppo_networks.make_inference_fn(networks)
  actions, _ = inference(params, deterministic=True)(
      observation, jax.random.PRNGKey(0),
  )
  actions = np.asarray(actions, dtype="<f4")
  expected_logits = np.load(
      FIXTURES / f"{model}_logits.npy", allow_pickle=False,
  )
  expected_actions = np.load(
      FIXTURES / f"{model}_actions.npy", allow_pickle=False,
  )
  assert logits.shape == (3, 24) and actions.shape == (3, 12)
  assert np.isfinite(logits).all() and np.isfinite(actions).all()
  assert np.array_equal(logits, expected_logits)
  assert np.array_equal(actions, expected_actions)
  np.testing.assert_allclose(logits, expected_logits, rtol=1e-6, atol=1e-7)
  np.testing.assert_allclose(actions, expected_actions, rtol=1e-6, atol=1e-7)
  assert array_sha256(logits) == EXPECTED[model]["logits"]
  assert array_sha256(actions) == EXPECTED[model]["actions"]


def test_expected_model_mismatch_is_rejected(artifact_root):
  with pytest.raises(ValueError, match="checkpoint model"):
    load_checkpoint(
        checkpoint_path(artifact_root, "native_mlp"),
        expected_model="gcn_match_lower",
    )
