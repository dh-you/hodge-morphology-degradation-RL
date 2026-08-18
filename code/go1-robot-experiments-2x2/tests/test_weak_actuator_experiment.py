from __future__ import annotations

import functools
import hashlib

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco_playground import registry

from go1_core import CANONICAL_MODELS, MODEL_SPECS, make_go1_ppo_networks
from go1_robot_experiments.constants import ENV_NAME
from go1_robot_experiments.protocol import assert_environment_contract

from experiments.weak_actuator.protocol import (
    CATEGORY_DEAD,
    CATEGORY_HEALTHY,
    CATEGORY_WEAK,
    CURRICULUM,
    CURRICULUM_SHA256,
    DEFAULT_EVALUATION_GRID,
    DEFAULT_MODEL,
    DEFAULT_NUM_EVALS,
    DEFAULT_NUM_TIMESTEPS,
    InjurySamples,
    OFFICIAL_PPO_CONFIG_SHA256,
    ORIGINAL_FULL_ACTUAL_TIMESTEPS,
    ORIGINAL_FULL_NUM_EVALS,
    ORIGINAL_FULL_NUM_TIMESTEPS,
    apply_injury,
    checkpoint_observation_size,
    config_sha256,
    parameter_count,
    resolved_ppo_config,
    sample_injuries,
    split_training_keys,
    validate_output_path,
    weak_actuator_randomize,
)
from experiments.weak_actuator.train import (
    _serialized_factory_kwargs,
    evaluation_axes,
    make_parser,
)


def test_curriculum_matches_original_weak_actuator_protocol():
  assert config_sha256(CURRICULUM) == CURRICULUM_SHA256
  assert CURRICULUM["category_probabilities"] == {
      "healthy": 0.30, "weak": 0.50, "dead": 0.20,
  }
  assert CURRICULUM["weak_strength_distribution"] == {
      "family": "log_uniform", "minimum": 0.05,
      "maximum_exclusive": 0.20,
  }


def test_prng_categories_strengths_and_force_range_mutation():
  rng = jax.random.split(jax.random.PRNGKey(11), 4096)
  keys = split_training_keys(rng)
  expected = jax.vmap(lambda key: jax.random.split(key, 4))(rng)
  for index, value in enumerate(keys):
    np.testing.assert_array_equal(value, expected[:, index])
  samples = sample_injuries(keys)
  assert set(np.asarray(samples.category).tolist()) == {
      CATEGORY_HEALTHY, CATEGORY_WEAK, CATEGORY_DEAD,
  }
  sampled = np.asarray(samples.sampled_weak_strength)
  assert np.all(sampled >= 0.05) and np.all(sampled < 0.20)
  effective = np.asarray(samples.effective_strength)
  category = np.asarray(samples.category)
  assert np.all(effective[category == CATEGORY_HEALTHY] == 1.0)
  assert np.all(effective[category == CATEGORY_DEAD] == 0.0)
  np.testing.assert_array_equal(
      effective[category == CATEGORY_WEAK], sampled[category == CATEGORY_WEAK],
  )

  force = jnp.asarray([[-float(i + 1), float(i + 1)] for i in range(12)])
  fixed = InjurySamples(
      category=jnp.asarray([CATEGORY_HEALTHY, CATEGORY_WEAK, CATEGORY_DEAD]),
      actuator_index=jnp.asarray([0, 3, 11]),
      sampled_weak_strength=jnp.asarray([0.1, 0.125, 0.1]),
      effective_strength=jnp.asarray([1.0, 0.125, 0.0]),
  )
  result = np.asarray(apply_injury(force, fixed))
  expected_force = np.broadcast_to(np.asarray(force), result.shape).copy()
  expected_force[1, 3] *= 0.125
  expected_force[2, 11] *= 0.0
  np.testing.assert_array_equal(result, expected_force)


def test_sampler_raw_hashes_match_frozen_pre_split_source():
  samples = sample_injuries(split_training_keys(
      jax.random.split(jax.random.PRNGKey(2026), 4096)
  ))
  expected = {
      "category": (
          "a0deb0f5029d445ae7fd5ddcdbd8a3f704431b6ec18cab861a8c7d0ee8c2ca56"
      ),
      "actuator_index": (
          "e7ab5561f2733e0c7c2fe6dfdaa2730007150d56f724dd7a89c555f03d296e7a"
      ),
      "sampled_weak_strength": (
          "78d81b154f7c64f40826ceed3274177a632643bf3fb22284cc64f51a17f43e0a"
      ),
      "effective_strength": (
          "839c7ea9d2a98148cf7711bfc4f52d1f7ce61973b1b1d6403d358a1ee3f8051e"
      ),
  }
  observed = {
      name: hashlib.sha256(
          np.ascontiguousarray(np.asarray(getattr(samples, name))).tobytes()
      ).hexdigest()
      for name in expected
  }
  assert observed == expected


def test_config_defaults_step_accounting_and_parser():
  with pytest.raises(ValueError, match="initial and terminal"):
    resolved_ppo_config(123, 0, 1)
  official, effective, accounting = resolved_ppo_config(
      DEFAULT_NUM_TIMESTEPS, 0, DEFAULT_NUM_EVALS,
  )
  assert config_sha256(official) == OFFICIAL_PPO_CONFIG_SHA256
  assert effective["num_evals"] == 2
  assert effective["seed"] == 0
  assert accounting["requested_environment_timesteps"] == 22_937_600
  assert accounting["actual_environment_timesteps"] == 22_937_600
  assert accounting["fresh_rollout_batches"] == 140
  assert accounting["optimizer_steps"] == 17_920
  args = make_parser().parse_args([])
  assert args.model == DEFAULT_MODEL
  assert args.seed == 0
  assert args.num_timesteps == DEFAULT_NUM_TIMESTEPS
  assert args.num_evals == DEFAULT_NUM_EVALS
  assert args.evaluation_grid == DEFAULT_EVALUATION_GRID
  assert tuple(CANONICAL_MODELS) == tuple(
      make_parser()._option_string_actions["--model"].choices
  )

  unused_official, original, original_accounting = resolved_ppo_config(
      ORIGINAL_FULL_NUM_TIMESTEPS, 0, ORIGINAL_FULL_NUM_EVALS,
  )
  assert original["num_evals"] == 19
  assert original_accounting == {
      "requested_environment_timesteps": 400_000_000,
      "actual_environment_timesteps": ORIGINAL_FULL_ACTUAL_TIMESTEPS,
      "rollout_steps": 163_840,
      "eval_intervals": 18,
      "resets_per_interval": 10,
      "steps_per_epoch": 14,
      "fresh_rollout_batches": 2_520,
      "optimizer_steps": 322_560,
  }


def test_original_full_evaluation_axes_and_smoke_subset():
  commands, resets = evaluation_axes("full")
  assert tuple(name for name, unused_value in commands) == (
      "stand", "forward", "backward", "left_lateral", "right_lateral",
      "left_yaw", "right_yaw", "forward_left_yaw", "forward_right_yaw",
  )
  assert resets == tuple(range(10))
  assert len(commands) * len(resets) * 109 == 9_810
  assert evaluation_axes("smoke") == ((commands[0],), (0,))


def test_output_path_is_commit_bound_and_non_overwriting(tmp_path):
  sha = "a" * 40
  expected = (
      tmp_path / sha / "weak-actuator" / DEFAULT_MODEL / "seed-3"
      / "steps-123" / "evals-2" / "grid-full"
  )
  assert validate_output_path(
      tmp_path, sha, DEFAULT_MODEL, 3, 123, 2, "full",
  ) == expected
  expected.mkdir(parents=True)
  with pytest.raises(FileExistsError):
    validate_output_path(tmp_path, sha, DEFAULT_MODEL, 3, 123, 2, "full")
  repository = tmp_path / "repo"
  with pytest.raises(ValueError, match="outside the repository"):
    validate_output_path(
        repository / "outputs", sha, DEFAULT_MODEL, 4, 123, 2, "full",
        repository=repository,
    )


def test_cpu_environment_network_randomizer_and_checkpoint_contract():
  assert jax.default_backend() == "cpu"
  env = registry.load(
      ENV_NAME,
      config=registry.get_default_config(ENV_NAME),
      config_overrides={"impl": "jax"},
  )
  assert_environment_contract(env)
  for model in CANONICAL_MODELS:
    networks = make_go1_ppo_networks(
        env.observation_size, env.action_size, model_name=model,
    )
    assert parameter_count(
        networks.policy_network.init(jax.random.PRNGKey(0))
    ) == MODEL_SPECS[model].actor_parameters
  network_factory = functools.partial(
      make_go1_ppo_networks,
      model_name=DEFAULT_MODEL,
      policy_hidden_layer_sizes=(512, 256, 128),
      value_hidden_layer_sizes=(512, 256, 128),
      policy_obs_key="state",
      value_obs_key="privileged_state",
  )
  checkpoint_config = ppo_checkpoint.network_config(
      checkpoint_observation_size(env.observation_size),
      env.action_size,
      True,
      network_factory,
  )
  assert _serialized_factory_kwargs(
      checkpoint_config, DEFAULT_MODEL,
  )["model_name"] == DEFAULT_MODEL
  randomized, in_axes = weak_actuator_randomize(
      env.mjx_model, jax.random.split(jax.random.PRNGKey(0), 4),
  )
  assert randomized.actuator_forcerange.shape == (4, 12, 2)
  assert in_axes.actuator_forcerange == 0
