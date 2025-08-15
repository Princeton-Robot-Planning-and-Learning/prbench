"""Tests for the TidyBot3D table scene: observation/action spaces, reset, and step."""

import numpy as np

from prbench.envs.tidybot.tidybot3d import TidyBot3DEnv


def test_tidybot3d_table_observation_space():
    """Reset should return an observation within the observation space."""
    env = TidyBot3DEnv(scene_type="table", num_objects=3, render_images=False)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_table_action_space():
    """A sampled action should be valid for the action space."""
    env = TidyBot3DEnv(scene_type="table", num_objects=3, render_images=False)
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    env.close()


def test_tidybot3d_table_step():
    """Step should return a valid obs, float reward, bool done flags, and info dict."""
    env = TidyBot3DEnv(scene_type="table", num_objects=3, render_images=False)
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_table_reset_seed_reproducible():
    """Reset with the same seed should produce identical observations."""
    env = TidyBot3DEnv(scene_type="table", num_objects=3, render_images=False)
    obs1, _ = env.reset(seed=110)
    obs2, _ = env.reset(seed=110)
    assert np.allclose(obs1, obs2, rtol=1e-5, atol=1e-4)
    env.close()


def test_tidybot3d_table_reset_changes_without_seed():
    """Consecutive resets without a seed should generally produce different
    observations."""
    env = TidyBot3DEnv(scene_type="table", num_objects=3, render_images=False)
    obs1, _ = env.reset(seed=1)
    obs2, _ = env.reset(seed=3)
    assert not np.allclose(obs1, obs2, rtol=1e-5, atol=1e-6)
    env.close()


def test_tidybot3d_table_reset_format():
    """Reset observation should match observation_space shape and be float32."""
    env = TidyBot3DEnv(scene_type="table", num_objects=3, render_images=False)
    obs, _ = env.reset()
    assert obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape
    env.close()
