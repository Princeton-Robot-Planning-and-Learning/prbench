"""Tests for obstruction2d.py."""

from gymnasium.spaces import Box

import prbench


def test_obstruction2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o2-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
