"""Tests for dyn_obstruction2d.py."""

# import imageio.v2 as iio
import numpy as np
from gymnasium.spaces import Box

import prbench


def test_dyn_obstruction2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o2-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_dyn_obstruction2d_action_space():
    """Tests that the actions are valid and the step function works."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o3-v0")
    _obs, _ = env.reset(seed=0)
    statble_move = np.array([0.05, 0.05, np.pi / 16, 0.05, -0.02], dtype=np.float32)
    # Check the control precision
    # zeros = np.zeros_like(obs)
    # zeros[0] += statble_move[0]
    # zeros[1] += statble_move[1]
    # zeros[2] += statble_move[2]
    # desired_obs_next = obs + statble_move
    _img = env.render()
    # iio.imwrite("unit_test_videos/init.png", img)
    for _ in range(10):
        _obs, _reward, _terminated, _truncated, _info = env.step(statble_move)
        # img = env.render()
        # iio.imwrite(f"unit_test_videos/step_{i}.png", img)


def test_dyn_obstruction2d_different_obstruction_counts():
    """Tests that different numbers of obstructions work."""
    prbench.register_all_environments()

    for num_obs in [0, 1, 2, 3, 4]:
        env = prbench.make(f"prbench/DynObstruction2D-o{num_obs}-v0")
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

        # Take a few steps to ensure environment works
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, _truncated, _info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (int, float))
            if terminated:
                break


def test_dyn_obstruction2d_reset_consistency():
    """Tests that reset produces consistent observations."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o2-v0")

    # Test multiple resets
    for _ in range(3):
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

        # Environment should not be terminated at start
        action = env.action_space.sample()
        _obs, reward, _terminated, _truncated, _info = env.step(action)
        # First step should give -1 reward (goal not satisfied immediately)
        assert reward == -1.0
