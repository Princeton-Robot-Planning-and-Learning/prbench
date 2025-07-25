"""Tests for motion2d.py."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import prbench
from prbench.envs.motion2d import Motion2DEnvSpec, ObjectCentricMotion2DEnv

prbench.register_all_environments()


def test_object_centric_motion2d_env():
    """Tests for ObjectCentricMotion2DEnv()."""

    # Test env creation and random actions.
    env = ObjectCentricMotion2DEnv(num_passages=5)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


def test_motion2d_stay_in_bounds():
    """Tests that the robot stays in bounds of the world."""
    env = prbench.make("prbench/Motion2D-p1-v0")

    # Cardinal directions.
    # The action space is (dx, dy, dtheta, darm, vacuum).
    up = (0.0, env.action_space.high[1], 0.0, 0.0, 0.0)
    down = (0.0, env.action_space.low[1], 0.0, 0.0, 0.0)
    left = (env.action_space.low[0], 0.0, 0.0, 0.0, 0.0)
    right = (env.action_space.high[0], 0.0, 0.0, 0.0, 0.0)
    directions = {"up": up, "down": down, "left": left, "right": right}

    # Get world bounds from the default spec
    default_spec = Motion2DEnvSpec()
    world_min_x, world_max_x = default_spec.world_min_x, default_spec.world_max_x
    world_min_y, world_max_y = default_spec.world_min_y, default_spec.world_max_y

    for _, direction in directions.items():
        obs, _ = env.reset(seed=123)
        for _ in range(100):
            obs, _, _, _, _ = env.step(np.array(direction, dtype=np.float32))
            # Robot position is in the first elements of the observation vector
            # (x, y, theta are the first 3 elements since robot is first object)
            robot_x = obs[0]
            robot_y = obs[1]
            assert world_min_x <= robot_x <= world_max_x
            assert world_min_y <= robot_y <= world_max_y

    env.close()


def test_motion2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    env = prbench.make("prbench/Motion2D-p5-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
