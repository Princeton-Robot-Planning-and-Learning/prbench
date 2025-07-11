"""Tests for stickbutton2d.py."""

from conftest import MAKE_VIDEOS
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import prbench
from prbench.envs.stickbutton2d import ObjectCentricStickButton2DEnv


def test_object_centric_stickbutton2d_env():
    """Tests for ObjectCentricMotion2DEnv()."""

    # Test env creation and random actions.
    env = ObjectCentricStickButton2DEnv(num_buttons=5)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


# def test_motion2d_observation_space():
#     """Tests that observations are vectors with fixed dimensionality."""
#     prbench.register_all_environments()
#     env = prbench.make("prbench/Motion2D-p5-v0")
#     assert isinstance(env.observation_space, Box)
#     for _ in range(5):
#         obs, _ = env.reset()
#         assert env.observation_space.contains(obs)
