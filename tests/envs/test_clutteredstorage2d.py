"""Tests for clutteredstoragel2d.py."""

from conftest import MAKE_VIDEOS
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import prbench
from prbench.envs.clutteredstorage2d import ObjectCentricClutteredStorage2DEnv


def test_object_centric_clutteredstoragel2d_env():
    """Tests for ObjectCentricClutteredStorage2DEnv()."""

    # Test env creation and random actions.
    env = ObjectCentricClutteredStorage2DEnv(num_blocks=9)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


# def test_clutteredretrieval2d_observation_space():
#     """Tests that observations are vectors with fixed dimensionality."""
#     prbench.register_all_environments()
#     env = prbench.make("prbench/ClutteredRetrieval2D-o10-v0")
#     assert isinstance(env.observation_space, Box)
#     for _ in range(5):
#         obs, _ = env.reset()
#         assert env.observation_space.contains(obs)
