"""Tests for clutter2d.py."""

from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from prbench.envs.clutter2d import ObjectCentricClutter2DEnv


def test_object_centric_clutter2d_env():
    """Tests for ObjectCentricClutter2DEnv()."""

    # Test env creation and random actions.
    env = ObjectCentricClutter2DEnv(num_obstructions=5)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()
