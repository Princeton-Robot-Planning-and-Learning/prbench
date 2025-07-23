"""Tests for clutteredstoragel2d.py."""

from conftest import MAKE_VIDEOS
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import prbench
from prbench.envs.clutteredstorage2d import (
    ObjectCentricClutteredStorage2DEnv,
    ShelfType,
    TargetBlockType,
)


def test_object_centric_clutteredstorage2d_env():
    """Tests for ObjectCentricClutteredStorage2DEnv()."""

    # Test env creation and random actions.
    env = ObjectCentricClutteredStorage2DEnv(num_blocks=1)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


def test_clutteredstorage2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/ClutteredStorage2D-b7-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_clutteredstorage2d_termination():
    """Tests that the environment terminates when all blocks are on the
    shelf."""
    env = ObjectCentricClutteredStorage2DEnv(num_blocks=3)
    state, _ = env.reset(seed=0)
    # Manually move the block into the shelf.
    shelf = state.get_objects(ShelfType)[0]
    blocks = state.get_objects(TargetBlockType)
    for block in blocks:
        # Move the block to the shelf.
        state.set(block, "x", state.get(shelf, "x"))
        state.set(block, "y", state.get(shelf, "y"))
        state.set(block, "theta", 0.0)

    env.reset(options={"init_state": state})
    # Any action should now result in termination.
    action = env.action_space.sample()
    state, reward, terminated, _, _ = env.step(action)
    assert reward == -1.0
    assert terminated
