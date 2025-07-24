"""Tests for stickbutton2d.py."""

from conftest import MAKE_VIDEOS
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

from geom2drobotenvs.object_types import CircleType

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


def test_stickbutton2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/StickButton2D-b5-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_stickbutton2d_termination():
    """Tests that the environment terminates when all buttons are pressed."""

    env = ObjectCentricStickButton2DEnv(num_buttons=5)
    env.reset()

    # Manually press all buttons.
    buttons = env._current_state.get_objects(CircleType)
    for button in buttons:
        env._current_state.set(button, "color_r", env._spec.button_pressed_rgb[0])
        env._current_state.set(button, "color_g", env._spec.button_pressed_rgb[1])
        env._current_state.set(button, "color_b", env._spec.button_pressed_rgb[2])

    # Any action should now result in termination.
    action = env.action_space.sample()
    state, reward, terminated, _, _ = env.step(action)
    assert reward == -1.0
    assert terminated
