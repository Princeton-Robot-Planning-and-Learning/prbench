"""Integration tests for the MujocoEnv class in prbench.envs.tidybot.tidybot_mujoco_env.

These tests verify environment initialization, reset, and the structure of observations.
"""

import os
import sys
from pathlib import Path

import pytest

from prbench.envs.tidybot.tidybot_mujoco_env import MujocoEnv


def test_mujoco_env_init_and_obs():
    """Test MujocoEnv initialization, reset, and get_obs with a minimal MJCF file."""
    # Detect headless mode (no DISPLAY) and set OSMesa if needed
    if not os.environ.get("DISPLAY"):
        if sys.platform == "darwin":
            os.environ["MUJOCO_GL"] = "glfw"
            os.environ["PYOPENGL_PLATFORM"] = "glfw"
        else:
            os.environ["MUJOCO_GL"] = "osmesa"
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    # Construct the MJCF path relative to this test file
    model_base_path = (
        Path(__file__).parent
        / ".."
        / ".."
        / "src"
        / "prbench"
        / "envs"
        / "tidybot"
        / "models"
        / "stanford_tidybot"
    )
    model_base_path = model_base_path.resolve()
    model_file = "scene.xml"
    absolute_model_path = model_base_path / model_file
    if not absolute_model_path.exists():
        pytest.skip(f"MJCF file not found: {absolute_model_path}")

    # Try to initialize MujocoEnv
    env = MujocoEnv(
        render_images=False,
        show_viewer=False,
        show_images=False,
        mjcf_path=str(absolute_model_path),
    )
    # Test reset
    env.reset(seed=123)
    # Test get_obs
    obs = env.get_obs()
    assert isinstance(obs, dict)
    assert "base_pose" in obs
    assert "arm_pos" in obs
    assert "arm_quat" in obs
    assert "gripper_pos" in obs
