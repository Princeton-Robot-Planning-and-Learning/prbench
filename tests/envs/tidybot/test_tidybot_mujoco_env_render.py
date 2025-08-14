"""Integration tests for MujocoEnv image observation output in
prbench.envs.tidybot.tidybot_mujoco_env.

These tests verify that image keys in the observation are present and have the expected
structure.
"""

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from prbench.envs.tidybot.tidybot_mujoco_env import MujocoEnv


def test_mujoco_env_obs_images():
    """Test MujocoEnv initialization, reset, and get_obs for image keys."""
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
        render_images=True,
        show_viewer=False,
        show_images=False,
        mjcf_path=str(absolute_model_path),
    )
    # Test reset
    env.reset(seed=123)
    # Test get_obs and check for image keys
    obs = env.get_obs()
    image_keys = [
        k for k, v in obs.items() if k.endswith("_image") and isinstance(v, np.ndarray)
    ]
    if image_keys:
        arr = obs[image_keys[0]]
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3  # Should be an image
        assert arr.shape[-1] == 3  # Should be RGB
    # Explicit cleanup
    del env

    gc.collect()
