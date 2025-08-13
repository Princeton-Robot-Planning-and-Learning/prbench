"""Integration tests for the MujocoSim class in prbench.envs.tidybot.tidybot_mujoco_env.

These tests verify simulation initialization, reset, and the presence of controller
attributes.
"""

import gc

# Create dummy shared memory state and command queue
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

from prbench.envs.tidybot.tidybot_mujoco_env import TidybotMujocoSim


def test_mujoco_sim_init_and_reset():
    """Test TidybotMujocoSim initialization and reset with a minimal MJCF file."""
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

    command_queue = mp.Queue(1)
    shm_state = None  # Let TidybotMujocoSim create its own state if needed

    # Try to initialize TidybotMujocoSim
    sim = TidybotMujocoSim(
        mjcf_path=str(absolute_model_path),
        command_queue=command_queue,
        shm_state=shm_state,
        show_viewer=False,
        seed=42,
    )

    try:
        # Test reset
        sim.reset(seed=123)
        # Test that base_controller and arm_controller exist
        assert hasattr(sim, "base_controller")
        assert hasattr(sim, "arm_controller")
        # Test that get attributes for base and arm positions
        assert isinstance(sim.qpos_base, np.ndarray)
        assert isinstance(sim.qpos_objects, list)
    finally:
        # Explicit cleanup to avoid nanobind/ruckig leaks
        if hasattr(sim, "base_controller"):
            del sim.base_controller
        if hasattr(sim, "arm_controller"):
            del sim.arm_controller
        del sim

        gc.collect()
