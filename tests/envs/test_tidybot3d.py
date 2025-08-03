"""Unit tests for the TidyBot3D environment and its inverse kinematics solver.

This module tests:
- The validity of the observation space
- The validity of the action space
- The basic functionality of the IKSolver
- The performance and accuracy of the IKSolver for a known pose
"""

import numpy as np

from prbench.envs.ik_solver import IKSolver
from prbench.envs.tidybot3d import TidyBot3DEnv


def test_tidybot3d_observation_space():
    """Test that the observation returned by TidyBot3DEnv.reset() is within the
    observation space."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    obs = env.reset()[0]
    # Check observation is within the observation space
    assert env.observation_space.contains(obs), "Observation not in observation space"
    env.close()


def test_tidybot3d_action_space():
    """Test that a sampled action is within the TidyBot3DEnv action space."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    action = env.action_space.sample()
    # Check action is within the action space
    assert env.action_space.contains(action), "Action not in action space"
    env.close()


def test_ik_solver_basic():
    """Test that the IKSolver returns a valid joint configuration for a simple
    target pose."""
    ik = IKSolver(ee_offset=0.12)
    # Use the retract configuration as a test target
    target_pos = ik.site_pos.copy()
    target_quat = np.array([0, 0, 0, 1])  # Identity quaternion (x, y, z, w)
    curr_qpos = ik.qpos0.copy()
    result_qpos = ik.solve(target_pos, target_quat, curr_qpos)
    # Should return a joint configuration of correct shape
    assert result_qpos.shape == curr_qpos.shape
    # Should not produce NaNs
    assert np.all(np.isfinite(result_qpos))


def test_ik_solver_performance_and_accuracy():
    """Test the performance and accuracy of the IKSolver for a known home pose
    over 1000 iterations."""
    ik_solver = IKSolver()
    home_pos = np.array([0.456, 0.0, 0.434])
    home_quat = np.array([0.5, 0.5, 0.5, 0.5])
    retract_qpos = np.deg2rad([0, -20, 180, -146, 0, -50, 90])

    for _ in range(1000):
        qpos = ik_solver.solve(home_pos, home_quat, retract_qpos)

    # Check that the output is finite and has correct shape
    assert qpos.shape == retract_qpos.shape
    assert np.all(np.isfinite(qpos))

    # Check that the solution is close to the expected home configuration
    expected_home_deg = np.array([0, 15, 180, -130, 0, 55, 90])
    qpos_deg = np.rad2deg(qpos)
    assert np.allclose(
        qpos_deg, expected_home_deg, atol=5
    ), f"IK solution deviates from expected: {qpos_deg} vs {expected_home_deg}"
