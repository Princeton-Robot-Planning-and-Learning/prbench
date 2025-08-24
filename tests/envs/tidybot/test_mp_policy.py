"""Unit tests for MotionPlannerPolicy helpers and safe operation."""

import numpy as np

from prbench.envs.tidybot.agent.mp_policy import MotionPlannerPolicy


def minimal_obs():
    """Create a minimal valid observation dict for the policy tests."""
    return {
        "base_pose": np.array([0.0, 0.0, 0.0], dtype=float),
        "arm_pos": np.array([0.0, 0.0, 0.0], dtype=float),
        "arm_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        "gripper_pos": np.array([0.0], dtype=float),
    }


def test_distance_and_heading_helpers():
    """Validate distance computation and heading normalization stay in [-pi, pi]."""
    policy = MotionPlannerPolicy()
    # Distance
    d = policy.distance((0.0, 0.0), (3.0, 4.0))
    assert abs(d - 5.0) < 1e-6
    # Heading wrap
    assert -np.pi <= policy.restrict_heading_range(10.0) <= np.pi


def test_get_end_effector_offset_default():
    """Ensure the default end-effector offset value is returned when unset."""
    policy = MotionPlannerPolicy()
    assert policy.get_end_effector_offset() == 0.55


def test_build_base_command_move():
    """The 'move' primitive should return a base-only command with tolerances."""
    policy = MotionPlannerPolicy()
    cmd = {"primitive_name": "move", "waypoints": [[0.0, 0.0], [1.0, 0.0]]}
    res = policy.build_base_command(cmd)
    assert res is not None
    assert res["target_ee_pos"] is None
    assert res["position_tolerance"] == 0.1


def test_step_returns_none_when_disabled():
    """Policy returns None when disabled, without running the state machine."""
    policy = MotionPlannerPolicy()
    policy.enabled = False
    obs = minimal_obs()
    assert policy.step(obs) is None
