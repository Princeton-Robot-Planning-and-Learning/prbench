"""Base interfaces and utilities for TidyBot agents.

This module defines `BaseAgent`, an abstract class that provides common
functionality for TidyBot policies, including IK initialization, pose utilities,
geometry helpers, and WebXR coordinate conversion. Concrete agents should
inherit from `BaseAgent` and implement `reset` and `step`.
"""

import math
from abc import ABC
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore

from prbench.envs.tidybot.ik_solver import TidybotIKSolver


class BaseAgent(ABC):
    """Abstract base class for all agents/policies."""

    def __init__(self) -> None:
        # Common utilities
        self.ik_solver: Optional[TidybotIKSolver] = None
        self.initialized: bool = False

    def _initialize_ik_solver(self, ee_offset: float = 0.12) -> None:
        """Initialize IK solver with end effector offset."""
        if self.ik_solver is None:
            self.ik_solver = TidybotIKSolver(ee_offset=ee_offset)

    def normalize_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """Ensure quaternion uniqueness (w >= 0)"""
        if quat[3] < 0.0:  # Assuming (x, y, z, w) format
            return -quat
        return quat

    def quat_to_rotation(self, quat: np.ndarray) -> R:
        """Convert quaternion to scipy Rotation object."""
        # Assuming input is (x, y, z, w), scipy expects (x, y, z, w)
        return R.from_quat(quat)

    def rotation_to_quat(self, rotation: R) -> np.ndarray:
        """Convert scipy Rotation to quaternion (x, y, z, w)"""
        return rotation.as_quat()

    def distance_2d(self, pt1: np.ndarray, pt2: np.ndarray) -> float:
        """Calculate 2D distance between two points."""
        return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    def distance_3d(self, pt1: np.ndarray, pt2: np.ndarray) -> float:
        """Calculate 3D distance between two points."""
        return float(np.linalg.norm(pt2 - pt1))

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π] range."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def solve_ik(
        self, target_pos: np.ndarray, target_quat: np.ndarray, current_qpos: np.ndarray
    ) -> np.ndarray:
        """Solve inverse kinematics for target pose."""
        if self.ik_solver is None:
            self._initialize_ik_solver()
        assert self.ik_solver is not None
        return self.ik_solver.solve(target_pos, target_quat, current_qpos)

    def transform_to_base_frame(
        self, world_pos: np.ndarray, base_pose: np.ndarray
    ) -> np.ndarray:
        """Transform world position to base local frame."""
        # Extract base position and orientation
        base_xy = base_pose[:2]
        base_theta = base_pose[2]

        # Compute relative position
        dx = world_pos[0] - base_xy[0]
        dy = world_pos[1] - base_xy[1]

        # Rotate to base frame
        cos_angle = math.cos(-base_theta)  # Negative for inverse rotation
        sin_angle = math.sin(-base_theta)

        local_pos = np.array(
            [
                cos_angle * dx - sin_angle * dy,
                sin_angle * dx + cos_angle * dy,
                world_pos[2] if len(world_pos) > 2 else 0.0,
            ]
        )

        return local_pos

    def check_position_reached(
        self, current_pos: np.ndarray, target_pos: np.ndarray, tolerance: float = 0.01
    ) -> bool:
        """Check if position is reached within tolerance."""
        return np.allclose(current_pos, target_pos, atol=tolerance)

    def check_orientation_reached(
        self,
        current_quat: np.ndarray,
        target_quat: np.ndarray,
        angle_tolerance: float = 0.1,
    ) -> bool:
        """Check if orientation is reached within tolerance (radians)"""
        dot_product = np.clip(np.dot(current_quat, target_quat), -1.0, 1.0)
        angle_diff = 2 * np.arccos(np.abs(dot_product))
        return angle_diff < angle_tolerance

    def interpolate_pose(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        current_quat: np.ndarray,
        target_quat: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate between current and target pose."""
        # Linear interpolation for position
        interp_pos = (1 - alpha) * current_pos + alpha * target_pos

        # SLERP for quaternion
        current_rot = self.quat_to_rotation(current_quat)
        target_rot = self.quat_to_rotation(target_quat)

        # Spherical linear interpolation
        interp_rot = current_rot * (current_rot.inv() * target_rot) ** alpha
        interp_quat = self.rotation_to_quat(interp_rot)

        return interp_pos, interp_quat

    def create_action(
        self,
        base_pose: Optional[np.ndarray] = None,
        arm_pos: Optional[np.ndarray] = None,
        arm_quat: Optional[np.ndarray] = None,
        gripper_pos: Optional[Union[float, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Create action dictionary with optional components."""
        action: Dict[str, Any] = {}

        if base_pose is not None:
            action["base_pose"] = np.array(base_pose)

        if arm_pos is not None:
            action["arm_pos"] = np.array(arm_pos)

        if arm_quat is not None:
            action["arm_quat"] = np.array(arm_quat)

        if gripper_pos is not None:
            if isinstance(gripper_pos, (int, float)):
                action["gripper_pos"] = np.array([gripper_pos])
            else:
                action["gripper_pos"] = np.array(gripper_pos)

        return action

    def hold_current_pose(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Create action to hold current pose."""
        return self.create_action(
            base_pose=obs["base_pose"].copy(),
            arm_pos=obs["arm_pos"].copy(),
            arm_quat=obs["arm_quat"].copy(),
            gripper_pos=obs["gripper_pos"].copy(),
        )

    # WebXR coordinate conversion utilities (for teleop policies)

    @staticmethod
    def convert_webxr_pose(
        pos: Dict[str, float],
        quat: Dict[str, float],
        device_camera_offset: np.ndarray = np.array([0.0, 0.02, -0.04]),
    ) -> Tuple[np.ndarray, R]:
        """Convert WebXR coordinate system to robot coordinate system."""
        # WebXR: +x right, +y up, +z back; Robot: +x forward, +y left, +z up
        pos_array = np.array([-pos["z"], -pos["x"], pos["y"]], dtype=np.float64)
        rot = R.from_quat([-quat["z"], -quat["x"], quat["y"], quat["w"]])

        # Apply offset for device center instead of camera
        pos_array = pos_array + rot.apply(device_camera_offset)

        return pos_array, rot

    # Path planning utilities

    def line_circle_intersection(
        self, d: tuple, f: tuple, r: float, use_t1: bool = False
    ) -> Optional[float]:
        """
        Find intersection of line with circle for path planning
        Based on: https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm # pylint: disable=line-too-long
        """
        a = d[0] * d[0] + d[1] * d[1]  # dot(d, d)
        b = 2 * (f[0] * d[0] + f[1] * d[1])  # 2 * dot(f, d)
        c = (f[0] * f[0] + f[1] * f[1]) - r * r  # dot(f, f) - r^2

        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            if use_t1:
                t1 = (-b - math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t1 <= 1:
                    return t1
            else:
                t2 = (-b + math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t2 <= 1:
                    return t2

        return None

    def dot(self, a: tuple, b: tuple) -> float:
        """Dot product helper function from controller.py."""
        return a[0] * b[0] + a[1] * b[1]

    def intersect(
        self, d: tuple, f: tuple, r: float, use_t1: bool = False
    ) -> Optional[float]:
        """Line-circle intersection from controller.py."""
        # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm/1084899%231084899
        a = self.dot(d, d)
        b = 2 * self.dot(f, d)
        c = self.dot(f, f) - r * r
        discriminant = (b * b) - (4 * a * c)
        if discriminant >= 0:
            if use_t1:
                t1 = (-b - math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t1 <= 1:
                    return t1
            else:
                t2 = (-b + math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t2 <= 1:
                    return t2
        return None

    def restrict_heading_range(self, h: float) -> float:
        """Normalize heading to [-π, π] range from controller.py."""
        return (h + math.pi) % (2 * math.pi) - math.pi
