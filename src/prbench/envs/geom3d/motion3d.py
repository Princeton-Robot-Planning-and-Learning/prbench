"""Environment where only 3D motion planning is needed to reach a goal
region."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from prpl_utils.spaces import FunctionalSpace
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot


@dataclass(frozen=True)
class Motion3DEnvSpec:
    """Spec for Motion3DEnv()."""

    # Robot.
    robot_name: str = "kinova-gen3"
    robot_base_pose: Pose = Pose.identity()
    # NOTE: the robot joints include 7 DOF for the arm and 6 DOF for the fingers. We
    # don't need to change the fingers this in the environment.
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            -4.3,  # "joint_1", starting at the robot base and going up to the gripper
            -1.6,  # "joint_2"
            -4.8,  # "joint_3"
            -1.8,  # "joint_4"
            -1.4,  # "joint_5"
            -1.1,  # "joint_6"
            1.6,  # "joint_7"
            # Finger joints (not used in this environment).
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    end_effector_viz_radius: float = 0.01
    end_effector_viz_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)
    max_action_mag: float = 0.05

    # Target.
    target_radius: float = 0.1
    target_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)
    target_lower_bound: tuple[float, float, float] = (0.0, 0.1, 0.0)
    target_upper_bound: tuple[float, float, float] = (0.5, 0.8, 0.5)

    # For rendering.
    render_dpi: int = 300
    render_fps: int = 20

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": self.robot_base_pose.position,
            "camera_yaw": 90,
            "camera_distance": 1.5,
            "camera_pitch": -20,
            # Use for fast testing.
            # "image_width": 32,
            # "image_height": 32,
        }


@dataclass(frozen=True)
class Motion3DState:
    """A state for Motion3DEnv()."""

    joint_positions: JointPositions
    target: tuple[float, float, float]  # 3D position to reach with end effector


@dataclass(frozen=True)
class Motion3DAction:
    """An action for Motion3DEnv().

    NOTE: the environment enforces a limit on the magnitude of the deltas.
    """

    # NOTE: this is only a delta on the 7 DOF of the arm, not the fingers, which do not
    # need to change in this environment.
    delta_arm_joints: JointPositions


class Motion3DEnv(gymnasium.Env[Motion3DState, Motion3DAction]):
    """Environment where only 3D motion planning is needed to reach a goal
    region."""

    # Only RGB rendering is implemented.
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        spec: Motion3DEnvSpec = Motion3DEnvSpec(),
        render_mode: str | None = None,
        use_gui: bool = False,
    ) -> None:
        super().__init__()
        self._spec = spec

        # Set up Gymnasium env fields.
        obs_md = self._create_observation_space_markdown_description()
        act_md = self._create_action_space_markdown_description()
        env_md = self._create_env_markdown_description()
        reward_md = self._create_reward_markdown_description()
        references_md = self._create_references_markdown_description()
        # Update the metadata. Note that we need to define the render_modes in the class
        # rather than in the instance because gym.make() extracts render_modes from cls.
        self.metadata = self.metadata.copy()
        self.metadata.update(
            {
                "description": env_md,
                "observation_space_description": obs_md,
                "action_space_description": act_md,
                "reward_description": reward_md,
                "references": references_md,
                "render_fps": self._spec.render_fps,
            }
        )
        self.render_mode = render_mode
        self.observation_space = FunctionalSpace(
            contains_fn=lambda o: isinstance(o, Motion3DState)
        )
        self.action_space = FunctionalSpace(
            contains_fn=lambda a: isinstance(a, Motion3DAction),
            sample_fn=self._sample_action,
        )

        # Create the PyBullet client.
        if use_gui:
            camera_info = self._spec.get_camera_kwargs()
            self.physics_client_id = create_gui_connection(**camera_info)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create robot.
        robot = create_pybullet_robot(
            self._spec.robot_name,
            self.physics_client_id,
            base_pose=self._spec.robot_base_pose,
            control_mode="reset",
            home_joint_positions=self._spec.initial_joints,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        self.robot = robot

        # Show a visualization of the end effector.
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self._spec.end_effector_viz_radius,
            rgbaColor=self._spec.end_effector_viz_color,
            physicsClientId=self.physics_client_id,
        )

        # Create the body.
        end_effector_pose = self.robot.get_end_effector_pose()
        self.end_effector_viz_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=end_effector_pose.position,
            baseOrientation=end_effector_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create target.
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self._spec.target_radius,
            rgbaColor=self._spec.target_color,
            physicsClientId=self.physics_client_id,
        )

        # Create the body.
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=(0, 0, 0),  # set in reset()
            baseOrientation=(0, 0, 0, 1),
            physicsClientId=self.physics_client_id,
        )

    def reset(
        self,
        *args,
        **kwargs,
    ) -> tuple[Motion3DState, dict]:
        super().reset(*args, **kwargs)  # necessary to reset RNG if seed is given

        # Reset the robot.
        self._set_robot_joints(self._spec.initial_joints)

        # Reset the target. Sample and check reachability.
        target_pose: Pose | None = None
        for _ in range(100_000):
            target_position = self.np_random.uniform(
                self._spec.target_lower_bound, self._spec.target_upper_bound
            )
            target_pose = Pose(tuple(target_position))
            try:
                inverse_kinematics(self.robot, target_pose, validate=True)
            except InverseKinematicsError:
                continue
            self._set_robot_joints(self._spec.initial_joints)
            # If the goal is already reached, keep sampling.
            if not self._goal_reached():
                break
        if target_pose is None:
            raise RuntimeError("Failed to find reachable target position")
        set_pose(self.target_id, target_pose, self.physics_client_id)

        return self._get_obs(), {}

    def step(
        self, action: Motion3DAction
    ) -> tuple[Motion3DState, float, bool, bool, dict]:
        delta_joints = np.clip(
            action.delta_arm_joints,
            -self._spec.max_action_mag,
            self._spec.max_action_mag,
        )
        current_joints = self.robot.get_joint_positions()
        current_joints_fingers = current_joints[7:]
        current_joints_no_fingers = current_joints[:7]
        next_joints_no_fingers = np.clip(
            current_joints_no_fingers + delta_joints,
            self.robot.joint_lower_limits[:7],
            self.robot.joint_upper_limits[:7],
        ).tolist()
        next_joints = next_joints_no_fingers + current_joints_fingers
        self._set_robot_joints(next_joints)
        reward = -1
        terminated = self._goal_reached()
        return self._get_obs(), reward, terminated, False, {}

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        return capture_image(self.physics_client_id, **self._spec.get_camera_kwargs())

    def _get_obs(self) -> Motion3DState:
        joint_positions = self.robot.get_joint_positions()
        target = get_pose(self.target_id, self.physics_client_id).position
        return Motion3DState(joint_positions, target)

    def _goal_reached(self) -> bool:
        target = get_pose(self.target_id, self.physics_client_id).position
        end_effector_pose = self.robot.get_end_effector_pose()
        dist = float(np.linalg.norm(np.subtract(target, end_effector_pose.position)))
        return dist < self._spec.target_radius

    def _set_robot_joints(self, joints: JointPositions) -> None:
        self.robot.set_joints(joints)
        self.robot.open_fingers()
        end_effector_pose = self.robot.get_end_effector_pose()
        set_pose(self.end_effector_viz_id, end_effector_pose, self.physics_client_id)

    def _sample_action(self, rng: np.random.Generator) -> Motion3DAction:
        num_dof = 7
        arr = rng.uniform(
            -self._spec.max_action_mag, self._spec.max_action_mag, size=(num_dof,)
        )
        return Motion3DAction(arr.tolist())

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return f"""A 3D motion planning environment where the goal is to reach a target sphere with the robot's end effector.

The robot is a Kinova Gen-3 with 7 degrees of freedom. The target is a sphere with radius {self._spec.target_radius:.3f}m positioned randomly within the workspace bounds.

The workspace bounds are:
- X: [{self._spec.target_lower_bound[0]:.1f}, {self._spec.target_upper_bound[0]:.1f}]
- Y: [{self._spec.target_lower_bound[1]:.1f}, {self._spec.target_upper_bound[1]:.1f}]
- Z: [{self._spec.target_lower_bound[2]:.1f}, {self._spec.target_upper_bound[2]:.1f}]

Only targets that are reachable via inverse kinematics are sampled.
"""

    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""
        # pylint: disable=line-too-long
        return f"""Observations consist of:
- **joint_positions**: Current joint positions of the {len(self._spec.initial_joints)}-DOF robot arm (list of floats)
- **target**: 3D position (x, y, z) of the target sphere to reach (tuple of 3 floats)

The observation is returned as a Motion3DState dataclass with these two fields.
"""

    def _create_action_space_markdown_description(self) -> str:
        """Create action space description."""
        # pylint: disable=line-too-long
        return f"""Actions control the change in joint positions:
- **delta_arm_joints**: Change in joint positions for all {len(self._spec.initial_joints)} joints (list of floats)

The action is a Motion3DAction dataclass with delta_arm_joints field. Each delta is clipped to the range [-{self._spec.max_action_mag:.3f}, {self._spec.max_action_mag:.3f}].

The resulting joint positions are clipped to the robot's joint limits before being applied.
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return f"""The reward structure is simple:
- **-1.0** penalty at every timestep until the goal is reached
- **Termination** occurs when the end effector is within {self._spec.target_radius:.3f}m of the target center

This encourages the robot to reach the target as quickly as possible while avoiding infinite episodes.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """This is a very common kind of environment."""
