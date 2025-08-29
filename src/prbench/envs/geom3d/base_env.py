"""Base environment class for all Geom3D environments."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, TypeVar

import gymnasium
import numpy as np
import pybullet as p
from gymnasium.spaces import Space
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot


@dataclass(frozen=True)
class Geom3DEnvSpec:
    """Spec for Geom3DEnv()."""

    # Robot.
    robot_name: str = "kinova-gen3"
    robot_base_pose: Pose = Pose.identity()
    # NOTE: the robot joints include 7 DOF for the arm and 6 DOF for the fingers. We
    # don't need to change the fingers this in the environment.
    initial_joints: JointPositions = field(
        # This is a retract position.
        default_factory=lambda: [
            0.0,  # "joint_1", starting at the robot base and going up to the gripper
            -0.35,  # "joint_2"
            -np.pi,  # "joint_3"
            -2.5,  # "joint_4"
            0.0,  # "joint_5"
            -0.87,  # "joint_6"
            np.pi / 2,  # "joint_7"
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
class Geom3DState:
    """A state for Geom3DEnv()."""

    joint_positions: JointPositions


@dataclass(frozen=True)
class Geom3DAction:
    """An action for Geom3DEnv().

    NOTE: the environment enforces a limit on the magnitude of the deltas.
    """

    # NOTE: this is only a delta on the 7 DOF of the arm, not the fingers, which do not
    # need to change in this environment.
    delta_arm_joints: JointPositions


_ObsType = TypeVar("_ObsType", bound=Geom3DState)
_ActType = TypeVar("_ActType", bound=Geom3DAction)


class Geom3DEnv(gymnasium.Env[_ObsType, _ActType], abc.ABC):
    """Environment where only 3D motion planning is needed to reach a goal region."""

    # Only RGB rendering is implemented.
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        spec: Geom3DEnvSpec,
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
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

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

        # Create the body for the end effector.
        end_effector_pose = self.robot.get_end_effector_pose()
        self.end_effector_viz_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=end_effector_pose.position,
            baseOrientation=end_effector_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

    @abc.abstractmethod
    def _create_observation_space(self) -> Space[_ObsType]:
        """Create the observation space."""

    @abc.abstractmethod
    def _create_action_space(self) -> Space[_ActType]:
        """Create the action space."""

    @abc.abstractmethod
    def _get_obs(self) -> _ObsType:
        """Get the current observation."""

    @abc.abstractmethod
    def _goal_reached(self) -> bool:
        """Check if the goal is currently reached."""

    def reset(
        self,
        *args,
        **kwargs,
    ) -> tuple[_ObsType, dict]:
        super().reset(*args, **kwargs)  # necessary to reset RNG if seed is given

        # Reset the robot. In the future, we may want to allow randomizing the initial
        # robot joint positions.
        self._set_robot_joints(self._spec.initial_joints)

        return self._get_obs(), {}

    def step(self, action: _ActType) -> tuple[_ObsType, float, bool, bool, dict]:
        # Clip the action to be within the allowed limits.
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

    def _set_robot_joints(self, joints: JointPositions) -> None:
        # Set the robot itself.
        self.robot.set_joints(joints)
        # NOTE: we will want to change this soon to allow for grasping.
        self.robot.open_fingers()
        # Update the end effector visualization.
        end_effector_pose = self.robot.get_end_effector_pose()
        set_pose(self.end_effector_viz_id, end_effector_pose, self.physics_client_id)

    @abc.abstractmethod
    def _create_env_markdown_description(self) -> str:
        """Create environment description."""

    @abc.abstractmethod
    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""

    @abc.abstractmethod
    def _create_action_space_markdown_description(self) -> str:
        """Create action space description."""

    @abc.abstractmethod
    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""

    @abc.abstractmethod
    def _create_references_markdown_description(self) -> str:
        """Create references description."""
