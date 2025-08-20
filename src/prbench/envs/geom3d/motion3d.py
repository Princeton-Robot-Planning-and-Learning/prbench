"""Environment where only 3D motion planning is needed to reach a goal region."""

from __future__ import annotations

from dataclasses import dataclass, field
import gymnasium
import numpy as np
from typing import Any

from prpl_utils.spaces import FunctionalSpace
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
import pybullet as p



@dataclass(frozen=True)
class Motion3DEnvSpec:
    """Spec for Motion3DEnv()."""

    # Robot.
    robot_name: str = "kinova-gen3"
    robot_base_pose: Pose = Pose.identity()
    initial_joints: JointPositions = field(
        default_factory=lambda: [-4.3, -1.6, -4.8, -1.8, -1.4, -1.1, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    end_effector_viz_radius: float = 0.01
    end_effector_viz_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)

    # Target.
    target_radius: float = 0.1
    target_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)

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
            "background_rgb": (250 / 255, 220 / 255, 255 / 255),
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

    TODO do the above.
    """

    delta_joints: JointPositions  # change in joint positions


class Motion3DEnv(gymnasium.Env[Motion3DState, Motion3DAction]):
    """Environment where only 3D motion planning is needed to reach a goal region."""

    # Only RGB rendering is implemented.
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 spec: Motion3DEnvSpec = Motion3DEnvSpec(),
                 use_gui: bool = False,
                 ) -> None:
        self._spec = spec

        # Set up Gymnasium env fields.
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": self._spec.render_fps,
        }
        self.render_mode = "rgb_array"
        self.observation_space = FunctionalSpace(contains_fn=lambda o: isinstance(o, Motion3DState))
        self.action_space = FunctionalSpace(contains_fn=lambda a: isinstance(a, Motion3DAction))

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
        self.cylinder_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=(0, 0, 0),  # TODO set in reset()
            baseOrientation=(0, 0, 0, 1),
            physicsClientId=self.physics_client_id,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Motion3DState, dict]:
        super().reset(seed=seed)

        # Reset the robot.
        self.robot.set_joints(self._spec.initial_joints)

        # TODO reset the target.

        return self._get_obs(), {}

    def step(self, action: Motion3DAction) -> tuple[Motion3DState, float, bool, bool, dict]:
        # TODO
        return self._get_obs(), 0.0, False, False, {}
    

    def _get_obs(self) -> Motion3DState:
        joint_positions = self.robot.get_joint_positions()
        target = (0, 0, 0)  # TODO
        return Motion3DState(joint_positions, target)
