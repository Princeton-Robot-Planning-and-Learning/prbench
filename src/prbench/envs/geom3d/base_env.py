"""Base environment class for all Geom3D environments."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import gymnasium
import numpy as np
import pybullet as p
from gymnasium.spaces import Space
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, set_pose, multiply_poses, get_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    check_collisions_with_held_object,
    set_robot_joints_with_held_object,
    check_body_collisions,
)
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

    # This is used to check whether a grasped object can be placed on a surface.
    min_placement_dist: float = 1e-3

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
    grasped_object: str | None
    grasped_object_transform: Pose | None  # end effector -> obj


@dataclass(frozen=True)
class Geom3DObjectState:
    """The state of a rigid object in Geom3DEnv()."""

    pose: Pose
    geometry: tuple[float, float, float]  # for now assuming cuboids; half extents


@dataclass(frozen=True)
class Geom3DAction:
    """An action for Geom3DEnv().

    NOTE: the environment enforces a limit on the magnitude of the deltas.
    """

    # NOTE: this is only a delta on the 7 DOF of the arm, not the fingers, which do not
    # need to change in this environment.
    delta_arm_joints: JointPositions
    gripper: Literal["open", "close", "none"] = "none"  # none = no change


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

        # Also create a collision body because we use it for grasp detection.
        collision_id = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self._spec.end_effector_viz_radius,
            physicsClientId=self.physics_client_id,
        )

        # Create the body for the end effector.
        end_effector_pose = self.robot.get_end_effector_pose()
        self.end_effector_viz_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=end_effector_pose.position,
            baseOrientation=end_effector_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Track current held object.
        self._grasped_object: str | None = None
        self._grasped_object_transform: Pose | None = None

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

    @abc.abstractmethod
    def _reset_objects(self) -> None:
        """Reset objects."""

    @abc.abstractmethod
    def _set_object_states(self, obs: _ObsType) -> None:
        """Reset the state of objects; helper for set_state()."""

    @abc.abstractmethod
    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        """Look up the PyBullet ID for a given object name."""

    @abc.abstractmethod
    def _get_collision_object_ids(self) -> set[int]:
        """Get the collision object IDs."""

    @abc.abstractmethod
    def _get_movable_object_names(self) -> set[str]:
        """The names of objects that can be moved by the robot (grasped and placed)."""

    @abc.abstractmethod
    def _get_surface_object_names(self) -> set[str]:
        """The names of objects that can be used as surfaces for other objects.
        
        Note that surfaces might be movable, for example, consider block stacking.
        """

    @property
    def _grasped_object_id(self) -> int | None:
        if self._grasped_object is not None:
            return self._object_name_to_pybullet_id(self._grasped_object)
        return None

    def reset(
        self,
        *args,
        **kwargs,
    ) -> tuple[_ObsType, dict]:
        super().reset(*args, **kwargs)  # necessary to reset RNG if seed is given

        # Reset the robot. In the future, we may want to allow randomizing the initial
        # robot joint positions.
        self._set_robot_and_held_object(self._spec.initial_joints)

        # Reset objects.
        self._reset_objects()

        return self._get_obs(), {}

    def set_state(self, obs: _ObsType) -> None:
        """Set the state of the environment to the given one.

        This is useful when treating the environment as a simulator.
        """
        self._set_robot_and_held_object(obs.joint_positions)
        self._grasped_object = obs.grasped_object
        self._grasped_object_transform = obs.grasped_object_transform
        self._set_object_states(obs)

    def step(self, action: _ActType) -> tuple[_ObsType, float, bool, bool, dict]:
        # Store the current robot joints because we may need to revert in collision.
        current_joints = self.robot.get_joint_positions()

        # Tentatively apply robot action.
        # Clip the action to be within the allowed limits.
        delta_joints = np.clip(
            action.delta_arm_joints,
            -self._spec.max_action_mag,
            self._spec.max_action_mag,
        )
        current_joints_fingers = current_joints[7:]
        current_joints_no_fingers = current_joints[:7]
        next_joints_no_fingers = np.clip(
            current_joints_no_fingers + delta_joints,
            self.robot.joint_lower_limits[:7],
            self.robot.joint_upper_limits[:7],
        ).tolist()
        next_joints = next_joints_no_fingers + current_joints_fingers
        self._set_robot_and_held_object(next_joints)

        # Check for collisions.
        if self._robot_or_held_object_collision_exists():
            # Revert!
            self._set_robot_and_held_object(current_joints)

        # Check for grasping.
        if action.gripper == "close" and self._grasped_object is None:
            # Check if an object is in collision with the end effector marker.
            # If multiple objects are in collision, treat this as a failed grasp.
            objects_in_grasp_zone : set[str] = set()
            # Perform collision detection one-time rather than once per check.
            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            for obj in sorted(self._get_movable_object_names()):
                obj_id = self._object_name_to_pybullet_id(obj)
                if check_body_collisions(obj_id, self.end_effector_viz_id,
                                         self.physics_client_id,
                                         perform_collision_detection=False):
                    objects_in_grasp_zone.add(obj)
            # There must be exactly one object in the grasp zone to succeed.
            if len(objects_in_grasp_zone) == 1:
                self._grasped_object = next(iter(objects_in_grasp_zone))
                # Create grasp transform.
                world_to_robot = self.robot.get_end_effector_pose()
                world_to_object = get_pose(self._grasped_object_id, self.physics_client_id)
                self._grasped_object_transform = multiply_poses(
                    world_to_robot.invert(), world_to_object
                )
                # Close the fingers until they are touching the object.
                while not check_body_collisions(self._grasped_object_id,
                                                self.robot.robot_id,
                                                self.physics_client_id):
                    # If the fingers are fully closed, stop.
                    current_finger_state = self.robot.get_finger_state()
                    closed_finger_state = self.robot.closed_fingers_state
                    assert isinstance(current_finger_state, float)
                    assert isinstance(closed_finger_state, float)
                    if current_finger_state >= closed_finger_state - 1e-2:
                        break
                    next_finger_state = current_finger_state + 1e-2
                    self.robot.set_finger_state(next_finger_state)

        # Check for ungrasping.
        elif action.gripper == "open" and self._grasped_object is not None:
            # Check if the held object is being placed on a surface. The rule is that
            # the distance between the object and the surface must be less than thresh.
            surface_supports = self._get_surfaces_supporting_object(self._grasped_object_id)
            # Placement is successful.
            if surface_supports:
                self._grasped_object = None
                self._grasped_object_transform = None
                self.robot.open_fingers()

        reward = -1
        terminated = self._goal_reached()
        return self._get_obs(), reward, terminated, False, {}

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        return capture_image(self.physics_client_id, **self._spec.get_camera_kwargs())

    def _set_robot_and_held_object(self, joints: JointPositions) -> None:
        set_robot_joints_with_held_object(
            self.robot,
            self.physics_client_id,
            self._grasped_object_id,
            self._grasped_object_transform,
            joints,
        )
        # Update the end effector visualization.
        end_effector_pose = self.robot.get_end_effector_pose()
        set_pose(self.end_effector_viz_id, end_effector_pose, self.physics_client_id)

    def _robot_or_held_object_collision_exists(self) -> bool:
        collision_bodies = self._get_collision_object_ids()
        if self._grasped_object_id is not None:
            collision_bodies.discard(self._grasped_object_id)
        return check_collisions_with_held_object(
            self.robot,
            collision_bodies,
            self.physics_client_id,
            self._grasped_object_id,
            self._grasped_object_transform,
            self.robot.get_joint_positions(),
        )
    
    def _get_surfaces_supporting_object(self, object_id: int) -> set[int]:
        thresh = self._spec.min_placement_dist
        supporting_surface_ids: set[int] = set()
        for surface in self._get_surface_object_names():
            surface_id = self._object_name_to_pybullet_id(surface)
            if check_body_collisions(object_id, surface_id,
                                        self.physics_client_id,
                                        distance_threshold=thresh):
                supporting_surface_ids.add(surface_id)
        return supporting_surface_ids

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
