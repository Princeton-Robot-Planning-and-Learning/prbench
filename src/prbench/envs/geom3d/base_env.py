"""Base environment class for all Geom3D environments."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import gymnasium
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
    check_collisions_with_held_object,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
    ObjectCentricStateSpace,
    Type,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.geom3d.object_types import (
    Geom3DCuboidType,
    Geom3DEnvTypeFeatures,
    Geom3DPointType,
    Geom3DRobotType,
)
from prbench.envs.geom3d.utils import Geom3DObjectCentricState, Geom3DRobotActionSpace


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
    render_image_width: int = 836
    render_image_height: int = 450

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": self.robot_base_pose.position,
            "camera_yaw": 90,
            "camera_distance": 1.5,
            "camera_pitch": -20,
        }


class Geom3DEnv(gymnasium.Env, abc.ABC):
    """Base class for Geom3D environments."""

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
        self._types = {Geom3DRobotType, Geom3DCuboidType, Geom3DPointType}
        self.render_mode = render_mode
        self.observation_space = ObjectCentricStateSpace(self._types)
        self.action_space = Geom3DRobotActionSpace(
            max_magnitude=self._spec.max_action_mag
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
    def _get_obs(self) -> Geom3DObjectCentricState:
        """Get the current observation."""

    @abc.abstractmethod
    def _goal_reached(self) -> bool:
        """Check if the goal is currently reached."""

    @abc.abstractmethod
    def _reset_objects(self) -> None:
        """Reset objects."""

    @abc.abstractmethod
    def _set_object_states(self, obs: Geom3DObjectCentricState) -> None:
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

    @abc.abstractmethod
    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        """Get the half extents for a cuboid object."""

    @property
    def _grasped_object_id(self) -> int | None:
        if self._grasped_object is not None:
            return self._object_name_to_pybullet_id(self._grasped_object)
        return None

    def reset(
        self,
        *args,
        **kwargs,
    ) -> tuple[Geom3DObjectCentricState, dict]:
        super().reset(*args, **kwargs)  # necessary to reset RNG if seed is given

        # Reset the robot. In the future, we may want to allow randomizing the initial
        # robot joint positions.
        self._set_robot_and_held_object(self._spec.initial_joints)

        # Reset objects.
        self._reset_objects()

        return self._get_obs(), {}

    def set_state(self, obs: Geom3DObjectCentricState) -> None:
        """Set the state of the environment to the given one.

        This is useful when treating the environment as a simulator.
        """
        self._set_robot_and_held_object(obs.joint_positions)
        self._grasped_object = obs.grasped_object
        self._grasped_object_transform = obs.grasped_object_transform
        self._set_object_states(obs)

    def step(
        self, action: Array
    ) -> tuple[Geom3DObjectCentricState, float, bool, bool, dict]:
        # Store the current robot joints because we may need to revert in collision.
        current_joints = self.robot.get_joint_positions()

        # Tentatively apply robot action.
        delta_arm_joints = action[:7]
        # Clip the action to be within the allowed limits.
        delta_joints = np.clip(
            delta_arm_joints,
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
        if action[7] < -0.5:
            gripper_action = "close"
        elif action[7] > 0.5:
            gripper_action = "open"
        else:
            gripper_action = "none"

        if gripper_action == "close" and self._grasped_object is None:
            # Check if an object is in collision with the end effector marker.
            # If multiple objects are in collision, treat this as a failed grasp.
            objects_in_grasp_zone: set[str] = set()
            # Perform collision detection one-time rather than once per check.
            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            for obj in sorted(self._get_movable_object_names()):
                obj_id = self._object_name_to_pybullet_id(obj)
                if check_body_collisions(
                    obj_id,
                    self.end_effector_viz_id,
                    self.physics_client_id,
                    perform_collision_detection=False,
                ):
                    objects_in_grasp_zone.add(obj)
            # There must be exactly one object in the grasp zone to succeed.
            if len(objects_in_grasp_zone) == 1:
                self._grasped_object = next(iter(objects_in_grasp_zone))
                assert self._grasped_object_id is not None
                # Create grasp transform.
                world_to_robot = self.robot.get_end_effector_pose()
                world_to_object = get_pose(
                    self._grasped_object_id, self.physics_client_id
                )
                self._grasped_object_transform = multiply_poses(
                    world_to_robot.invert(), world_to_object
                )
                # Close the fingers until they are touching the object.
                while not check_body_collisions(
                    self._grasped_object_id, self.robot.robot_id, self.physics_client_id
                ):
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
        elif gripper_action == "open" and self._grasped_object_id is not None:
            # Check if the held object is being placed on a surface. The rule is that
            # the distance between the object and the surface must be less than thresh.
            surface_supports = self._get_surfaces_supporting_object(
                self._grasped_object_id
            )
            # Placement is successful.
            if surface_supports:
                self._grasped_object = None
                self._grasped_object_transform = None
                self.robot.open_fingers()

        reward = -1
        terminated = self._goal_reached()
        return self._get_obs(), reward, terminated, False, {}

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        return capture_image(
            self.physics_client_id,
            image_width=self._spec.render_image_width,
            image_height=self._spec.render_image_height,
            **self._spec.get_camera_kwargs(),
        )

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
            if check_body_collisions(
                object_id, surface_id, self.physics_client_id, distance_threshold=thresh
            ):
                supporting_surface_ids.add(surface_id)
        return supporting_surface_ids

    def _create_state_dict(
        self, objects: list[tuple[str, Type]]
    ) -> dict[Object, dict[str, float]]:
        state_dict: dict[Object, dict[str, float]] = {}
        for object_name, object_type in objects:
            obj = Object(object_name, object_type)
            feats: dict[str, float] = {}
            # Handle robots.
            if object_type == Geom3DRobotType:
                # Add joints.
                joints = self.robot.get_joint_positions()
                for i, v in enumerate(joints):
                    feats[f"joint_{i+1}"] = v
                # Add grasp.
                grasp_tf_feat_names = [
                    "grasp_tf_x",
                    "grasp_tf_y",
                    "grasp_tf_z",
                    "grasp_tf_qx",
                    "grasp_tf_qy",
                    "grasp_tf_qz",
                    "grasp_tf_qw",
                ]
                if self._grasped_object_transform is None:
                    feats["grasp_active"] = 0
                    for feat_name in grasp_tf_feat_names:
                        feats[feat_name] = 0
                else:
                    feats["grasp_active"] = 1
                    grasp_tf_feats = list(
                        self._grasped_object_transform.position
                    ) + list(self._grasped_object_transform.orientation)
                    for feat_name, feat in zip(
                        grasp_tf_feat_names, grasp_tf_feats, strict=True
                    ):
                        feats[feat_name] = feat
            # Handle cuboids.
            elif object_type == Geom3DCuboidType:
                # Add pose.
                body_id = self._object_name_to_pybullet_id(object_name)
                pose = get_pose(body_id, self.physics_client_id)
                pose_feat_names = [
                    "pose_x",
                    "pose_y",
                    "pose_z",
                    "pose_qx",
                    "pose_qy",
                    "pose_qz",
                    "pose_qw",
                ]
                pose_feats = list(pose.position) + list(pose.orientation)
                for feat_name, feat in zip(pose_feat_names, pose_feats, strict=True):
                    feats[feat_name] = feat
                # Add grasp active.
                if self._grasped_object == object_name:
                    feats["grasp_active"] = 1
                else:
                    feats["grasp_active"] = 0
                # Add half extents.
                half_extent_names = ["half_extent_x", "half_extent_y", "half_extent_z"]
                half_extents = self._get_half_extents(object_name)
                for feat_name, feat in zip(
                    half_extent_names, half_extents, strict=True
                ):
                    feats[feat_name] = feat
            # Handle points.
            elif object_type == Geom3DPointType:
                # Add position.
                body_id = self._object_name_to_pybullet_id(object_name)
                pose = get_pose(body_id, self.physics_client_id)
                feats["x"] = pose.position[0]
                feats["y"] = pose.position[1]
                feats["z"] = pose.position[2]
            else:
                raise NotImplementedError(f"Unsupported object type: {object_type}")
            # Add feats to state dict.
            state_dict[obj] = feats
        return state_dict


class ConstantObjectGeom3DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Defined by an object-centric Geom3D environment and a constant object set.

    The point of this pattern is to allow implementing object-centric environments with
    variable numbers of objects, but then also create versions of the environment with a
    constant number of objects so it is easy to apply, e.g., RL approaches that use
    fixed-dimensional observation and action spaces.
    """

    # NOTE: we need to define render_modes in the class instead of the instance because
    # gym.make extracts render_modes from the class (entry_point) before instantiation.
    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(self, *args, render_mode: str | None = None, **kwargs) -> None:
        super().__init__()
        self._geom3d_env = self._create_object_centric_geom3d_env(*args, **kwargs)
        # Create a Box version of the observation space by extracting the constant
        # objects from an exemplar state.
        assert isinstance(self._geom3d_env.observation_space, ObjectCentricStateSpace)
        exemplar_object_centric_state, _ = self._geom3d_env.reset()
        obj_name_to_obj = {o.name: o for o in exemplar_object_centric_state}
        obj_names = self._get_constant_object_names(exemplar_object_centric_state)
        self._constant_objects = [obj_name_to_obj[o] for o in obj_names]
        # This is a Box space with some extra functionality to allow easy vectorizing.
        self.observation_space = self._geom3d_env.observation_space.to_box(
            self._constant_objects, Geom3DEnvTypeFeatures
        )
        self.action_space = self._geom3d_env.action_space
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        # The action space already inherits from Box, so we don't need to change it.
        assert isinstance(self.action_space, Geom3DRobotActionSpace)
        # Add descriptions to metadata for doc generation.
        obs_md = self.observation_space.create_markdown_description()
        act_md = self.action_space.create_markdown_description()
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
                "render_fps": self._geom3d_env.metadata.get("render_fps", 20),
            }
        )
        self.render_mode = render_mode

    @abc.abstractmethod
    def _create_object_centric_geom3d_env(self, *args, **kwargs) -> Geom3DEnv:
        """Create the underlying object-centric environment."""

    @abc.abstractmethod
    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        """The ordered names of the constant objects extracted from the observations."""

    @abc.abstractmethod
    def _create_env_markdown_description(self) -> str:
        """Create a markdown description of the overall environment."""

    @abc.abstractmethod
    def _create_reward_markdown_description(self) -> str:
        """Create a markdown description of the environment rewards."""

    @abc.abstractmethod
    def _create_references_markdown_description(self) -> str:
        """Create a markdown description of the reference (e.g. papers) for this env."""

    def reset(self, *args, **kwargs) -> tuple[NDArray[np.float32], dict]:
        super().reset(*args, **kwargs)  # necessary to reset RNG if seed is given
        obs, info = self._geom3d_env.reset(*args, **kwargs)
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        vec_obs = self.observation_space.vectorize(obs)
        return vec_obs, info

    def step(
        self, *args, **kwargs
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        obs, reward, terminated, truncated, done = self._geom3d_env.step(
            *args, **kwargs
        )
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        vec_obs = self.observation_space.vectorize(obs)
        return vec_obs, reward, terminated, truncated, done

    def render(self):
        return self._geom3d_env.render()

    def get_action_from_gui_input(
        self, gui_input: dict[str, Any]
    ) -> NDArray[np.float32]:
        """Get the mapping from human inputs to actions."""
        # This will be implemented later
        del gui_input
        return np.array([], dtype=np.float32)
