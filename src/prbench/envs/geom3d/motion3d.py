"""Environment where only 3D motion planning is needed to reach a goal region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prbench.core import ConstantObjectPRBenchEnv
from prbench.envs.geom3d.base_env import (
    Geom3DEnvConfig,
    ObjectCentricGeom3DRobotEnv,
)
from prbench.envs.geom3d.object_types import (
    Geom3DEnvTypeFeatures,
    Geom3DPointType,
    Geom3DRobotType,
)
from prbench.envs.geom3d.utils import Geom3DObjectCentricState


@dataclass(frozen=True)
class Motion3DEnvConfig(Geom3DEnvConfig):
    """Config for Motion3DEnv()."""

    # Target.
    target_radius: float = 0.1
    target_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)
    target_lower_bound: tuple[float, float, float] = (0.0, 0.1, 0.0)
    target_upper_bound: tuple[float, float, float] = (0.5, 0.8, 0.5)


class Motion3DObjectCentricState(Geom3DObjectCentricState):
    """A state in the Motion3DEnv().

    Adds convenience methods on top of Geom3DObjectCentricState().
    """

    @property
    def target_position(self) -> tuple[float, float, float]:
        """The position of the target, assuming the name "target"."""
        target = self.get_object_from_name("target")
        return (self.get(target, "x"), self.get(target, "y"), self.get(target, "z"))


class ObjectCentricMotion3DEnv(
    ObjectCentricGeom3DRobotEnv[Motion3DObjectCentricState, Motion3DEnvConfig]
):
    """Environment where only 3D motion planning is needed to reach a goal region."""

    def __init__(
        self, config: Motion3DEnvConfig = Motion3DEnvConfig(), **kwargs
    ) -> None:
        super().__init__(config=config, **kwargs)

        # Create target.
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.config.target_radius,
            rgbaColor=self.config.target_color,
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

    @property
    def state_cls(self) -> TypingType[Geom3DObjectCentricState]:
        return Motion3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        # Neither the target nor the robot are constant in this env.
        return {}

    def _reset_objects(self) -> None:
        # Reset the target. Sample and check reachability.
        target_pose: Pose | None = None
        for _ in range(100_000):
            target_position = self.np_random.uniform(
                self.config.target_lower_bound, self.config.target_upper_bound
            )
            target_pose = Pose(tuple(target_position))
            try:
                inverse_kinematics(self.robot, target_pose, validate=True)
            except InverseKinematicsError:
                continue
            self._set_robot_and_held_object(
                self.config.initial_joints, self.config.initial_finger_state
            )
            # If the goal is already reached, keep sampling.
            if not self._goal_reached():
                break
        if target_pose is None:
            raise RuntimeError("Failed to find reachable target position")
        set_pose(self.target_id, target_pose, self.physics_client_id)

    def _set_object_states(self, obs: Motion3DObjectCentricState) -> None:
        assert self.target_id is not None
        set_pose(self.target_id, Pose(obs.target_position), self.physics_client_id)

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "target":
            return self.target_id
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return set()

    def _get_movable_object_names(self) -> set[str]:
        return set()

    def _get_surface_object_names(self) -> set[str]:
        return set()

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        raise NotImplementedError("No objects have half extents")

    def _get_obs(self) -> Motion3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Geom3DRobotType), ("target", Geom3DPointType)]
        )
        state = create_state_from_dict(
            state_dict, Geom3DEnvTypeFeatures, state_cls=Motion3DObjectCentricState
        )
        assert isinstance(state, Motion3DObjectCentricState)
        return state

    def _goal_reached(self) -> bool:
        target = get_pose(self.target_id, self.physics_client_id).position
        end_effector_pose = self.robot.get_end_effector_pose()
        dist = float(np.linalg.norm(np.subtract(target, end_effector_pose.position)))
        return dist < self.config.target_radius


class Motion3DEnv(ConstantObjectPRBenchEnv):
    """Motion 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricGeom3DRobotEnv:
        return ObjectCentricMotion3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return ["robot", "target"]

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Motion3DEnvConfig)
        return f"""A 3D motion planning environment where the goal is to reach a target sphere with the robot's end effector.

The robot is a Kinova Gen-3 with 7 degrees of freedom. The target is a sphere with radius {config.target_radius:.3f}m positioned randomly within the workspace bounds.

The workspace bounds are:
- X: [{config.target_lower_bound[0]:.1f}, {config.target_upper_bound[0]:.1f}]
- Y: [{config.target_lower_bound[1]:.1f}, {config.target_upper_bound[1]:.1f}]
- Z: [{config.target_lower_bound[2]:.1f}, {config.target_upper_bound[2]:.1f}]

Only targets that are reachable via inverse kinematics are sampled.
"""

    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Motion3DEnvConfig)
        return f"""Observations consist of:
- **joint_positions**: Current joint positions of the {len(config.initial_joints)}-DOF robot arm (list of floats)
- **target**: 3D position (x, y, z) of the target sphere to reach (tuple of 3 floats)

The observation is returned as a Motion3DState dataclass with these two fields.
"""

    def _create_action_space_markdown_description(self) -> str:
        """Create action space description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Motion3DEnvConfig)
        return f"""Actions control the change in joint positions:
- **delta_arm_joints**: Change in joint positions for all {len(config.initial_joints)} joints (list of floats)

The action is a Motion3DAction dataclass with delta_arm_joints field. Each delta is clipped to the range [-{config.max_action_mag:.3f}, {config.max_action_mag:.3f}].

The resulting joint positions are clipped to the robot's joint limits before being applied.
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Motion3DEnvConfig)
        return f"""The reward structure is simple:
- **-1.0** penalty at every timestep until the goal is reached
- **Termination** occurs when the end effector is within {config.target_radius:.3f}m of the target center

This encourages the robot to reach the target as quickly as possible while avoiding infinite episodes.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """This is a very common kind of environment."""
