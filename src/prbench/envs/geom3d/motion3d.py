"""Environment where only 3D motion planning is needed to reach a goal region."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from relational_structs import ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prbench.envs.geom3d.base_env import (
    ConstantObjectGeom3DEnv,
    Geom3DEnv,
    Geom3DEnvSpec,
)
from prbench.envs.geom3d.object_types import (
    Geom3DEnvTypeFeatures,
    Geom3DPointType,
    Geom3DRobotType,
)
from prbench.envs.geom3d.utils import Geom3DObjectCentricState


@dataclass(frozen=True)
class Motion3DEnvSpec(Geom3DEnvSpec):
    """Spec for Motion3DEnv()."""

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


class ObjectCentricMotion3DEnv(Geom3DEnv):
    """Environment where only 3D motion planning is needed to reach a goal region."""

    def __init__(self, spec: Motion3DEnvSpec = Motion3DEnvSpec(), **kwargs) -> None:
        super().__init__(spec, **kwargs)

        # The spec is of the right type.
        self._spec: Motion3DEnvSpec

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

    def _reset_objects(self) -> None:
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
            self._set_robot_and_held_object(
                self._spec.initial_joints, self._spec.initial_finger_state
            )
            # If the goal is already reached, keep sampling.
            if not self._goal_reached():
                break
        if target_pose is None:
            raise RuntimeError("Failed to find reachable target position")
        set_pose(self.target_id, target_pose, self.physics_client_id)

    def _set_object_states(self, obs: Geom3DObjectCentricState) -> None:
        assert isinstance(obs, Motion3DObjectCentricState)
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
        s = create_state_from_dict(state_dict, Geom3DEnvTypeFeatures)
        return Motion3DObjectCentricState(s.data, Geom3DEnvTypeFeatures)

    def _goal_reached(self) -> bool:
        target = get_pose(self.target_id, self.physics_client_id).position
        end_effector_pose = self.robot.get_end_effector_pose()
        dist = float(np.linalg.norm(np.subtract(target, end_effector_pose.position)))
        return dist < self._spec.target_radius


class Motion3DEnv(ConstantObjectGeom3DEnv):
    """Motion 3D env with a constant number of objects."""

    def __init__(self, spec: Motion3DEnvSpec = Motion3DEnvSpec(), **kwargs) -> None:
        self._spec = spec
        super().__init__(spec=spec, **kwargs)

    def _create_object_centric_geom3d_env(self, *args, **kwargs) -> Geom3DEnv:
        return ObjectCentricMotion3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return ["robot", "target"]

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
