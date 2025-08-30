"""Environment where only 3D motion planning is needed to reach a goal region."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p
from gymnasium.spaces import Space
from prpl_utils.spaces import FunctionalSpace
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)

from prbench.envs.geom3d.base_env import (
    Geom3DAction,
    Geom3DEnv,
    Geom3DEnvSpec,
    Geom3DState,
)


@dataclass(frozen=True)
class Motion3DEnvSpec(Geom3DEnvSpec):
    """Spec for Motion3DEnv()."""

    # Target.
    target_radius: float = 0.1
    target_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)
    target_lower_bound: tuple[float, float, float] = (0.0, 0.1, 0.0)
    target_upper_bound: tuple[float, float, float] = (0.5, 0.8, 0.5)


@dataclass(frozen=True)
class Motion3DState(Geom3DState):
    """A state for Motion3DEnv()."""

    target: tuple[float, float, float]  # 3D position to reach with end effector


@dataclass(frozen=True)
class Motion3DAction(Geom3DAction):
    """An action for Motion3DEnv()."""


class Motion3DEnv(Geom3DEnv[Motion3DState, Motion3DAction]):
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

    def _create_observation_space(self) -> Space[Motion3DState]:
        return FunctionalSpace(contains_fn=lambda o: isinstance(o, Motion3DState))

    def _create_action_space(self) -> Space[Motion3DAction]:
        return FunctionalSpace(
            contains_fn=lambda a: isinstance(a, Motion3DAction),
            sample_fn=self._sample_action,
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
            self._set_robot_and_held_object(self._spec.initial_joints)
            # If the goal is already reached, keep sampling.
            if not self._goal_reached():
                break
        if target_pose is None:
            raise RuntimeError("Failed to find reachable target position")
        set_pose(self.target_id, target_pose, self.physics_client_id)

    def _set_object_states(self, obs: Motion3DState) -> None:
        assert self.target_id is not None
        set_pose(self.target_id, Pose(obs.target), self.physics_client_id)

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

    def _get_obs(self) -> Motion3DState:
        joint_positions = self.robot.get_joint_positions()
        target = get_pose(self.target_id, self.physics_client_id).position
        return Motion3DState(
            joint_positions,
            grasped_object_transform=None,
            grasped_object=None,
            target=target,
        )

    def _goal_reached(self) -> bool:
        target = get_pose(self.target_id, self.physics_client_id).position
        end_effector_pose = self.robot.get_end_effector_pose()
        dist = float(np.linalg.norm(np.subtract(target, end_effector_pose.position)))
        return dist < self._spec.target_radius

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
