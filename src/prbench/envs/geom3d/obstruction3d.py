"""Environment where obstructions must be cleared to place a target on a region."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p
from gymnasium.spaces import Space
from prpl_utils.spaces import FunctionalSpace
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.utils import create_pybullet_block

from prbench.envs.geom3d.base_env import (
    Geom3DAction,
    Geom3DEnv,
    Geom3DEnvSpec,
    Geom3DState,
)
from prbench.envs.geom3d.utils import PURPLE


@dataclass(frozen=True)
class Obstruction3DEnvSpec(Geom3DEnvSpec):
    """Spec for Obstruction3DEnv()."""

    # Table.
    table_pose: Pose = Pose((0.3, 0.0, -0.175))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.2, 0.4, 0.25)

    # Target region.
    target_region_half_extents_lb: tuple[float, float, float] = (0.02, 0.02, 0.005)
    target_region_half_extents_ub: tuple[float, float, float] = (0.05, 0.05, 0.005)
    target_region_rgba: tuple[float, float, float, float] = PURPLE + (1.0,)

    # Target block.
    target_block_size_scale: float = 0.8  # x, y -- relative to target region
    target_block_height: float = 0.025
    target_block_rgba: tuple[float, float, float, float] = target_region_rgba

    # Obstructions.
    obstruction_half_extents_lb: tuple[float, float, float] = (0.01, 0.01, 0.01)
    obstruction_half_extents_ub: tuple[float, float, float] = (0.02, 0.02, 0.03)
    obstruction_rgba: tuple[float, float, float, float] = (0.75, 0.1, 0.1, 1.0)
    # NOTE: this is not the "real" probability, but rather, the probability
    # that we will attempt to sample the obstruction somewhere on the target
    # surface during each round of rejection sampling during reset().
    obstruction_init_on_target_prob: float = 0.9

    def _sample_block_on_block_pose(
        self,
        top_block_half_extents: tuple[float, float, float],
        bottom_block_half_extents: tuple[float, float, float],
        bottom_block_pose: Pose,
        rng: np.random.Generator,
    ) -> Pose:
        """Sample one block pose on top of another one, with no hanging allowed."""
        assert np.allclose(bottom_block_pose.orientation, (0, 0, 0, 1)), "Not implemented"

        lb = (
            bottom_block_pose.position[0]
            - bottom_block_half_extents[0]
            + top_block_half_extents[0],
            bottom_block_pose.position[1]
            - bottom_block_half_extents[1]
            + top_block_half_extents[1],
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        ub = (
            bottom_block_pose.position[0]
            + bottom_block_half_extents[0]
            - top_block_half_extents[0],
            bottom_block_pose.position[1]
            + bottom_block_half_extents[1]
            - top_block_half_extents[1],
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        x, y, z = rng.uniform(lb, ub)

        return Pose((x, y, z))
    
    def _sample_block_on_block_pose_with_overhang(
        self,
        top_block_half_extents: tuple[float, float, float],
        bottom_block_half_extents: tuple[float, float, float],
        bottom_block_pose: Pose,
        rng: np.random.Generator,
    ) -> Pose:
        """Sample one block pose on top of another one, where hanging is allowed."""
        assert np.allclose(bottom_block_pose.orientation, (0, 0, 0, 1)), "Not implemented"

        overhang_pad = 1e-3

        lb = (
            bottom_block_pose.position[0]
            - bottom_block_half_extents[0]
            - top_block_half_extents[0]
            + overhang_pad,
            bottom_block_pose.position[1]
            - bottom_block_half_extents[1]
            - top_block_half_extents[1]
            + overhang_pad,
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        ub = (
            bottom_block_pose.position[0]
            + bottom_block_half_extents[0]
            + top_block_half_extents[0]
            - overhang_pad,
            bottom_block_pose.position[1]
            + bottom_block_half_extents[1]
            + top_block_half_extents[1]
            - overhang_pad,
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        x, y, z = rng.uniform(lb, ub)

        return Pose((x, y, z))

    def sample_block_on_table_pose(
        self, block_half_extents: tuple[float, float, float], rng: np.random.Generator
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        return self._sample_block_on_block_pose(
            block_half_extents, self.table_half_extents, self.table_pose, rng
        )

    def get_target_block_half_extents(
        self, target_region_half_extents: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Calculate the target block half extents based on the target region."""
        return (
            self.target_block_size_scale * target_region_half_extents[0],
            self.target_block_size_scale * target_region_half_extents[1],
            self.target_block_height,
        )

    def sample_obstruction_pose_on_target(
        self,
        obstruction_half_extents: tuple[float, float, float],
        target_region_half_extents: tuple[float, float, float],
        target_region_pose: Pose,
        rng: np.random.Generator,
    ) -> Pose:
        """Sample a pose for the obstruction on top of the target region."""
        return self._sample_block_on_block_pose_with_overhang(
            obstruction_half_extents,
            target_region_half_extents,
            target_region_pose,
            rng,
        )


@dataclass(frozen=True)
class Obstruction3DState(Geom3DState):
    """A state for Obstruction3DEnv()."""


@dataclass(frozen=True)
class Obstruction3DAction(Geom3DAction):
    """An action for Obstruction3DEnv()."""


class Obstruction3DEnv(Geom3DEnv[Obstruction3DState, Obstruction3DAction]):
    """Environment where obstructions must be cleared to place a target on a region."""

    def __init__(
        self,
        num_obstructions: int = 2,
        spec: Obstruction3DEnvSpec = Obstruction3DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__(spec, **kwargs)

        # The spec is of the right type.
        self._spec: Obstruction3DEnvSpec
        self._num_obstructions = num_obstructions

        # Create table.
        self.table_id = create_pybullet_block(
            self._spec.table_rgba,
            half_extents=self._spec.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.table_id, self._spec.table_pose, self.physics_client_id)

        # The objects are created in reset() because they have geometries that change
        # in each episode.
        self._target_region_id: int | None = None
        self._target_block_id: int | None = None
        self._obstruction_ids: set[int] = set()

    def _create_observation_space(self) -> Space[Obstruction3DState]:
        return FunctionalSpace(contains_fn=lambda o: isinstance(o, Obstruction3DState))

    def _create_action_space(self) -> Space[Obstruction3DAction]:
        return FunctionalSpace(
            contains_fn=lambda a: isinstance(a, Obstruction3DAction),
            sample_fn=self._sample_action,
        )

    def reset(
        self,
        *args,
        **kwargs,
    ) -> tuple[Obstruction3DState, dict]:
        super().reset(*args, **kwargs)  # reset the robot

        # Destroy old objects that have varying geometries.
        for old_id in {
            self._target_region_id,
            self._target_block_id,
        } | self._obstruction_ids:
            if old_id is not None:
                p.removeBody(old_id, physicsClientID=self.physics_client_id)

        # Recreate the target region.
        target_region_half_extents: tuple[float, float, float] = tuple(
            self.np_random.uniform(
                self._spec.target_region_half_extents_lb,
                self._spec.target_region_half_extents_ub,
            )
        )
        target_region_pose = self._spec.sample_block_on_table_pose(
            target_region_half_extents, self.np_random
        )
        self._target_region_id = create_pybullet_block(
            self._spec.target_region_rgba,
            half_extents=target_region_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self._target_region_id, target_region_pose, self.physics_client_id)

        # Recreate the target block.
        target_block_half_extents = self._spec.get_target_block_half_extents(
            target_region_half_extents
        )
        self._target_block_id = create_pybullet_block(
            self._spec.target_block_rgba,
            half_extents=target_block_half_extents,
            physics_client_id=self.physics_client_id,
        )
        for _ in range(100_000):
            target_block_pose = self._spec.sample_block_on_table_pose(
                target_block_half_extents, self.np_random
            )
            set_pose(self._target_block_id, target_block_pose, self.physics_client_id)
            # Make sure the target block is not touching the target region at all.
            if not check_body_collisions(
                self._target_block_id,
                self._target_region_id,
                self.physics_client_id,
            ):
                break
        else:
            raise RuntimeError("Failed to sample target block pose")

        # Recreate the obstructions.
        self._obstruction_ids.clear()
        for _ in range(self._num_obstructions):
            obstruction_half_extents: tuple[float, float, float] = tuple(
                self.np_random.uniform(
                    self._spec.obstruction_half_extents_lb,
                    self._spec.obstruction_half_extents_ub,
                )
            )
            obstruction_id = create_pybullet_block(
                self._spec.obstruction_rgba,
                half_extents=obstruction_half_extents,
                physics_client_id=self.physics_client_id,
            )
            self._obstruction_ids.add(obstruction_id)
            for _ in range(100_000):
                obstruction_init_on_target = (
                    self.np_random.uniform()
                    < self._spec.obstruction_init_on_target_prob
                )
                collision_ids = ({self._target_block_id} | self._obstruction_ids) - {obstruction_id}
                if obstruction_init_on_target:
                    obstruction_pose = self._spec.sample_obstruction_pose_on_target(
                        obstruction_half_extents,
                        target_region_half_extents,
                        target_region_pose,
                        self.np_random,
                    )
                else:
                    obstruction_pose = self._spec.sample_block_on_table_pose(
                        obstruction_half_extents, self.np_random
                    )
                    collision_ids.add(self._target_region_id)
                set_pose(obstruction_id, obstruction_pose, self.physics_client_id)
                # Make sure the target block is not touching the target region at all.
                collision_exists = False
                for collision_id in collision_ids:
                    if check_body_collisions(
                        obstruction_id,
                        collision_id,
                        self.physics_client_id,
                    ):
                        collision_exists = True
                        break
                if not collision_exists:
                    break
            else:
                raise RuntimeError("Failed to sample target block pose")

        return self._get_obs(), {}

    def _get_obs(self) -> Obstruction3DState:
        joint_positions = self.robot.get_joint_positions()
        return Obstruction3DState(joint_positions)

    def _goal_reached(self) -> bool:
        # TODO
        return False

    def _sample_action(self, rng: np.random.Generator) -> Obstruction3DAction:
        num_dof = 7
        arr = rng.uniform(
            -self._spec.max_action_mag, self._spec.max_action_mag, size=(num_dof,)
        )
        return Obstruction3DAction(arr.tolist())

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return f"""TODO
"""

    def _create_observation_space_markdown_description(self) -> str:
        """Create observation space description."""
        # pylint: disable=line-too-long
        return f"""TODO
"""

    def _create_action_space_markdown_description(self) -> str:
        """Create action space description."""
        # pylint: disable=line-too-long
        return f"""TODO
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return f"""TODO
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """Similar environments have been used many times, especially in the task and motion planning literature. We took inspiration especially from the "1D Continuous TAMP" environment in [PDDLStream](https://github.com/caelan/pddlstream).
"""
