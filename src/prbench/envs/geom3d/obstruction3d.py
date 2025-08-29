"""Environment where obstructions must be cleared to place a target on a region."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium.spaces import Space
from prpl_utils.spaces import FunctionalSpace
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.utils import create_pybullet_block

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
class Obstruction3DEnvSpec(Geom3DEnvSpec):
    """Spec for Obstruction3DEnv()."""

    # Table.
    table_pose: Pose = Pose((0.5, 0.0, -0.175))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.25, 0.4, 0.25)


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
        spec: Obstruction3DEnvSpec = Obstruction3DEnvSpec(),
        render_mode: str | None = None,
        use_gui: bool = False,
    ) -> None:
        super().__init__(spec, render_mode=render_mode, use_gui=use_gui)

        # The spec is of the right type.
        self._spec: Obstruction3DEnvSpec

        # Create table.
        self.table_id = create_pybullet_block(
            self._spec.table_rgba,
            half_extents=self._spec.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(
            self.table_id, self._spec.table_pose, self.physics_client_id
        )

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

