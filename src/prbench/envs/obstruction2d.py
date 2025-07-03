"""Obstruction 2D env (more description coming soon)."""

import gymnasium
import numpy as np
from geom2drobotenvs.envs.obstruction_2d_env import (
    Geom2DRobotEnvTypeFeatures,
)
from geom2drobotenvs.envs.obstruction_2d_env import Obstruction2DEnv as G2DOE
from geom2drobotenvs.envs.obstruction_2d_env import (
    Obstruction2DEnvSpec,
)
from gymnasium.spaces import Box
from numpy.typing import NDArray
from relational_structs import ObjectCentricStateSpace
from relational_structs.spaces import ObjectCentricBoxSpace


class Obstruction2DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Obstruction 2D env (more description coming soon)."""

    def __init__(
        self,
        num_obstructions: int = 2,
        spec: Obstruction2DEnvSpec = Obstruction2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # At the moment, all the real logic for this environment is defined
        # externally. We create that environment and then add some additional
        # code to vectorize observations, making it easier for RL approaches.
        self._geom2d_env = G2DOE(num_obstructions=num_obstructions, spec=spec, **kwargs)
        # Create a Box version of the observation space by assuming a constant
        # number of obstructions (and thus a constant number of objects).
        assert isinstance(self._geom2d_env.observation_space, ObjectCentricStateSpace)
        # Make observation vectors start with the robot, then target surface,
        # then target block, then obstruction blocks. Don't include the walls
        # or the table because those are universally constant.
        exemplar_object_centric_state, _ = self._geom2d_env.reset()
        obj_name_to_obj = {o.name: o for o in exemplar_object_centric_state}
        self._constant_objects = [
            obj_name_to_obj["robot"],
            obj_name_to_obj["target_surface"],
            obj_name_to_obj["target_block"],
        ]
        obstruction_names = {o for o in obj_name_to_obj if o.startswith("obstruct")}
        assert len(obstruction_names) == num_obstructions
        for obstruction_name in sorted(obstruction_names):
            self._constant_objects.append(obj_name_to_obj[obstruction_name])
        self.observation_space = self._geom2d_env.observation_space.to_box(
            self._constant_objects, Geom2DRobotEnvTypeFeatures
        )
        self.action_space = self._geom2d_env.action_space
        assert isinstance(self.observation_space, Box)
        assert isinstance(self.action_space, Box)

    def reset(self, *args, **kwargs) -> tuple[NDArray[np.float32], dict]:
        super().reset(*args, **kwargs)  # necessary to reset RNG if seed is given
        obs, info = self._geom2d_env.reset(*args, **kwargs)
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        vec_obs = self.observation_space.vectorize(obs)
        return vec_obs, info

    def step(
        self, *args, **kwargs
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        obs, reward, terminated, truncated, done = self._geom2d_env.step(
            *args, **kwargs
        )
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        vec_obs = self.observation_space.vectorize(obs)
        return vec_obs, reward, terminated, truncated, done

    def render(self):
        return self._geom2d_env.render()
