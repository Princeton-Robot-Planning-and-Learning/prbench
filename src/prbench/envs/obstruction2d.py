"""Obstruction 2D env."""

import inspect
from typing import Any

import gymnasium
import numpy as np
from geom2drobotenvs.concepts import is_on
from geom2drobotenvs.envs.obstruction_2d_env import (
    Geom2DRobotEnvTypeFeatures,
)
from geom2drobotenvs.envs.obstruction_2d_env import Obstruction2DEnv as G2DOE
from geom2drobotenvs.envs.obstruction_2d_env import (
    Obstruction2DEnvSpec,
)
from geom2drobotenvs.utils import CRVRobotActionSpace
from numpy.typing import NDArray
from relational_structs import ObjectCentricStateSpace
from relational_structs.spaces import ObjectCentricBoxSpace


def create_env_description(num_obstructions: int = 2) -> str:
    """Create a human-readable environment description."""
    # pylint: disable=line-too-long
    if num_obstructions > 0:
        obstruction_sentence = f"\nThe target surface may be initially obstructed. In this environment, there are always {num_obstructions} obstacle blocks.\n"
    else:
        obstruction_sentence = ""

    return f"""A 2D environment where the goal is to place a target block onto a target surface. The block must be completely contained within the surface boundaries.
{obstruction_sentence}    
The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.
"""


def create_reward_description() -> str:
    """Create a human-readable description of environment rewards."""
    # pylint: disable=line-too-long
    return f"""A penalty of -1.0 is given at every time step until termination, which occurs when the target block is "on" the target surface. The definition of "on" is given below:
```python
{inspect.getsource(is_on)}```
"""


def create_references() -> str:
    """Create a human-readable reference section."""
    # pylint: disable=line-too-long
    return """Similar environments have been used many times, especially in the task and motion planning literature. We took inspiration especially from the "1D Continuous TAMP" environment in [PDDLStream](https://github.com/caelan/pddlstream).
"""


class Obstruction2DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Obstruction 2D env."""

    def __init__(
        self,
        num_obstructions: int = 2,
        spec: Obstruction2DEnvSpec = Obstruction2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__()
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
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        assert isinstance(self.action_space, CRVRobotActionSpace)
        # Add descriptions to metadata for doc generation.
        obs_md = self.observation_space.create_markdown_description()
        act_md = self.action_space.create_markdown_description()
        reward_md = create_reward_description()
        references_md = create_references()
        self.metadata = {
            "description": create_env_description(num_obstructions),
            "observation_space_description": obs_md,
            "action_space_description": act_md,
            "reward_description": reward_md,
            "references": references_md,
            "render_modes": self._geom2d_env.metadata["render_modes"],
            "render_fps": 10,
        }

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

    def get_action_from_gui_input(
        self, gui_input: dict[str, Any]
    ) -> NDArray[np.float32]:
        """Get the mapping from human inputs to actions, derived from action
        space."""
        # Unpack the input.
        keys_pressed = gui_input["keys"]
        right_x, right_y = gui_input["right_stick"]
        left_x, _ = gui_input["left_stick"]

        # Initialize the action.
        low = self.action_space.low
        high = self.action_space.high
        action = np.zeros(self.action_space.shape, self.action_space.dtype)

        def _rescale(x: float, lb: float, ub: float) -> float:
            """Rescale from [-1, 1] to [lb, ub]."""
            return lb + (x + 1) * (ub - lb) / 2

        # The right stick controls the x, y movement of the base.
        action[0] = _rescale(right_x, low[0], high[0])
        action[1] = _rescale(right_y, low[1], high[1])

        # The left stick controls the rotation of the base. Only the x axis
        # is used right now.
        action[2] = _rescale(left_x, low[2], high[2])

        # The up/down mouse keys are used to adjust the robot arm.
        if "up" in keys_pressed:
            action[3] = low[3]
        if "down" in keys_pressed:
            action[3] = high[3]

        # The space bar is used to turn on the vacuum.
        if "space" in keys_pressed:
            action[4] = 1.0

        return action
