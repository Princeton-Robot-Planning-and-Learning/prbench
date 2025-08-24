"""Obstruction 2D env."""

import inspect

from geom2drobotenvs.envs.base_env import Geom2DRobotEnv
from geom2drobotenvs.envs.obstruction_2d_env import Obstruction2DEnv as G2DOE
from relational_structs import ObjectCentricState

from prbench.envs.geom2d.geom2d_utils import ConstantObjectGeom2DEnv
from prbench.envs.geom2d.utils import is_on


class Obstruction2DEnv(ConstantObjectGeom2DEnv):
    """Obstruction 2D env with a constant number of objects."""

    def _create_object_centric_geom2d_env(self, *args, **kwargs) -> Geom2DRobotEnv:
        return G2DOE(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "target_surface", "target_block"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstruct"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        num_obstructions = len(self._constant_objects) - 3
        # pylint: disable=line-too-long
        if num_obstructions > 0:
            obstruction_sentence = f"\nThe target surface may be initially obstructed. In this environment, there are always {num_obstructions} obstacle blocks.\n"
        else:
            obstruction_sentence = ""

        return f"""A 2D environment where the goal is to place a target block onto a target surface. The block must be completely contained within the surface boundaries.
{obstruction_sentence}
The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.
"""

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return f"""A penalty of -1.0 is given at every time step until termination, which occurs when the target block is "on" the target surface. The definition of "on" is given below:
```python
{inspect.getsource(is_on)}```
"""

    def _create_references_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """Similar environments have been used many times, especially in the task and motion planning literature. We took inspiration especially from the "1D Continuous TAMP" environment in [PDDLStream](https://github.com/caelan/pddlstream).
"""
