"""Environment where a block must be retrieved amidst clutter."""

from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
from geom2drobotenvs.envs.base_env import Geom2DRobotEnv, Geom2DRobotEnvSpec
from geom2drobotenvs.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from geom2drobotenvs.structs import ZOrder
from geom2drobotenvs.utils import (
    PURPLE,
    CRVRobotActionSpace,
    SE2Pose,
    create_walls_from_world_boundaries,
    get_suctioned_objects,
    sample_se2_pose,
    state_has_collision,
)
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState, ObjectCentricStateSpace, Type
from relational_structs.spaces import ObjectCentricBoxSpace
from relational_structs.utils import create_state_from_dict

from prbench.utils import get_geom2d_crv_robot_action_from_gui_input

TargetBlockType = Type("target_block", parent=RectangleType)
Geom2DRobotEnvTypeFeatures[TargetBlockType] = list(
    Geom2DRobotEnvTypeFeatures[RectangleType]
)


@dataclass(frozen=True)
class Clutter2DEnvSpec(Geom2DRobotEnvSpec):
    """Scene specification for Clutter2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 2.5
    world_min_y: float = 0.0
    world_max_y: float = 2.5

    # Action space parameters.
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_vac: float = 0.0
    max_vac: float = 1.0

    # Robot hyperparameters.
    robot_base_radius: float = 0.1
    robot_arm_length: float = 2 * robot_base_radius
    robot_gripper_height: float = 0.07
    robot_gripper_width: float = 0.01
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + 4 * robot_base_radius,
            world_min_y + 4 * robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            world_max_x - 4 * robot_base_radius,
            world_max_y - 4 * robot_base_radius,
            np.pi,
        ),
    )

    # Target block hyperparameters.
    target_block_rgb: tuple[float, float, float] = PURPLE
    target_block_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + robot_base_radius,
            world_min_y + robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            world_max_x - robot_base_radius,
            world_max_y - robot_base_radius,
            np.pi,
        ),
    )
    target_block_shape: tuple[float, float] = (
        2 * robot_gripper_height,
        2 * robot_gripper_height,
    )

    # Obstruction hyperparameters.
    obstruction_rgb: tuple[float, float, float] = (0.75, 0.1, 0.1)

    obstruction_height_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    obstruction_width_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    # NOTE: obstruction poses are sampled using a 2D gaussian that is centered
    # at the target location. This hyperparameter controls the variance.
    obstruction_pose_init_distance_scale: float = 0.25

    # For initial state sampling.
    max_init_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 150


class ObjectCentricClutter2DEnv(Geom2DRobotEnv):
    """Environment where a block must be retrieved amidst clutter.

    This is an object-centric environment. The vectorized version with
    Box spaces is defined below.
    """

    def __init__(
        self,
        num_obstructions: int = 2,
        spec: Clutter2DEnvSpec = Clutter2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__(spec, **kwargs)
        self._num_obstructions = num_obstructions
        self._spec: Clutter2DEnvSpec = spec  # for type checking
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": 10,
        }

    def _sample_initial_state(self) -> ObjectCentricState:
        initial_state_dict = self._create_constant_initial_state_dict()
        static_objects = set(initial_state_dict)
        robot_pose = sample_se2_pose(self._spec.robot_init_pose_bounds, self.np_random)
        state = self._create_initial_state(initial_state_dict, robot_pose)
        robot = state.get_objects(CRVRobotType)[0]
        assert not state_has_collision(state, {robot}, static_objects, {})
        # Sample target pose and check for collisions with robot and static objects.
        for _ in range(self._spec.max_init_sampling_attempts):
            target_pose = sample_se2_pose(
                self._spec.target_block_init_bounds, self.np_random
            )
            state = self._create_initial_state(
                initial_state_dict,
                robot_pose,
                target_pose=target_pose,
            )
            target_block = state.get_objects(TargetBlockType)[0]
            if not state_has_collision(
                state, {target_block}, {robot} | static_objects, {}
            ):
                break
        else:
            raise RuntimeError("Failed to sample target pose.")
        # Sample obstructions one by one. Assume that the scene is never so dense
        # that we need to resample earlier choices.
        obstructions: list[tuple[SE2Pose, tuple[float, float]]] = []
        for _ in range(self._num_obstructions):
            for _ in range(self._spec.max_init_sampling_attempts):
                # Sample xy, relative to the target.
                x, y = self.np_random.normal(
                    loc=(target_pose.x, target_pose.y),
                    scale=self._spec.obstruction_pose_init_distance_scale,
                    size=(2,),
                )
                # Make sure in bounds.
                if not (
                    self._spec.world_min_x < x < self._spec.world_max_x
                    and self._spec.world_min_y < y < self._spec.world_max_y
                ):
                    continue
                # Sample theta.
                theta = self.np_random.uniform(-np.pi, np.pi)
                # Check for collisions.
                obstruction_pose = SE2Pose(x, y, theta)
                # Sample shape.
                obstruction_shape = (
                    self.np_random.uniform(*self._spec.obstruction_width_bounds),
                    self.np_random.uniform(*self._spec.obstruction_height_bounds),
                )
                possible_obstructions = obstructions + [
                    (obstruction_pose, obstruction_shape)
                ]
                state = self._create_initial_state(
                    initial_state_dict,
                    robot_pose,
                    target_pose=target_pose,
                    obstructions=possible_obstructions,
                )
                obj_name_to_obj = {o.name: o for o in state}
                new_obstruction = obj_name_to_obj[f"obstruction{len(obstructions)}"]
                assert new_obstruction.name.startswith("obstruction")
                if not state_has_collision(state, {new_obstruction}, set(state), {}):
                    break
            else:
                raise RuntimeError("Failed to sample obstruction pose.")
            # Update obstructions.
            obstructions = possible_obstructions
        # The state should already be finalized.
        return state

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create room walls.
        assert isinstance(self.action_space, CRVRobotActionSpace)
        min_dx, min_dy = self.action_space.low[:2]
        max_dx, max_dy = self.action_space.high[:2]
        wall_state_dict = create_walls_from_world_boundaries(
            self._spec.world_min_x,
            self._spec.world_max_x,
            self._spec.world_min_y,
            self._spec.world_max_y,
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        )
        init_state_dict.update(wall_state_dict)

        return init_state_dict

    def _create_initial_state(
        self,
        constant_initial_state_dict: dict[Object, dict[str, float]],
        robot_pose: SE2Pose,
        target_pose: SE2Pose | None = None,
        obstructions: list[tuple[SE2Pose, tuple[float, float]]] | None = None,
    ) -> ObjectCentricState:

        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        init_state_dict = constant_initial_state_dict.copy()

        # Create the robot.
        robot = Object("robot", CRVRobotType)
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "y": robot_pose.y,
            "theta": robot_pose.theta,
            "base_radius": self._spec.robot_base_radius,
            "arm_joint": self._spec.robot_base_radius,  # arm is fully retracted
            "arm_length": self._spec.robot_arm_length,
            "vacuum": 0.0,  # vacuum is off
            "gripper_height": self._spec.robot_gripper_height,
            "gripper_width": self._spec.robot_gripper_width,
        }

        # Create the target block.
        if target_pose is not None:
            target_block = Object("target_block", TargetBlockType)
            init_state_dict[target_block] = {
                "x": target_pose.x,
                "y": target_pose.y,
                "theta": target_pose.theta,
                "width": self._spec.target_block_shape[0],
                "height": self._spec.target_block_shape[1],
                "static": False,
                "color_r": self._spec.target_block_rgb[0],
                "color_g": self._spec.target_block_rgb[1],
                "color_b": self._spec.target_block_rgb[2],
                "z_order": ZOrder.ALL.value,
            }

        # Create the obstructions.
        if obstructions:
            for i, (obstruction_pose, obstruction_shape) in enumerate(obstructions):
                obstruction = Object(f"obstruction{i}", RectangleType)
                init_state_dict[obstruction] = {
                    "x": obstruction_pose.x,
                    "y": obstruction_pose.y,
                    "theta": obstruction_pose.theta,
                    "width": obstruction_shape[0],
                    "height": obstruction_shape[1],
                    "static": False,
                    "color_r": self._spec.obstruction_rgb[0],
                    "color_g": self._spec.obstruction_rgb[1],
                    "color_b": self._spec.obstruction_rgb[2],
                    "z_order": ZOrder.ALL.value,
                }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        # Terminate when the target object is suctioned.
        assert self._current_state is not None
        robot = self._current_state.get_objects(CRVRobotType)[0]
        target_object = self._current_state.get_objects(TargetBlockType)[0]
        suctioned_objs = {
            o for o, _ in get_suctioned_objects(self._current_state, robot)
        }
        terminated = target_object in suctioned_objs
        return -1.0, terminated


class Clutter2DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Clutter 2D env."""

    def __init__(
        self,
        num_obstructions: int = 2,
        spec: Clutter2DEnvSpec = Clutter2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__()
        # At the moment, all the real logic for this environment is defined
        # externally. We create that environment and then add some additional
        # code to vectorize observations, making it easier for RL approaches.
        self._geom2d_env = ObjectCentricClutter2DEnv(
            num_obstructions=num_obstructions, spec=spec, **kwargs
        )
        # Create a Box version of the observation space by assuming a constant
        # number of obstructions (and thus a constant number of objects).
        assert isinstance(self._geom2d_env.observation_space, ObjectCentricStateSpace)
        # Make observation vectors start with the robot, then target block,
        # then obstruction blocks. Don't include the walls because those are
        # universally constant.
        exemplar_object_centric_state, _ = self._geom2d_env.reset()
        obj_name_to_obj = {o.name: o for o in exemplar_object_centric_state}
        self._constant_objects = [
            obj_name_to_obj["robot"],
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
        env_md = "TODO"
        obs_md = self.observation_space.create_markdown_description()
        act_md = self.action_space.create_markdown_description()
        reward_md = "TODO"
        references_md = "TODO"
        self.metadata = {
            "description": env_md,
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
        """Get the mapping from human inputs to actions."""
        assert isinstance(self.action_space, CRVRobotActionSpace)
        return get_geom2d_crv_robot_action_from_gui_input(self.action_space, gui_input)
