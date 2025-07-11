"""Environment with a stick and buttons that need to be pressed."""

from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
from geom2drobotenvs.envs.base_env import Geom2DRobotEnv, Geom2DRobotEnvSpec
from geom2drobotenvs.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
    CircleType,
)
from geom2drobotenvs.structs import ZOrder
from geom2drobotenvs.utils import (
    BLACK,
    PURPLE,
    CRVRobotActionSpace,
    SE2Pose,
    create_walls_from_world_boundaries,
    rectangle_object_to_geom,
    sample_se2_pose,
    state_has_collision,
)
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState, ObjectCentricStateSpace, Type
from relational_structs.spaces import ObjectCentricBoxSpace
from relational_structs.utils import create_state_from_dict

from prbench.utils import get_geom2d_crv_robot_action_from_gui_input


@dataclass(frozen=True)
class StickButton2DEnvSpec(Geom2DRobotEnvSpec):
    """Spec for StickButton2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 3.5
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
    # The robot starts on the bottom (off the table).
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + 3 * robot_base_radius,
            world_min_y + 3 * robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            world_max_x - 3 * robot_base_radius,
            world_min_y + (world_max_y - world_min_y) / 2 - 3 * robot_base_radius,
            np.pi,
        ),
    )

    # Table hyperparameters.
    table_rgb: tuple[float, float, float] = BLACK
    table_pose: SE2Pose = SE2Pose(
        x=world_min_x,
        y=world_min_y + (world_max_y - world_min_y) / 2,
        theta=0,
    )
    table_shape: tuple[float, float] = (
        world_max_x - world_min_x,
        (world_max_y - world_min_y) / 2,
    )

    # Stick hyperparameters.
    stick_rgb: tuple[float, float, float] = (0.4, 0.2, 0.1)
    stick_shape: tuple[float, float] = (robot_base_radius / 2, table_shape[1])
    stick_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(world_min_x, table_pose.y - stick_shape[1] / 2, 0),
        SE2Pose(world_max_x - stick_shape[0],
                table_pose.y - stick_shape[1] / 10, 0),
    )

    # Button hyperparameters.
    button_unpressed_rgb: tuple[float, float, float] = (0.9, 0.0, 0.0)
    button_pressed_rgb: tuple[float, float, float] = (0.0, 0.9, 0.0)
    button_radius: float = robot_base_radius / 2
    button_init_position_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (world_min_x + button_radius, world_min_y + button_radius),
        (world_max_x - button_radius, world_max_y - button_radius),
    )

    # For initial state sampling.
    max_init_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 150


class ObjectCentricStickButton2DEnv(Geom2DRobotEnv):
    """Environment with a stick and buttons that need to be pressed.

    The robot cannot directly press buttons that are on the table but can
    directly press buttons that are on the floor (by touching them).

    The stick can be used to press buttons on the table (by touch).

    This is an object-centric environment. The vectorized version with
    Box spaces is defined below.
    """

    def __init__(
        self,
        num_buttons: int = 2,
        spec: StickButton2DEnvSpec = StickButton2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__(spec, **kwargs)
        self._num_buttons = num_buttons
        self._spec: StickButton2DEnvSpec = spec  # for type checking
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": 10,
        }

    def _sample_initial_state(self) -> ObjectCentricState:
        initial_state_dict = self._create_constant_initial_state_dict()
        # Sample initial robot pose.
        robot_pose = sample_se2_pose(self._spec.robot_init_pose_bounds, self.np_random)
        # Sample stick pose.
        for _ in range(self._spec.max_init_sampling_attempts):
            stick_pose = sample_se2_pose(self._spec.stick_init_pose_bounds, self.np_random)
            state = self._create_initial_state(
                initial_state_dict,
                robot_pose,
                stick_pose=stick_pose,
                button_positions=[],
            )
            obj_name_to_obj = {o.name: o for o in state}
            stick = obj_name_to_obj["stick"]
            if not state_has_collision(
                state, {stick}, set(state), {}
            ):
                break
        else:
            raise RuntimeError("Failed to sample target pose.")

        # Sample button positions. Assume that the scene is never so dense
        # that we need to resample earlier choices.
        button_positions: list[tuple[float, float]] = []
        for _ in range(self._num_buttons):
            while True:
                button_position = tuple(np.random.uniform(*self._spec.button_init_position_bounds))
                new_button_positions = button_positions + [button_position]
                state = self._create_initial_state(
                    initial_state_dict,
                    robot_pose,
                    stick_pose=stick_pose,
                    button_positions=new_button_positions,
                    button_z_order=ZOrder.SURFACE.value,
                )
                obj_name_to_obj = {o.name: o for o in state}
                new_button = obj_name_to_obj[f"button{len(button_positions)}"]
                if not state_has_collision(
                    state, {new_button}, set(state), {}
                ):
                    button_positions.append(button_position)
                    break

        # Recreate state now with no-collision buttons.
        state = self._create_initial_state(
            initial_state_dict,
            robot_pose,
            stick_pose=stick_pose,
            button_positions=button_positions,
            button_z_order=ZOrder.NONE.value,
        )
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

        # Create the table.
        table = Object("table", RectangleType)
        init_state_dict[table] = {
            "x": self._spec.table_pose.x,
            "y": self._spec.table_pose.y,
            "theta": self._spec.table_pose.theta,
            "width": self._spec.table_shape[0],
            "height": self._spec.table_shape[1],
            "static": True,
            "color_r": self._spec.table_rgb[0],
            "color_g": self._spec.table_rgb[1],
            "color_b": self._spec.table_rgb[2],
            "z_order": ZOrder.FLOOR.value,
        }

        return init_state_dict

    def _create_initial_state(
        self,
        constant_initial_state_dict: dict[Object, dict[str, float]],
        robot_pose: SE2Pose,
        stick_pose: SE2Pose,
        button_positions: list[tuple[float, float]],
        button_z_order: ZOrder = ZOrder.NONE.value,
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

        # Create the stick.
        stick = Object("stick", RectangleType)
        init_state_dict[stick] = {
            "x": stick_pose.x,
            "y": stick_pose.y,
            "theta": stick_pose.theta,
            "width": self._spec.stick_shape[0],
            "height": self._spec.stick_shape[1],
            "static": False,
            "color_r": self._spec.stick_rgb[0],
            "color_g": self._spec.stick_rgb[1],
            "color_b": self._spec.stick_rgb[2],
            "z_order": ZOrder.SURFACE.value,
        }

        # Create the buttons.
        for button_idx, button_position in enumerate(button_positions):
            button = Object(f"button{button_idx}", CircleType)
            init_state_dict[button] = {
                "x": button_position[0],
                "y": button_position[1],
                "theta": 0,
                "radius": self._spec.button_radius,
                "static": True,
                "color_r": self._spec.button_unpressed_rgb[0],
                "color_g": self._spec.button_unpressed_rgb[1],
                "color_b": self._spec.button_unpressed_rgb[2],
                "z_order": button_z_order,
            }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        # TODO
        terminated = False
        return -1.0, terminated


# class Motion2DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
#     """Motion 2D env."""

#     def __init__(
#         self,
#         num_passages: int = 2,
#         spec: Motion2DEnvSpec = Motion2DEnvSpec(),
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         # At the moment, all the real logic for this environment is defined
#         # externally. We create that environment and then add some additional
#         # code to vectorize observations, making it easier for RL approaches.
#         self._geom2d_env = ObjectCentricMotion2DEnv(
#             num_passages=num_passages, spec=spec, **kwargs
#         )
#         # Create a Box version of the observation space by assuming a constant
#         # number of obstacles (and thus a constant number of objects).
#         assert isinstance(self._geom2d_env.observation_space, ObjectCentricStateSpace)
#         # Make observation vectors start with the robot, then target region,
#         # then obstacle blocks. Don't include the walls because those are
#         # universally constant.
#         exemplar_object_centric_state, _ = self._geom2d_env.reset()
#         obj_name_to_obj = {o.name: o for o in exemplar_object_centric_state}
#         self._constant_objects = [
#             obj_name_to_obj["robot"],
#             obj_name_to_obj["target_region"],
#         ]
#         obstacle_names = {o for o in obj_name_to_obj if o.startswith("obstacle")}
#         assert len(obstacle_names) == 2 * num_passages
#         for obstacle_name in sorted(obstacle_names):
#             self._constant_objects.append(obj_name_to_obj[obstacle_name])
#         self.observation_space = self._geom2d_env.observation_space.to_box(
#             self._constant_objects, Geom2DRobotEnvTypeFeatures
#         )
#         self.action_space = self._geom2d_env.action_space
#         assert isinstance(self.observation_space, ObjectCentricBoxSpace)
#         assert isinstance(self.action_space, CRVRobotActionSpace)
#         # Add descriptions to metadata for doc generation.
#         env_md = create_env_description(num_passages)
#         obs_md = self.observation_space.create_markdown_description()
#         act_md = self.action_space.create_markdown_description()
#         reward_md = "A penalty of -1.0 is given at every time step until termination, which occurs when the robot's position is within the target region.\n"  # pylint: disable=line-too-long
#         references_md = "Narrow passages are a classic challenge in motion planning.\n"  # pylint: disable=line-too-long
#         self.metadata = {
#             "description": env_md,
#             "observation_space_description": obs_md,
#             "action_space_description": act_md,
#             "reward_description": reward_md,
#             "references": references_md,
#             "render_modes": self._geom2d_env.metadata["render_modes"],
#             "render_fps": 10,
#         }

#     def reset(self, *args, **kwargs) -> tuple[NDArray[np.float32], dict]:
#         super().reset(*args, **kwargs)  # necessary to reset RNG if seed is given
#         obs, info = self._geom2d_env.reset(*args, **kwargs)
#         assert isinstance(self.observation_space, ObjectCentricBoxSpace)
#         vec_obs = self.observation_space.vectorize(obs)
#         return vec_obs, info

#     def step(
#         self, *args, **kwargs
#     ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
#         obs, reward, terminated, truncated, done = self._geom2d_env.step(
#             *args, **kwargs
#         )
#         assert isinstance(self.observation_space, ObjectCentricBoxSpace)
#         vec_obs = self.observation_space.vectorize(obs)
#         return vec_obs, reward, terminated, truncated, done

#     def render(self):
#         return self._geom2d_env.render()

#     def get_action_from_gui_input(
#         self, gui_input: dict[str, Any]
#     ) -> NDArray[np.float32]:
#         """Get the mapping from human inputs to actions."""
#         assert isinstance(self.action_space, CRVRobotActionSpace)
#         return get_geom2d_crv_robot_action_from_gui_input(self.action_space, gui_input)


# def create_env_description(num_passages: int = 2) -> str:
#     """Create a human-readable environment description."""
#     # pylint: disable=line-too-long
#     if num_passages > 0:
#         obstacle_sentence = (
#             f"\nIn this environment, there are always {num_passages} narrow passages.\n"
#         )
#     else:
#         obstacle_sentence = ""

#     return f"""A 2D environment where the goal is to reach a target region while avoiding static obstacles.
# {obstacle_sentence}
# The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. The arm and vacuum do not need to be used in this environment.
# """
