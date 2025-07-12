"""Environment with a stick and buttons that need to be pressed."""

from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
from geom2drobotenvs.envs.base_env import Geom2DRobotEnv, Geom2DRobotEnvSpec
from geom2drobotenvs.object_types import (
    CircleType,
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from geom2drobotenvs.structs import ZOrder
from geom2drobotenvs.utils import (
    BLACK,
    CRVRobotActionSpace,
    SE2Pose,
    create_walls_from_world_boundaries,
    sample_se2_pose,
    state_has_collision,
)
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState, ObjectCentricStateSpace
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
        SE2Pose(world_max_x - stick_shape[0], table_pose.y - stick_shape[1] / 10, 0),
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
    render_fps: int = 20


class ObjectCentricStickButton2DEnv(Geom2DRobotEnv):
    """Environment with a stick and buttons that need to be pressed.

    The robot cannot directly press buttons that are on the table but
    can directly press buttons that are on the floor (by touching them).

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
            "render_fps": self._spec.render_fps,
        }

    def _sample_initial_state(self) -> ObjectCentricState:
        initial_state_dict = self._create_constant_initial_state_dict()
        # Sample initial robot pose.
        robot_pose = sample_se2_pose(self._spec.robot_init_pose_bounds, self.np_random)
        # Sample stick pose.
        for _ in range(self._spec.max_init_sampling_attempts):
            stick_pose = sample_se2_pose(
                self._spec.stick_init_pose_bounds, self.np_random
            )
            state = self._create_initial_state(
                initial_state_dict,
                robot_pose,
                stick_pose=stick_pose,
                button_positions=[],
            )
            obj_name_to_obj = {o.name: o for o in state}
            stick = obj_name_to_obj["stick"]
            if not state_has_collision(state, {stick}, set(state), {}):
                break
        else:
            raise RuntimeError("Failed to sample target pose.")

        # Sample button positions. Assume that the scene is never so dense
        # that we need to resample earlier choices.
        button_positions: list[tuple[float, float]] = []
        for _ in range(self._num_buttons):
            while True:
                button_position = tuple(
                    self.np_random.uniform(*self._spec.button_init_position_bounds)
                )
                new_button_positions = button_positions + [button_position]
                state = self._create_initial_state(
                    initial_state_dict,
                    robot_pose,
                    stick_pose=stick_pose,
                    button_positions=new_button_positions,
                    button_z_order=ZOrder.SURFACE,
                )
                obj_name_to_obj = {o.name: o for o in state}
                new_button = obj_name_to_obj[f"button{len(button_positions)}"]
                if not state_has_collision(state, {new_button}, set(state), {}):
                    button_positions.append(button_position)
                    break

        # Recreate state now with no-collision buttons.
        state = self._create_initial_state(
            initial_state_dict,
            robot_pose,
            stick_pose=stick_pose,
            button_positions=button_positions,
            button_z_order=ZOrder.NONE,
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
        button_z_order: ZOrder = ZOrder.NONE,
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
                "z_order": button_z_order.value,
            }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        # For any button in contact with either the robot or the stick, change
        # color to pressed.
        super().step(action)
        assert self._current_state is not None
        newly_pressed_buttons: set[Object] = set()
        obj_name_to_obj = {o.name: o for o in self._current_state}
        robot = obj_name_to_obj["robot"]
        stick = obj_name_to_obj["stick"]
        for button in self._current_state.get_objects(CircleType):
            if state_has_collision(
                self._current_state,
                {button},
                {robot, stick},
                self._static_object_body_cache,
                ignore_z_orders=True,
            ):
                newly_pressed_buttons.add(button)
        # Change colors.
        for button in newly_pressed_buttons:
            self._current_state.set(button, "color_r", self._spec.button_pressed_rgb[0])
            self._current_state.set(button, "color_g", self._spec.button_pressed_rgb[1])
            self._current_state.set(button, "color_b", self._spec.button_pressed_rgb[2])
            # This is hacky, but it's the easiest way to force re-rendering.
            del self._static_object_body_cache[button]

        reward, terminated = self._get_reward_and_done()
        truncated = False  # no maximum horizon, by default
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_reward_and_done(self) -> tuple[float, bool]:
        terminated = True
        assert self._current_state is not None
        for button in self._current_state.get_objects(CircleType):
            color = (
                self._current_state.get(button, "color_r"),
                self._current_state.get(button, "color_g"),
                self._current_state.get(button, "color_b"),
            )
            if not np.allclose(color, self._spec.button_pressed_rgb):
                terminated = False
                break
        return -1.0, terminated


class StickButton2DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Stick button 2D env."""

    def __init__(
        self,
        num_buttons: int = 2,
        spec: StickButton2DEnvSpec = StickButton2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__()
        # At the moment, all the real logic for this environment is defined
        # externally. We create that environment and then add some additional
        # code to vectorize observations, making it easier for RL approaches.
        self._geom2d_env = ObjectCentricStickButton2DEnv(
            num_buttons=num_buttons, spec=spec, **kwargs
        )
        # Create a Box version of the observation space by assuming a constant
        # number of buttons (and thus a constant number of objects).
        assert isinstance(self._geom2d_env.observation_space, ObjectCentricStateSpace)
        # Make observation vectors start with the robot, then the stick,
        # then buttons. Don't include the walls or table because those are
        # universally constant.
        exemplar_object_centric_state, _ = self._geom2d_env.reset()
        obj_name_to_obj = {o.name: o for o in exemplar_object_centric_state}
        self._constant_objects = [
            obj_name_to_obj["robot"],
            obj_name_to_obj["stick"],
        ]
        button_names = {o for o in obj_name_to_obj if o.startswith("button")}
        assert len(button_names) == num_buttons
        for button_name in sorted(button_names):
            self._constant_objects.append(obj_name_to_obj[button_name])
        self.observation_space = self._geom2d_env.observation_space.to_box(
            self._constant_objects, Geom2DRobotEnvTypeFeatures
        )
        self.action_space = self._geom2d_env.action_space
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        assert isinstance(self.action_space, CRVRobotActionSpace)
        # Add descriptions to metadata for doc generation.
        env_md = create_env_description(num_buttons)
        obs_md = self.observation_space.create_markdown_description()
        act_md = self.action_space.create_markdown_description()
        reward_md = "A penalty of -1.0 is given at every time step until termination, which occurs when all buttons have been pressed.\n"  # pylint: disable=line-too-long
        references_md = 'This environment is based on the Stick Button environment that was originally introduced in "Learning Neuro-Symbolic Skills for Bilevel Planning" (Silver et al., CoRL 2022). This version is simplified in that the robot or stick need only make contact with a button to press it, rather than explicitly pressing. Also, the full stick works for pressing, not just the tip.\n'  # pylint: disable=line-too-long
        self.metadata = {
            "description": env_md,
            "observation_space_description": obs_md,
            "action_space_description": act_md,
            "reward_description": reward_md,
            "references": references_md,
            "render_modes": self._geom2d_env.metadata["render_modes"],
            "render_fps": self._geom2d_env.metadata["render_fps"],
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


def create_env_description(num_buttons: int = 2) -> str:
    """Create a human-readable environment description."""
    # pylint: disable=line-too-long
    return f"""A 2D environment where the goal is to touch all buttons, possibly by using a stick for buttons that are out of the robot's direct reach.

In this environment, there are always {num_buttons} buttons.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector.
"""
