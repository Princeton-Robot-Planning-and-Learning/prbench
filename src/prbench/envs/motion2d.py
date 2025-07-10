"""Environment where only 2D motion planning is needed to reach a goal region."""

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
from geom2drobotenvs.utils import rectangle_object_to_geom

from geom2drobotenvs.structs import ZOrder
from geom2drobotenvs.utils import (
    PURPLE,
    BLACK,
    CRVRobotActionSpace,
    SE2Pose,
    create_walls_from_world_boundaries,
    sample_se2_pose,
    state_has_collision,
)
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState, ObjectCentricStateSpace, Type
from relational_structs.spaces import ObjectCentricBoxSpace
from relational_structs.utils import create_state_from_dict

from prbench.utils import get_geom2d_crv_robot_action_from_gui_input

TargetRegionType = Type("target_region", parent=RectangleType)
Geom2DRobotEnvTypeFeatures[TargetRegionType] = list(
    Geom2DRobotEnvTypeFeatures[RectangleType]
)


@dataclass(frozen=True)
class Motion2DEnvSpec(Geom2DRobotEnvSpec):
    """Spec for Motion2DEnv()."""

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
    # NOTE: these should not be needed, but they are included for consistency
    # with the other geom2d environments.
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

    # Target region hyperparameters.
    target_region_rgb: tuple[float, float, float] = PURPLE
    target_region_init_bounds: tuple[SE2Pose, SE2Pose] = (
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
    target_region_shape: tuple[float, float] = (
        2.5 * robot_base_radius,
        2.5 * robot_base_radius,
    )

    # Obstacle hyperparameters.
    obstacle_rgb: tuple[float, float, float] = BLACK

    obstacle_height_bounds: tuple[float, float] = (
        robot_base_radius,
        5 * robot_base_radius,
    )
    obstacle_width_bounds: tuple[float, float] = (
        robot_base_radius,
        5 * robot_base_radius,
    )
    obstacle_pose_init_bounds: tuple[SE2Pose, SE2Pose] = (
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

    # For initial state sampling.
    max_init_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 150


class ObjectCentricMotion2DEnv(Geom2DRobotEnv):
    """Only 2D motion planning is needed to reach a goal region.

    This is an object-centric environment. The vectorized version with
    Box spaces is defined below.
    """

    def __init__(
        self,
        num_obstacles: int = 2,
        spec: Motion2DEnvSpec = Motion2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__(spec, **kwargs)
        self._num_obstacles = num_obstacles
        self._spec: Motion2DEnvSpec = spec  # for type checking
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
        # Sample target region and check for collisions with robot and static objects.
        for _ in range(self._spec.max_init_sampling_attempts):
            target_region_pose = sample_se2_pose(
                self._spec.target_region_init_bounds, self.np_random
            )
            state = self._create_initial_state(
                initial_state_dict,
                robot_pose,
                target_region_pose=target_region_pose,
            )
            target_region = state.get_objects(TargetRegionType)[0]
            if not state_has_collision(
                state, {target_region}, {robot} | static_objects, {}
            ):
                break
        else:
            raise RuntimeError("Failed to sample target pose.")
        # Sample obstacles one by one. Assume that the scene is never so dense
        # that we need to resample earlier choices.
        obstacles: list[tuple[SE2Pose, tuple[float, float]]] = []
        for _ in range(self._num_obstacles):
            for _ in range(self._spec.max_init_sampling_attempts):
                obstacle_pose = sample_se2_pose(self._spec.obstacle_pose_init_bounds, self.np_random)
                # Sample shape.
                obstacle_shape = (
                    self.np_random.uniform(*self._spec.obstacle_width_bounds),
                    self.np_random.uniform(*self._spec.obstacle_height_bounds),
                )
                possible_obstacles = obstacles + [
                    (obstacle_pose, obstacle_shape)
                ]
                state = self._create_initial_state(
                    initial_state_dict,
                    robot_pose,
                    target_region_pose=target_region_pose,
                    obstacles=possible_obstacles,
                )
                obj_name_to_obj = {o.name: o for o in state}
                new_obstacle = obj_name_to_obj[f"obstacle{len(obstacles)}"]
                assert new_obstacle.name.startswith("obstacle")
                if not state_has_collision(state, {new_obstacle}, set(state), {}):
                    break
            else:
                raise RuntimeError("Failed to sample obstacle pose.")
            # Update obstacles.
            obstacles = possible_obstacles
        
        # TODO HERE!!

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
        target_region_pose: SE2Pose | None = None,
        obstacles: list[tuple[SE2Pose, tuple[float, float]]] | None = None,
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

        # Create the target region.
        if target_region_pose is not None:
            target_region = Object("target_region", TargetRegionType)
            init_state_dict[target_region] = {
                "x": target_region_pose.x,
                "y": target_region_pose.y,
                "theta": target_region_pose.theta,
                "width": self._spec.target_region_shape[0],
                "height": self._spec.target_region_shape[1],
                "static": True,  # NOTE
                "color_r": self._spec.target_region_rgb[0],
                "color_g": self._spec.target_region_rgb[1],
                "color_b": self._spec.target_region_rgb[2],
                "z_order": ZOrder.ALL.value,
            }

        # Create the obstacles.
        if obstacles:
            for i, (obstacle_pose, obstacle_shape) in enumerate(obstacles):
                obstacle = Object(f"obstacle{i}", RectangleType)
                init_state_dict[obstacle] = {
                    "x": obstacle_pose.x,
                    "y": obstacle_pose.y,
                    "theta": obstacle_pose.theta,
                    "width": obstacle_shape[0],
                    "height": obstacle_shape[1],
                    "static": True,  # NOTE
                    "color_r": self._spec.obstacle_rgb[0],
                    "color_g": self._spec.obstacle_rgb[1],
                    "color_b": self._spec.obstacle_rgb[2],
                    "z_order": ZOrder.ALL.value,
                }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        # Terminate when the robot position is in the target region.
        assert self._current_state is not None
        robot = self._current_state.get_objects(CRVRobotType)[0]
        x = self._current_state.get(robot, "x")
        y = self._current_state.get(robot, "y")
        target_region = self._current_state.get_objects(TargetRegionType)[0]
        target_region_geom = rectangle_object_to_geom(self._current_state, target_region, self._static_object_body_cache)
        terminated = target_region_geom.contains_point(x, y)
        return -1.0, terminated
