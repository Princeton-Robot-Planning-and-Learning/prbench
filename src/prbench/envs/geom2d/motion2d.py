"""Environment where only 2D motion planning is needed to reach a goal region."""

from dataclasses import dataclass

import numpy as np
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict

from prbench.core import ConstantObjectPRBenchEnv, FinalConfigMeta
from prbench.envs.geom2d.base_env import (
    Geom2DRobotEnvConfig,
    ObjectCentricGeom2DRobotEnv,
)
from prbench.envs.geom2d.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from prbench.envs.geom2d.structs import SE2Pose, ZOrder
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    create_walls_from_world_boundaries,
    rectangle_object_to_geom,
)
from prbench.envs.utils import BLACK, PURPLE, sample_se2_pose, state_2d_has_collision

TargetRegionType = Type("target_region", parent=RectangleType)
Geom2DRobotEnvTypeFeatures[TargetRegionType] = list(
    Geom2DRobotEnvTypeFeatures[RectangleType]
)


@dataclass(frozen=True)
class Motion2DEnvConfig(Geom2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Config for Motion2DEnv()."""

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
    # The robot starts on the left side of the screen.
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + 2.5 * robot_base_radius,
            world_min_y + 3 * robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            world_min_x + 3 * robot_base_radius,
            world_max_y - 3 * robot_base_radius,
            np.pi,
        ),
    )

    # Target region hyperparameters.
    target_region_rgb: tuple[float, float, float] = PURPLE
    # The target region starts on the right side of the screen.
    target_region_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_max_x - 3 * robot_base_radius,
            world_min_y + 3 * robot_base_radius,
            0,
        ),
        SE2Pose(
            world_max_x - 2.5 * robot_base_radius,
            world_max_y - 3 * robot_base_radius,
            0,
        ),
    )
    target_region_shape: tuple[float, float] = (
        2.5 * robot_base_radius,
        2.5 * robot_base_radius,
    )

    # Obstacle hyperparameters.
    obstacle_rgb: tuple[float, float, float] = BLACK
    obstacle_width: float = robot_base_radius / 10
    obstacle_min_x: float = robot_init_pose_bounds[1].x + 2 * robot_base_radius
    obstacle_max_x: float = target_region_init_bounds[0].x - (
        2 * robot_base_radius + obstacle_width
    )
    obstacle_passage_height_bounds: tuple[float, float] = (
        2.5 * robot_base_radius,
        4.0 * robot_base_radius,
    )
    obstacle_passage_y_bounds: tuple[float, float] = (
        world_min_y + 2 * robot_base_radius,
        world_max_y - 2 * robot_base_radius,
    )

    # For rendering.
    render_dpi: int = 300
    render_fps: int = 20


class ObjectCentricMotion2DEnv(ObjectCentricGeom2DRobotEnv[Motion2DEnvConfig]):
    """Only 2D motion planning is needed to reach a goal region.

    This is an object-centric environment. The vectorized version with Box spaces is
    defined below.
    """

    def __init__(
        self,
        num_passages: int = 2,
        config: Motion2DEnvConfig = Motion2DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._num_passages = num_passages

    def _sample_initial_state(self) -> ObjectCentricState:
        # Sample initial robot pose.
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)
        # Sample initial target region pose.
        target_region_pose = sample_se2_pose(
            self.config.target_region_init_bounds, self.np_random
        )
        # Sample obstacles to form vertical narrow passages.
        obstacles: list[tuple[SE2Pose, tuple[float, float]]] = []
        min_x = self.config.obstacle_min_x
        max_x = self.config.obstacle_max_x
        if self._num_passages > 1:
            x_dist_between_passages = (
                max_x - min_x - self._num_passages * self.config.obstacle_width
            ) / (self._num_passages - 1)
            assert x_dist_between_passages > 2 * self.config.robot_base_radius
        else:
            x_dist_between_passages = 0.0  # not used
        for i in range(self._num_passages):
            # Sample the passage parameters.
            passage_y = self.np_random.uniform(*self.config.obstacle_passage_y_bounds)
            passage_height = self.np_random.uniform(
                *self.config.obstacle_passage_height_bounds
            )
            x = min_x + i * (self.config.obstacle_width + x_dist_between_passages)
            # Create the bottom obstacle.
            y = self.config.world_min_y
            height = passage_y - y
            pose = SE2Pose(x, y, 0.0)
            shape = (self.config.obstacle_width, height)
            obstacles.append((pose, shape))
            # Create the top obstacle.
            y = y + height + passage_height
            pose = SE2Pose(x, y, 0.0)
            height = self.config.world_max_y - y
            shape = (self.config.obstacle_width, height)
            obstacles.append((pose, shape))

        state = self._create_initial_state(robot_pose, target_region_pose, obstacles)

        # Sanity check.
        robot = state.get_objects(CRVRobotType)[0]
        target_region = state.get_objects(TargetRegionType)[0]
        assert not state_2d_has_collision(state, {robot, target_region}, set(state), {})

        return state

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create room walls.
        assert isinstance(self.action_space, CRVRobotActionSpace)
        min_dx, min_dy = self.action_space.low[:2]
        max_dx, max_dy = self.action_space.high[:2]
        wall_state_dict = create_walls_from_world_boundaries(
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        )
        init_state_dict.update(wall_state_dict)

        return init_state_dict

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        target_region_pose: SE2Pose | None = None,
        obstacles: list[tuple[SE2Pose, tuple[float, float]]] | None = None,
    ) -> ObjectCentricState:
        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the robot.
        robot = Object("robot", CRVRobotType)
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "y": robot_pose.y,
            "theta": robot_pose.theta,
            "base_radius": self.config.robot_base_radius,
            "arm_joint": self.config.robot_base_radius,  # arm is fully retracted
            "arm_length": self.config.robot_arm_length,
            "vacuum": 0.0,  # vacuum is off
            "gripper_height": self.config.robot_gripper_height,
            "gripper_width": self.config.robot_gripper_width,
        }

        # Create the target region.
        if target_region_pose is not None:
            target_region = Object("target_region", TargetRegionType)
            init_state_dict[target_region] = {
                "x": target_region_pose.x,
                "y": target_region_pose.y,
                "theta": target_region_pose.theta,
                "width": self.config.target_region_shape[0],
                "height": self.config.target_region_shape[1],
                "static": True,  # NOTE
                "color_r": self.config.target_region_rgb[0],
                "color_g": self.config.target_region_rgb[1],
                "color_b": self.config.target_region_rgb[2],
                "z_order": ZOrder.NONE.value,
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
                    "color_r": self.config.obstacle_rgb[0],
                    "color_g": self.config.obstacle_rgb[1],
                    "color_b": self.config.obstacle_rgb[2],
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
        # NOTE: the type: ignore can be removed after refactor.
        target_region_geom = rectangle_object_to_geom(
            self._current_state,
            target_region,
            self._static_object_body_cache,
        )
        terminated = target_region_geom.contains_point(x, y)
        return -1.0, terminated


class Motion2DEnv(ConstantObjectPRBenchEnv):
    """Motion 2D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricGeom2DRobotEnv:
        return ObjectCentricMotion2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "target_region"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstacle"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        num_passages = (len(self._constant_objects) - 2) // 2
        # pylint: disable=line-too-long
        if num_passages > 0:
            obstacle_sentence = f"\nIn this environment, there are always {num_passages} narrow passages.\n"
        else:
            obstacle_sentence = ""

        return f"""A 2D environment where the goal is to reach a target region while avoiding static obstacles.
{obstacle_sentence}
The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. The arm and vacuum do not need to be used in this environment.
"""

    def _create_reward_markdown_description(self) -> str:
        return "A penalty of -1.0 is given at every time step until termination, which occurs when the robot's position is within the target region.\n"  # pylint: disable=line-too-long

    def _create_references_markdown_description(self) -> str:
        return "Narrow passages are a classic challenge in motion planning.\n"  # pylint: disable=line-too-long
