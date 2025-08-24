"""Environment where a block must be placed on an obstructed target."""

from dataclasses import dataclass

import numpy as np
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict

from prbench.envs.geom2d.base_env import Geom2DRobotEnv, Geom2DRobotEnvSpec
from prbench.envs.geom2d.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from prbench.envs.geom2d.structs import MultiBody2D, ZOrder
from prbench.envs.geom2d.utils import (
    PURPLE,
    CRVRobotActionSpace,
    SE2Pose,
    create_walls_from_world_boundaries,
    is_on,
    sample_se2_pose,
    state_has_collision,
)

TargetBlockType = Type("target_block", parent=RectangleType)
TargetSurfaceType = Type("target_surface", parent=RectangleType)
Geom2DRobotEnvTypeFeatures[TargetBlockType] = list(
    Geom2DRobotEnvTypeFeatures[RectangleType]
)
Geom2DRobotEnvTypeFeatures[TargetSurfaceType] = list(
    Geom2DRobotEnvTypeFeatures[RectangleType]
)


@dataclass(frozen=True)
class Obstruction2DEnvSpec(Geom2DRobotEnvSpec):
    """Scene specification for Obstruction2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = (1 + np.sqrt(5)) / 2  # golden ratio :)
    world_min_y: float = 0.0
    world_max_y: float = 1.0

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
            world_min_x + robot_base_radius,
            world_max_y - 2 * robot_base_radius,
            -np.pi / 2,
        ),
        SE2Pose(
            world_max_x - robot_base_radius, world_max_y - robot_base_radius, -np.pi / 2
        ),
    )

    # Table hyperparameters.
    table_rgb: tuple[float, float, float] = (0.75, 0.75, 0.75)
    table_height: float = 0.1
    table_width: float = world_max_x - world_min_x
    # The table pose is defined relative to the bottom left hand corner.
    table_pose: SE2Pose = SE2Pose(world_min_x, world_min_y, 0.0)

    # Target surface hyperparameters.
    target_surface_rgb: tuple[float, float, float] = PURPLE
    target_surface_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(world_min_x + robot_base_radius, table_pose.y, 0.0),
        SE2Pose(world_max_x - robot_base_radius, table_pose.y, 0.0),
    )
    target_surface_height: float = table_height
    # This adds to the width of the target block.
    target_surface_width_addition_bounds: tuple[float, float] = (
        robot_base_radius / 5,
        robot_base_radius / 2,
    )

    # Target block hyperparameters.
    target_block_rgb: tuple[float, float, float] = PURPLE
    target_block_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
        SE2Pose(
            world_max_x - robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
    )
    target_block_height_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    target_block_width_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )

    # Obstruction hyperparameters.
    obstruction_rgb: tuple[float, float, float] = (0.75, 0.1, 0.1)
    obstruction_init_pose_bounds = (
        SE2Pose(
            world_min_x + robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
        SE2Pose(
            world_max_x - robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
    )
    obstruction_height_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    obstruction_width_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    # NOTE: this is not the "real" probability, but rather, the probability
    # that we will attempt to sample the obstruction somewhere on the target
    # surface during each round of rejection sampling during reset().
    obstruction_init_on_target_prob: float = 0.9

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 250


class Obstruction2DEnv(Geom2DRobotEnv):
    """Environment where a block must be placed on an obstructed target."""

    def __init__(
        self,
        num_obstructions: int = 2,
        spec: Obstruction2DEnvSpec = Obstruction2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__(spec, **kwargs)
        self._num_obstructions = num_obstructions
        self._spec: Obstruction2DEnvSpec = spec  # for type checking
        initial_state_dict = self._create_constant_initial_state_dict()
        self._initial_constant_state = create_state_from_dict(
            initial_state_dict, Geom2DRobotEnvTypeFeatures
        )

    def _sample_initial_state(self) -> ObjectCentricState:
        assert self._initial_constant_state is not None
        static_objects = set(self._initial_constant_state)
        assert not state_has_collision(
            self._initial_constant_state,
            static_objects,
            static_objects,
            {},
        )
        n = self._spec.max_initial_state_sampling_attempts
        for _ in range(n):
            # Sample all randomized values.
            robot_pose = sample_se2_pose(
                self._spec.robot_init_pose_bounds, self.np_random
            )
            target_block_pose = sample_se2_pose(
                self._spec.target_block_init_pose_bounds, self.np_random
            )
            target_block_shape = (
                self.np_random.uniform(*self._spec.target_block_width_bounds),
                self.np_random.uniform(*self._spec.target_block_height_bounds),
            )
            target_surface_pose = sample_se2_pose(
                self._spec.target_surface_init_pose_bounds, self.np_random
            )
            target_surface_width_addition = self.np_random.uniform(
                *self._spec.target_surface_width_addition_bounds
            )
            target_surface_shape = (
                target_block_shape[0] + target_surface_width_addition,
                self._spec.target_surface_height,
            )
            obstructions: list[tuple[SE2Pose, tuple[float, float]]] = []
            for _ in range(self._num_obstructions):
                obstruction_shape = (
                    self.np_random.uniform(*self._spec.obstruction_width_bounds),
                    self.np_random.uniform(*self._spec.obstruction_height_bounds),
                )
                obstruction_init_on_target = (
                    self.np_random.uniform()
                    < self._spec.obstruction_init_on_target_prob
                )
                if obstruction_init_on_target:
                    old_lb, old_ub = self._spec.obstruction_init_pose_bounds
                    new_x_lb = target_surface_pose.x - obstruction_shape[0]
                    new_x_ub = target_surface_pose.x + target_surface_shape[0]
                    new_lb = SE2Pose(new_x_lb, old_lb.y, old_lb.theta)
                    new_ub = SE2Pose(new_x_ub, old_ub.y, old_ub.theta)
                    pose_bounds = (new_lb, new_ub)
                else:
                    pose_bounds = self._spec.obstruction_init_pose_bounds
                obstruction_pose = sample_se2_pose(pose_bounds, self.np_random)
                obstructions.append((obstruction_pose, obstruction_shape))
            state = self._create_initial_state(
                robot_pose,
                target_surface_pose,
                target_surface_shape,
                target_block_pose,
                target_block_shape,
                obstructions,
            )
            # Check initial state validity: goal not satisfied and no collisions.
            if self._target_satisfied(state, {}):
                continue
            full_state = state.copy()
            full_state.data.update(self._initial_constant_state.data)
            all_objects = set(full_state)
            if state_has_collision(full_state, all_objects, all_objects, {}):
                continue
            return state
        raise RuntimeError(f"Failed to sample initial state after {n} attempts")

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the table.
        table = Object("table", RectangleType)
        init_state_dict[table] = {
            "x": self._spec.table_pose.x,
            "y": self._spec.table_pose.y,
            "theta": self._spec.table_pose.theta,
            "width": self._spec.table_width,
            "height": self._spec.table_height,
            "static": True,
            "color_r": self._spec.table_rgb[0],
            "color_g": self._spec.table_rgb[1],
            "color_b": self._spec.table_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

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
        robot_pose: SE2Pose,
        target_surface_pose: SE2Pose,
        target_surface_shape: tuple[float, float],
        target_block_pose: SE2Pose,
        target_block_shape: tuple[float, float],
        obstructions: list[tuple[SE2Pose, tuple[float, float]]],
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
            "base_radius": self._spec.robot_base_radius,
            "arm_joint": self._spec.robot_base_radius,  # arm is fully retracted
            "arm_length": self._spec.robot_arm_length,
            "vacuum": 0.0,  # vacuum is off
            "gripper_height": self._spec.robot_gripper_height,
            "gripper_width": self._spec.robot_gripper_width,
        }

        # Create the target surface.
        target_surface = Object("target_surface", TargetSurfaceType)
        init_state_dict[target_surface] = {
            "x": target_surface_pose.x,
            "y": target_surface_pose.y,
            "theta": target_surface_pose.theta,
            "width": target_surface_shape[0],
            "height": target_surface_shape[1],
            "static": True,
            "color_r": self._spec.target_surface_rgb[0],
            "color_g": self._spec.target_surface_rgb[1],
            "color_b": self._spec.target_surface_rgb[2],
            "z_order": ZOrder.NONE.value,
        }

        # Create the target block.
        target_block = Object("target_block", TargetBlockType)
        init_state_dict[target_block] = {
            "x": target_block_pose.x,
            "y": target_block_pose.y,
            "theta": target_block_pose.theta,
            "width": target_block_shape[0],
            "height": target_block_shape[1],
            "static": False,
            "color_r": self._spec.target_block_rgb[0],
            "color_g": self._spec.target_block_rgb[1],
            "color_b": self._spec.target_block_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

        # Create obstructions.
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

    def _target_satisfied(
        self,
        state: ObjectCentricState,
        static_object_body_cache: dict[Object, MultiBody2D],
    ) -> bool:
        target_objects = state.get_objects(TargetBlockType)
        assert len(target_objects) == 1
        target_object = target_objects[0]
        target_surfaces = state.get_objects(TargetSurfaceType)
        assert len(target_surfaces) == 1
        target_surface = target_surfaces[0]
        return is_on(state, target_object, target_surface, static_object_body_cache)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        # Terminate when target object is on the target surface. Give -1 reward
        # at every step until then to encourage fast completion.
        assert self._current_state is not None
        terminated = self._target_satisfied(
            self._current_state,
            self._static_object_body_cache,
        )
        return -1.0, terminated
