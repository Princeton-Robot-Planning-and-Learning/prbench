"""Environment where a block must be retrieved amidst clutter."""

import inspect
from typing import Any

import gymnasium
import numpy as np
from geom2drobotenvs.object_types import Geom2DRobotEnvTypeFeatures
from geom2drobotenvs.utils import CRVRobotActionSpace
from numpy.typing import NDArray
from relational_structs import ObjectCentricStateSpace
from relational_structs.spaces import ObjectCentricBoxSpace

from dataclasses import dataclass

import numpy as np
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict

from geom2drobotenvs.concepts import is_on
from geom2drobotenvs.envs.base_env import Geom2DRobotEnv, Geom2DRobotEnvSpec
from geom2drobotenvs.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from geom2drobotenvs.structs import MultiBody2D, ZOrder
from geom2drobotenvs.utils import (
    PURPLE,
    CRVRobotActionSpace,
    SE2Pose,
    create_walls_from_world_boundaries,
    sample_se2_pose,
    state_has_collision,
)

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
    target_block_shape: tuple[float, float] = (2 * robot_gripper_height, 2 * robot_gripper_height)


    # For rendering.
    render_dpi: int = 150


class ObjectCentricClutter2DEnv(Geom2DRobotEnv):
    """Environment where a block must be retrieved amidst clutter.
    
    This is an object-centric environment. The vectorized version with Box
    spaces is defined below (TODO!!!!).
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
        robot_pose = sample_se2_pose(
            self._spec.robot_init_pose_bounds, self.np_random
        )
        state = self._create_initial_state(initial_state_dict, robot_pose)
        robot = state.get_objects(CRVRobotType)[0]
        assert not state_has_collision(state, {robot}, static_objects, {})
        # Sample target pose and check for collisions with robot and static objects.
        while True:
            target_pose = sample_se2_pose(
                self._spec.target_block_init_bounds, self.np_random
            )
            state = self._create_initial_state(
                initial_state_dict,
                robot_pose,
                target_pose=target_pose,
            )
            target_block = state.get_objects(TargetBlockType)[0]
            if not state_has_collision(state, {target_block}, {robot} | static_objects, {}):
                break
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
    
    def _create_initial_state(self, constant_initial_state_dict: dict[Object, dict[str, float]],
        robot_pose: SE2Pose,
        target_pose: SE2Pose | None = None) -> ObjectCentricState:
        
        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        init_state_dict = constant_initial_state_dict.copy()

        # Create the robot.
        robot = CRVRobotType("robot")
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
            target_block = TargetBlockType("target_block")
            init_state_dict[target_block] = {
                "x": target_pose.x,
                "y": target_pose.y,
                "theta": target_pose.theta,
                "width": self._spec.target_block_shape[0],
                "height":self._spec.target_block_shape[1],
                "static": False,
                "color_r": self._spec.target_block_rgb[0],
                "color_g": self._spec.target_block_rgb[1],
                "color_b": self._spec.target_block_rgb[2],
                "z_order": ZOrder.ALL.value,
            }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        # TODO
        terminated = False
        return -1.0, terminated