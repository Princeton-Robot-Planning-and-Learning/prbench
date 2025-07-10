"""Environment where a robot must make coffee by pouring and placing objects."""

from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
from geom2drobotenvs.concepts import is_on, is_inside
from geom2drobotenvs.envs.base_env import Geom2DRobotEnv, Geom2DRobotEnvSpec
from geom2drobotenvs.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from geom2drobotenvs.structs import ZOrder
from geom2drobotenvs.utils import (
    BLACK,
    PURPLE,
    CRVRobotActionSpace,
    SE2Pose,
    create_walls_from_world_boundaries,
    get_suctioned_objects,
    sample_se2_pose,
    state_has_collision,
)

# Define custom colors for coffee making objects
BROWN = (0.6, 0.3, 0.1)  # Coffee pot
WHITE = (1.0, 1.0, 1.0)  # Coffee cup, sugar
CREAM_COLOR = (1.0, 0.95, 0.8)  # Cream container
LIGHT_BLUE = (0.6, 0.8, 1.0)  # Water pitcher
GREEN = (0.0, 0.8, 0.0)  # Coaster
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState, ObjectCentricStateSpace, Type
from relational_structs.spaces import ObjectCentricBoxSpace
from relational_structs.utils import create_state_from_dict

from prbench.utils import get_geom2d_crv_robot_action_from_gui_input

# Define new object types for coffee making (all rectangular for simplicity)
CoffeePotType = Type("coffee_pot", parent=RectangleType)
CoffeeCupType = Type("coffee_cup", parent=RectangleType)
CreamContainerType = Type("cream_container", parent=RectangleType)
SugarContainerType = Type("sugar_container", parent=RectangleType)
WaterPitcherType = Type("water_pitcher", parent=RectangleType)
CoasterType = Type("coaster", parent=RectangleType)

# Register type features
Geom2DRobotEnvTypeFeatures[CoffeePotType] = list(Geom2DRobotEnvTypeFeatures[RectangleType])
Geom2DRobotEnvTypeFeatures[CoffeeCupType] = list(Geom2DRobotEnvTypeFeatures[RectangleType])
Geom2DRobotEnvTypeFeatures[CreamContainerType] = list(Geom2DRobotEnvTypeFeatures[RectangleType])
Geom2DRobotEnvTypeFeatures[SugarContainerType] = list(Geom2DRobotEnvTypeFeatures[RectangleType])
Geom2DRobotEnvTypeFeatures[WaterPitcherType] = list(Geom2DRobotEnvTypeFeatures[RectangleType])
Geom2DRobotEnvTypeFeatures[CoasterType] = list(Geom2DRobotEnvTypeFeatures[RectangleType])


@dataclass(frozen=True)
class CoffeeMaking2DEnvSpec(Geom2DRobotEnvSpec):
    """Scene specification for CoffeeMaking2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 3.0
    world_min_y: float = 0.0
    world_max_y: float = 2.0

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
    robot_base_radius: float = 0.08
    robot_arm_length: float = 2 * robot_base_radius
    robot_gripper_height: float = 0.06
    robot_gripper_width: float = 0.01
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(0.2, 0.2, -np.pi),
        SE2Pose(0.8, 0.8, np.pi),
    )

    # Coffee pot hyperparameters.
    coffee_pot_rgb: tuple[float, float, float] = BROWN
    coffee_pot_shape: tuple[float, float] = (0.15, 0.15)
    coffee_pot_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(0.5, 0.5, -np.pi),
        SE2Pose(1.5, 1.0, np.pi),
    )

    # Coffee cup hyperparameters.
    coffee_cup_rgb: tuple[float, float, float] = WHITE
    coffee_cup_shape: tuple[float, float] = (0.12, 0.12)  # width, height
    coffee_cup_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(0.3, 0.3, -np.pi),
        SE2Pose(1.0, 1.0, np.pi),
    )

    # Water pitcher hyperparameters.
    water_pitcher_rgb: tuple[float, float, float] = LIGHT_BLUE
    water_pitcher_shape: tuple[float, float] = (0.12, 0.18)
    water_pitcher_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(1.8, 0.3, -np.pi),
        SE2Pose(2.5, 1.0, np.pi),
    )

    # Cream container hyperparameters.
    cream_container_rgb: tuple[float, float, float] = CREAM_COLOR
    cream_container_shape: tuple[float, float] = (0.10, 0.10)  # width, height
    cream_container_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(0.3, 1.2, -np.pi),
        SE2Pose(1.0, 1.7, np.pi),
    )

    # Sugar container hyperparameters.
    sugar_container_rgb: tuple[float, float, float] = WHITE
    sugar_container_shape: tuple[float, float] = (0.10, 0.10)  # width, height
    sugar_container_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(1.0, 1.2, -np.pi),
        SE2Pose(1.7, 1.7, np.pi),
    )

    # Green coaster (target) hyperparameters.
    coaster_rgb: tuple[float, float, float] = GREEN
    coaster_shape: tuple[float, float] = (0.16, 0.16)  # width, height
    coaster_pose: SE2Pose = SE2Pose(2.3, 0.5, 0.0)  # Fixed position near edge

    # For initial state sampling.
    max_init_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 150


class ObjectCentricCoffeeMaking2DEnv(Geom2DRobotEnv):
    """Environment where a robot must make coffee with pouring actions.

    This is an object-centric environment. The vectorized version with
    Box spaces is defined below.
    """

    def __init__(
        self,
        include_cream: bool = True,
        include_sugar: bool = True,
        spec: CoffeeMaking2DEnvSpec = CoffeeMaking2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__(spec, **kwargs)
        self._include_cream = include_cream
        self._include_sugar = include_sugar
        self._spec: CoffeeMaking2DEnvSpec = spec  # for type checking
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": 10,
        }

    def _sample_initial_state(self) -> ObjectCentricState:
        initial_state_dict = self._create_constant_initial_state_dict()
        
        # Sample robot pose
        robot_pose = sample_se2_pose(self._spec.robot_init_pose_bounds, self.np_random)
        
        # Sample other object poses and check for collisions
        for _ in range(self._spec.max_init_sampling_attempts):
            coffee_pot_pose = sample_se2_pose(
                self._spec.coffee_pot_init_bounds, self.np_random
            )
            coffee_cup_pose = sample_se2_pose(
                self._spec.coffee_cup_init_bounds, self.np_random
            )
            water_pitcher_pose = sample_se2_pose(
                self._spec.water_pitcher_init_bounds, self.np_random
            )
            
            poses = {
                'coffee_pot': coffee_pot_pose,
                'coffee_cup': coffee_cup_pose,
                'water_pitcher': water_pitcher_pose,
            }
            
            if self._include_cream:
                cream_pose = sample_se2_pose(
                    self._spec.cream_container_init_bounds, self.np_random
                )
                poses['cream'] = cream_pose
            
            if self._include_sugar:
                sugar_pose = sample_se2_pose(
                    self._spec.sugar_container_init_bounds, self.np_random
                )
                poses['sugar'] = sugar_pose
            
            state = self._create_initial_state(initial_state_dict, robot_pose, poses)
            
            # Check for collisions between all objects
            if not self._has_collisions(state):
                return state
        
        raise RuntimeError("Failed to sample collision-free initial state.")

    def _has_collisions(self, state: ObjectCentricState) -> bool:
        """Check if any objects in the state have collisions."""
        objects = list(state)
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                if state_has_collision(state, {obj1}, {obj2}, {}):
                    return True
        return False

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

        # Create the green coaster (target) at a fixed position
        coaster = Object("coaster", CoasterType)
        init_state_dict[coaster] = {
            "x": self._spec.coaster_pose.x,
            "y": self._spec.coaster_pose.y,
            "theta": self._spec.coaster_pose.theta,
            "width": self._spec.coaster_shape[0],
            "height": self._spec.coaster_shape[1],
            "static": True,
            "color_r": self._spec.coaster_rgb[0],
            "color_g": self._spec.coaster_rgb[1],
            "color_b": self._spec.coaster_rgb[2],
            "z_order": ZOrder.SURFACE.value,
        }

        return init_state_dict

    def _create_initial_state(
        self,
        constant_initial_state_dict: dict[Object, dict[str, float]],
        robot_pose: SE2Pose,
        object_poses: dict[str, SE2Pose],
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

        # Create coffee pot
        if 'coffee_pot' in object_poses:
            coffee_pot = Object("coffee_pot", CoffeePotType)
            pose = object_poses['coffee_pot']
            init_state_dict[coffee_pot] = {
                "x": pose.x,
                "y": pose.y,
                "theta": pose.theta,
                "width": self._spec.coffee_pot_shape[0],
                "height": self._spec.coffee_pot_shape[1],
                "static": False,
                "color_r": self._spec.coffee_pot_rgb[0],
                "color_g": self._spec.coffee_pot_rgb[1],
                "color_b": self._spec.coffee_pot_rgb[2],
                "z_order": ZOrder.ALL.value,
            }

        # Create coffee cup
        if 'coffee_cup' in object_poses:
            coffee_cup = Object("coffee_cup", CoffeeCupType)
            pose = object_poses['coffee_cup']
            init_state_dict[coffee_cup] = {
                "x": pose.x,
                "y": pose.y,
                "theta": pose.theta,
                "width": self._spec.coffee_cup_shape[0],
                "height": self._spec.coffee_cup_shape[1],
                "static": False,
                "color_r": self._spec.coffee_cup_rgb[0],
                "color_g": self._spec.coffee_cup_rgb[1],
                "color_b": self._spec.coffee_cup_rgb[2],
                "z_order": ZOrder.ALL.value,
            }

        # Create water pitcher
        if 'water_pitcher' in object_poses:
            water_pitcher = Object("water_pitcher", WaterPitcherType)
            pose = object_poses['water_pitcher']
            init_state_dict[water_pitcher] = {
                "x": pose.x,
                "y": pose.y,
                "theta": pose.theta,
                "width": self._spec.water_pitcher_shape[0],
                "height": self._spec.water_pitcher_shape[1],
                "static": False,
                "color_r": self._spec.water_pitcher_rgb[0],
                "color_g": self._spec.water_pitcher_rgb[1],
                "color_b": self._spec.water_pitcher_rgb[2],
                "z_order": ZOrder.ALL.value,
            }

        # Create cream container if included
        if self._include_cream and 'cream' in object_poses:
            cream_container = Object("cream_container", CreamContainerType)
            pose = object_poses['cream']
            init_state_dict[cream_container] = {
                "x": pose.x,
                "y": pose.y,
                "theta": pose.theta,
                "width": self._spec.cream_container_shape[0],
                "height": self._spec.cream_container_shape[1],
                "static": False,
                "color_r": self._spec.cream_container_rgb[0],
                "color_g": self._spec.cream_container_rgb[1],
                "color_b": self._spec.cream_container_rgb[2],
                "z_order": ZOrder.ALL.value,
            }

        # Create sugar container if included
        if self._include_sugar and 'sugar' in object_poses:
            sugar_container = Object("sugar_container", SugarContainerType)
            pose = object_poses['sugar']
            init_state_dict[sugar_container] = {
                "x": pose.x,
                "y": pose.y,
                "theta": pose.theta,
                "width": self._spec.sugar_container_shape[0],
                "height": self._spec.sugar_container_shape[1],
                "static": False,
                "color_r": self._spec.sugar_container_rgb[0],
                "color_g": self._spec.sugar_container_rgb[1],
                "color_b": self._spec.sugar_container_rgb[2],
                "z_order": ZOrder.ALL.value,
            }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Reward function for coffee making task.
        
        The task is complete when the coffee cup is placed on the green coaster.
        For simplicity, we assume the coffee making process is successful if
        the cup reaches the coaster (the actual pouring/mixing is abstracted).
        """
        assert self._current_state is not None
        
        # Get the coffee cup and coaster
        try:
            coffee_cup = self._current_state.get_objects(CoffeeCupType)[0]
            coaster = self._current_state.get_objects(CoasterType)[0]
        except IndexError:
            # If objects don't exist, something went wrong
            return -1.0, False
        
        # Check if coffee cup is on the coaster
        cup_on_coaster = is_on(self._current_state, coffee_cup, coaster, {})
        
        # Task is complete when cup is on coaster
        terminated = cup_on_coaster
        
        # Reward structure: -1 per timestep until completion
        reward = 0.0 if terminated else -1.0
        
        return reward, terminated


class CoffeeMaking2DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Coffee Making 2D environment."""

    def __init__(
        self,
        include_cream: bool = True,
        include_sugar: bool = True,
        spec: CoffeeMaking2DEnvSpec = CoffeeMaking2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Create the object-centric environment
        self._geom2d_env = ObjectCentricCoffeeMaking2DEnv(
            include_cream=include_cream,
            include_sugar=include_sugar,
            spec=spec,
            **kwargs
        )
        
        # Create a Box version of the observation space
        assert isinstance(self._geom2d_env.observation_space, ObjectCentricStateSpace)
        
        # Get an exemplar state to determine object ordering
        exemplar_object_centric_state, _ = self._geom2d_env.reset()
        obj_name_to_obj = {o.name: o for o in exemplar_object_centric_state}
        
        # Define constant object ordering (exclude walls as they're constant)
        self._constant_objects = [obj_name_to_obj["robot"]]
        
        # Add objects that are always present
        required_objects = ["coffee_pot", "coffee_cup", "water_pitcher", "coaster"]
        for obj_name in required_objects:
            if obj_name in obj_name_to_obj:
                self._constant_objects.append(obj_name_to_obj[obj_name])
        
        # Add optional objects
        if include_cream and "cream_container" in obj_name_to_obj:
            self._constant_objects.append(obj_name_to_obj["cream_container"])
        if include_sugar and "sugar_container" in obj_name_to_obj:
            self._constant_objects.append(obj_name_to_obj["sugar_container"])
        
        self.observation_space = self._geom2d_env.observation_space.to_box(
            self._constant_objects, Geom2DRobotEnvTypeFeatures
        )
        self.action_space = self._geom2d_env.action_space
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        assert isinstance(self.action_space, CRVRobotActionSpace)
        
        # Add descriptions to metadata for doc generation.
        env_md = create_env_description(include_cream, include_sugar)
        obs_md = self.observation_space.create_markdown_description()
        act_md = self.action_space.create_markdown_description()
        reward_md = create_reward_description()
        references_md = create_references()
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


def create_env_description(include_cream: bool = True, include_sugar: bool = True) -> str:
    """Create a human-readable environment description."""
    
    ingredients = []
    if include_cream:
        ingredients.append("cream")
    if include_sugar:
        ingredients.append("sugar")
    
    ingredient_str = ""
    if ingredients:
        if len(ingredients) == 1:
            ingredient_str = f" The robot can optionally add {ingredients[0]} to the coffee."
        elif len(ingredients) == 2:
            ingredient_str = f" The robot can optionally add {ingredients[0]} and {ingredients[1]} to the coffee."
    
    return f"""A 2D kitchen environment where the goal is to make coffee and place the finished cup on a green coaster.

The robot must manipulate various objects including a coffee pot, water pitcher, coffee cup, and optional ingredients.{ingredient_str}

The task is complete when the coffee cup is successfully placed on the green coaster near the edge of the workspace.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.
"""


def create_reward_description() -> str:
    """Create a human-readable description of environment rewards."""
    return """A penalty of -1.0 is given at every time step until termination, which occurs when the coffee cup is placed on the green coaster. The task encourages efficient completion of the coffee making process."""


def create_references() -> str:
    """Create a human-readable reference section."""
    return """This environment is inspired by the Kitchen2D framework from "Active model learning and diverse action sampling for task and motion planning" (Wang et al., IROS 2018) and coffee-making robotics research. The task demonstrates complex manipulation requiring sequential actions and precise pouring behaviors.""" 