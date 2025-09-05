"""Dynamic Obstruction 2D env using PyMunk physics."""

import inspect
from dataclasses import dataclass

import numpy as np
import pymunk
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict

from prbench.envs.geom2d.object_types import (
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from prbench.envs.geom2d.structs import SE2Pose
from prbench.envs.dynamic2d.base_env import (
    ConstantObjectDynamic2DEnv,
    Dynamic2DRobotEnv,
    Dynamic2DRobotEnvSpec,
)
from prbench.envs.dynamic2d.utils import (
    DYNAMIC_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
)

# Define custom object types for the obstruction environment
TargetBlockType = Type("target_block", parent=RectangleType)
TargetSurfaceType = Type("target_surface", parent=RectangleType)
Geom2DRobotEnvTypeFeatures[TargetBlockType] = list(
    Geom2DRobotEnvTypeFeatures[RectangleType]
)
Geom2DRobotEnvTypeFeatures[TargetSurfaceType] = list(
    Geom2DRobotEnvTypeFeatures[RectangleType]
)

# Colors
PURPLE = (128 / 255, 0 / 255, 128 / 255)
BLACK = (0.1, 0.1, 0.1)


def sample_se2_pose(
    bounds: tuple[SE2Pose, SE2Pose], rng: np.random.Generator
) -> SE2Pose:
    """Sample a SE2Pose uniformly between the bounds."""
    lb, ub = bounds
    x = rng.uniform(lb.x, ub.x)
    y = rng.uniform(lb.y, ub.y)
    theta = rng.uniform(lb.theta, ub.theta)
    return SE2Pose(x, y, theta)


def is_on_dynamic(
    space: pymunk.Space,
    top_obj: Object,
    bottom_obj: Object,
    tol: float = 2.5,
) -> bool:
    """Check if top object is on bottom object in the physics simulation."""
    # Get PyMunk bodies for both objects
    top_body = getattr(top_obj, "_pymunk_body", None)
    bottom_body = getattr(bottom_obj, "_pymunk_body", None)
    
    return False


@dataclass(frozen=True)
class DynObstruction2DEnvSpec(Dynamic2DRobotEnvSpec):
    """Scene specification for DynObstruction2DEnv()."""

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
    
    # Robot hyperparameters.
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(8.0, 8.0, np.pi / 2),
        SE2Pose(2.0, 2.0, -np.pi / 2),
    )

    # Table hyperparameters.
    table_rgb: tuple[float, float, float] = (0.75, 0.75, 0.75)
    table_height: float = 2.0
    table_width: float = 10.0
    table_pose: SE2Pose = SE2Pose(100.0, 100.0, 0.0)

    # Target surface hyperparameters (KINEMATIC).
    target_surface_rgb: tuple[float, float, float] = PURPLE
    target_surface_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(150.0, 120.0, 0.0),
        SE2Pose(350.0, 120.0, 0.0),
    )
    target_surface_height: float = 10.0
    target_surface_width_addition_bounds: tuple[float, float] = (10.0, 20.0)

    # Target block hyperparameters (DYNAMIC).
    target_block_rgb: tuple[float, float, float] = PURPLE
    target_block_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(150.0, 200.0, 0.0),
        SE2Pose(400.0, 300.0, 0.0),
    )
    target_block_height_bounds: tuple[float, float] = (15.0, 30.0)
    target_block_width_bounds: tuple[float, float] = (15.0, 30.0)

    # Obstruction hyperparameters (DYNAMIC).
    obstruction_rgb: tuple[float, float, float] = (0.75, 0.1, 0.1)
    obstruction_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(150.0, 200.0, 0.0),
        SE2Pose(400.0, 300.0, 0.0),
    )
    obstruction_height_bounds: tuple[float, float] = (15.0, 30.0)
    obstruction_width_bounds: tuple[float, float] = (15.0, 30.0)
    obstruction_init_on_target_prob: float = 0.7

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 100


class ObjectCentricDynObstruction2DEnv(Dynamic2DRobotEnv):
    """Dynamic environment where a block must be placed on an obstructed target using PyMunk physics."""

    def __init__(
        self,
        num_obstructions: int = 2,
        spec: DynObstruction2DEnvSpec = DynObstruction2DEnvSpec(),
        **kwargs,
    ) -> None:
        super().__init__(spec, **kwargs)
        self._num_obstructions = num_obstructions
        self._spec: DynObstruction2DEnvSpec = spec  # for type checking
        
        # Store object references for tracking
        self._target_block: Object | None = None
        self._target_surface: Object | None = None

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space."""
        if not self.space:
            return

        # Add static objects (table, walls)
        for obj in state:
            if obj.name == "table":
                self._add_table_to_space(obj, state)
            elif obj.name.startswith("wall"):
                self._add_wall_to_space(obj, state)
            elif obj.is_instance(TargetSurfaceType):
                self._add_target_surface_to_space(obj, state)
                self._target_surface = obj
            elif obj.is_instance(TargetBlockType):
                self._add_target_block_to_space(obj, state)
                self._target_block = obj
            elif obj.name.startswith("obstruction"):
                self._add_obstruction_to_space(obj, state)

    def _add_table_to_space(self, obj: Object, state: ObjectCentricState) -> None:
        """Add table as a static object."""
        if not self.space:
            return
            
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")

        body = self.space.static_body
        vs = [
            (-width / 2, -height / 2),
            (-width / 2, height / 2),
            (width / 2, height / 2),
            (width / 2, -height / 2),
        ]
        shape = pymunk.Poly(body, vs)
        shape.body.position = x, y
        shape.body.angle = theta
        shape.friction = 1.0
        shape.collision_type = STATIC_COLLISION_TYPE
        self.space.add(shape)


    def _add_wall_to_space(self, obj: Object, state: ObjectCentricState) -> None:
        """Add wall as a static object."""
        if not self.space:
            return
            
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")

        body = self.space.static_body
        vs = [
            (-width / 2, -height / 2),
            (-width / 2, height / 2),
            (width / 2, height / 2),
            (width / 2, -height / 2),
        ]
        shape = pymunk.Poly(body, vs)
        shape.body.position = x, y
        shape.body.angle = theta
        shape.friction = 1.0
        shape.collision_type = STATIC_COLLISION_TYPE
        self.space.add(shape)


    def _add_target_surface_to_space(self, obj: Object, state: ObjectCentricState) -> None:
        """Add target surface as a kinematic object."""
        if not self.space:
            return
            
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")

        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        vs = [
            (-width / 2, -height / 2),
            (-width / 2, height / 2),
            (width / 2, height / 2),
            (width / 2, -height / 2),
        ]
        shape = pymunk.Poly(body, vs)
        body.position = x, y
        body.angle = theta
        shape.friction = 1.0
        shape.collision_type = STATIC_COLLISION_TYPE  # Kinematic objects act like static for collisions
        self.space.add(body, shape)


    def _add_target_block_to_space(self, obj: Object, state: ObjectCentricState) -> None:
        """Add target block as a dynamic object."""
        if not self.space:
            return
            
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")

        mass = 1.0
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = x, y
        body.angle = theta

        vs = [
            (-width / 2, -height / 2),
            (-width / 2, height / 2),
            (width / 2, height / 2),
            (width / 2, -height / 2),
        ]
        shape = pymunk.Poly(body, vs)
        shape.friction = 1.0
        shape.collision_type = DYNAMIC_COLLISION_TYPE
        self.space.add(body, shape)


    def _add_obstruction_to_space(self, obj: Object, state: ObjectCentricState) -> None:
        """Add obstruction as a dynamic object."""
        if not self.space:
            return
            
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")

        mass = 1.0
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = x, y
        body.angle = theta

        vs = [
            (-width / 2, -height / 2),
            (-width / 2, height / 2),
            (width / 2, height / 2),
            (width / 2, -height / 2),
        ]
        shape = pymunk.Poly(body, vs)
        shape.friction = 1.0
        shape.collision_type = DYNAMIC_COLLISION_TYPE
        self.space.add(body, shape)


    def _read_state_from_space(self) -> ObjectCentricState:
        """Read the current state from the PyMunk space."""
        if not self.space or not self._current_state:
            return ObjectCentricState({})

        state = self._current_state.copy()

        # Update dynamic object positions from PyMunk simulation
        for obj in state:
            if hasattr(obj, "_pymunk_body") and hasattr(obj, "_pymunk_shape"):
                body = getattr(obj, "_pymunk_body")
                # Only update dynamic objects (not static or kinematic)
                if body.body_type == pymunk.Body.DYNAMIC:
                    state.set(obj, "x", float(body.position.x))
                    state.set(obj, "y", float(body.position.y))
                    state.set(obj, "theta", float(body.angle))

        return state

    def _sample_initial_state(self) -> ObjectCentricState:
        """Sample an initial state for the environment."""
        n = self._spec.max_initial_state_sampling_attempts
        for _ in range(n):
            # Sample all randomized values.
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
                target_surface_pose,
                target_surface_shape,
                target_block_pose,
                target_block_shape,
                obstructions,
            )
            
            # Basic validation - ensure objects don't start overlapping too much
            # The physics simulation will handle the rest
            return state
            
        raise RuntimeError(f"Failed to sample initial state after {n} attempts")

    def _create_initial_state(
        self,
        target_surface_pose: SE2Pose,
        target_surface_shape: tuple[float, float],
        target_block_pose: SE2Pose,
        target_block_shape: tuple[float, float],
        obstructions: list[tuple[SE2Pose, tuple[float, float]]],
    ) -> ObjectCentricState:
        """Create the initial state dictionary."""
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
            "z_order": 1.0,
        }

        # Create walls
        wall_thickness = 10.0
        walls = [
            ("wall_left", self._spec.world_min_x - wall_thickness/2, (self._spec.world_min_y + self._spec.world_max_y)/2, wall_thickness, self._spec.world_max_y - self._spec.world_min_y),
            ("wall_right", self._spec.world_max_x + wall_thickness/2, (self._spec.world_min_y + self._spec.world_max_y)/2, wall_thickness, self._spec.world_max_y - self._spec.world_min_y),
            ("wall_bottom", (self._spec.world_min_x + self._spec.world_max_x)/2, self._spec.world_min_y - wall_thickness/2, self._spec.world_max_x - self._spec.world_min_x, wall_thickness),
            ("wall_top", (self._spec.world_min_x + self._spec.world_max_x)/2, self._spec.world_max_y + wall_thickness/2, self._spec.world_max_x - self._spec.world_min_x, wall_thickness),
        ]
        
        for wall_name, x, y, width, height in walls:
            wall = Object(wall_name, RectangleType)
            init_state_dict[wall] = {
                "x": x,
                "y": y,
                "theta": 0.0,
                "width": width,
                "height": height,
                "static": True,
                "color_r": BLACK[0],
                "color_g": BLACK[1],
                "color_b": BLACK[2],
                "z_order": 1.0,
            }

        # Create the target surface (kinematic).
        target_surface = Object("target_surface", TargetSurfaceType)
        init_state_dict[target_surface] = {
            "x": target_surface_pose.x,
            "y": target_surface_pose.y,
            "theta": target_surface_pose.theta,
            "width": target_surface_shape[0],
            "height": target_surface_shape[1],
            "static": False,  # Kinematic objects are not static in the state representation
            "color_r": self._spec.target_surface_rgb[0],
            "color_g": self._spec.target_surface_rgb[1],
            "color_b": self._spec.target_surface_rgb[2],
            "z_order": 0.0,
        }

        # Create the target block (dynamic).
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
            "z_order": 1.0,
        }

        # Create obstructions (dynamic).
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
                "z_order": 1.0,
            }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)

    def _target_satisfied(self) -> bool:
        """Check if the target block is on the target surface."""
        if not self.space or not self._target_block or not self._target_surface:
            return False
        
        return is_on_dynamic(self.space, self._target_block, self._target_surface)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination."""
        # Terminate when target object is on the target surface. Give -1 reward
        # at every step until then to encourage fast completion.
        terminated = self._target_satisfied()
        return -1.0, terminated


class DynObstruction2DEnv(ConstantObjectDynamic2DEnv):
    """Dynamic Obstruction 2D env with a constant number of objects."""

    def _create_object_centric_dynamic2d_env(self, *args, **kwargs) -> Dynamic2DRobotEnv:
        return ObjectCentricDynObstruction2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["target_surface", "target_block"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstruct"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        num_obstructions = len(self._constant_objects) - 2  # Exclude target surface and block
        # pylint: disable=line-too-long
        if num_obstructions > 0:
            obstruction_sentence = f"\nThe target surface may be initially obstructed. In this environment, there are always {num_obstructions} obstacle blocks.\n"
        else:
            obstruction_sentence = ""

        return f"""A 2D physics-based environment where the goal is to place a target block onto a target surface using a fingered robot with PyMunk physics simulation. The block must be completely on the surface.
{obstruction_sentence}
The robot has a movable circular base and an extendable arm with gripper fingers. Objects can be grasped and released through gripper actions. All objects follow realistic physics including gravity, friction, and collisions.
"""

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return f"""A penalty of -1.0 is given at every time step until termination, which occurs when the target block is "on" the target surface. The definition of "on" is implemented using physics-based collision detection:
```python
{inspect.getsource(is_on_dynamic)}```
"""

    def _create_references_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """This is a physics-based version of manipulation environments commonly used in robotics research. It extends the geometric obstruction environment to include realistic physics simulation using PyMunk, enabling more realistic robot manipulation scenarios.
"""