"""Base class for Dynamic2D (PyMunk) robot environments."""

import abc
import time
from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
import pymunk
from numpy.typing import NDArray
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
    ObjectCentricStateSpace,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.geom2d.structs import MultiBody2D
from prbench.envs.dynamic2d.object_types import (
    Dynamic2DRobotEnvTypeFeatures,
    KinRobotType,
    RectangleType,
    DynRectangleType,
)
from prbench.envs.dynamic2d.utils import (
    DYNAMIC_COLLISION_TYPE,
    ROBOT_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
    KinRobotActionSpace,
    KinRobot,
    PDController,
    get_fingered_robot_action_from_gui_input,
    on_collision_w_static,
    on_gripper_grasp,
    render_state
)

@dataclass(frozen=True)
class Dynamic2DRobotEnvSpec:
    """Scene specification for a Dynamic2DRobotEnv."""

    # The world is oriented like a standard X/Y coordinate frame.
    world_min_x: float = 0.0
    world_max_x: float = 10.0
    world_min_y: float = 0.0
    world_max_y: float = 10.0

    # Action space parameters.
    min_dx: float = -1e-2
    max_dx: float = 1e-2
    min_dy: float = -1e-2
    max_dy: float = 1e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -5e-2
    max_darm: float = 5e-2
    min_dgripper: float = -0.02
    max_dgripper: float = 0.02

    # Robot parameters
    init_robot_pos: tuple[float, float] = (5.0, 5.0)
    robot_base_radius: float = 0.4
    robot_arm_length_max: float = 0.8
    gripper_base_width: float = 0.01
    gripper_base_height: float = 0.1
    gripper_finger_width: float = 0.1
    gripper_finger_height: float = 0.01

    # Controller parameters
    kp_pos: float = 50.0
    kv_pos: float = 5.0
    kp_rot: float = 50.0
    kv_rot: float = 5.0

    # Physics parameters
    gravity_y: float = -9.8
    control_freq: int = 20  # Control frequency (actions per second)
    sim_freq: int = 120     # Simulation frequency (physics steps per second)

    # For rendering.
    render_dpi: int = 50


# Define a simple robot type for Dynamic2D environments
FingeredRobotType = Object("fingered_robot", None)


class Dynamic2DRobotEnv(gymnasium.Env):
    """Base class for Dynamic2D robot environments using PyMunk physics.

    This environment uses PyMunk for physics simulation with a KinRobot.
    """

    # Only RGB rendering is implemented.
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self, spec: Dynamic2DRobotEnvSpec, render_mode: str | None = "rgb_array"
    ) -> None:
        self._spec = spec
        self._types = {KinRobotType, RectangleType, DynRectangleType}
        self.render_mode = render_mode
        self.observation_space = ObjectCentricStateSpace(self._types)
        self.action_space = KinRobotActionSpace(
            min_dx=self._spec.min_dx,
            max_dx=self._spec.max_dx,
            min_dy=self._spec.min_dy,
            max_dy=self._spec.max_dy,
            min_dtheta=self._spec.min_dtheta,
            max_dtheta=self._spec.max_dtheta,
            min_darm=self._spec.min_darm,
            max_darm=self._spec.max_darm,
            min_dgripper=self._spec.min_dgripper,
            max_dgripper=self._spec.max_dgripper,
        )

        # PyMunk physics space
        self.space: pymunk.Space | None = None
        self.robot: KinRobot | None = None
        self.pd_controller = PDController(
            kp_pos=self._spec.kp_pos,
            kv_pos=self._spec.kv_pos,
            kp_rot=self._spec.kp_rot,
            kv_rot=self._spec.kv_rot,
        )

        # Initialized by reset().
        self._current_state: ObjectCentricState | None = None
        # Some objects may map to multiple pymunk bodies (e.g., the robot).
        self._state_obj_to_pymunk_body_idx: dict[Object, int] = {}
        # Maintain an independent initial_constant_state, including static objects
        # that never change throughout the lifetime of the environment.
        self._initial_constant_state: ObjectCentricState | None = None
        self._static_object_body_cache: dict[Object, MultiBody2D] = {}

        super().__init__()

    def _setup_physics_space(self) -> None:
        """Set up the PyMunk physics space."""
        self.space = pymunk.Space()
        self.space.gravity = 0, self._spec.gravity_y

        # Create robot
        self.robot = KinRobot(
            init_pos=self._spec.init_robot_pos,
            base_radius=self._spec.robot_base_radius,
            arm_length_max=self._spec.robot_arm_length_max,
            gripper_base_width=self._spec.gripper_base_width,
            gripper_base_height=self._spec.gripper_base_height,
            gripper_finger_width=self._spec.gripper_finger_width,
            gripper_finger_height=self._spec.gripper_finger_height,
            kp_pos=self._spec.kp_pos,
            kv_pos=self._spec.kv_pos,
            kp_rot=self._spec.kp_rot,
            kv_rot=self._spec.kv_rot,
        )
        self.robot.add_to_space(self.space)

        # Set up collision handlers
        self.space.on_collision(
            DYNAMIC_COLLISION_TYPE, ROBOT_COLLISION_TYPE,
            post_solve=on_gripper_grasp, data=self.robot
        )
        self.space.on_collision(
            STATIC_COLLISION_TYPE, ROBOT_COLLISION_TYPE,
            pre_solve=on_collision_w_static, data=self.robot
        )

    def _reset_robot_in_space(self, obj: Object, state: ObjectCentricState) -> None:
        """Reset the robot in the PyMunk space."""
        if not self.space:
            return
        robot_base_x = state.get(obj, "x")
        robot_base_y = state.get(obj, "y")
        robot_theta = state.get(obj, "theta")
        robot_arm = state.get(obj, "arm_joint")
        robot_gripper = state.get(obj, "finger_gap")
        assert self.robot is not None, "Robot not initialized"
        self.robot.reset_positions(
            base_x=robot_base_x,
            base_y=robot_base_y,
            base_theta=robot_theta,
            arm_length=robot_arm,
            gripper_gap=robot_gripper,
        )
        self._state_obj_to_pymunk_body_idx[obj] = self.robot.body_id

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space."""
        if not self.space:
            return
        for obj in state:
            if obj.type == KinRobotType:
                self._reset_robot_in_space(obj, state)
            else:
                # Add other objects (e.g., rectangles) to the space.
                # This will be implemented in subclasses.
                pass

    @abc.abstractmethod
    def _read_state_from_space(self) -> None:
        """Read the current state from the PyMunk space."""

    @abc.abstractmethod
    def _sample_initial_state(self) -> ObjectCentricState:
        """Use self.np_random to sample an initial state."""

    @abc.abstractmethod
    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination based on self._current_state."""

    def _get_obs(self) -> ObjectCentricState:
        """Get observation by reading from the physics simulation."""
        assert self._current_state is not None, "Need to call reset()"
        return self._current_state.copy()

    def _get_info(self) -> dict:
        return {}  # no extra info provided right now

    @property
    def initial_constant_state(self) -> ObjectCentricState:
        """Get the initial constant state, which includes static objects."""
        assert (
            self._initial_constant_state is not None
        ), "This env has no initial constant state"
        return self._initial_constant_state.copy()

    @property
    def full_state(self) -> ObjectCentricState:
        """Get the full state, which includes both dynamic and static objects."""
        full_state = self._current_state.copy()
        if self._initial_constant_state is not None:
            # Merge the initial constant state with the current state.
            full_state.data.update(self._initial_constant_state.data)
        return full_state

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObjectCentricState, dict]:
        super().reset(seed=seed)

        # Clear existing physics space
        if self.space:
            # Remove all bodies and shapes
            for body in list(self.space.bodies):
                for shape in body.shapes:
                    self.space.remove(body, shape)
            for shape in list(self.space.shapes):
                if shape.body == self.space.static_body:
                    self.space.remove(shape)

        # Set up new physics space
        self._setup_physics_space()
        self._static_object_body_cache = {}
        self._state_obj_to_pymunk_body_idx = {}

        # For testing purposes only, the options may specify an initial scene.
        if options is not None and "init_state" in options:
            self._current_state = options["init_state"].copy()
        # Otherwise, set up the initial scene here.
        else:
            self._current_state = self._sample_initial_state()

        # Add objects to physics space
        self._add_state_to_space(self.full_state)

        # Calculate simulation parameters
        dt = 1.0 / self._spec.control_freq
        n_steps = self._spec.sim_freq // self._spec.control_freq

        # Stepping physics to let things settle
        if self.space:
            for _ in range(n_steps):
                self.space.step(dt)

        self._read_state_from_space()
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Array) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        dx, dy, dtheta, darm, dgripper = action
        assert self._current_state is not None, "Need to call reset()"
        assert self.space is not None, "Space not initialized"
        assert self.robot is not None, "Robot not initialized"

        # Calculate simulation parameters
        n_steps = self._spec.sim_freq // self._spec.control_freq
        dt = 1.0 / self._spec.control_freq

        # Calculate target positions
        tgt_x = self.robot.base_pose.x + dx
        tgt_y = self.robot.base_pose.y + dy
        tgt_theta = self.robot.base_pose.theta + dtheta
        tgt_arm = max(min(self.robot.curr_arm_length + darm, self.robot.arm_length_max), 
                      self.robot.base_radius)
        tgt_gripper = max(
            min(self.robot.curr_gripper + dgripper, self.robot.gripper_gap_max),
            self.robot.gripper_finger_height * 2,
        )

        # Multi-step simulation like basic_pymunk.py
        s = time.time()
        for _ in range(n_steps):
            # Use PD control to compute base and gripper velocities
            base_vel, base_ang_vel, gripper_base_vel, finger_vel = self.pd_controller.compute_control(
                self.robot, tgt_x, tgt_y, tgt_theta, tgt_arm, tgt_gripper, dt)
            # Update robot with the vel (PD control updates velocities)
            self.robot.update(base_vel, 
                              base_ang_vel, 
                              gripper_base_vel, 
                              finger_vel)
            # Step physics simulation
            self.space.step(dt)
        e = time.time()
        print(f"Stepped {n_steps} physics steps in {e - s:.4f} seconds")
        # Drop objects after internal steps (like basic_pymunk.py)
        self.robot.drop_held_objects(self.space)

        # Update current state from simulation
        self._read_state_from_space()

        reward, terminated = self._get_reward_and_done()
        truncated = False  # no maximum horizon, by default
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Render the current state. To be implemented."""
        # This will be implemented later
        assert self.render_mode == "rgb_array"
        assert self._current_state is not None, "Need to call reset()"
        render_input_state = self._current_state.copy()
        if self._initial_constant_state is not None:
            # Merge the initial constant state with the current state.
            render_input_state.data.update(self._initial_constant_state.data)
        return render_state(
            render_input_state,
            self._static_object_body_cache,
            self._spec.world_min_x,
            self._spec.world_max_x,
            self._spec.world_min_y,
            self._spec.world_max_y,
            self._spec.render_dpi,
        )


class ConstantObjectDynamic2DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Defined by an object-centric Dynamic2D environment and a constant object set.

    The point of this pattern is to allow implementing object-centric environments with
    variable numbers of objects, but then also create versions of the environment with a
    constant number of objects so it is easy to apply, e.g., RL approaches that use
    fixed-dimensional observation and action spaces.
    """

    # NOTE: we need to define render_modes in the class instead of the instance because
    # gym.make extracts render_modes from the class (entry_point) before instantiation.
    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(self, *args, render_mode: str | None = None, **kwargs) -> None:
        super().__init__()
        self._dynamic2d_env = self._create_object_centric_dynamic2d_env(*args, **kwargs)
        # Create a Box version of the observation space by extracting the constant
        # objects from an exemplar state.
        assert isinstance(self._dynamic2d_env.observation_space, ObjectCentricStateSpace)
        exemplar_object_centric_state, _ = self._dynamic2d_env.reset()
        obj_name_to_obj = {o.name: o for o in exemplar_object_centric_state}
        obj_names = self._get_constant_object_names(exemplar_object_centric_state)
        self._constant_objects = [obj_name_to_obj[o] for o in obj_names]
        # This is a Box space with some extra functionality to allow easy vectorizing.
        self.observation_space = self._dynamic2d_env.observation_space.to_box(
            self._constant_objects, Dynamic2DRobotEnvTypeFeatures
        )
        self.action_space = self._dynamic2d_env.action_space
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        # The action space already inherits from Box, so we don't need to change it.
        assert isinstance(self.action_space, KinRobotActionSpace)
        # Add descriptions to metadata for doc generation.
        obs_md = self.observation_space.create_markdown_description()
        act_md = self.action_space.create_markdown_description()
        env_md = self._create_env_markdown_description()
        reward_md = self._create_reward_markdown_description()
        references_md = self._create_references_markdown_description()
        # Update the metadata. Note that we need to define the render_modes in the class
        # rather than in the instance because gym.make() extracts render_modes from cls.
        self.metadata = self.metadata.copy()
        self.metadata.update(
            {
                "description": env_md,
                "observation_space_description": obs_md,
                "action_space_description": act_md,
                "reward_description": reward_md,
                "references": references_md,
                "render_fps": self._dynamic2d_env.metadata.get("render_fps", 20),
            }
        )
        self.render_mode = render_mode

    @abc.abstractmethod
    def _create_object_centric_dynamic2d_env(self, *args, **kwargs) -> Dynamic2DRobotEnv:
        """Create the underlying object-centric environment."""

    @abc.abstractmethod
    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        """The ordered names of the constant objects extracted from the observations."""

    @abc.abstractmethod
    def _create_env_markdown_description(self) -> str:
        """Create a markdown description of the overall environment."""

    @abc.abstractmethod
    def _create_reward_markdown_description(self) -> str:
        """Create a markdown description of the environment rewards."""

    @abc.abstractmethod
    def _create_references_markdown_description(self) -> str:
        """Create a markdown description of the reference (e.g. papers) for this env."""

    def reset(self, *args, **kwargs) -> tuple[NDArray[np.float32], dict]:
        super().reset(*args, **kwargs)  # necessary to reset RNG if seed is given
        obs, info = self._dynamic2d_env.reset(*args, **kwargs)
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        vec_obs = self.observation_space.vectorize(obs)
        return vec_obs, info

    def step(
        self, *args, **kwargs
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        obs, reward, terminated, truncated, done = self._dynamic2d_env.step(
            *args, **kwargs
        )
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        vec_obs = self.observation_space.vectorize(obs)
        return vec_obs, reward, terminated, truncated, done

    def render(self):
        return self._dynamic2d_env.render()

    def get_action_from_gui_input(
        self, gui_input: dict[str, Any]
    ) -> NDArray[np.float32]:
        """Get the mapping from human inputs to actions."""
        # This will be implemented later
        assert isinstance(self.action_space, KinRobotActionSpace)
        return get_fingered_robot_action_from_gui_input(self.action_space, gui_input)