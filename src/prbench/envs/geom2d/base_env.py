"""Base class for Geom2D robot environments."""

import abc
from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray
from prpl_utils.utils import wrap_angle
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
    ObjectCentricStateSpace,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.geom2d.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
    RectangleType,
)
from prbench.envs.geom2d.structs import MultiBody2D, SE2Pose
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    get_geom2d_crv_robot_action_from_gui_input,
    get_suctioned_objects,
    snap_suctioned_objects,
)
from prbench.envs.utils import render_2dstate, state_2d_has_collision


@dataclass(frozen=True)
class Geom2DRobotEnvSpec:
    """Scene specification for a Geom2DRobotEnv."""

    # The world is oriented like a standard X/Y coordinate frame.
    world_min_x: float = 0.0
    world_max_x: float = 10.0
    world_min_y: float = 0.0
    world_max_y: float = 10.0

    # Action space parameters.
    min_dx: float = -5e-1
    max_dx: float = 5e-1
    min_dy: float = -5e-1
    max_dy: float = 5e-1
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_vac: float = 0.0
    max_vac: float = 1.0

    # For rendering.
    render_dpi: int = 50


class Geom2DRobotEnv(gymnasium.Env):
    """Base class for Geom2D robot environments.

    NOTE: this implementation currently assumes we are using CRVRobotType.
    If we add other robot types in the future, we will need to refactor a bit.
    """

    # Only RGB rendering is implemented.
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self, spec: Geom2DRobotEnvSpec, render_mode: str | None = "rgb_array"
    ) -> None:
        self._spec = spec
        self._types = {RectangleType, CRVRobotType}
        self.render_mode = render_mode
        self.observation_space = ObjectCentricStateSpace(self._types)
        self.action_space = CRVRobotActionSpace(
            min_dx=self._spec.min_dx,
            max_dx=self._spec.max_dx,
            min_dy=self._spec.min_dy,
            max_dy=self._spec.max_dy,
            min_dtheta=self._spec.min_dtheta,
            max_dtheta=self._spec.max_dtheta,
            min_darm=self._spec.min_darm,
            max_darm=self._spec.max_darm,
            min_vac=self._spec.min_vac,
            max_vac=self._spec.max_vac,
        )

        # Initialized by reset().
        self._current_state: ObjectCentricState | None = None
        # Maintain an independent initial_constant_state, including static objects
        # that never change throughout the lifetime of the environment.
        self._initial_constant_state: ObjectCentricState | None = None
        self._static_object_body_cache: dict[Object, MultiBody2D] = {}

        super().__init__()

    @abc.abstractmethod
    def _sample_initial_state(self) -> ObjectCentricState:
        """Use self.np_random to sample an initial state."""

    @abc.abstractmethod
    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination based on self._current_state."""

    def _get_obs(self) -> ObjectCentricState:
        assert self._current_state is not None, "Need to call reset()"
        # NOTE: Based on the discussion, we commit to providing
        # only the changeable objects in the state.
        # A learning-based algorithm has no access to the
        # initial constant state, as the algorithm should learn
        # to handle them if they affect decision making.

        # That being said, we still want to provide an interface
        # for accessing the static objects, as some baselines
        # (planner model) requires such information.
        full_state = self._current_state.copy()
        return full_state

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
        assert self._current_state is not None, "Need to call reset()"
        full_state = self._current_state.copy()
        if self._initial_constant_state is not None:
            # Merge the initial constant state with the current state.
            full_state.data.update(self._initial_constant_state.data)
        return full_state

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObjectCentricState, dict]:
        super().reset(seed=seed)

        # Need to flush the cache in case static objects move.
        self._static_object_body_cache = {}

        # For testing purposes only, the options may specify an initial scene.
        if options is not None and "init_state" in options:
            self._current_state = options["init_state"].copy()

        # Otherwise, set up the initial scene here.
        else:
            self._current_state = self._sample_initial_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Array) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        dx, dy, dtheta, darm, vac = action
        assert self._current_state is not None, "Need to call reset()"
        state = self._current_state.copy()
        robots = [o for o in state if o.is_instance(CRVRobotType)]
        assert len(robots) == 1, "Multi-robot not yet supported"
        robot = robots[0]

        # NOTE: xy clipping is not needed because world boundaries are assumed
        # handled by collision detection with walls.
        new_x = state.get(robot, "x") + dx
        new_y = state.get(robot, "y") + dy
        new_theta = wrap_angle(state.get(robot, "theta") + dtheta)
        min_arm = state.get(robot, "base_radius")
        max_arm = state.get(robot, "arm_length")
        new_arm = np.clip(state.get(robot, "arm_joint") + darm, min_arm, max_arm)
        state.set(robot, "x", new_x)
        state.set(robot, "y", new_y)
        state.set(robot, "arm_joint", new_arm)
        state.set(robot, "theta", new_theta)
        state.set(robot, "vacuum", vac)

        # The order here is subtle and important:
        # 1) Look at which objects were suctioned in the *previous* time step.
        # 2) Get the transform between gripper and object in the *previous*.
        # 3) Update the position of the object to snap to the robot *now*.
        # 4) When checking collisions, make sure to include all objects that
        #    may have moved. This cannot be derived from `state` alone!
        # The last point was previously overlook and led to bugs where the held
        # objects could come into collision with other objects if the suction is
        # disabled at the right time.

        # Update the state of any objects that are currently suctioned.
        # NOTE: this is both objects and their SE2 transforms.
        suctioned_objs = get_suctioned_objects(self._current_state, robot)
        snap_suctioned_objects(state, robot, suctioned_objs)

        # Update non-static objects if contact is detected between them
        # and the suctioned objects.
        state, moved_objects = self.get_objects_to_move(state, suctioned_objs)

        # Check for collisions, and only update the state if none exist.
        moving_objects = (
            {robot} | {o for o, _ in suctioned_objs} | {o for o, _ in moved_objects}
        )
        full_state = state.copy()
        if self._initial_constant_state is not None:
            # Merge the initial constant state with the current state.
            full_state.data.update(self._initial_constant_state.data)
        obstacles = set(full_state) - moving_objects
        if not state_2d_has_collision(
            full_state, moving_objects, obstacles, self._static_object_body_cache
        ):
            self._current_state = state

        reward, terminated = self._get_reward_and_done()
        truncated = False  # no maximum horizon, by default
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def get_objects_to_move(
        self,
        state: ObjectCentricState,
        suctioned_objs: list[tuple[Object, SE2Pose]],
    ) -> tuple[ObjectCentricState, set[tuple[Object, SE2Pose]]]:
        """Get the set of objects that should be moved based on the current state and
        robot actions.

        Implement this in the derived class.
        """
        del suctioned_objs  # not used, but subclasses may use
        # Explicitly type the set to ensure it's set[Object]
        moved_objects: set[tuple[Object, SE2Pose]] = set()
        return state, moved_objects

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        assert self.render_mode == "rgb_array"
        assert self._current_state is not None, "Need to call reset()"
        render_input_state = self._current_state.copy()
        if self._initial_constant_state is not None:
            # Merge the initial constant state with the current state.
            render_input_state.data.update(self._initial_constant_state.data)
        return render_2dstate(
            render_input_state,
            self._static_object_body_cache,
            self._spec.world_min_x,
            self._spec.world_max_x,
            self._spec.world_min_y,
            self._spec.world_max_y,
            self._spec.render_dpi,
        )


class ConstantObjectGeom2DEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Defined by an object-centric Geom2D environment and a constant object set.

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
        self._geom2d_env = self._create_object_centric_geom2d_env(*args, **kwargs)
        # Create a Box version of the observation space by extracting the constant
        # objects from an exemplar state.
        assert isinstance(self._geom2d_env.observation_space, ObjectCentricStateSpace)
        exemplar_object_centric_state, _ = self._geom2d_env.reset()
        obj_name_to_obj = {o.name: o for o in exemplar_object_centric_state}
        obj_names = self._get_constant_object_names(exemplar_object_centric_state)
        self._constant_objects = [obj_name_to_obj[o] for o in obj_names]
        # This is a Box space with some extra functionality to allow easy vectorizing.
        self.observation_space = self._geom2d_env.observation_space.to_box(
            self._constant_objects, Geom2DRobotEnvTypeFeatures
        )
        self.action_space = self._geom2d_env.action_space
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        # The action space already inherits from Box, so we don't need to change it.
        assert isinstance(self.action_space, CRVRobotActionSpace)
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
                "render_fps": self._geom2d_env.metadata.get("render_fps", 20),
            }
        )
        self.render_mode = render_mode

    @abc.abstractmethod
    def _create_object_centric_geom2d_env(self, *args, **kwargs) -> Geom2DRobotEnv:
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
