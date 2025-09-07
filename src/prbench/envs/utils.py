"""Utility functions shared across different types of environments."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from prpl_utils.utils import fig2data
from relational_structs import (
    Object,
    ObjectCentricState,
)
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

from prbench.envs.dynamic2d.object_types import (
    DynRectangleType,
    KinRectangleType,
    KinRobotType,
)
from prbench.envs.dynamic2d.utils import (
    kin_robot_to_multibody2d,
)
from prbench.envs.geom2d.object_types import (
    CircleType,
    CRVRobotType,
    DoubleRectType,
    LObjectType,
    RectangleType,
)
from prbench.envs.geom2d.structs import (
    Body2D,
    MultiBody2D,
    SE2Pose,
    ZOrder,
    z_orders_may_collide,
)
from prbench.envs.geom2d.utils import (
    crv_robot_to_multibody2d,
    geom2d_double_rectangle_to_multibody2d,
    geom2d_lobject_to_multibody2d,
)

PURPLE: tuple[float, float, float] = (128 / 255, 0 / 255, 128 / 255)
BLACK: tuple[float, float, float] = (0.1, 0.1, 0.1)


def get_se2_pose(state: ObjectCentricState, obj: Object) -> SE2Pose:
    """Get the SE2Pose of an object in a given state."""
    return SE2Pose(
        x=state.get(obj, "x"),
        y=state.get(obj, "y"),
        theta=state.get(obj, "theta"),
    )


def get_relative_se2_transform(
    state: ObjectCentricState, obj1: Object, obj2: Object
) -> SE2Pose:
    """Get the pose of obj2 in the frame of obj1."""
    world_to_obj1 = get_se2_pose(state, obj1)
    world_to_obj2 = get_se2_pose(state, obj2)
    return world_to_obj1.inverse * world_to_obj2


def sample_se2_pose(
    bounds: tuple[SE2Pose, SE2Pose], rng: np.random.Generator
) -> SE2Pose:
    """Sample a SE2Pose uniformly between the bounds."""
    lb, ub = bounds
    x = rng.uniform(lb.x, ub.x)
    y = rng.uniform(lb.y, ub.y)
    theta = rng.uniform(lb.theta, ub.theta)
    return SE2Pose(x, y, theta)


def state_2d_has_collision(
    state: ObjectCentricState,
    group1: set[Object],
    group2: set[Object],
    static_object_cache: dict[Object, MultiBody2D],
    ignore_z_orders: bool = False,
) -> bool:
    """Check for collisions between any objects in two groups."""
    # Create multibodies once.
    obj_to_multibody = {
        o: object_to_multibody2d(o, state, static_object_cache) for o in state
    }
    # Check pairwise collisions.
    for obj1 in group1:
        for obj2 in group2:
            if obj1 == obj2:
                continue
            multibody1 = obj_to_multibody[obj1]
            multibody2 = obj_to_multibody[obj2]
            for body1 in multibody1.bodies:
                for body2 in multibody2.bodies:
                    if not (
                        ignore_z_orders
                        or z_orders_may_collide(body1.z_order, body2.z_order)
                    ):
                        continue
                    if geom2ds_intersect(body1.geom, body2.geom):
                        return True
    return False


def render_2dstate_on_ax(
    state: ObjectCentricState,
    ax: plt.Axes,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
) -> None:
    """Render a state on an existing plt.Axes."""
    if static_object_body_cache is None:
        static_object_body_cache = {}

    # Sort objects by ascending z order, with the robot first.
    def _render_order(obj: Object) -> int:
        if obj.is_instance(CRVRobotType):
            return -1
        return int(state.get(obj, "z_order"))

    for obj in sorted(state, key=_render_order):
        body = object_to_multibody2d(obj, state, static_object_body_cache)
        body.plot(ax)


def render_2dstate(
    state: ObjectCentricState,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
    world_min_x: float = 0.0,
    world_max_x: float = 10.0,
    world_min_y: float = 0.0,
    world_max_y: float = 10.0,
    render_dpi: int = 150,
) -> NDArray[np.uint8]:
    """Render a state.

    Useful for viz and debugging.
    """
    if static_object_body_cache is None:
        static_object_body_cache = {}

    figsize = (
        world_max_x - world_min_x,
        world_max_y - world_min_y,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=render_dpi)

    render_2dstate_on_ax(state, ax, static_object_body_cache)

    pad_x = (world_max_x - world_min_x) / 25
    pad_y = (world_max_y - world_min_y) / 25
    ax.set_xlim(world_min_x - pad_x, world_max_x + pad_x)
    ax.set_ylim(world_min_y - pad_y, world_max_y + pad_y)
    ax.axis("off")
    plt.tight_layout()
    img = fig2data(fig)
    plt.close()
    return img


def object_to_multibody2d(
    obj: Object,
    state: ObjectCentricState,
    static_object_cache: dict[Object, MultiBody2D],
) -> MultiBody2D:
    """Create a Body2D instance for objects of standard geom types."""
    if obj.is_instance(CRVRobotType):
        return crv_robot_to_multibody2d(obj, state)
    if obj.is_instance(KinRobotType):
        return kin_robot_to_multibody2d(obj, state)
    is_static = state.get(obj, "static") > 0.5
    if is_static and obj in static_object_cache:
        return static_object_cache[obj]
    geom: Geom2D  # rectangle or circle
    if obj.is_instance(RectangleType):
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")
        geom = Rectangle(x, y, width, height, theta)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    elif obj.is_instance(DynRectangleType) or obj.is_instance(KinRectangleType):
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")
        # Different from RectangleType, use from_center.
        geom = Rectangle.from_center(x, y, width, height, theta)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    elif obj.is_instance(CircleType):
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        radius = state.get(obj, "radius")
        geom = Circle(x, y, radius)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    elif obj.is_instance(LObjectType):
        multibody = geom2d_lobject_to_multibody2d(obj, state)
    elif obj.is_instance(DoubleRectType):
        multibody = geom2d_double_rectangle_to_multibody2d(obj, state)
    else:
        raise NotImplementedError
    if is_static:
        static_object_cache[obj] = multibody
    return multibody


def rectangle_object_to_geom(
    state: ObjectCentricState,
    rect_obj: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> Rectangle:
    """Helper to extract a rectangle for an object."""
    assert (
        rect_obj.is_instance(RectangleType)
        or rect_obj.is_instance(DynRectangleType)
        or rect_obj.is_instance(KinRectangleType)
    )
    multibody = object_to_multibody2d(rect_obj, state, static_object_cache)
    assert len(multibody.bodies) == 1
    geom = multibody.bodies[0].geom
    assert isinstance(geom, Rectangle)
    return geom
