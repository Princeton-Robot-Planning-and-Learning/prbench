"""Utilities."""

from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from prpl_utils.motion_planning import BiRRT
from prpl_utils.utils import fig2data, get_signed_angle_distance, wrap_angle
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
)
from tomsgeoms2d.structs import Circle, Geom2D, Lobject, Rectangle
from tomsgeoms2d.utils import find_closest_points, geom2ds_intersect

from prbench.envs.geom2d.object_types import (
    CircleType,
    CRVRobotType,
    DoubleRectType,
    Geom2DType,
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

PURPLE: tuple[float, float, float] = (128 / 255, 0 / 255, 128 / 255)
BLACK: tuple[float, float, float] = (0.1, 0.1, 0.1)


class CRVRobotActionSpace(Box):
    """An action space for a CRV robot.

    Actions are bounded relative movements of the base and the arm, as well as an
    absolute setting for the vacuum.
    """

    def __init__(
        self,
        min_dx: float = -5e-1,
        max_dx: float = 5e-1,
        min_dy: float = -5e-1,
        max_dy: float = 5e-1,
        min_dtheta: float = -np.pi / 16,
        max_dtheta: float = np.pi / 16,
        min_darm: float = -1e-1,
        max_darm: float = 1e-1,
        min_vac: float = 0.0,
        max_vac: float = 1.0,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_vac])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_vac])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        # pylint: disable=line-too-long
        features = [
            ("dx", "Change in robot x position (positive is right)"),
            ("dy", "Change in robot y position (positive is up)"),
            ("dtheta", "Change in robot angle in radians (positive is ccw)"),
            ("darm", "Change in robot arm length (positive is out)"),
            ("vac", "Directly sets the vacuum (0.0 is off, 1.0 is on)"),
        ]
        md_table_str = (
            "| **Index** | **Feature** | **Description** | **Min** | **Max** |"
        )
        md_table_str += "\n| --- | --- | --- | --- | --- |"
        for idx, (feature, description) in enumerate(features):
            lb = self.low[idx]
            ub = self.high[idx]
            md_table_str += (
                f"\n| {idx} | {feature} | {description} | {lb:.3f} | {ub:.3f} |"
            )
        return f"The entries of an array in this Box space correspond to the following action features:\n{md_table_str}\n"


def object_to_multibody2d(
    obj: Object,
    state: ObjectCentricState,
    static_object_cache: dict[Object, MultiBody2D],
) -> MultiBody2D:
    """Create a Body2D instance for objects of standard geom types."""
    if obj.is_instance(CRVRobotType):
        return _robot_to_multibody2d(obj, state)
    assert obj.is_instance(Geom2DType)
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
        multibody = _lobject_to_multibody2d(obj, state)
    elif obj.is_instance(DoubleRectType):
        multibody = _double_rectangle_to_multibody2d(obj, state)
    else:
        raise NotImplementedError
    if is_static:
        static_object_cache[obj] = multibody
    return multibody


def _robot_to_multibody2d(obj: Object, state: ObjectCentricState) -> MultiBody2D:
    """Helper for object_to_multibody2d()."""
    assert obj.is_instance(CRVRobotType)
    bodies: list[Body2D] = []

    # Base.
    base_x = state.get(obj, "x")
    base_y = state.get(obj, "y")
    base_radius = state.get(obj, "base_radius")
    circ = Circle(
        x=base_x,
        y=base_y,
        radius=base_radius,
    )
    z_order = ZOrder.ALL
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    base = Body2D(circ, z_order, rendering_kwargs, name="base")
    bodies.append(base)

    # Gripper.
    theta = state.get(obj, "theta")
    arm_joint = state.get(obj, "arm_joint")
    gripper_cx = base_x + np.cos(theta) * arm_joint
    gripper_cy = base_y + np.sin(theta) * arm_joint
    gripper_height = state.get(obj, "gripper_height")
    gripper_width = state.get(obj, "gripper_width")
    rect = Rectangle.from_center(
        center_x=gripper_cx,
        center_y=gripper_cy,
        height=gripper_height,
        width=gripper_width,
        rotation_about_center=theta,
    )
    z_order = ZOrder.SURFACE
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    gripper = Body2D(rect, z_order, rendering_kwargs, name="gripper")
    bodies.append(gripper)

    # Arm.
    rect = Rectangle.from_center(
        center_x=(base_x + gripper_cx) / 2,
        center_y=(base_y + gripper_cy) / 2,
        height=np.sqrt((base_x - gripper_cx) ** 2 + (base_y - gripper_cy) ** 2),
        width=(0.5 * gripper_width),
        rotation_about_center=(theta + np.pi / 2),
    )
    z_order = ZOrder.SURFACE
    silver = (128 / 255, 128 / 255, 128 / 255)
    rendering_kwargs = {"facecolor": silver, "edgecolor": BLACK}
    arm = Body2D(rect, z_order, rendering_kwargs, name="arm")
    bodies.append(arm)

    # If the vacuum is on, add a suction area.
    if state.get(obj, "vacuum") > 0.5:
        suction_height = gripper_height
        suction_width = gripper_width
        suction_cx = base_x + np.cos(theta) * (
            arm_joint + gripper_width + suction_width / 2
        )
        suction_cy = base_y + np.sin(theta) * (
            arm_joint + gripper_width + suction_width / 2
        )
        rect = Rectangle.from_center(
            center_x=suction_cx,
            center_y=suction_cy,
            height=suction_height,
            width=suction_width,
            rotation_about_center=theta,
        )
        z_order = ZOrder.NONE  # NOTE: suction collides with nothing
        rendering_kwargs = {"facecolor": PURPLE}
        suction = Body2D(rect, z_order, rendering_kwargs, name="suction")
        bodies.append(suction)

    return MultiBody2D(obj.name, bodies)


def _lobject_to_multibody2d(obj: Object, state: ObjectCentricState) -> MultiBody2D:
    """Helper to create a MultiBody2D for an LObjectType object."""
    assert obj.is_instance(LObjectType)
    # Get parameters
    x = state.get(obj, "x")
    y = state.get(obj, "y")
    theta = state.get(obj, "theta")
    width = state.get(obj, "width")
    length_side1 = state.get(obj, "length_side1")
    length_side2 = state.get(obj, "length_side2")
    color = (
        state.get(obj, "color_r"),
        state.get(obj, "color_g"),
        state.get(obj, "color_b"),
    )
    z_order = ZOrder(int(state.get(obj, "z_order")))

    geom = Lobject(x, y, width, (length_side1, length_side2), theta)

    rendering_kwargs = {
        "facecolor": color,
        "edgecolor": BLACK,
    }
    body = Body2D(geom, z_order, rendering_kwargs, name="hook")

    return MultiBody2D(obj.name, [body])


def _double_rectangle_to_multibody2d(
    obj: Object, state: ObjectCentricState
) -> MultiBody2D:
    """Helper to create a MultiBody2D for a DoubleRectType object."""
    assert obj.is_instance(DoubleRectType)
    # Note: We need to assume the two rectangles are aligned now.
    # This means theta is the same, relative dy == 0, 0 <= dx < width0 - width1.
    # Such that we can create two obstacles from the base rectangle.
    bodies: list[Body2D] = []

    # First rectangle.
    x0 = state.get(obj, "x")
    y0 = state.get(obj, "y")
    theta0 = state.get(obj, "theta")
    height0 = state.get(obj, "height")
    width0 = state.get(obj, "width")
    pose0 = SE2Pose(x0, y0, theta0)
    # Second rectangle.
    x1 = state.get(obj, "x1")
    y1 = state.get(obj, "y1")
    theta1 = state.get(obj, "theta1")
    width1 = state.get(obj, "width1")
    height1 = state.get(obj, "height1")
    pose1 = SE2Pose(x1, y1, theta1)
    assert theta0 == theta1, f"Expected theta0 == theta1, got {theta0} != {theta1}"
    relative_pose = pose0.inverse * pose1
    assert relative_pose.y == 0.0, f"Expected relative y == 0, got {relative_pose.y}"
    assert relative_pose.x >= 0.0, f"Expected relative x >= 0, got {relative_pose.x}"
    assert relative_pose.x + width1 < width0, "Expected relative x + width1 < width0"
    right_bookend_width = width0 - width1 - relative_pose.x

    # Left bookend.
    geom0 = Rectangle(x0, y0, relative_pose.x, height0, theta0)
    z_order0 = ZOrder(int(state.get(obj, "z_order")))
    rendering_kwargs0 = {
        "facecolor": (
            state.get(obj, "color_r"),
            state.get(obj, "color_g"),
            state.get(obj, "color_b"),
        ),
        "edgecolor": BLACK,
    }
    body0 = Body2D(geom0, z_order0, rendering_kwargs0, name=f"{obj.name}_base0")
    bodies.append(body0)
    # Right bookend.
    right_bookend_pose = pose0 * SE2Pose(relative_pose.x + width1, 0.0, theta0)
    geom0_ = Rectangle(
        right_bookend_pose.x, right_bookend_pose.y, right_bookend_width, height0, theta0
    )
    z_order0_ = ZOrder(int(state.get(obj, "z_order")))
    rendering_kwargs0_ = {
        "facecolor": (
            state.get(obj, "color_r"),
            state.get(obj, "color_g"),
            state.get(obj, "color_b"),
        ),
        "edgecolor": BLACK,
    }
    body0_ = Body2D(geom0_, z_order0_, rendering_kwargs0_, name=f"{obj.name}_base1")
    bodies.append(body0_)

    # Second rectangle.
    x1 = state.get(obj, "x1")
    y1 = state.get(obj, "y1")
    width1 = state.get(obj, "width1")
    height1 = state.get(obj, "height1")
    theta1 = state.get(obj, "theta1")
    geom1 = Rectangle(x1, y1, width1, height1, theta1)
    z_order1 = ZOrder(int(state.get(obj, "z_order1")))
    rendering_kwargs1 = {
        "facecolor": (
            state.get(obj, "color_r1"),
            state.get(obj, "color_g1"),
            state.get(obj, "color_b1"),
        ),
        "edgecolor": BLACK,
        "alpha": 0.5,
    }
    body1 = Body2D(geom1, z_order1, rendering_kwargs1, name=f"{obj.name}_part")
    bodies.append(body1)

    return MultiBody2D(obj.name, bodies)


def rectangle_object_to_geom(
    state: ObjectCentricState,
    rect_obj: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> Rectangle:
    """Helper to extract a rectangle for an object."""
    assert rect_obj.is_instance(RectangleType)
    multibody = object_to_multibody2d(rect_obj, state, static_object_cache)
    assert len(multibody.bodies) == 1
    geom = multibody.bodies[0].geom
    assert isinstance(geom, Rectangle)
    return geom


def double_rectangle_object_to_part_geom(
    state: ObjectCentricState,
    double_rect_obj: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> Rectangle:
    """Helper to extract the second rectangle for a DoubleRectType object."""
    assert double_rect_obj.is_instance(DoubleRectType)
    multibody = object_to_multibody2d(double_rect_obj, state, static_object_cache)
    assert len(multibody.bodies) == 3
    # The second body is the "part" rectangle.
    assert "part" in multibody.bodies[2].name
    geom = multibody.bodies[2].geom
    assert isinstance(geom, Rectangle)
    return geom


def create_walls_from_world_boundaries(
    world_min_x: float,
    world_max_x: float,
    world_min_y: float,
    world_max_y: float,
    min_dx: float,
    max_dx: float,
    min_dy: float,
    max_dy: float,
) -> dict[Object, dict[str, float]]:
    """Create wall objects and feature dicts based on world boundaries.

    Velocities are used to determine how large the walls need to be to avoid the
    possibility that the robot will transport over the wall.
    """
    state_dict: dict[Object, dict[str, float]] = {}
    # Right wall.
    right_wall = Object("right_wall", RectangleType)
    side_wall_height = world_max_y - world_min_y
    state_dict[right_wall] = {
        "x": world_max_x,
        "y": world_min_y,
        "width": 2 * max_dx,  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Left wall.
    left_wall = Object("left_wall", RectangleType)
    state_dict[left_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_min_y,
        "width": 2 * abs(min_dx),  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Top wall.
    top_wall = Object("top_wall", RectangleType)
    horiz_wall_width = 2 * 2 * abs(min_dx) + world_max_x - world_min_x
    state_dict[top_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_max_y,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Bottom wall.
    bottom_wall = Object("bottom_wall", RectangleType)
    state_dict[bottom_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_min_y + 2 * min_dy,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    return state_dict


def render_state_on_ax(
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


def render_state(
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

    render_state_on_ax(state, ax, static_object_body_cache)

    pad_x = (world_max_x - world_min_x) / 25
    pad_y = (world_max_y - world_min_y) / 25
    ax.set_xlim(world_min_x - pad_x, world_max_x + pad_x)
    ax.set_ylim(world_min_y - pad_y, world_max_y + pad_y)
    ax.axis("off")
    plt.tight_layout()
    img = fig2data(fig)
    plt.close()
    return img


def state_has_collision(
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


def get_tool_tip_position(
    state: ObjectCentricState, robot: Object
) -> tuple[float, float]:
    """Get the tip of the tool for the robot, which is defined as the center of the
    bottom edge of the gripper."""
    multibody = _robot_to_multibody2d(robot, state)
    gripper_geom = multibody.get_body("gripper").geom
    assert isinstance(gripper_geom, Rectangle)
    # Transform the x, y point.
    tool_tip = np.array([1.0, 0.5])
    scale_matrix = np.array(
        [
            [gripper_geom.width, 0],
            [0, gripper_geom.height],
        ]
    )
    translate_vector = np.array([gripper_geom.x, gripper_geom.y])
    tool_tip = tool_tip @ scale_matrix.T
    tool_tip = tool_tip @ gripper_geom.rotation_matrix.T
    tool_tip = translate_vector + tool_tip
    return (tool_tip[0], tool_tip[1])


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


def get_suctioned_objects(
    state: ObjectCentricState, robot: Object
) -> list[tuple[Object, SE2Pose]]:
    """Find objects that are in the suction zone of a CRVRobot and return the associated
    transform from gripper tool tip to suctioned object."""
    # If the robot's vacuum is not on, there are no suctioned objects.
    if state.get(robot, "vacuum") <= 0.5:
        return []
    robot_multibody = _robot_to_multibody2d(robot, state)
    suction_body = robot_multibody.get_body("suction")
    gripper_x, gripper_y = get_tool_tip_position(state, robot)
    gripper_theta = state.get(robot, "theta")
    world_to_gripper = SE2Pose(gripper_x, gripper_y, gripper_theta)
    # Find MOVABLE objects in collision with the suction geom.
    movable_objects = [o for o in state if o != robot and state.get(o, "static") < 0.5]
    suctioned_objects: list[tuple[Object, SE2Pose]] = []
    for obj in movable_objects:
        # No point in using a static object cache because these objects are
        # not static by definition.
        obj_multibody = object_to_multibody2d(obj, state, {})
        for obj_body in obj_multibody.bodies:
            if geom2ds_intersect(suction_body.geom, obj_body.geom):
                world_to_obj = get_se2_pose(state, obj)
                gripper_to_obj = world_to_gripper.inverse * world_to_obj
                suctioned_objects.append((obj, gripper_to_obj))
    return suctioned_objects


def snap_suctioned_objects(
    state: ObjectCentricState,
    robot: Object,
    suctioned_objs: list[tuple[Object, SE2Pose]],
) -> None:
    """Updates the state in-place."""
    gripper_x, gripper_y = get_tool_tip_position(state, robot)
    gripper_theta = state.get(robot, "theta")
    world_to_gripper = SE2Pose(gripper_x, gripper_y, gripper_theta)
    for obj, gripper_to_obj in suctioned_objs:
        world_to_obj = world_to_gripper * gripper_to_obj
        state.set(obj, "x", world_to_obj.x)
        state.set(obj, "y", world_to_obj.y)
        state.set(obj, "theta", world_to_obj.theta)


def move_objects_in_contact(
    state: ObjectCentricState,
    robot: Object,
    suctioned_objs: list[tuple[Object, SE2Pose]],
) -> tuple[ObjectCentricState, set[tuple[Object, SE2Pose]]]:
    """Move objects that are in contact with the robot's suctioned objects."""
    moved_objects = []
    moving_objects = {robot} | {o for o, _ in suctioned_objs}
    nonstatic_objects = {
        o for o in state if (o not in moving_objects) and (not state.get(o, "static"))
    }

    for contact_obj in nonstatic_objects:
        for suctioned_obj, _ in suctioned_objs:
            suctioned_body = object_to_multibody2d(suctioned_obj, state, {})
            contact_body = object_to_multibody2d(contact_obj, state, {})
            for b1 in suctioned_body.bodies:
                for b2 in contact_body.bodies:
                    if geom2ds_intersect(b1.geom, b2.geom):
                        closest_points_b1, closest_points_b2, _ = find_closest_points(
                            b1.geom, b2.geom
                        )
                        contact_vec = np.array(closest_points_b2) - np.array(
                            closest_points_b1
                        )

                        current_x = state.get(contact_obj, "x")
                        current_y = state.get(contact_obj, "y")

                        new_x = current_x + contact_vec[0]
                        new_y = current_y + contact_vec[1]

                        state.set(contact_obj, "x", new_x)
                        state.set(contact_obj, "y", new_y)

                        moved_objects.append(
                            (contact_obj, get_se2_pose(state, contact_obj))
                        )

                        return state, set(moved_objects)  # Stop checking other objects

    return state, set(moved_objects)


def run_motion_planning_for_crv_robot(
    state: ObjectCentricState,
    robot: Object,
    target_pose: SE2Pose,
    action_space: CRVRobotActionSpace,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
) -> list[SE2Pose] | None:
    """Run motion planning in an environment with a CRV action space."""
    if static_object_body_cache is None:
        static_object_body_cache = {}

    rng = np.random.default_rng(seed)

    # Use the object positions in the state to create a rough room boundary.
    x_lb, x_ub, y_lb, y_ub = np.inf, -np.inf, np.inf, -np.inf
    for obj in state:
        pose = get_se2_pose(state, obj)
        x_lb = min(x_lb, pose.x)
        x_ub = max(x_ub, pose.x)
        y_lb = min(y_lb, pose.y)
        y_ub = max(y_ub, pose.y)

    # Create a static version of the state so that the geoms only need to be
    # instantiated once during motion planning (except for the robot). Make
    # sure to not update the global cache because we don't want to carry over
    # static things that are not actually static.
    static_object_body_cache = static_object_body_cache.copy()
    suctioned_objects = get_suctioned_objects(state, robot)
    moving_objects = {robot} | {o for o, _ in suctioned_objects}
    static_state = state.copy()
    for o in static_state:
        if o in moving_objects:
            continue
        static_state.set(o, "static", 1.0)

    # Uncomment to visualize the scene.
    # import matplotlib.pyplot as plt
    # import imageio.v2 as iio
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # render_state_on_ax(static_state, ax)
    # goal_state = static_state.copy()
    # goal_state.set(robot, "x", target_pose.x)
    # goal_state.set(robot, "y", target_pose.y)
    # goal_state.set(robot, "theta", target_pose.theta)
    # snap_suctioned_objects(goal_state, robot, suctioned_objects)
    # goal_robot_mb = _robot_to_multibody2d(robot, goal_state)
    # for body in goal_robot_mb.bodies:
    #     body.rendering_kwargs["facecolor"] = "pink"
    #     body.rendering_kwargs["alpha"] = 0.5
    # goal_robot_mb.plot(ax)
    # ax.set_xlim(-1, 11)
    # ax.set_ylim(-1, 11)
    # img = fig2data(fig)
    # import ipdb; ipdb.set_trace()

    # Set up the RRT methods.
    def sample_fn(_: SE2Pose) -> SE2Pose:
        """Sample a robot pose."""
        x = rng.uniform(x_lb, x_ub)
        y = rng.uniform(y_lb, y_ub)
        theta = rng.uniform(-np.pi, np.pi)
        return SE2Pose(x, y, theta)

    def extend_fn(pt1: SE2Pose, pt2: SE2Pose) -> Iterable[SE2Pose]:
        """Interpolate between the two poses."""
        # Make sure that we obey the bounds on actions.
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        assert isinstance(action_space, CRVRobotActionSpace)
        abs_x = action_space.high[0] if dx > 0 else action_space.low[0]
        abs_y = action_space.high[1] if dy > 0 else action_space.low[1]
        abs_theta = action_space.high[2] if dtheta > 0 else action_space.low[2]
        x_num_steps = int(dx / abs_x) + 1
        assert x_num_steps > 0
        y_num_steps = int(dy / abs_y) + 1
        assert y_num_steps > 0
        theta_num_steps = int(dtheta / abs_theta) + 1
        assert theta_num_steps > 0
        num_steps = max(x_num_steps, y_num_steps, theta_num_steps)
        x = pt1.x
        y = pt1.y
        theta = pt1.theta
        yield SE2Pose(x, y, theta)
        for _ in range(num_steps):
            x += dx / num_steps
            y += dy / num_steps
            theta = wrap_angle(theta + dtheta / num_steps)
            yield SE2Pose(x, y, theta)

    def collision_fn(pt: SE2Pose) -> bool:
        """Check for collisions if the robot were at this pose."""

        # Update the static state with the robot's new hypothetical pose.
        static_state.set(robot, "x", pt.x)
        static_state.set(robot, "y", pt.y)
        static_state.set(robot, "theta", pt.theta)

        # Update the suctioned objects in the static state.
        snap_suctioned_objects(static_state, robot, suctioned_objects)
        obstacle_objects = set(static_state) - moving_objects

        return state_has_collision(
            static_state, moving_objects, obstacle_objects, static_object_body_cache
        )

    def distance_fn(pt1: SE2Pose, pt2: SE2Pose) -> float:
        """Return a distance between the two points."""
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        return np.sqrt(dx**2 + dy**2) + abs(dtheta)

    birrt = BiRRT(
        sample_fn,
        extend_fn,
        collision_fn,
        distance_fn,
        rng,
        num_attempts,
        num_iters,
        smooth_amt,
    )

    initial_pose = get_se2_pose(state, robot)
    return birrt.query(initial_pose, target_pose)


def crv_pose_plan_to_action_plan(
    pose_plan: list[SE2Pose],
    action_space: CRVRobotActionSpace,
    vacuum_while_moving: bool = False,
) -> list[Array]:
    """Convert a CRV robot pose plan into corresponding actions."""
    action_plan: list[Array] = []
    for pt1, pt2 in zip(pose_plan[:-1], pose_plan[1:]):
        action = np.zeros_like(action_space.high)
        action[0] = pt2.x - pt1.x
        action[1] = pt2.y - pt1.y
        action[2] = get_signed_angle_distance(pt2.theta, pt1.theta)
        action[4] = 1.0 if vacuum_while_moving else 0.0
        action_plan.append(action)
    return action_plan


def is_inside(
    state: ObjectCentricState,
    inner: Object,
    outer: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> bool:
    """Checks if the inner object is completely inside the outer one.

    Only rectangles are currently supported.
    """
    inner_geom = rectangle_object_to_geom(state, inner, static_object_cache)
    outer_geom = rectangle_object_to_geom(state, outer, static_object_cache)
    for x, y in inner_geom.vertices:
        if not outer_geom.contains_point(x, y):
            return False
    return True


def is_inside_shelf(
    state: ObjectCentricState,
    inner: Object,
    outer: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> bool:
    """Checks if the inner object is completely inside the outer shelf.

    The outer object is assumed to be a double rectangle type. (shelf)
    """
    assert outer.is_instance(
        DoubleRectType
    ), "Outer object must be a shelf (DoubleRectType)."
    inner_geom = rectangle_object_to_geom(state, inner, static_object_cache)
    outer_geom = double_rectangle_object_to_part_geom(state, outer, static_object_cache)
    for x, y in inner_geom.vertices:
        if not outer_geom.contains_point(x, y):
            return False
    return True


def is_on(
    state: ObjectCentricState,
    top: Object,
    bottom: Object,
    static_object_cache: dict[Object, MultiBody2D],
    tol: float = 0.025,
) -> bool:
    """Checks top object is completely on the bottom one.

    Only rectangles are currently supported.

    Assumes that "up" is positive y.
    """
    top_geom = rectangle_object_to_geom(state, top, static_object_cache)
    bottom_geom = rectangle_object_to_geom(state, bottom, static_object_cache)
    # The bottom-most vertices of top_geom should be contained within the bottom
    # geom when those vertices are offset by tol.
    sorted_vertices = sorted(top_geom.vertices, key=lambda v: v[1])
    for x, y in sorted_vertices[:2]:
        offset_y = y - tol
        if not bottom_geom.contains_point(x, offset_y):
            return False
    return True


def is_movable_rectangle(state: ObjectCentricState, obj: Object) -> bool:
    """Checks if an object is a movable rectangle."""
    return obj.is_instance(RectangleType) and state.get(obj, "static") < 0.5


def get_geom2d_crv_robot_action_from_gui_input(
    action_space: CRVRobotActionSpace, gui_input: dict[str, Any]
) -> NDArray[np.float32]:
    """Get the mapping from human inputs to actions, derived from action space."""
    # Unpack the input.
    keys_pressed = gui_input["keys"]
    right_x, right_y = gui_input["right_stick"]
    left_x, _ = gui_input["left_stick"]

    # Initialize the action.
    low = action_space.low
    high = action_space.high
    action = np.zeros(action_space.shape, action_space.dtype)

    def _rescale(x: float, lb: float, ub: float) -> float:
        """Rescale from [-1, 1] to [lb, ub]."""
        return lb + (x + 1) * (ub - lb) / 2

    # The right stick controls the x, y movement of the base.
    action[0] = _rescale(right_x, low[0], high[0])
    action[1] = _rescale(right_y, low[1], high[1])

    # The left stick controls the rotation of the base. Only the x axis
    # is used right now.
    action[2] = _rescale(left_x, low[2], high[2])

    # The w/s mouse keys are used to adjust the robot arm.
    if "w" in keys_pressed:
        action[3] = low[3]
    if "s" in keys_pressed:
        action[3] = high[3]

    # The space bar is used to turn on the vacuum.
    if "space" in keys_pressed:
        action[4] = 1.0

    return action
