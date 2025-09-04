"""Utilities for Dynamic2D PyMunk-based environments."""

from typing import Any

import numpy as np
import pymunk
from gymnasium.spaces import Box
from numpy.typing import NDArray
from pymunk.vec2d import Vec2d

from prbench.envs.geom2d.structs import SE2Pose

# Collision types from the basic_pymunk.py script
STATIC_COLLISION_TYPE = 0
DYNAMIC_COLLISION_TYPE = 1
ROBOT_COLLISION_TYPE = 2
GRIPPER_COLLISION_TYPE = 3
HELD_OBJECT_COLLISION_TYPE = 4


class FingeredRobotActionSpace(Box):
    """An action space for a fingered robot with gripper control.

    Actions are bounded relative movements of the base, arm extension, and gripper opening/closing.
    """

    def __init__(
        self,
        min_dx: float = -2.5,
        max_dx: float = 2.5,
        min_dy: float = -2.5,
        max_dy: float = 2.5,
        min_dtheta: float = -np.pi / 16,
        max_dtheta: float = np.pi / 16,
        min_darm: float = -5.0,
        max_darm: float = 5.0,
        min_dgripper: float = -1.0,
        max_dgripper: float = 1.0,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_dgripper])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_dgripper])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        features = [
            ("dx", "Change in robot x position (positive is right)"),
            ("dy", "Change in robot y position (positive is up)"),
            ("dtheta", "Change in robot angle in radians (positive is ccw)"),
            ("darm", "Change in robot arm length (positive is out)"),
            ("dgripper", "Change in gripper gap (positive is open)"),
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


class KinRobot:
    """Kinematic robot implementation using PyMunk physics engine."""

    def __init__(
        self,
        init_pos: Vec2d = Vec2d(100, 300),
        base_radius: float = 30,
        arm_length_max: float = 60,
        gripper_base_width: float = 4,
        gripper_base_height: float = 50,
        gripper_finger_width: float = 24,
        gripper_finger_height: float = 4,
        gripper_gap_max: float = 50,
    ) -> None:
        # Robot parameters
        self.base_radius = base_radius
        self.gripper_base_width = gripper_base_width
        self.gripper_base_height = gripper_base_height
        self.gripper_finger_width = gripper_finger_width
        self.gripper_finger_height = gripper_finger_height
        self.arm_length_max = arm_length_max
        self.gripper_gap_max = gripper_gap_max

        # Track last robot state
        self._base_position = init_pos
        self._base_angle = 0.0
        self._arm_length = base_radius
        self._gripper_gap = gripper_finger_height
        self.held_objects: list[tuple[pymunk.Body, pymunk.Shape]] = []

        # Body and shape references
        self.base_body: pymunk.Body | None = None
        self.base_shape: pymunk.Shape | None = None
        self.gripper_base_body: pymunk.Body | None = None
        self.gripper_base_shape: pymunk.Shape | None = None
        self.left_finger_body: pymunk.Body | None = None
        self.left_finger_shape: pymunk.Shape | None = None
        self.right_finger_body: pymunk.Body | None = None
        self.right_finger_shape: pymunk.Shape | None = None

        self.create_components()

    def create_components(self) -> None:
        """Create all robot components."""
        self.create_base()
        self.create_gripper_base()
        self.left_finger_body, self.left_finger_shape = self.create_finger()
        self.right_finger_body, self.right_finger_shape = self.create_finger(left=False)

    def add_to_space(self, space: pymunk.Space) -> None:
        """Add robot components to the PyMunk space."""
        if self.base_body and self.base_shape:
            space.add(self.base_body, self.base_shape)
        if self.gripper_base_body and self.gripper_base_shape:
            space.add(self.gripper_base_body, self.gripper_base_shape)
        if self.left_finger_body and self.left_finger_shape:
            space.add(self.left_finger_body, self.left_finger_shape)
        if self.right_finger_body and self.right_finger_shape:
            space.add(self.right_finger_body, self.right_finger_shape)

    def create_base(self) -> tuple[pymunk.Body, pymunk.Shape]:
        """Create the robot base."""
        self.base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.base_shape = pymunk.Circle(self.base_body, self.base_radius)
        self.base_shape.color = (255, 50, 50, 255)
        self.base_shape.friction = 1
        self.base_shape.collision_type = ROBOT_COLLISION_TYPE
        self.base_shape.density = 1.0
        self.base_body.position = self._base_position
        self.base_body.angle = self._base_angle
        return self.base_body, self.base_shape

    @property
    def base_pose(self) -> SE2Pose:
        """Get the base pose as SE2Pose."""
        if self.base_body is None:
            return SE2Pose(x=0.0, y=0.0, theta=0.0)
        return SE2Pose(
            x=self.base_body.position.x,
            y=self.base_body.position.y,
            theta=self.base_body.angle,
        )

    def create_gripper_base(self) -> tuple[pymunk.Body, pymunk.Shape]:
        """Create the gripper base."""
        half_w = self.gripper_base_width / 2
        half_h = self.gripper_base_height / 2
        vs = [(-half_w, half_h), (-half_w, -half_h), (half_w, -half_h), (half_w, half_h)]
        self.gripper_base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.gripper_base_shape = pymunk.Poly(self.gripper_base_body, vs)
        self.gripper_base_shape.friction = 1
        self.gripper_base_shape.collision_type = ROBOT_COLLISION_TYPE
        self.gripper_base_shape.density = 1.0

        init_rel_pos = SE2Pose(x=self._arm_length, y=0.0, theta=0.0)
        init_pose = self.base_pose * init_rel_pos
        self.gripper_base_body.position = (init_pose.x, init_pose.y)
        self.gripper_base_body.angle = init_pose.theta
        return self.gripper_base_body, self.gripper_base_shape

    @property
    def gripper_base_pose(self) -> SE2Pose:
        """Get the gripper base pose as SE2Pose."""
        if self.gripper_base_body is None:
            return SE2Pose(x=0.0, y=0.0, theta=0.0)
        return SE2Pose(
            x=self.gripper_base_body.position.x,
            y=self.gripper_base_body.position.y,
            theta=self.gripper_base_body.angle,
        )

    def create_finger(self, left: bool = True) -> tuple[pymunk.Body, pymunk.Shape]:
        """Create a gripper finger."""
        half_w = self.gripper_finger_width / 2
        half_h = self.gripper_finger_height / 2
        vs = [(-half_w, half_h), (-half_w, -half_h), (half_w, -half_h), (half_w, half_h)]
        finger_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        finger_shape = pymunk.Poly(finger_body, vs)
        finger_shape.friction = 1
        finger_shape.density = 1.0

        if left:
            init_rel_pos = SE2Pose(
                x=half_w, y=self._gripper_gap / 2, theta=0.0
            )
            finger_shape.collision_type = GRIPPER_COLLISION_TYPE
        else:
            init_rel_pos = SE2Pose(
                x=half_w, y=-self._gripper_gap / 2, theta=0.0
            )
            finger_shape.collision_type = ROBOT_COLLISION_TYPE
        init_pose = self.gripper_base_pose * init_rel_pos
        finger_body.position = (init_pose.x, init_pose.y)
        finger_body.angle = init_pose.theta
        return finger_body, finger_shape

    @property
    def left_finger_pose(self) -> SE2Pose:
        """Get the left finger pose as SE2Pose."""
        if self.left_finger_body is None:
            return SE2Pose(x=0.0, y=0.0, theta=0.0)
        return SE2Pose(
            x=self.left_finger_body.position.x,
            y=self.left_finger_body.position.y,
            theta=self.left_finger_body.angle,
        )

    @property
    def is_opening_finger(self) -> bool:
        """Check if the gripper is opening."""
        current_relative_finger_pose = self.gripper_base_pose.inverse * self.left_finger_pose
        return (current_relative_finger_pose.y - 0.1) >= (self._gripper_gap / 2)

    @property
    def is_closing_finger(self) -> bool:
        """Check if the gripper is closing."""
        current_relative_finger_pose = self.gripper_base_pose.inverse * self.left_finger_pose
        return (current_relative_finger_pose.y + 0.1) <= (self._gripper_gap / 2)

    def reset_last_state(self) -> None:
        """Reset to last state when collide with static objects."""
        if self.base_body:
            self.base_body.position = self._base_position
            self.base_body.angle = self._base_angle
        
        gripper_base_rel_pos = SE2Pose(x=self._arm_length, y=0.0, theta=0.0)
        gripper_base_pose = self.base_pose * gripper_base_rel_pos
        if self.gripper_base_body:
            self.gripper_base_body.position = (gripper_base_pose.x, gripper_base_pose.y)
            self.gripper_base_body.angle = gripper_base_pose.theta
        
        left_finger_rel_pos = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=self._gripper_gap / 2,
            theta=0.0,
        )
        left_finger_pose = gripper_base_pose * left_finger_rel_pos
        if self.left_finger_body:
            self.left_finger_body.position = (left_finger_pose.x, left_finger_pose.y)
            self.left_finger_body.angle = left_finger_pose.theta
        
        right_finger_rel_pos = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=-self._gripper_gap / 2,
            theta=0.0,
        )
        right_finger_pose = gripper_base_pose * right_finger_rel_pos
        if self.right_finger_body:
            self.right_finger_body.position = (right_finger_pose.x, right_finger_pose.y)
            self.right_finger_body.angle = right_finger_pose.theta

        # Update held objects
        for obj, relative_pose in self.held_objects:
            new_obj_pose = gripper_base_pose * relative_pose
            obj.position = (new_obj_pose.x, new_obj_pose.y)
            obj.angle = new_obj_pose.theta

    def update_last_state(self) -> None:
        """Update the last state tracking variables."""
        if self.base_body:
            self._base_position = Vec2d(self.base_body.position.x, self.base_body.position.y)
            self._base_angle = self.base_body.angle
        
        relative_pose = self.base_pose.inverse * self.gripper_base_pose
        self._arm_length = relative_pose.x
        relative_finger_pose = self.gripper_base_pose.inverse * self.left_finger_pose
        self._gripper_gap = relative_finger_pose.y * 2.0

    def update(
        self,
        dx: float,
        dy: float,
        dtheta: float,
        darm: float,
        dgripper: float,
        space: pymunk.Space,
    ) -> None:
        """Update robot position and configuration."""
        # Update robot last state
        self.update_last_state()
        # Clip arm length and gripper gap
        relative_y = max(
            min(self._gripper_gap / 2 + dgripper, self.gripper_gap_max // 2),
            self.gripper_finger_height,
        )
        relative_x = max(
            min(self._arm_length + darm, self.arm_length_max), self.base_radius
        )
        # Update positions in simulation
        self.update_positions(dx, dy, dtheta, relative_x, relative_y, space)

    def update_positions(
        self,
        dx: float,
        dy: float,
        dtheta: float,
        arm_length: float,
        finger_relative_y: float,
        space: pymunk.Space,
    ) -> None:
        """Update robot component positions in the simulation."""
        if not self.base_body:
            return
            
        curr_base_x, curr_base_y = self.base_body.position
        curr_base_theta = self.base_body.angle
        self.base_body.position = (curr_base_x + dx, curr_base_y + dy)
        self.base_body.angle = curr_base_theta + dtheta

        body_pose = SE2Pose(
            x=self.base_body.position.x,
            y=self.base_body.position.y,
            theta=self.base_body.angle,
        )

        relative_pose_gripper_base = SE2Pose(x=arm_length, y=0.0, theta=0.0)
        gripper_base_pose = body_pose * relative_pose_gripper_base
        if self.gripper_base_body:
            self.gripper_base_body.position = (gripper_base_pose.x, gripper_base_pose.y)
            self.gripper_base_body.angle = gripper_base_pose.theta

        new_relative_finger_pose_l = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=finger_relative_y,
            theta=0.0,
        )
        new_relative_finger_pose_r = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=-finger_relative_y,
            theta=0.0,
        )

        l_finger_pose = gripper_base_pose * new_relative_finger_pose_l
        if self.left_finger_body:
            self.left_finger_body.position = (l_finger_pose.x, l_finger_pose.y)
            self.left_finger_body.angle = l_finger_pose.theta

        r_finger_pose = gripper_base_pose * new_relative_finger_pose_r
        if self.right_finger_body:
            self.right_finger_body.position = (r_finger_pose.x, r_finger_pose.y)
            self.right_finger_body.angle = r_finger_pose.theta

        # Check if we should release held objects
        if self.is_opening_finger:
            for obj, _ in self.held_objects:
                self.del_from_hand_space((obj, obj.shapes[0]), space)
            self.held_objects = []

        # Update held objects
        for obj, relative_pose in self.held_objects:
            new_obj_pose = gripper_base_pose * relative_pose
            obj.position = (new_obj_pose.x, new_obj_pose.y)
            obj.angle = new_obj_pose.theta

    def is_grasping(self, contact_point_set: pymunk.ContactPointSet, tgt_body: pymunk.Body) -> bool:
        """Check if robot is grasping a target body."""
        # Checker 0: If robot is closing gripper
        if not self.is_closing_finger:
            return False
        # Checker 1: If contact normal is roughly perpendicular 
        # to gripper_base_body base
        normal = contact_point_set.normal
        if not self.gripper_base_body:
            return False
        dtheta = abs(self.gripper_base_body.angle - normal.angle)
        dtheta = min(dtheta, 2 * np.pi - dtheta)
        theta_ok = abs(dtheta - np.pi / 2) < 0.1
        if not theta_ok:
            return False
        # Checker 2: If the body is within the grasping area
        p_a = SE2Pose(x=tgt_body.position.x, y=tgt_body.position.y, theta=0.0)
        rel_a = self.gripper_base_pose.inverse * p_a
        if (abs(rel_a.y) < self.gripper_base_height / 4) and (
            abs(rel_a.x) < self.gripper_finger_width
        ):
            print("Grasped!")
            return True
        return False

    def add_to_hand(self, obj: tuple[pymunk.Body, pymunk.Shape]) -> None:
        """Add an object to the robot's hand."""
        obj_body, _ = obj
        obj_pose = SE2Pose(x=obj_body.position.x, y=obj_body.position.y, theta=obj_body.angle)
        gripper_base_pose = SE2Pose(
            x=self.gripper_base_body.position.x,
            y=self.gripper_base_body.position.y,
            theta=self.gripper_base_body.angle,
        )
        relative_obj_pose = gripper_base_pose.inverse * obj_pose
        self.held_objects.append((obj_body, relative_obj_pose))

    def del_from_hand_space(self, obj: tuple[pymunk.Body, pymunk.Shape], space: pymunk.Space) -> None:
        """Remove an object from hand and make it dynamic."""
        mass = 1.0
        kinematic_body, kinematic_shape = obj
        points = kinematic_shape.get_vertices()
        moment = pymunk.moment_for_poly(mass, points, (0, 0))
        dynamic_body = pymunk.Body(mass, moment)
        dynamic_body.position = kinematic_body.position
        dynamic_body.angle = kinematic_body.angle
        shape = pymunk.Poly(dynamic_body, points)
        shape.friction = 1
        shape.density = 1.0
        shape.collision_type = DYNAMIC_COLLISION_TYPE
        space.add(dynamic_body, shape)
        space.remove(kinematic_body, kinematic_shape)


def on_gripper_grasp(arbiter: pymunk.Arbiter, space: pymunk.Space, robot: KinRobot) -> bool:
    """Collision callback for gripper grasping objects."""
    print("Gripper Collision detected!")
    dynamic_body = arbiter.bodies[0]
    if robot.is_grasping(arbiter.contact_point_set, dynamic_body):
        # Create a new kinematic object
        kinematic_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        kinematic_body.position = dynamic_body.position
        kinematic_body.angle = dynamic_body.angle
        points = arbiter.shapes[0].get_vertices()
        shape = pymunk.Poly(kinematic_body, points)
        shape.friction = 1
        shape.density = 1.0
        shape.collision_type = HELD_OBJECT_COLLISION_TYPE
        space.add(kinematic_body, shape)
        robot.add_to_hand((kinematic_body, shape))
        # Remove the dynamic body from the space
        space.remove(dynamic_body, arbiter.shapes[0])
        return True
    return False


def on_collision_w_static(arbiter: pymunk.Arbiter, 
                          space: pymunk.Space, 
                          robot: KinRobot) -> bool:
    """Collision callback for robot colliding with static objects."""
    del arbiter
    del space
    print("Static Collision detected!")
    robot.reset_last_state()
    return True


def get_fingered_robot_action_from_gui_input(
    action_space: FingeredRobotActionSpace, gui_input: dict[str, Any]
) -> NDArray[np.float32]:
    """Get the mapping from human inputs to actions, derived from action space."""
    # This will be implemented later - placeholder for now
    del gui_input  # Unused for now
    action = np.zeros(action_space.shape, action_space.dtype)
    return action