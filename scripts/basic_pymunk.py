"""Showcase of flying arrows that can stick to objects in a somewhat
realistic looking way.
"""

import sys
import numpy as np

import pygame

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

from prbench.envs.geom2d.structs import SE2Pose

STATIC_COLLISION_TYPE = 0
DYNAMIC_COLLISION_TYPE = 1
ROBOT_COLLISION_TYPE = 2
GRIPPER_COLLISION_TYPE = 3
HELD_OBJECT_COLLISION_TYPE = 4



class KinRobot:
    def __init__(self, 
                 init_pos=Vec2d(100, 500),
                 base_radius=30, 
                 arm_length_max=60, 
                 gripper_base_width=4,
                 gripper_base_height=50,
                 gripper_finger_width=24,
                 gripper_finger_height=4,
                 gripper_gap_max=50):
        self.base_radius = base_radius
        self.gripper_base_width = gripper_base_width
        self.gripper_base_height = gripper_base_height
        self.gripper_finger_width = gripper_finger_width
        self.gripper_finger_height = gripper_finger_height
        self.arm_length_max = arm_length_max
        self.gripper_gap_max = gripper_gap_max
        self.base_position = init_pos
        self.base_angle = 0.0
        self.held_objects = []
        
        self.base_body = None
        self.base_shape = None
        self.gripper_base_body = None
        self.gripper_base_shape = None
        self.left_finger_body = None
        self.left_finger_shape = None
        self.right_finger_body = None
        self.right_finger_shape = None
        
        self.create_components()
    
    def create_base(self):
        self.base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.base_shape = pymunk.Circle(self.base_body, self.base_radius)
        self.base_shape.color = (255, 50, 50, 255)
        self.base_shape.friction = 1
        self.base_shape.collision_type = ROBOT_COLLISION_TYPE
        self.base_shape.density = 1.0
        self.base_body.position = self.base_position
        self.base_body.angle = self.base_angle
        return self.base_body, self.base_shape
    
    def create_gripper_base(self):
        half_w = self.gripper_base_width / 2
        half_h = self.gripper_base_height / 2
        vs = [(-half_w, half_h), (-half_w, -half_h), (half_w, -half_h), (half_w, half_h)]
        self.gripper_base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.gripper_base_shape = pymunk.Poly(self.gripper_base_body, vs)
        self.gripper_base_shape.friction = 1
        self.gripper_base_shape.collision_type = ROBOT_COLLISION_TYPE
        self.gripper_base_shape.density = 1.0

        init_rel_pos = SE2Pose(x=self.base_radius, y=0.0, theta=0.0)
        init_pose = SE2Pose(x=self.base_body.position.x, y=self.base_body.position.y, \
                            theta=self.base_body.angle) * init_rel_pos
        self.gripper_base_body.position = (init_pose.x, init_pose.y)
        self.gripper_base_body.angle = init_pose.theta
        return self.gripper_base_body, self.gripper_base_shape
    
    def create_finger(self, left=True):
        half_w = self.gripper_finger_width / 2
        half_h = self.gripper_finger_height / 2
        vs = [(-half_w, half_h), (-half_w, -half_h), (half_w, -half_h), (half_w, half_h)]
        finger_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        finger_shape = pymunk.Poly(finger_body, vs)
        finger_shape.friction = 1
        finger_shape.density = 1.0

        if left:
            init_rel_pos = SE2Pose(x=half_w, \
                                   y=self.gripper_gap_max / 2, theta=0.0)
            finger_shape.collision_type = GRIPPER_COLLISION_TYPE
        else:
            init_rel_pos = SE2Pose(x=half_w, \
                                   y=-self.gripper_gap_max / 2, theta=0.0)
            finger_shape.collision_type = ROBOT_COLLISION_TYPE
        init_pose = SE2Pose(x=self.gripper_base_body.position.x, y=self.gripper_base_body.position.y, \
                            theta=self.gripper_base_body.angle) * init_rel_pos
        finger_body.position = (init_pose.x, init_pose.y)
        finger_body.angle = init_pose.theta
        return finger_body, finger_shape
    
    def create_components(self):
        self.create_base()
        self.create_gripper_base()
        self.left_finger_body, self.left_finger_shape = self.create_finger()
        self.right_finger_body, self.right_finger_shape = self.create_finger(left=False)
    
    def add_to_space(self, space):
        space.add(self.base_body, self.base_shape)
        space.add(self.gripper_base_body, self.gripper_base_shape)
        space.add(self.left_finger_body, self.left_finger_shape)
        space.add(self.right_finger_body, self.right_finger_shape)
    
    def update(self, dx, dy, dtheta, darm, dgripper):        
        current_finger_pose = SE2Pose(
            x=self.left_finger_body.position.x,
            y=self.left_finger_body.position.y,
            theta=self.left_finger_body.angle,
        )
        current_body_pose = SE2Pose(x=self.base_body.position.x, y=self.base_body.position.y, \
                            theta=self.base_body.angle)
        current_finger_base_pose = SE2Pose(x=self.gripper_base_body.position.x, y=self.gripper_base_body.position.y, \
                            theta=self.gripper_base_body.angle)
        
        current_relative_finger_pose = current_finger_base_pose.inverse * current_finger_pose
        relative_y = max(min(current_relative_finger_pose.y + dgripper, self.gripper_gap_max // 2), self.gripper_finger_height)

        current_relative_finger_base_pose = current_body_pose.inverse * current_finger_base_pose
        relative_x = max(min(current_relative_finger_base_pose.x + darm, self.arm_length_max), self.base_radius)
        
        self.update_positions(dx, dy, dtheta, relative_x, relative_y)
    
    def update_positions(self, dx, dy, dtheta, arm_length, finger_relative_y):
        curr_base_x, curr_base_y = self.base_body.position
        curr_base_theta = self.base_body.angle
        self.base_body.position = (curr_base_x + dx, curr_base_y + dy)
        self.base_body.angle = curr_base_theta + dtheta
        
        body_pose = SE2Pose(x=self.base_body.position.x, \
                            y=self.base_body.position.y, \
                                theta=self.base_body.angle)
        
        relative_pose_gripper_base = SE2Pose(x=arm_length, y=0.0, theta=0.0)
        gripper_base_pose = body_pose * relative_pose_gripper_base
        self.gripper_base_body.position = (gripper_base_pose.x, gripper_base_pose.y)
        self.gripper_base_body.angle = gripper_base_pose.theta
        
        new_relative_finger_pose_l = SE2Pose(x=self.gripper_finger_width / 2, \
                                             y=finger_relative_y, theta=0.0)
        new_relative_finger_pose_r = SE2Pose(x=self.gripper_finger_width / 2, \
                                             y=-finger_relative_y, theta=0.0)
        
        l_finger_pose = gripper_base_pose * new_relative_finger_pose_l
        self.left_finger_body.position = (l_finger_pose.x, l_finger_pose.y)
        self.left_finger_body.angle = l_finger_pose.theta
        
        r_finger_pose = gripper_base_pose * new_relative_finger_pose_r
        self.right_finger_body.position = (r_finger_pose.x, r_finger_pose.y)
        self.right_finger_body.angle = r_finger_pose.theta

        # Update held objects
        for i, (obj, relative_pose) in enumerate(self.held_objects):
            new_obj_pose = gripper_base_pose * relative_pose
            obj[0].position = (new_obj_pose.x, new_obj_pose.y)
            obj[0].angle = new_obj_pose.theta

    def is_grasping(self, contact_point_set):
        normal = contact_point_set.normal
        dtheta = abs(self.gripper_base_body.angle - normal.angle)
        dtheta = min(dtheta, 2 * np.pi - dtheta)
        # TODO: also check contact point is within finger bounds and far from the gripper border
        return abs(dtheta - np.pi / 2) < 0.1

    def add_to_hand(self, obj):
        obj_pose = SE2Pose(x=obj[0].position.x, y=obj[0].position.y, theta=obj[0].angle)
        gripper_base_pose = SE2Pose(x=self.gripper_base_body.position.x, \
                                    y=self.gripper_base_body.position.y, \
                                    theta=self.gripper_base_body.angle)
        relative_obj_pose = gripper_base_pose.inverse * obj_pose
        self.held_objects.append((obj, relative_obj_pose))

    def del_from_hand(self, obj):
        for i, (held_obj, _) in enumerate(self.held_objects):
            if held_obj == obj:
                self.held_objects.pop(i)
                return

def on_gripper_grasp(arbiter, space, data):
    print(f"Collision detected!")
    dynamic_body = arbiter.bodies[0]
    if data[0].is_grasping(arbiter.contact_point_set):
        # Create a new kinematic object
        kinematic_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        kinematic_body.position = dynamic_body.position
        kinematic_body.angle = dynamic_body.angle
        points = data[1]
        shape = pymunk.Poly(kinematic_body, points)
        shape.friction = 1
        shape.density = 1.0
        shape.collision_type = HELD_OBJECT_COLLISION_TYPE
        space.add(kinematic_body, shape)
        data[0].add_to_hand((kinematic_body, shape))
        # Remove the dynamic body from the space
        space.remove(dynamic_body, arbiter.shapes[0])
        return True

def on_gripper_leave(arbiter, space, data):
    print(f"Gripper leave detected!")
    kinematic_body = arbiter.bodies[0]
    # Create a new kinematic object
    points = data[1]
    mass = 1.0
    moment = pymunk.moment_for_poly(mass, points, (0, 0))
    dynamic_body = pymunk.Body(mass, moment)
    dynamic_body.position = kinematic_body.position
    dynamic_body.angle = kinematic_body.angle
    shape = pymunk.Poly(dynamic_body, points)
    shape.friction = 1
    shape.density = 1.0
    shape.collision_type = DYNAMIC_COLLISION_TYPE
    space.add(dynamic_body, shape)
    data[0].del_from_hand((kinematic_body, arbiter.shapes[0]))
    # Remove the dynamic body from the space
    space.remove(kinematic_body, arbiter.shapes[0])
    return True

width, height = 690, 600


def main():
    ### PyGame init
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    running = True
    font = pygame.font.SysFont("Arial", 16)

    ### Physics stuff
    space = pymunk.Space()
    space.gravity = 0, 1000
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Ground
    shape = pymunk.Segment(space.static_body, (5, 500), (595, 500), 1.0)
    shape.friction = 1.0
    shape.collision_type = STATIC_COLLISION_TYPE
    space.add(shape)
    size = 15

    points = [(-size, -size), (-size, size), (size, size), (size, -size)]
    mass = 1.0
    moment = pymunk.moment_for_poly(mass, points, (0, 0))
    b1 = pymunk.Body(mass, moment)
    b1.position = Vec2d(50, 450)
    shape = pymunk.Poly(b1, points)
    shape.friction = 1
    shape.density = 1.0
    shape.collision_type = DYNAMIC_COLLISION_TYPE
    space.add(b1, shape)

    b2 = pymunk.Body(mass, moment)
    b2.position = Vec2d(200, 450)
    shape = pymunk.Poly(b2, points)
    shape.friction = 1
    shape.density = 1.0
    shape.collision_type = DYNAMIC_COLLISION_TYPE
    space.add(b2, shape)

    # Create robot
    robot = KinRobot()
    robot.add_to_space(space)

    # Collision handler
    space.on_collision(DYNAMIC_COLLISION_TYPE, GRIPPER_COLLISION_TYPE, post_solve=on_gripper_grasp, data=[robot, points])

    space.on_collision(HELD_OBJECT_COLLISION_TYPE, GRIPPER_COLLISION_TYPE, separate=on_gripper_leave, data=[robot, points])


    while running:
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and (event.key in [pygame.K_ESCAPE, pygame.K_q])
            ):
                running = False

        keys = pygame.key.get_pressed()
        mouse_position = pymunk.pygame_util.from_pygame(
            Vec2d(*pygame.mouse.get_pos()), screen
        )
        mouse_pressed = pygame.mouse.get_pressed()
        
        # Calculate movement deltas from inputs
        dx, dy = 0.0, 0.0
        speed = 2.5
        if keys[pygame.K_w]:
            dy = -speed
        if keys[pygame.K_s]:
            dy = speed
        if keys[pygame.K_a]:
            dx = -speed
        if keys[pygame.K_d]:
            dx = speed
        
        # Calculate rotation from mouse position
        target_angle = (mouse_position - robot.base_body.position).angle
        dtheta = target_angle - robot.base_body.angle
        
        # Calculate gripper movement
        dgripper = 0.0
        gripper_speed = 1.0
        if mouse_pressed[0]:
            dgripper = -gripper_speed
        else:
            dgripper = gripper_speed

        # Calculate arm movement
        darm = 0.0
        arm_speed = 5.0
        if keys[pygame.K_SPACE]:
            darm = arm_speed
        else:
            darm = -arm_speed
        
        robot.update(dx, dy, dtheta, darm, dgripper)

        ### Clear screen
        screen.fill(pygame.Color("black"))

        ### Draw stuff
        space.debug_draw(draw_options)
        # draw(screen, space)

        # Info and flip screen
        screen.blit(
            font.render("fps: " + str(clock.get_fps()), True, pygame.Color("white")),
            (0, 0),
        )
        screen.blit(
            font.render("Press ESC or Q to quit", True, pygame.Color("darkgrey")),
            (5, height - 20),
        )

        pygame.display.flip()

        ### Update physics
        fps = 60
        dt = 1.0 / fps
        space.step(dt)

        clock.tick(fps)


if __name__ == "__main__":
    sys.exit(main())