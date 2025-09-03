"""Showcase of flying arrows that can stick to objects in a somewhat
realistic looking way.
"""

import sys

import pygame

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

from prbench.envs.geom2d.structs import SE2Pose


class KinRobot:
    def __init__(self, base_radius=30, arm_length=32, gripper_gap_max=50):
        self.base_radius = base_radius
        self.arm_length = arm_length
        self.gripper_gap_max = gripper_gap_max
        self.gripper_gap = gripper_gap_max
        self.position = Vec2d(100, 500)
        self.angle = 0.0
        self.speed = 2.5
        self.gripper_speed = 1.0
        
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
        self.base_body.position = self.position
        return self.base_body, self.base_shape
    
    def create_gripper_base(self):
        vs = [(-2, 25), (-2, -25), (2, -25), (2, 25)]
        self.gripper_base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        
        self.gripper_base_shape = pymunk.Poly(self.gripper_base_body, vs)
        self.gripper_base_shape.friction = 1
        self.gripper_base_shape.collision_type = 0
        self.gripper_base_shape.density = 1.0
        return self.gripper_base_body, self.gripper_base_shape
    
    def create_finger(self):
        vs = [(-12, 2), (-12, -2), (12, -2), (12, 2)]
        finger_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        
        finger_shape = pymunk.Poly(finger_body, vs)
        finger_shape.friction = 1
        finger_shape.collision_type = 2
        finger_shape.density = 1.0
        return finger_body, finger_shape
    
    def create_components(self):
        self.create_base()
        self.create_gripper_base()
        self.left_finger_body, self.left_finger_shape = self.create_finger()
        self.right_finger_body, self.right_finger_shape = self.create_finger()
    
    def add_to_space(self, space):
        space.add(self.base_body, self.base_shape)
        space.add(self.gripper_base_body, self.gripper_base_shape)
        space.add(self.left_finger_body, self.left_finger_shape)
        space.add(self.right_finger_body, self.right_finger_shape)
    
    def update(self, keys, mouse_pos, mouse_pressed):
        if keys[pygame.K_w]:
            self.position += Vec2d(0, -1) * self.speed
        if keys[pygame.K_s]:
            self.position += Vec2d(0, 1) * self.speed
        if keys[pygame.K_a]:
            self.position += Vec2d(-1, 0) * self.speed
        if keys[pygame.K_d]:
            self.position += Vec2d(1, 0) * self.speed
        
        self.angle = (mouse_pos - self.position).angle
        
        gripper_dy = 0.0
        if mouse_pressed[0]:
            gripper_dy = -self.gripper_speed
        else:
            gripper_dy = self.gripper_speed
        
        current_finger_pose = SE2Pose(
            x=self.left_finger_body.position.x,
            y=self.left_finger_body.position.y,
            theta=self.left_finger_body.angle,
        )
        body_pose = SE2Pose(x=self.position.x, y=self.position.y, theta=self.angle)
        current_relative_finger_pose = body_pose.inverse * current_finger_pose
        relative_y = abs(min(current_relative_finger_pose.y + gripper_dy, self.gripper_gap_max // 2))
        
        self.update_positions(relative_y)
    
    def update_positions(self, finger_relative_y):
        self.base_body.position = self.position
        self.base_body.angle = self.angle
        
        body_pose = SE2Pose(x=self.position.x, y=self.position.y, theta=self.angle)
        
        relative_pose_gripper_base = SE2Pose(x=self.arm_length, y=0.0, theta=0.0)
        gripper_base_pose = body_pose * relative_pose_gripper_base
        self.gripper_base_body.position = (gripper_base_pose.x, gripper_base_pose.y)
        self.gripper_base_body.angle = gripper_base_pose.theta
        
        new_relative_finger_pose_l = SE2Pose(x=45.0, y=finger_relative_y, theta=0.0)
        new_relative_finger_pose_r = SE2Pose(x=45.0, y=-finger_relative_y, theta=0.0)
        
        l_finger_pose = body_pose * new_relative_finger_pose_l
        self.left_finger_body.position = (l_finger_pose.x, l_finger_pose.y)
        self.left_finger_body.angle = l_finger_pose.theta
        
        r_finger_pose = body_pose * new_relative_finger_pose_r
        self.right_finger_body.position = (r_finger_pose.x, r_finger_pose.y)
        self.right_finger_body.angle = r_finger_pose.theta

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
    space.add(shape)
    size = 20
    points = [(-size, -size), (-size, size), (size, size), (size, -size)]
    mass = 1.0
    moment = pymunk.moment_for_poly(mass, points, (0, 0))

    b1 = pymunk.Body(mass, moment)
    b1.position = Vec2d(50, 450)
    shape = pymunk.Poly(b1, points)
    shape.friction = 1
    shape.density = 1.0
    space.add(b1, shape)

    b2 = pymunk.Body(mass, moment)
    b2.position = Vec2d(200, 450)
    shape = pymunk.Poly(b2, points)
    shape.friction = 1
    shape.density = 1.0
    space.add(b2, shape)

    # Create robot
    robot = KinRobot(base_radius=30, arm_length=32, gripper_gap_max=50)
    robot.add_to_space(space)

    start_time = 0
    while running:
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and (event.key in [pygame.K_ESCAPE, pygame.K_q])
            ):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                start_time = pygame.time.get_ticks()

        keys = pygame.key.get_pressed()
        mouse_position = pymunk.pygame_util.from_pygame(
            Vec2d(*pygame.mouse.get_pos()), screen
        )
        mouse_pressed = pygame.mouse.get_pressed()
        
        robot.update(keys, mouse_position, mouse_pressed)

        ### Clear screen
        screen.fill(pygame.Color("black"))

        ### Draw stuff
        space.debug_draw(draw_options)
        # draw(screen, space)

        # Power meter
        if pygame.mouse.get_pressed()[0]:
            current_time = pygame.time.get_ticks()
            diff = current_time - start_time
            power = max(min(diff, 1000), 10)
            h = power // 2
            pygame.draw.line(screen, pygame.Color("red"), (30, 550), (30, 550 - h), 10)

        # Info and flip screen
        screen.blit(
            font.render("fps: " + str(clock.get_fps()), True, pygame.Color("white")),
            (0, 0),
        )
        screen.blit(
            font.render(
                "Aim with mouse, hold LMB to powerup, release to fire",
                True,
                pygame.Color("darkgrey"),
            ),
            (5, height - 35),
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