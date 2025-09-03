"""Showcase of flying arrows that can stick to objects in a somewhat
realistic looking way.
"""

import sys

import pygame

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

from prbench.envs.geom2d.structs import SE2Pose


def create_gripper_base():
    vs = [(-2, 25), (-2, -25), (2, -25), (2, 25)]
    # mass = 1
    # moment = pymunk.moment_for_poly(mass, vs)
    gripper_base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    gripper_base_shape = pymunk.Poly(gripper_base_body, vs)
    gripper_base_shape.friction = 1
    gripper_base_shape.collision_type = 0
    gripper_base_shape.density = 1.0
    return gripper_base_body, gripper_base_shape

def create_finger():
    vs = [(-12, 2), (-12, -2), (12, -2), (12, 2)]
    finger_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    finger_shape = pymunk.Poly(finger_body, vs)
    finger_shape.friction = 1
    finger_shape.collision_type = 2
    finger_shape.density = 1.0
    return finger_body, finger_shape

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

    # "Cannon" that can fire arrows
    cannon_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    cannon_shape = pymunk.Circle(cannon_body, 30)
    cannon_shape.color = (255, 50, 50, 255)
    cannon_body.position = 100, 500
    space.add(cannon_body, cannon_shape)

    gripper_base_body, gripper_base_shape = create_gripper_base()
    space.add(gripper_base_body, gripper_base_shape)
    gripper_base_body.position = 132, 500

    l_gripper_finger_body, l_gripper_finger_shape = create_finger()
    space.add(l_gripper_finger_body, l_gripper_finger_shape)
    l_gripper_finger_body.position = 145, 525

    r_gripper_finger_body, r_gripper_finger_shape = create_finger()
    space.add(r_gripper_finger_body, r_gripper_finger_shape)
    r_gripper_finger_body.position = 145, 475

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

        gripper_dy = 0.0
        gripper_speed = 1.0
        if pygame.mouse.get_pressed()[0]:
            gripper_dy = -gripper_speed
        else:
            gripper_dy = gripper_speed

        relative_pose_gripper_base = SE2Pose(
            x=32.0, y=0.0, theta=0.0
        )
        curr_body_pose = SE2Pose(
            x=cannon_body.position.x,
            y=cannon_body.position.y,
            theta=cannon_body.angle,
        )
        curr_finger_pose = SE2Pose(
            x=l_gripper_finger_body.position.x,
            y=l_gripper_finger_body.position.y,
            theta=l_gripper_finger_body.angle,
        )
        curr_relative_finger_pose = curr_body_pose.inverse * curr_finger_pose
        relative_y = abs(min(curr_relative_finger_pose.y + gripper_dy, 25.0))

        new_relative_finger_pose_l = SE2Pose(
            x=45.0,
            y=relative_y,
            theta=0.0,
        )
        new_relative_finger_pose_r = SE2Pose(
            x=45.0,
            y=-relative_y,
            theta=0.0,
        )

        speed = 2.5
        if keys[pygame.K_w]:
            cannon_body.position += Vec2d(0, -1) * speed
        if keys[pygame.K_s]:
            cannon_body.position += Vec2d(0, 1) * speed
        if keys[pygame.K_a]:
            cannon_body.position += Vec2d(-1, 0) * speed
        if keys[pygame.K_d]:
            cannon_body.position += Vec2d(1, 0) * speed

        mouse_position = pymunk.pygame_util.from_pygame(
            Vec2d(*pygame.mouse.get_pos()), screen
        )
        cannon_body.angle = (mouse_position - cannon_body.position).angle
        # move the unfired arrow together with the cannon
        body_pose = SE2Pose(
            x=cannon_body.position.x,
            y=cannon_body.position.y,
            theta=cannon_body.angle,
        )
        gripper_base_pose = body_pose * relative_pose_gripper_base
        gripper_base_body.position = (gripper_base_pose.x, gripper_base_pose.y)
        gripper_base_body.angle = gripper_base_pose.theta

        l_finger_pose = body_pose * new_relative_finger_pose_l
        l_gripper_finger_body.position = (l_finger_pose.x, l_finger_pose.y)
        l_gripper_finger_body.angle = l_finger_pose.theta

        r_finger_pose = body_pose * new_relative_finger_pose_r
        r_gripper_finger_body.position = (r_finger_pose.x, r_finger_pose.y)
        r_gripper_finger_body.angle = r_finger_pose.theta

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