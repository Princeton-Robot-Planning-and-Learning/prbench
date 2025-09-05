"""Tests for motion3d.py."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
)

from prbench.envs.geom3d.motion3d import Motion3DAction, Motion3DEnv, Motion3DState


def test_motion3d_env():
    """Tests for basic methods in motion3D env."""

    env = Motion3DEnv(use_gui=False)  # set use_gui=True to debug
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, Motion3DState)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, Motion3DAction)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.physics_client_id)


def test_motion_planning_in_motion3d_env():
    """Proof of concept that motion planning works in this environment."""

    # Create the real environment.
    env = Motion3DEnv(render_mode="rgb_array")
    spec = env._spec  # pylint: disable=protected-access
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=123)

    # Create a simulator for planning.
    sim = Motion3DEnv(spec=spec)

    # Run motion planning.
    if MAKE_VIDEOS:  # make a smooth motion plan for videos
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    joint_plan = run_smooth_motion_planning_to_pose(
        Pose(obs.target),
        sim.robot,
        collision_ids=set(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=max_candidate_plans,
    )
    assert joint_plan is not None
    # Make sure we stay below the required max_action_mag by a fair amount.
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot, max_distance=spec.max_action_mag / 2
    )

    env.action_space.seed(123)
    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints, obs.joint_positions)[:7]
        delta_lst = [wrap_angle(a) for a in delta]
        action = Motion3DAction(delta_lst)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    else:
        assert False, "Plan did not reach goal"
    env.close()
