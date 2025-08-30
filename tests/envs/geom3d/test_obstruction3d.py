"""Tests for obstruction3d.py."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from pybullet_helpers.geometry import Pose, get_pose
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
)
from prpl_utils.utils import wrap_angle

from prbench.envs.geom3d.obstruction3d import (
    Obstruction3DAction,
    Obstruction3DEnv,
    Obstruction3DState,
)


def test_obstruction3d_env():
    """Tests for basic methods in obstruction3d env."""

    env = Obstruction3DEnv(use_gui=False)  # set use_gui=True to debug
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, Obstruction3DState)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, Obstruction3DAction)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.physics_client_id)


def test_pick_place_no_obstructions():
    """Test that picking and placing succeeds when there are no obstructions."""
    # Create the real environment.
    env = Obstruction3DEnv(num_obstructions=0, use_gui=True, render_mode="rgb_array")
    spec = env._spec  # pylint: disable=protected-access
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset(seed=123)

    # Create a simulator for planning.
    sim = Obstruction3DEnv(spec=spec)
    sim.set_state(obs)

    # Run motion planning.
    if MAKE_VIDEOS:  # make a smooth motion plan for videos
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    # First, move to pre-grasp pose (top-down).
    x, y, z = obs.target_block.pose.position
    dz = 0.025
    pre_grasp_pose = Pose.from_rpy((x, y, z + dz), (np.pi, 0, np.pi / 2))
    joint_plan = run_smooth_motion_planning_to_pose(
        pre_grasp_pose,
        sim.robot,
        collision_ids=sim._get_collision_object_ids(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=max_candidate_plans,
    )
    assert joint_plan is not None

    # Make sure we stay below the required max_action_mag by a fair amount.
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot, max_distance=spec.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints, obs.joint_positions)[:7]
        delta_lst = [wrap_angle(a) for a in delta]
        action = Obstruction3DAction(delta_lst)
        obs, _, _, _, _ = env.step(action)

    # Close the gripper to grasp.
    action = Obstruction3DAction(delta_arm_joints=[0.] * 7, gripper="close")
    obs, _, _, _, _ = env.step(action)

    # The target block should now be grasped.
    assert obs.grasped_object == "target_block"

    import pybullet as p

    # from pybullet_helpers.gui import visualize_pose
    # visualize_pose(pre_grasp_pose, env.physics_client_id)
    while True:
        p.getMouseEvents(env.physics_client_id)

    env.close()
