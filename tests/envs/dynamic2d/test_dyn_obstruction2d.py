"""Tests for dyn_obstruction2d.py."""

# import imageio.v2 as iio
import numpy as np
from gymnasium.spaces import Box
from relational_structs import ObjectCentricState

import prbench


def test_dyn_obstruction2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o2-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_dyn_obstruction2d_action_space():
    """Tests that the actions are valid and the step function works."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o3-v0")
    obs, _ = env.reset(seed=0)
    stable_move = np.array([0.05, 0.05, np.pi / 16, 0.0, 0.0], dtype=np.float32)
    # Check the control precision of base movements
    for s in range(5):
        obs, _ = env.reset(seed=s)
        state: ObjectCentricState = env.observation_space.devectorize(obs)
        name_to_object = {obj.name: obj for obj in state.data}
        robot_object = name_to_object["robot"]
        robot_x = state.get(robot_object, "x")
        robot_y = state.get(robot_object, "y")
        robot_theta = state.get(robot_object, "theta")
        robot_arm_length = state.get(robot_object, "arm_length")
        robot_finger_gap = state.get(robot_object, "finger_gap")
        obs_, _, _, _, _ = env.step(stable_move)
        state_: ObjectCentricState = env.observation_space.devectorize(obs_)
        robot_x_ = state_.get(robot_object, "x")
        robot_y_ = state_.get(robot_object, "y")
        robot_theta_ = state_.get(robot_object, "theta")
        robot_arm_length_ = state_.get(robot_object, "arm_length")
        robot_finger_gap_ = state_.get(robot_object, "finger_gap")
        assert np.isclose(robot_x + stable_move[0], robot_x_, atol=1e-3)
        assert np.isclose(robot_y + stable_move[1], robot_y_, atol=1e-3)
        assert np.isclose(
            robot_theta + stable_move[2], robot_theta_, atol=1e-2
        )  # 0.5 degree
        assert np.isclose(
            robot_arm_length + stable_move[3], robot_arm_length_, atol=1e-3
        )
        assert np.isclose(
            robot_finger_gap - stable_move[4], robot_finger_gap_, atol=1e-3
        )

    obs, _ = env.reset(seed=0)
    stable_move = np.array([0.0, 0.0, 0.0, 0.05, -0.02], dtype=np.float32)
    # Check the control precision of base movements
    for s in [1, 2]:
        obs, _ = env.reset(seed=s)
        state: ObjectCentricState = env.observation_space.devectorize(obs)
        name_to_object = {obj.name: obj for obj in state.data}
        robot_object = name_to_object["robot"]
        robot_x = state.get(robot_object, "x")
        robot_y = state.get(robot_object, "y")
        robot_theta = state.get(robot_object, "theta")
        robot_arm_length = state.get(robot_object, "arm_joint")
        robot_finger_gap = state.get(robot_object, "finger_gap")
        obs_, _, _, _, _ = env.step(stable_move)
        state_: ObjectCentricState = env.observation_space.devectorize(obs_)
        robot_x_ = state_.get(robot_object, "x")
        robot_y_ = state_.get(robot_object, "y")
        robot_theta_ = state_.get(robot_object, "theta")
        robot_arm_length_ = state_.get(robot_object, "arm_joint")
        robot_finger_gap_ = state_.get(robot_object, "finger_gap")
        assert np.isclose(robot_x + stable_move[0], robot_x_, atol=1e-3)
        assert np.isclose(robot_y + stable_move[1], robot_y_, atol=1e-3)
        assert np.isclose(
            robot_theta + stable_move[2], robot_theta_, atol=1e-2
        )  # 0.5 degree
        assert np.isclose(
            robot_arm_length + stable_move[3], robot_arm_length_, atol=1e-3
        )
        assert np.isclose(
            robot_finger_gap + stable_move[4], robot_finger_gap_, atol=1e-3
        )


def test_dyn_obstruction2d_different_obstruction_counts():
    """Tests that different numbers of obstructions work."""
    prbench.register_all_environments()

    for num_obs in [0, 1, 2, 3, 4]:
        env = prbench.make(f"prbench/DynObstruction2D-o{num_obs}-v0")
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

        # Take a few steps to ensure environment works
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, _truncated, _info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (int, float))
            if terminated:
                break


def test_dyn_obstruction2d_reset_consistency():
    """Tests that reset produces consistent observations."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o2-v0")

    # Test multiple resets
    for _ in range(3):
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

        # Environment should not be terminated at start
        action = env.action_space.sample()
        _obs, reward, _terminated, _truncated, _info = env.step(action)
        # First step should give -1 reward (goal not satisfied immediately)
        assert reward == -1.0
