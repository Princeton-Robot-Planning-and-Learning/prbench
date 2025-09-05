"""Tests for dyn_obstruction2d.py."""

from gymnasium.spaces import Box

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
    env = prbench.make("prbench/DynObstruction2D-o4-v0")
    obs, _ = env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(5):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


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
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (int, float))
            if terminated:
                break


def test_dyn_obstruction2d_rendering():
    """Tests that rendering works without errors."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o1-v0")
    obs, _ = env.reset()
    
    # Test rendering
    img = env.render()
    assert img is not None
    assert len(img.shape) == 3  # Should be RGB image
    assert img.shape[2] == 3   # RGB channels
    
    # Take a step and render again
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    img = env.render()
    assert img is not None
    assert len(img.shape) == 3


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
        obs, reward, terminated, truncated, info = env.step(action)
        # First step should give -1 reward (goal not satisfied immediately)
        assert reward == -1.0