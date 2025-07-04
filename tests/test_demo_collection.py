"""Tests for demo collection functionality."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import prbench


def test_human_input_mapping():
    """Test that environments implement human input mapping correctly."""
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o2-v0")
    
    # Get the underlying environment (unwrap gymnasium wrappers)
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    # Test that environment has the required method
    assert hasattr(unwrapped_env, 'get_human_input_mapping')
    assert hasattr(unwrapped_env, 'map_human_input_to_action')
    
    # Test mapping structure
    mapping = unwrapped_env.get_human_input_mapping()
    assert 'key_mappings' in mapping
    assert 'description' in mapping
    assert 'action_bounds' in mapping
    
    # Test action mapping
    keys_pressed = {'w', 'a'}  # up and left
    action = unwrapped_env.map_human_input_to_action(keys_pressed)
    assert isinstance(action, np.ndarray)
    assert action.shape == (5,)
    assert np.isclose(action[0], -0.05)  # dx (left)
    assert np.isclose(action[1], 0.05)   # dy (up)


def test_demo_save_format():
    """Test that demo data is saved in the correct format."""
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o2-v0")
    
    # Create some fake demo data
    demo_data = {
        "env_id": "prbench/Obstruction2D-o2-v0",
        "timestamp": 1234567890,
        "observations": [np.zeros(29).tolist(), np.ones(29).tolist()],
        "actions": [np.zeros(5).tolist(), np.ones(5).tolist()],
        "rewards": [0.0, -1.0],
        "terminated": True,
        "truncated": False,
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(demo_data, f, indent=2)
        temp_path = f.name
    
    try:
        # Load and verify
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['env_id'] == demo_data['env_id']
        assert loaded_data['timestamp'] == demo_data['timestamp']
        assert len(loaded_data['observations']) == 2
        assert len(loaded_data['actions']) == 2
        assert len(loaded_data['rewards']) == 2
        assert loaded_data['terminated'] == True
        assert loaded_data['truncated'] == False
        
    finally:
        Path(temp_path).unlink()


def test_environment_action_bounds():
    """Test that action bounds match the environment's action space."""
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o2-v0")
    
    # Get the underlying environment (unwrap gymnasium wrappers)
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    mapping = unwrapped_env.get_human_input_mapping()
    action_bounds = mapping['action_bounds']
    
    # Check that bounds match action space
    assert len(action_bounds) == env.action_space.shape[0]
    
    # Check that bounds are reasonable
    for i, (min_val, max_val) in enumerate(action_bounds):
        assert min_val <= max_val
        assert env.action_space.low[i] <= min_val
        assert env.action_space.high[i] >= max_val 