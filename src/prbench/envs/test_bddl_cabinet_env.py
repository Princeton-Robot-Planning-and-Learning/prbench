"""Test script for the BDDL Cabinet Environment."""

import numpy as np
from bddl_cabinet_env import BDDLCabinetEnv


def test_bddl_cabinet_env():
    """Test the BDDL cabinet environment."""
    print("Creating BDDL Cabinet Environment...")
    
    # Create environment
    env = BDDLCabinetEnv(
        show_viewer=True,  # Set to True to see the simulation
        show_images=False,
        render_mode="rgb_array"
    )
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a few random steps
    print("\nRunning random actions...")
    for step in range(10):
        # Generate random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: reward={reward:.3f}, terminated={terminated}")
        
        if terminated:
            print("Task completed!")
            break
    
    # Close environment
    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_bddl_cabinet_env() 