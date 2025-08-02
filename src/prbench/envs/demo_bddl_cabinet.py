"""Demo script for the BDDL Cabinet Environment.

This script demonstrates how to use the BDDL Cabinet Environment
based on the example_1_parsed.json BDDL specification.

Task: Put the white bowl in the top drawer of the wooden cabinet.
"""

import time
import numpy as np
from bddl_cabinet_env import BDDLCabinetEnv


def demo_random_actions():
    """Demo with random actions."""
    print("=== BDDL Cabinet Environment Demo ===")
    print("Task: Put the white bowl in the top drawer of the wooden cabinet")
    print("\nCreating environment...")
    
    # Create environment
    env = BDDLCabinetEnv(
        show_viewer=True,  # Show MuJoCo viewer
        show_images=False,  # Don't show camera images separately
        render_mode="rgb_array"
    )
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    try:
        # Reset environment
        print("\nResetting environment...")
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        print("\nStarting random action demo...")
        print("The robot will perform random actions.")
        print("Watch the simulation to see the robot, table, cabinet, and bowl!")
        
        total_reward = 0
        for step in range(100):  # Run for more steps to see the environment
            # Generate random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 10 == 0:  # Print every 10 steps
                print(f"Step {step+1}: reward={reward:.3f}, total_reward={total_reward:.3f}, terminated={terminated}")
            
            if terminated:
                print(f"\nTask completed at step {step+1}!")
                print(f"Total reward: {total_reward:.3f}")
                break
            
            # Small delay to make the simulation visible
            time.sleep(0.05)
        
        if not terminated:
            print(f"\nDemo completed after {step+1} steps")
            print(f"Total reward: {total_reward:.3f}")
            print("Task was not completed with random actions (this is expected)")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Close environment
        env.close()
        print("\nEnvironment closed. Demo finished!")


def demo_simple_policy():
    """Demo with a simple heuristic policy."""
    print("\n=== Simple Policy Demo ===")
    print("This demo uses a simple heuristic to approach the bowl")
    
    env = BDDLCabinetEnv(
        show_viewer=True,
        show_images=False,
        render_mode="rgb_array"
    )
    
    try:
        obs, info = env.reset()
        
        print("Running simple approach policy...")
        total_reward = 0
        
        for step in range(50):
            # Simple policy: try to move towards the bowl and open gripper
            action = np.array([
                0.1,  # base x (move forward slowly)
                0.0,  # base y
                0.0,  # base theta
                0.5,  # arm x (reach forward)
                0.0,  # arm y
                0.5,  # arm z (reach up)
                1.0,  # arm quat x
                0.0,  # arm quat y
                0.0,  # arm quat z
                0.0,  # arm quat w
                1.0,  # gripper (open)
            ], dtype=np.float32)
            
            # Add some randomness
            action += np.random.normal(0, 0.1, action.shape)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 5 == 0:
                print(f"Step {step+1}: reward={reward:.3f}, total_reward={total_reward:.3f}")
            
            if terminated:
                print(f"Task completed at step {step+1}!")
                break
            
            time.sleep(0.1)
        
        print(f"Simple policy demo completed. Total reward: {total_reward:.3f}")
        
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    
    finally:
        env.close()


def print_environment_info():
    """Print detailed information about the environment."""
    print("\n=== Environment Information ===")
    
    env = BDDLCabinetEnv()
    
    print("Description:")
    print(env.metadata["description"])
    
    print("\nObservation Space Description:")
    print(env.metadata["observation_space_description"])
    
    print("\nAction Space Description:")
    print(env.metadata["action_space_description"])
    
    print("\nReward Description:")
    print(env.metadata["reward_description"])
    
    print("\nReferences:")
    print(env.metadata["references"])
    
    env.close()


if __name__ == "__main__":
    print("BDDL Cabinet Environment Demo")
    print("=============================")
    
    # Print environment information
    print_environment_info()
    
    # Run demos
    demo_random_actions()
    
    # Uncomment to run the simple policy demo as well
    # demo_simple_policy()
    
    print("\nDemo completed! The environment is ready for use.")
    print("\nTo use this environment in your code:")
    print("from prbench.envs import BDDLCabinetEnv")
    print("env = BDDLCabinetEnv()") 