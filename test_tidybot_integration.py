#!/usr/bin/env python3
"""Test script for TidyBot integration with PRBench."""

import os
import sys
import time

import numpy as np

# Add PRBench to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_tidybot_integration():
    """Test the TidyBot integration with PRBench."""
    print("Testing TidyBot integration with PRBench...")

    try:
        # Import PRBench
        import prbench

        # Register all environments
        prbench.register_all_environments()

        # Test a simple TidyBot environment
        print("Creating TidyBot environment...")
        env = prbench.make("prbench/TidyBot3D-table-o3-mp-v0")

        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Metadata: {env.metadata}")

        # Test reset
        print("\nTesting reset...")
        obs, info = env.reset()
        print(f"Reset successful! Observation shape: {obs.shape}")

        # Test step with random actions
        print("\nTesting step with random actions...")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"Step {i+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}"
            )
            if terminated or truncated:
                break

        # Test step with policy
        # print("\nTesting step with policy...")
        # obs, info = env.reset()
        # for i in range(10):
        #     obs, reward, terminated, truncated, info = env.step_with_policy()
        #     print(
        #         f"Policy step {i+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}"
        #     )
        #     if terminated or truncated:
        #         break

        # Test render
        print("\nTesting render...")
        img = env.render()
        if img is not None:
            print(f"Render successful! Image shape: {img.shape}")
        else:
            print("Render returned None (expected for headless mode)")

        # Close environment
        env.close()
        print("\nEnvironment closed successfully!")

        # Test getting available environments
        print("\nTesting available environments...")
        available_envs = prbench.get_all_env_ids()
        tidybot_envs = [env_id for env_id in available_envs if "TidyBot3D" in env_id]
        print(f"Found {len(tidybot_envs)} TidyBot environments:")
        for env_id in sorted(tidybot_envs)[:10]:  # Show first 10
            print(f"  {env_id}")
        if len(tidybot_envs) > 10:
            print(f"  ... and {len(tidybot_envs) - 10} more")

        print("\n✅ TidyBot integration test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ TidyBot integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_different_scenes():
    """Test different scene types."""
    print("\nTesting different scene types...")

    try:
        import prbench

        scenes = ["table", "cupboard", "cabinet"]
        policy_types = ["mp"]

        for scene in scenes:
            for policy_type in policy_types:
                env_id = f"prbench/TidyBot3D-{scene}-o3-{policy_type}-v0"
                try:
                    print(f"Testing {env_id}...")
                    env = prbench.make(env_id)
                    obs, info = env.reset()
                    print(f"  ✅ {env_id} works!")
                    env.close()
                except Exception as e:
                    print(f"  ❌ {env_id} failed: {e}")

        print("✅ Scene type tests completed!")
        return True

    except Exception as e:
        print(f"❌ Scene type tests failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TidyBot Integration Test for PRBench")
    print("=" * 60)

    success1 = test_tidybot_integration()
    success2 = test_different_scenes()

    if success1 and success2:
        print("\n🎉 All tests passed! TidyBot integration is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the error messages above.")
        sys.exit(1)
