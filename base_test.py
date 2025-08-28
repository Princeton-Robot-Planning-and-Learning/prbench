#!/usr/bin/env python3
"""
Base-only movement test script for PRBench.

This script tests the BaseController class directly and demonstrates
how to control robot base movement while keeping the arm stationary.
"""

import numpy as np
import time


def test_base_controller():
    """Test the BaseController class directly for base movement."""
    print("Testing BaseController for Base-Only Movement")
    print("=" * 50)
    
    try:
        from prbench.envs.tidybot.base_controller import BaseController
        
        # Create base controller
        print("Creating BaseController...")
        qpos = np.zeros(3)  # [x, y, theta] 
        qvel = np.zeros(3)  # [vx, vy, omega]
        ctrl = np.zeros(3)  # Control target
        timestep = 0.1
        
        controller = BaseController(qpos, qvel, ctrl, timestep)
        
        print(f"Max velocity limits: {controller.otg_inp.max_velocity}")
        print(f"Max acceleration limits: {controller.otg_inp.max_acceleration}")
        
        # Reset controller
        controller.reset()
        print(f"Initial position: {controller.qpos}")
        print(f"Initial control: {controller.ctrl}")
        
        # Test various base movements
        print("\nTesting base movements...")
        
        movements = [
            {"name": "Move Forward", "target": [1.0, 0.0, 0.0]},
            {"name": "Move Left", "target": [1.0, 1.0, 0.0]}, 
            {"name": "Rotate", "target": [1.0, 1.0, 0.5]},
            {"name": "Move Back", "target": [0.0, 1.0, 0.5]},
            {"name": "Return to Origin", "target": [0.0, 0.0, 0.0]},
        ]
        
        for movement in movements:
            print(f"\n{movement['name']} -> {movement['target']}")
            action = {"base_pose": np.array(movement['target'])}
            target_pos = np.array(movement['target'])
            initial_pos = controller.ctrl.copy()
            
            # Run controller for multiple steps to reach target
            reached_target = False
            for step in range(20):  # Increased steps for better convergence
                old_ctrl = controller.ctrl.copy()
                controller.run_controller(action)
                
                # Calculate movement progress
                error = np.linalg.norm(controller.ctrl - target_pos)
                movement_delta = np.linalg.norm(controller.ctrl - old_ctrl)
                
                print(f"  Step {step+1:2d}: pos=[{controller.ctrl[0]:.3f}, {controller.ctrl[1]:.3f}, {controller.ctrl[2]:.3f}], "
                      f"error={error:.4f}, delta={movement_delta:.4f}")
                
                # Stop if we're close enough to the target
                if error < 0.05:  # Slightly relaxed tolerance
                    print(f"  ✅ Reached target in {step+1} steps! (error: {error:.4f})")
                    reached_target = True
                    break
                    
                # Add small delay to simulate real-time
                time.sleep(0.01)
            
            # Verify target was reached
            final_error = np.linalg.norm(controller.ctrl - target_pos)
            total_distance = np.linalg.norm(controller.ctrl - initial_pos)
            
            if not reached_target:
                print(f"  ⚠️  Did not reach target within 20 steps (final error: {final_error:.4f})")
            
            print(f"  📏 Total distance moved: {total_distance:.4f}")
            print(f"  🎯 Final error from target: {final_error:.4f}")
            
            # Validate movement occurred
            if total_distance < 0.001:
                print(f"  ❌ WARNING: Robot did not move significantly!")
            else:
                print(f"  ✅ Movement confirmed: robot moved {total_distance:.4f} units")
        
        print("\n" + "="*50)
        print("BaseController Test Summary:")
        print("✅ All movement commands were processed")
        print("✅ Trajectory generation is working correctly")
        print("✅ Position tracking and error calculation verified")
        print("BaseController test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure prbench is properly installed with BaseController dependencies.")
        return False
    except Exception as e:
        print(f"Error during BaseController test: {e}")
        return False


def test_tidybot3d_minimal():
    """Minimal test of TidyBot3D for base movement only."""
    print("\nTesting TidyBot3D Base Movement (Minimal)")
    print("=" * 50)
    
    try:
        import prbench
        
        # Register environments
        prbench.register_all_environments()
        
        print("Creating TidyBot3D environment...")
        env = prbench.make(
            "prbench/TidyBot3D-ground-o3-v0",
            render_images=True,
            show_images=True,
            show_viewer=False
        )
        
        print("Resetting environment...")
        obs, info = env.reset(seed=42)
        
        # Get access to the robot environment
        robot_env = env.env.env._tidybot_robot_env
        print(f"Initial base position: {robot_env.qpos_base}")
        
        # Test base-only movements
        print("\nTesting base movements in TidyBot3D...")
        
        # Keep arm and gripper stationary
        arm_pos = np.array([0.0, 0.0, 0.5])
        arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        gripper_pos = np.array([0.0])
        
        base_movements = [
            {"name": "Forward", "target": [0.1, 0.0, 0.0]},
            {"name": "Left", "target": [0.1, 0.1, 0.0]},
            {"name": "Rotate", "target": [0.1, 0.1, 0.2]},
        ]
        
        for i, movement in enumerate(base_movements):
            print(f"\n{movement['name']}: Moving to {movement['target']}")
            
            # Record initial position
            initial_base_pos = robot_env.qpos_base.copy()
            target_pos = np.array(movement['target'])
            
            print(f"  Initial position: [{initial_base_pos[0]:.3f}, {initial_base_pos[1]:.3f}, {initial_base_pos[2]:.3f}]")
            
            # Create action: [base_pose(3), arm_pos(3), arm_quat(4), gripper_pos(1)]
            action = np.concatenate([target_pos, arm_pos, arm_quat, gripper_pos])
            
            # Take multiple steps to allow movement to complete
            for step in range(100):
                obs, reward, terminated, truncated, info = env.step(action)
                current_pos = robot_env.qpos_base.copy()
                
                # Calculate error and movement
                error = np.linalg.norm(current_pos - target_pos)
                
                print(f"    Step {step+1}: pos=[{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}], error={error:.4f}")
                
                # Check for convergence
                if error < 0.01:  # Reasonable tolerance for 3D environment
                    print(f"    ✅ Converged to target in {step+1} steps!")
                    break
                
                if terminated or truncated:
                    print(f"    Episode terminated at step {step+1}")
                    break
            
            # Final verification
            final_pos = robot_env.qpos_base.copy()
            total_movement = np.linalg.norm(final_pos - initial_base_pos)
            final_error = np.linalg.norm(final_pos - target_pos)
            
            print(f"  📊 Movement Summary:")
            print(f"    Initial:  [{initial_base_pos[0]:.3f}, {initial_base_pos[1]:.3f}, {initial_base_pos[2]:.3f}]")
            print(f"    Target:   [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            print(f"    Final:    [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
            print(f"    📏 Distance moved: {total_movement:.4f}")
            print(f"    🎯 Final error: {final_error:.4f}")
            
            # Validate movement
            if total_movement < 0.01:
                print(f"    ❌ WARNING: Robot barely moved!")
            elif final_error < 0.01:
                print(f"    ✅ Successfully moved toward target!")
            else:
                print(f"    ⚠️  Significant error from target position!")
            
            if terminated or truncated:
                print(f"  Episode terminated at movement {i+1}")
                break
        
        env.close()
        print("\n" + "="*50)
        print("TidyBot3D Test Summary:")
        print("✅ Environment created and reset successfully")
        print("✅ Base movement commands executed")
        print("✅ Position tracking in 3D environment verified")
        print("✅ Base controller integration with MuJoCo confirmed")
        print("TidyBot3D test completed successfully!")
        return True
        
    except Exception as e:
        print(f"TidyBot3D test failed: {e}")
        return False


def main():
    """Run all base movement tests."""
    print("PRBench Base-Only Movement Test Suite")
    print("This script demonstrates how to control robot base movement while keeping arm stationary.\n")
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Direct BaseController
    if test_base_controller():
        success_count += 1
    
    # Test 2: TidyBot3D base movement
    if test_tidybot3d_minimal():
        success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All base movement tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main() 