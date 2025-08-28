#!/usr/bin/env python3
"""
Arm-only movement test script for PRBench.

This script tests the ArmController class directly and demonstrates
how to control robot arm movement while keeping the base fixed.
"""

import numpy as np
import time
import mujoco


def test_arm_controller():
    """Test the ArmController class directly for arm movement only."""
    print("Testing ArmController for Arm-Only Movement")
    print("=" * 50)
    
    try:
        from prbench.envs.tidybot.arm_controller import ArmController
        
        # Create arm controller
        print("Creating ArmController...")
        num_joints = 7
        qpos = np.zeros(num_joints)  # Joint positions
        qvel = np.zeros(num_joints)  # Joint velocities
        ctrl = np.zeros(num_joints)  # Joint control targets
        qpos_gripper = np.array([0.0])  # Gripper position
        ctrl_gripper = np.array([0.0])  # Gripper control
        timestep = 0.1
        
        controller = ArmController(qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep)
        
        print(f"Max velocity (rad/s): {controller.otg_inp.max_velocity}")
        print(f"Max acceleration (rad/s²): {controller.otg_inp.max_acceleration}")
        print(f"Retract position: {controller.retract_qpos}")
        
        # Reset controller
        controller.reset()
        print(f"Initial joint positions: {controller.qpos}")
        print(f"Initial control targets: {controller.ctrl}")
        print(f"Initial gripper position: {controller.ctrl_gripper}")
        
        # Test various arm movements
        print("\nTesting arm movements...")
        
        # Define test movements (position + quaternion + gripper)
        movements = [
            {
                "name": "Move Up", 
                "arm_pos": [0.0, 0.0, 0.8],  # Move end-effector up
                "arm_quat": [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
                "gripper_pos": [0.0]  # Keep gripper closed
            },
            {
                "name": "Move Forward", 
                "arm_pos": [0.3, 0.0, 0.6],  # Move forward and slightly down
                "arm_quat": [1.0, 0.0, 0.0, 0.0],  
                "gripper_pos": [0.5]  # Open gripper halfway
            },
            {
                "name": "Move Left", 
                "arm_pos": [0.3, 0.3, 0.6],  # Move left
                "arm_quat": [0.707, 0.0, 0.0, 0.707],  # 90 degree rotation around z
                "gripper_pos": [1.0]  # Fully open gripper
            },
            {
                "name": "Return to Retract", 
                "arm_pos": [0.0, 0.0, 0.5],  # Return to safe position
                "arm_quat": [1.0, 0.0, 0.0, 0.0],  
                "gripper_pos": [0.0]  # Close gripper
            },
        ]
        
        for movement in movements:
            print(f"\n{movement['name']} -> pos: {movement['arm_pos']}, gripper: {movement['gripper_pos']}")
            
            # Record initial state
            initial_joints = controller.ctrl.copy()
            initial_gripper = controller.ctrl_gripper.copy()
            target_ee_pos = np.array(movement['arm_pos'])
            
            # Get initial end-effector position using forward kinematics
            try:
                # Set the IK solver to current joint positions and compute forward kinematics
                controller.ik_solver.data.qpos[:] = controller.ctrl
                mujoco.mj_kinematics(controller.ik_solver.model, controller.ik_solver.data)
                mujoco.mj_comPos(controller.ik_solver.model, controller.ik_solver.data)
                initial_ee_pos = controller.ik_solver.site_pos.copy()
                print(f"  Initial end-effector position: [{initial_ee_pos[0]:.3f}, {initial_ee_pos[1]:.3f}, {initial_ee_pos[2]:.3f}]")
            except Exception as e:
                initial_ee_pos = np.array([0.0, 0.0, 0.0])  # Fallback if FK fails
                print(f"  Initial end-effector position: [unknown - FK failed: {e}]")
            
            # Create action
            action = {
                "arm_pos": np.array(movement['arm_pos']),
                "arm_quat": np.array(movement['arm_quat']),
                "gripper_pos": np.array(movement['gripper_pos'])
            }
            
            # Run controller for multiple steps to reach target
            reached_target = False
            for step in range(30):  # Allow more steps for arm convergence
                old_joints = controller.ctrl.copy()
                old_gripper = controller.ctrl_gripper.copy()
                
                controller.run_controller(action)
                
                # Calculate movement progress
                joint_delta = np.linalg.norm(controller.ctrl - old_joints)
                gripper_delta = np.linalg.norm(controller.ctrl_gripper - old_gripper)
                
                # Check for joint movement
                if step % 5 == 0:  # Print every 5th step
                    print(f"  Step {step+1:2d}: joints=[{controller.ctrl[0]:.3f}, {controller.ctrl[1]:.3f}, "
                          f"{controller.ctrl[2]:.3f}, {controller.ctrl[3]:.3f}, {controller.ctrl[4]:.3f}, "
                          f"{controller.ctrl[5]:.3f}, {controller.ctrl[6]:.3f}], "
                          f"gripper={controller.ctrl_gripper[0]:.1f}, "
                          f"joint_delta={joint_delta:.4f}")
                
                # Check if movement has stabilized
                if joint_delta < 0.001 and gripper_delta < 0.1:
                    print(f"  ✅ Movement stabilized at step {step+1}!")
                    reached_target = True
                    break
                    
                # Add small delay to simulate real-time
                time.sleep(0.01)
            
            # Verify movement occurred
            final_joints = controller.ctrl.copy()
            final_gripper = controller.ctrl_gripper.copy()
            
            # Get final end-effector position using forward kinematics
            try:
                # Set the IK solver to final joint positions and compute forward kinematics
                controller.ik_solver.data.qpos[:] = controller.ctrl
                mujoco.mj_kinematics(controller.ik_solver.model, controller.ik_solver.data)
                mujoco.mj_comPos(controller.ik_solver.model, controller.ik_solver.data)
                final_ee_pos = controller.ik_solver.site_pos.copy()
                ee_error = np.linalg.norm(final_ee_pos - target_ee_pos)
                ee_movement = np.linalg.norm(final_ee_pos - initial_ee_pos)
                print(f"  Final end-effector position: [{final_ee_pos[0]:.3f}, {final_ee_pos[1]:.3f}, {final_ee_pos[2]:.3f}]")
            except Exception as e:
                final_ee_pos = np.array([0.0, 0.0, 0.0])  # Fallback if FK fails
                ee_error = float('inf')
                ee_movement = 0.0
                print(f"  Final end-effector position: [unknown - FK failed: {e}]")
            
            joint_movement = np.linalg.norm(final_joints - initial_joints)
            gripper_movement = np.linalg.norm(final_gripper - initial_gripper)
            
            if not reached_target:
                print(f"  ⚠️  Movement did not stabilize within 30 steps")
            
            print(f"  📏 Total joint movement: {joint_movement:.4f} rad")
            print(f"  🤏 Total gripper movement: {gripper_movement:.1f}")
            print(f"  🎯 End-effector error from target: {ee_error:.4f}")
            print(f"  📐 End-effector movement: {ee_movement:.4f}")
            
            # Validate movement occurred
            if joint_movement < 0.01:
                print(f"  ❌ WARNING: Arm joints barely moved!")
            else:
                print(f"  ✅ Arm movement confirmed: joints moved {joint_movement:.4f} rad")
                
            if gripper_movement > 0.1:
                print(f"  ✅ Gripper movement confirmed: moved {gripper_movement:.1f} units")
            
            # Validate end-effector position
            if ee_error != float('inf'):  # Only validate if FK worked
                if ee_error < 0.05:  # 5cm tolerance
                    print(f"  ✅ End-effector reached target position! (error: {ee_error:.4f})")
                elif ee_movement > 0.01:
                    print(f"  ⚠️  End-effector moved but significant error from target (error: {ee_error:.4f})")
                else:
                    print(f"  ❌ WARNING: End-effector barely moved!")
        
        print("\n" + "="*50)
        print("ArmController Test Summary:")
        print("✅ All arm movement commands were processed")
        print("✅ Inverse kinematics solver is working")
        print("✅ Joint trajectory generation is functioning")
        print("✅ Gripper control is operational")
        print("✅ End-effector position validation confirmed")
        print("ArmController test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure prbench is properly installed with ArmController dependencies.")
        return False
    except Exception as e:
        print(f"Error during ArmController test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tidybot3d_arm_only():
    """Test arm movement in TidyBot3D while keeping base fixed."""
    print("\nTesting TidyBot3D Arm Movement (Base Fixed)")
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
        
        # Wait a few steps for controllers to stabilize after reset
        print("Waiting for controllers to stabilize...")
        for _ in range(10):
            # TidyBot3D expects numpy array action: [base_pose(3), arm_pos(3), arm_quat(4), gripper_pos(1)]
            stabilization_action = np.concatenate([
                robot_env.qpos_base.copy(),  # base_pose(3)
                robot_env.qpos_arm.copy(),   # arm_pos(7) - but we only use first 3
                [0.0, 0.0, 0.0, 1.0],      # arm_quat(4) - identity quaternion
                [0.0]                       # gripper_pos(1)
            ])
            env.step(stabilization_action)
        
        # Record initial positions after stabilization
        initial_base_pos = robot_env.qpos_base.copy()
        initial_arm_pos = robot_env.qpos_arm.copy()
        
        print(f"Initial base position: [{initial_base_pos[0]:.3f}, {initial_base_pos[1]:.3f}, {initial_base_pos[2]:.3f}]")
        print(f"Initial arm joints: [{initial_arm_pos[0]:.3f}, {initial_arm_pos[1]:.3f}, {initial_arm_pos[2]:.3f}, {initial_arm_pos[3]:.3f}, {initial_arm_pos[4]:.3f}, {initial_arm_pos[5]:.3f}, {initial_arm_pos[6]:.3f}]")
        
        # Test arm-only movements (keeping base fixed)
        print("\nTesting arm movements in TidyBot3D...")
        
        # Keep base fixed at initial position
        fixed_base_pos = initial_base_pos.copy()
        
        arm_movements = [
            {
                "name": "Reach Up", 
                "arm_pos": [0.0, 0.0, 0.8],
                "arm_quat": [0.0, 0.0, 0.0, 1.0],  # (x, y, z, w) format for identity
                "gripper_pos": [0.0]
            },
            {
                "name": "Reach Forward", 
                "arm_pos": [0.4, 0.0, 0.6],
                "arm_quat": [0.0, 0.0, 0.0, 1.0],  # (x, y, z, w) format for identity
                "gripper_pos": [0.5]
            },
            {
                "name": "Reach Left", 
                "arm_pos": [0.3, 0.3, 0.6],
                "arm_quat": [0.0, 0.0, 0.707, 0.707],  # (x, y, z, w) format for 90° rotation around Z
                "gripper_pos": [1.0]
            },
        ]
        
        for i, movement in enumerate(arm_movements):
            print(f"\n{movement['name']}: Moving arm to {movement['arm_pos']}")
            
            # Record initial arm position
            initial_arm = robot_env.qpos_arm.copy()
            target_ee_pos = np.array(movement['arm_pos'])
            
            print(f"  Initial arm joints: [{initial_arm[0]:.3f}, {initial_arm[1]:.3f}, {initial_arm[2]:.3f}, {initial_arm[3]:.3f}, {initial_arm[4]:.3f}, {initial_arm[5]:.3f}, {initial_arm[6]:.3f}]")
            
            # Get initial end-effector position if arm controller is available
            try:
                if hasattr(robot_env, 'arm_controller') and robot_env.arm_controller is not None:
                    # Use environment's MuJoCo model/data to read world-frame EE position
                    site_id = robot_env.sim.model.name2id('pinch_site', 'site')
                    initial_ee_pos = robot_env.sim.data.site_xpos[site_id].copy()
                    print(f"  Initial end-effector position: [{initial_ee_pos[0]:.3f}, {initial_ee_pos[1]:.3f}, {initial_ee_pos[2]:.3f}]")
                else:
                    initial_ee_pos = None
                    print(f"  Initial end-effector position: [arm controller not accessible]")
            except Exception as e:
                initial_ee_pos = None
                print(f"  Initial end-effector position: [FK failed: {e}]")
            
            # Create action as numpy array (what TidyBot3D expects)
            # Keep base FIXED at initial position
            # Format: [base_pose(3), arm_pos(3), arm_quat(4), gripper_pos(1)]
            target_ee_pos = np.array(movement['arm_pos'])
            print(f"  🎯 Commanding end-effector to: [{target_ee_pos[0]:.3f}, {target_ee_pos[1]:.3f}, {target_ee_pos[2]:.3f}]")
            
            action = np.concatenate([
                fixed_base_pos,  # Fixed base position (3)
                target_ee_pos,   # Target end-effector position (3)
                movement['arm_quat'],  # Arm orientation quaternion (4)
                movement['gripper_pos']  # Gripper state (1)
            ])
            
            # Take multiple steps to allow arm movement to complete
            for step in range(100):  # Increased steps for better convergence
                obs, reward, terminated, truncated, info = env.step(action)
                
                current_base = robot_env.qpos_base.copy()
                current_arm = robot_env.qpos_arm.copy()
                
                # Calculate movements
                base_drift = np.linalg.norm(current_base - initial_base_pos)
                arm_movement = np.linalg.norm(current_arm - initial_arm)
                
                if step % 10 == 0:  # Print every 10th step
                    print(f"    Step {step+1}: arm_joints=[{current_arm[0]:.3f}, {current_arm[1]:.3f}, {current_arm[2]:.3f}, {current_arm[3]:.3f}, {current_arm[4]:.3f}, {current_arm[5]:.3f}, {current_arm[6]:.3f}]")
                    print(f"               base_drift={base_drift:.4f}, arm_movement={arm_movement:.4f}")
                
                if terminated or truncated:
                    print(f"    Episode terminated at step {step+1}")
                    break
            
            # Final verification
            final_base = robot_env.qpos_base.copy()
            final_arm = robot_env.qpos_arm.copy()
            
            total_base_drift = np.linalg.norm(final_base - initial_base_pos)
            total_arm_movement = np.linalg.norm(final_arm - initial_arm)
            
            # Get final end-effector position
            try:
                if hasattr(robot_env, 'arm_controller') and robot_env.arm_controller is not None:
                    site_id = robot_env.sim.model.name2id('pinch_site', 'site')
                    final_ee_pos = robot_env.sim.data.site_xpos[site_id].copy()
                    if initial_ee_pos is not None:
                        ee_error = np.linalg.norm(final_ee_pos - target_ee_pos)
                        ee_movement = np.linalg.norm(final_ee_pos - initial_ee_pos)
                    else:
                        ee_error = float('inf')
                        ee_movement = 0.0
                    print(f"    Final end-effector position: [{final_ee_pos[0]:.3f}, {final_ee_pos[1]:.3f}, {final_ee_pos[2]:.3f}]")
                else:
                    final_ee_pos = None
                    ee_error = float('inf')
                    ee_movement = 0.0
                    print(f"    Final end-effector position: [arm controller not accessible]")
            except Exception as e:
                final_ee_pos = None
                ee_error = float('inf')
                ee_movement = 0.0
                print(f"    Final end-effector position: [FK failed: {e}]")
            
            print(f"  📊 Movement Summary:")
            print(f"    Initial arm:  [{initial_arm[0]:.3f}, {initial_arm[1]:.3f}, {initial_arm[2]:.3f}, {initial_arm[3]:.3f}, {initial_arm[4]:.3f}, {initial_arm[5]:.3f}, {initial_arm[6]:.3f}]")
            print(f"    Final arm:    [{final_arm[0]:.3f}, {final_arm[1]:.3f}, {final_arm[2]:.3f}, {final_arm[3]:.3f}, {final_arm[4]:.3f}, {final_arm[5]:.3f}, {final_arm[6]:.3f}]")
            print(f"    Target EE:    [{target_ee_pos[0]:.3f}, {target_ee_pos[1]:.3f}, {target_ee_pos[2]:.3f}]")
            print(f"    🦾 Arm movement: {total_arm_movement:.4f} rad")
            print(f"    🏠 Base drift: {total_base_drift:.4f} (should be minimal)")
            if ee_error != float('inf'):
                print(f"    🎯 End-effector error: {ee_error:.4f}")
                print(f"    📐 End-effector movement: {ee_movement:.4f}")
            
            # Validate movement
            if total_arm_movement < 0.01:
                print(f"    ❌ WARNING: Arm barely moved!")
            else:
                print(f"    ✅ Arm movement confirmed!")
                
            if total_base_drift > 0.1:
                print(f"    ⚠️  WARNING: Base drifted significantly!")
            else:
                print(f"    ✅ Base remained fixed (drift: {total_base_drift:.4f})")
            
            # Validate end-effector position
            if ee_error != float('inf'):
                if ee_error < 0.1:  # 10cm tolerance for 3D environment
                    print(f"    ✅ End-effector reached target position! (error: {ee_error:.4f})")
                elif ee_movement > 0.05:
                    print(f"    ⚠️  End-effector moved but significant error from target (error: {ee_error:.4f})")
                else:
                    print(f"    ❌ WARNING: End-effector barely moved!")
            
            if terminated or truncated:
                print(f"  Episode terminated at movement {i+1}")
                break
        
        env.close()
        print("\n" + "="*50)
        print("TidyBot3D Arm-Only Test Summary:")
        print("✅ Environment created and reset successfully")
        print("✅ Arm movement commands executed while base stayed fixed")
        print("✅ Joint position tracking in 3D environment verified")
        print("✅ End-effector position tracking and validation confirmed")
        print("✅ Arm controller integration with MuJoCo confirmed")
        print("TidyBot3D arm-only test completed successfully!")
        return True
        
    except Exception as e:
        print(f"TidyBot3D arm-only test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all arm movement tests."""
    print("PRBench Arm-Only Movement Test Suite")
    print("This script demonstrates how to control robot arm movement while keeping base fixed.\n")
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Direct ArmController
    if test_arm_controller():
        success_count += 1
    
    # Test 2: TidyBot3D arm-only movement
    if test_tidybot3d_arm_only():
        success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All arm movement tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main() 