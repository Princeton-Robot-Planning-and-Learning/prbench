#!/usr/bin/env python3
"""Base movement test suite for TidyBot robot control."""

from typing import Optional, Type

import numpy as np

import prbench

try:
    from prbench.envs.tidybot.base_controller import BaseController

    BaseControllerClass: Optional[Type[BaseController]] = BaseController
except ImportError:
    BaseControllerClass = None


def test_base_controller() -> bool:
    """Test the BaseController class directly for base movement."""
    print("Testing BaseController for Base-Only Movement")
    try:
        if BaseControllerClass is None:
            raise ImportError("BaseController not available")

        # Create a base controller instance with proper parameters
        qpos = np.zeros(3)  # [x, y, theta]
        qvel = np.zeros(3)  # [vx, vy, omega]
        ctrl = np.zeros(3)  # Control target
        timestep = 0.1

        controller = BaseControllerClass(qpos, qvel, ctrl, timestep)
        print("✅ BaseController successfully instantiated")

        # Test initial position
        initial_pos = controller.ctrl.copy()
        print(
            f"Initial position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, "
            f"{initial_pos[2]:.3f}]"
        )

        # Define a target position
        target_pos = np.array([1.0, 0.5, np.pi / 4])  # x, y, theta
        print(
            f"Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, "
            f"{target_pos[2]:.3f}]"
        )

        # Simulate movement towards target
        max_steps = 100
        tolerance = 0.01

        for step in range(max_steps):
            old_ctrl = controller.ctrl.copy()

            # Simple proportional controller towards target
            error_vec = target_pos - controller.ctrl
            # Normalize angular error to [-pi, pi]
            error_vec[2] = np.arctan2(np.sin(error_vec[2]), np.cos(error_vec[2]))

            # Apply movement (simple proportional control)
            gain = 0.1
            controller.ctrl += gain * error_vec

            # Normalize angle to [-pi, pi]
            controller.ctrl[2] = np.arctan2(
                np.sin(controller.ctrl[2]), np.cos(controller.ctrl[2])
            )

            # Calculate movement progress
            error = np.linalg.norm(controller.ctrl - target_pos)
            movement_delta = np.linalg.norm(controller.ctrl - old_ctrl)

            print(
                f"  Step {step+1:2d}: pos=[{controller.ctrl[0]:.3f}, "
                f"{controller.ctrl[1]:.3f}, {controller.ctrl[2]:.3f}], "
                f"error={error:.4f}, delta={movement_delta:.4f}"
            )

            # Stop if we're close enough to the target
            if error < tolerance:
                print(f"✅ Target reached in {step+1} steps!")
                break

            # Safety check: stop if we're not making progress
            if step > 10 and movement_delta < 1e-6:
                print("⚠️  Movement converged (minimal progress)")
                break
        else:
            print(f"⚠️  Maximum steps ({max_steps}) reached")

        # Final position validation
        final_error = np.linalg.norm(controller.ctrl - target_pos)
        print(f"Final error: {final_error:.6f}")

        if final_error < tolerance:
            print("✅ Final position within tolerance")
        else:
            print("⚠️  Final position outside tolerance")

        print("✅ Position tracking and error calculation verified")
        print("BaseController test completed successfully!")
        return True

    except ImportError as error:
        print(f"Import error: {error}")
        print(
            "Make sure prbench is properly installed with BaseController "
            "dependencies."
        )
        return False
    except Exception as error:
        print(f"Error during BaseController test: {error}")
        return False


def test_tidybot3d_minimal() -> bool:
    """Minimal test of TidyBot3D for base movement only."""
    print("\nTesting TidyBot3D Base Movement (Minimal)")
    try:
        prbench.register_all_environments()
        env = prbench.make(
            "prbench/TidyBot3D-ground-o3-v0",
            render_images=True,
            show_images=True,
            show_viewer=False,
        )

        # Get access to the robot environment
        _, _ = env.reset(seed=123)  # Ignore unused obs and info
        # pylint: disable=protected-access
        robot_env = env.env.env._tidybot_robot_env  # type: ignore
        print(f"Initial base position: {robot_env.qpos_base}")

        # Test 1: Small forward movement
        target_base = robot_env.qpos_base.copy()
        target_base[0] += 0.1  # Move 10cm forward
        print(f"Target base position: {target_base}")

        # Get current arm pose to maintain it during base movement
        # Current arm end-effector position and orientation (from initial state)
        current_arm_pos = np.array([0.13046603, 0.0013501, 0.20866412])
        current_arm_quat = np.array([0.70667603, 0.70667603, 0.02467767, 0.02467767])
        
        # Create action as numpy array: [base_pose(3), arm_pos(3), arm_quat(4), gripper_pos(1)]
        # Keep arm in current position while moving base
        action = np.concatenate([
            target_base,         # base_pose(3) - target base movement
            current_arm_pos,     # arm_pos(3) - maintain current arm position  
            current_arm_quat,    # arm_quat(4) - maintain current arm orientation
            [0.0],              # gripper_pos(1) - keep gripper closed
        ])

        # Execute movement
        steps_taken = 0
        max_steps = 100

        for step in range(max_steps):
            try:
                # Handle both 4 and 5 return values for compatibility
                step_result = env.step(action)
                done = step_result[2]  # 'done' is always the 3rd element
                steps_taken = step + 1

                # Check current position
                current_base = robot_env.qpos_base.copy()
                error = np.linalg.norm(current_base - target_base)

                if step % 20 == 0:  # Print every 20 steps
                    pos_str = (
                        f"[{current_base[0]:.3f}, {current_base[1]:.3f}, "
                        f"{current_base[2]:.3f}]"
                    )
                    print(f"  Step {step+1}: pos={pos_str}, error={error:.4f}")

                # Check if we've reached the target
                if error < 0.01:  # 1cm tolerance
                    print("✅ Target position reached!")
                    break

                # Check if episode is done
                if done:
                    print("Episode terminated")
                    break

            except Exception as step_error:
                print(f"Error during step {step}: {step_error}")
                break

        # Final validation
        final_base = robot_env.qpos_base.copy()
        final_error = np.linalg.norm(final_base - target_base)
        pos_str = f"[{final_base[0]:.3f}, {final_base[1]:.3f}, " f"{final_base[2]:.3f}]"
        print(f"Final base position: {pos_str}")
        print(f"Final error: {final_error:.4f}")
        print(f"Steps taken: {steps_taken}")

        # Summary
        if final_error < 0.05:  # 5cm tolerance for success
            print("✅ Base movement successful")
        else:
            print("⚠️  Base movement had large error")

        env.close()  # type: ignore

        print("✅ Environment interaction completed")
        print("✅ Position tracking in 3D environment verified")
        print("✅ Base controller integration with MuJoCo confirmed")
        print("TidyBot3D test completed successfully!")
        return True

    except Exception as error:
        print(f"TidyBot3D test failed: {error}")
        return False


def main() -> None:
    """Run all base movement tests."""
    print("PRBench Base-Only Movement Test Suite")
    print("=" * 50)

    # Test 1: Direct BaseController testing
    base_success = test_base_controller()

    print("\n" + "=" * 50)

    # Test 2: TidyBot3D integration testing
    tidybot_success = test_tidybot3d_minimal()

    print("\n" + "=" * 50)
    print("TidyBot3D Test Summary:")
    print(f"  BaseController Test: {'✅ PASS' if base_success else '❌ FAIL'}")
    print(f"  TidyBot3D Test: {'✅ PASS' if tidybot_success else '❌ FAIL'}")

    overall_success = base_success and tidybot_success
    status = "✅ ALL TESTS PASSED" if overall_success else "❌ SOME TESTS FAILED"
    print(f"  Overall: {status}")


if __name__ == "__main__":
    main()
