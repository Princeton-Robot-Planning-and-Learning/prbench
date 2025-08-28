#!/usr/bin/env python3
import numpy as np
import prbench

TARGETS = [
    np.array([0.0, -0.35, 3.14, -2.55, 0.0, -0.87, 1.57]),
    np.array([0.5, -0.2, 2.8, -2.0, -0.3, -0.2, 1.9]),
    np.array([-0.8, 0.3, 2.2, -1.2, 0.4, 0.1, 1.1]),
]

def main():
    prbench.register_all_environments()
    env = prbench.make('prbench/TidyBot3D-ground-o3-v0', render_images=False, show_images=False, show_viewer=False)
    obs, info = env.reset(seed=123)
    robot_env = env.env.env._tidybot_robot_env

    print("Joint Command Tracking Test (TidyBot3D)")
    for idx, target in enumerate(TARGETS, 1):
        print(f"\nTarget {idx}: {np.round(target, 3)}")
        # Build action dict: keep base fixed, command joint targets
        action = {
            "base_pose": robot_env.qpos_base.copy(),
            "arm_joints": target,
            "gripper_pos": [0.0],
        }
        # Run for N control steps using dict-only path to avoid vector action override
        for step in range(300):
            # Use robot_env.step() directly with action dict to avoid vector conversion
            robot_env.step(action)
        # Evaluate error
        current = robot_env.qpos_arm.copy()
        err = current - target
        # Wrap error to [-pi, pi]
        err = (err + np.pi) % (2 * np.pi) - np.pi
        err_norm = np.linalg.norm(err)
        print(f"  Current: {np.round(current, 3)}")
        print(f"  Error:   {np.round(err, 4)} | L2: {err_norm:.4f}")
        # Success criteria per joint < 0.03 rad
        if np.all(np.abs(err) < 0.03):
            print("  ✅ Joint target achieved (< 0.03 rad per joint)")
        else:
            print("  ❌ Joint target not reached within tolerance")

    env.close()

if __name__ == "__main__":
    main() 