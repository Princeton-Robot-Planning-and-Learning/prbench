# Robot Movement Test Scripts

This directory contains test scripts for testing both robot base and arm movement in the PRBench environments.

## Available Scripts

### 1. `base_test.py` (Base Movement Only)

A focused test script that demonstrates base movement control while keeping arm stationary:

```bash
source ~/miniconda3/bin/activate pr_env
python base_test.py
```

**Features:**
- Tests the `BaseController` class directly for base movement
- Demonstrates smooth trajectory generation using Ruckig for base
- Tests base movement in TidyBot3D environment while keeping arm stationary
- Shows position tracking and error calculation for base system
- Validates actual robot base movement vs. commanded positions
- No complex dependencies or visualizations

**Output:**
- Shows position updates for each base movement step
- Displays velocity and acceleration limits for base controller
- Reports test success/failure for base components
- Validates that robot base actually reaches commanded positions
- Measures actual movement distances and convergence errors

### 2. `arm_only_test.py` (Arm Movement Only)

A dedicated test script that demonstrates arm movement control while keeping base fixed:

```bash
source ~/miniconda3/bin/activate pr_env
python arm_only_test.py
```

**Features:**
- Tests the `ArmController` class directly for arm movement
- Demonstrates inverse kinematics and trajectory generation for 7-DOF arm
- Tests arm movement in TidyBot3D environment while keeping base fixed
- Shows joint position tracking and gripper control
- Validates actual robot arm movement vs. commanded positions
- **Verifies end-effector position accuracy using forward kinematics**
- Tracks end-effector error from target positions
- No complex dependencies or visualizations

**Output:**
- Shows joint position updates for each arm movement step
- Displays velocity and acceleration limits for arm controller
- Reports test success/failure for arm and gripper components
- **Shows actual end-effector positions and validates against target positions**
- Measures actual joint movement distances and gripper operation
- Provides end-effector position error and movement tracking

### 3. `test_base_movement.py` (Legacy - Comprehensive)

A legacy comprehensive test script with multiple environments and visualization:

```bash
source ~/miniconda3/bin/activate pr_env
python test_base_movement.py
```

**Features:**
- Tests Motion2D and TidyBot3D environments
- Includes visualization capabilities (if matplotlib available)
- More detailed movement patterns and analysis
- Comprehensive error handling

**Note:** This is a legacy script. For focused testing, use `base_test.py` or `arm_only_test.py` instead.

## Robot Movement Control

### BaseController Class

The `BaseController` class provides smooth base movement using online trajectory generation:

```python
from prbench.envs.tidybot.base_controller import BaseController

# Create controller
qpos = np.zeros(3)  # [x, y, theta]
qvel = np.zeros(3)  # [vx, vy, omega] 
ctrl = np.zeros(3)  # Control target
controller = BaseController(qpos, qvel, ctrl, timestep=0.1)

# Reset to origin
controller.reset()

# Move to target position
action = {"base_pose": np.array([1.0, 0.5, 0.3])}
controller.run_controller(action)

# Check current position
print(f"Current position: {controller.ctrl}")
```

### ArmController Class

The `ArmController` class provides arm movement using inverse kinematics and trajectory generation:

```python
from prbench.envs.tidybot.arm_controller import ArmController

# Create controller
num_joints = 7
qpos = np.zeros(num_joints)      # Joint positions
qvel = np.zeros(num_joints)      # Joint velocities  
ctrl = np.zeros(num_joints)      # Joint control targets
qpos_gripper = np.array([0.0])   # Gripper position
ctrl_gripper = np.array([0.0])   # Gripper control
controller = ArmController(qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep=0.1)

# Reset to retract position
controller.reset()

# Move to target end-effector position
action = {
    "arm_pos": np.array([0.3, 0.0, 0.6]),        # End-effector position
    "arm_quat": np.array([1.0, 0.0, 0.0, 0.0]),  # End-effector orientation
    "gripper_pos": np.array([0.5])               # Gripper state (0-1)
}
controller.run_controller(action)

# Check current joint positions
print(f"Current joint positions: {controller.ctrl}")
print(f"Current gripper position: {controller.ctrl_gripper}")
```

### TidyBot3D Base Movement

For TidyBot3D environment, base movement is controlled through action vectors:

```python
# Action format: [base_pose(3), arm_pos(3), arm_quat(4), gripper_pos(1)]
base_target = [0.1, 0.0, 0.0]  # Move forward
arm_pos = [0.0, 0.0, 0.5]      # Keep arm stationary
arm_quat = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
gripper_pos = [0.0]             # Gripper closed

action = np.concatenate([base_target, arm_pos, arm_quat, gripper_pos])
obs, reward, terminated, truncated, info = env.step(action)
```

## Configuration

### Velocity and Acceleration Limits

The BaseController uses the following default limits:
- Max velocity: [0.5 m/s, 0.5 m/s, 3.14 rad/s] for [x, y, theta]
- Max acceleration: [0.5 m/s², 0.5 m/s², 2.36 rad/s²] for [x, y, theta]

These can be customized when creating the controller:

```python
controller = BaseController(
    qpos, qvel, ctrl, timestep,
    max_velocity=[1.0, 1.0, 2.0],
    max_acceleration=[1.0, 1.0, 1.0]
)
```

## Coordinate System

- **X-axis**: Forward/backward movement
- **Y-axis**: Left/right movement  
- **Theta**: Rotation (counterclockwise positive)

## Troubleshooting

1. **Import errors**: Make sure the `pr_env` conda environment is activated
2. **Environment not found**: Ensure `prbench.register_all_environments()` is called first
3. **Visualization errors**: Set `visualize=False` or install matplotlib if needed
4. **Trajectory not converging**: Check that target positions are within reasonable bounds

## Example Output

```
Testing BaseController for Base-Only Movement
==================================================
Max velocity limits: [0.5, 0.5, 3.14]
Max acceleration limits: [0.5, 0.5, 2.36]
Initial position: [0. 0. 0.]

Move Forward -> [1.0, 0.0, 0.0]
  Step  1: pos=[0.003, 0.000, 0.000], error=0.9975, delta=0.0025
  Step  2: pos=[0.010, 0.000, 0.000], error=0.9900, delta=0.0075
  ...
  -> Reached target in X steps!
``` 