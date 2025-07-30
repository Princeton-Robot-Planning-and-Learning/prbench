# TidyBot Integration with PRBench

This document describes the integration of TidyBot 3D mobile manipulation environments into PRBench.

## Overview

The TidyBot integration adds 3D mobile manipulation capabilities to PRBench, featuring:

- **Holonomic Mobile Base**: Powered casters for full planar motion control
- **Kinova Gen3 Arm**: 7-DOF robotic arm with gripper
- **Multiple Scene Types**: Table, drawer, cupboard, and cabinet environments
- **Various Policy Types**: Teleoperation, motion planning, and stacking policies
- **Task-Specific Rewards**: Custom reward functions for different tasks

## Installation

### Dependencies

The TidyBot integration requires additional dependencies beyond the base PRBench requirements:

```bash
# Install TidyBot dependencies
pip install mujoco==3.2.4 pin>=2.7.0 ruckig>=0.12.2
pip install opencv-python>=4.9.0 flask>=3.0.2 flask_socketio>=5.3.6
pip install pyzmq>=25.1.2 redis>=5.0.6 phoenix6 pygame>=2.5.2
```

### Model Files

The TidyBot model files are automatically included in the PRBench installation. These include:
- MuJoCo XML scene files for different environments
- Robot model definitions
- Camera configurations

## Usage

### Basic Usage

```python
import prbench

# Register all environments (including TidyBot)
prbench.register_all_environments()

# Create a TidyBot environment
env = prbench.make("prbench/TidyBot3D-table-o3-mp-v0")

# Standard Gymnasium interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
img = env.render()
env.close()
```

### Environment Naming Convention

TidyBot environments follow the naming pattern:
```
prbench/TidyBot3D-{scene_type}-o{num_objects}-{policy_type}-v0
```

Where:
- `scene_type`: `table`, `drawer`, `cupboard`, `cabinet`
- `num_objects`: Number of objects in the scene (varies by scene)
- `policy_type`: Type of policy to use

### Available Scenes

#### Table Scene
- **Description**: Object stacking and manipulation on a table
- **Object counts**: 3, 5, 7 objects
- **Tasks**: Stack objects, pick and place, organize items

#### Drawer Scene
- **Description**: Opening/closing drawers and placing objects inside
- **Object counts**: 2, 4, 6 objects
- **Tasks**: Open drawer, place objects inside, close drawer

#### Cupboard Scene
- **Description**: Opening cupboards and organizing objects
- **Object counts**: 3, 5, 8 objects
- **Tasks**: Open cupboard, place objects on shelves, organize items

#### Cabinet Scene
- **Description**: Cabinet manipulation tasks
- **Object counts**: 2, 4, 6 objects
- **Tasks**: Open cabinet, place objects inside, close cabinet

### Available Policy Types

#### Teleoperation Policies
- `teleop`: Phone-based teleoperation using WebXR
- `remote`: Remote policy execution via network

#### Motion Planning Policies
- `mp`: General motion planning policies
- `mp_three`: Three-sequential motion planning
- `mp_cabinet_two_phase`: Two-phase cabinet manipulation
- `mp_n_cupboard`: N-object cupboard manipulation

#### Stacking Policies
- `stack`: Object stacking policies
- `stack_three`: Three-object stacking policies

#### Custom Grasping Policies
- `custom_grasp`: Custom grasping policies
- `custom_grasp_three`: Three-sequential custom grasping

### Using Policies

#### Standard Step (Manual Control)
```python
# Manual action control
action = env.action_space.sample()  # Random action
obs, reward, terminated, truncated, info = env.step(action)
```

#### Policy-Based Step (Automatic Control)
```python
# Use the environment's internal policy
obs, reward, terminated, truncated, info = env.step_with_policy()
```

### Action Space

The action space is 11-dimensional:
- `base_pose[3]`: Mobile base position (x, y) and orientation (theta)
- `arm_pos[3]`: End effector position (x, y, z)
- `arm_quat[4]`: End effector orientation as quaternion (x, y, z, w)
- `gripper_pos[1]`: Gripper position (0=closed, 1=open)

### Observation Space

The observation space includes:
- Robot state: base pose, arm position/orientation, gripper state
- Object states: positions and orientations of all objects
- Camera images: RGB images from base and wrist cameras
- Scene-specific features: handle positions for cabinets/drawers

## Reward System

Each environment has task-specific reward functions:

### Table Stacking
- Reward for stacking objects
- Bonus for reaching target height
- Penalty for time steps

### Drawer/Cabinet Tasks
- Reward for opening/closing
- Reward for placing objects inside
- Penalty for time steps

### Motion Planning
- Reward for reaching target locations
- Reward for successful pick-and-place
- Penalty for time steps

## Testing

Run the integration test to verify everything works:

```bash
cd /path/to/prbench
python test_tidybot_integration.py
```

This will test:
- Environment creation and registration
- Basic functionality (reset, step, render)
- Different scene types
- Policy execution

## Examples

### Example 1: Table Stacking with Teleoperation
```python
import prbench

prbench.register_all_environments()
env = prbench.make_unwrapped("prbench/TidyBot3D-table-o3-mp-v0")

obs, info = env.reset()
for _ in range(10000):
    obs, reward, terminated, truncated, info = env.step_with_policy()
    print(f"Reward: {reward:.3f}")
    if truncated:
        break

env.close()
```

### Example 2: Motion Planning in Cupboard
```python
import prbench

prbench.register_all_environments()
env = prbench.make("prbench/TidyBot3D-cupboard-o5-mp-v0")

obs, info = env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step_with_policy()
    print(f"Reward: {reward:.3f}")
    if terminated or truncated:
        break

env.close()
```

### Example 3: Custom Grasping
```python
import prbench

prbench.register_all_environments()
env = prbench.make("prbench/TidyBot3D-cupboard-o3-custom_grasp-v0")

obs, info = env.reset()
for _ in range(150):
    obs, reward, terminated, truncated, info = env.step_with_policy()
    print(f"Reward: {reward:.3f}")
    if terminated or truncated:
        break

env.close()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all TidyBot dependencies are installed
2. **Model File Errors**: Verify model files are in the correct location
3. **Policy Errors**: Check that the policy type is compatible with the scene type
4. **Memory Issues**: TidyBot environments can be memory-intensive; close environments properly

### Getting Help

- Check the main PRBench documentation
- Review the TidyBot original documentation
- Run the integration test to identify issues
- Check the environment metadata for configuration details

## References

- **TidyBot++ Paper**: [Conference on Robot Learning (CoRL), 2024](https://arxiv.org/abs/2412.10447)
- **TidyBot Repository**: [https://github.com/tidybot2/tidybot2](https://github.com/tidybot2/tidybot2)
- **PRBench Documentation**: See main README.md 