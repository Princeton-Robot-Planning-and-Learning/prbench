# TidyBot Integration with PRBench

This document describes the integration of TidyBot 3D mobile manipulation environments into PRBench.

## Overview

The TidyBot integration adds 3D mobile manipulation capabilities to PRBench, featuring:

- **Holonomic Mobile Base**: Powered casters for full planar motion control
- **Kinova Gen3 Arm**: 7-DOF robotic arm with gripper
- **Multiple Scene Types**: Table, drawer, cupboard, and cabinet environments
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

# Create a TidyBot environment (no policy_type)
env = prbench.make(
    "prbench/TidyBot3D-table-o5-v0",
    show_viewer=True,
    show_images=True
    )

# Standard Gymnasium interface
obs, info = env.reset(seed=123)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
img = env.render()
env.close()
```

check_env 

'''
import gymnasium
from gymnasium.utils.env_checker import check_env
import prbench
prbench.register_all_environments()
env = prbench.make("prbench/TidyBot3D-cabinet-o3-v0",render_mode="rgb_array")
check_env(env.unwrapped)
'''

### Environment Naming Convention

TidyBot environments follow the naming pattern:
```
prbench/TidyBot3D-{scene_type}-o{num_objects}-v0
```

Where:
- `scene_type`: `table`, `drawer`, `cupboard`, `cabinet`
- `num_objects`: Number of objects in the scene (varies by scene)

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

### Example 0: ground random actions
```python
import prbench
prbench.register_all_environments()
env = prbench.make_unwrapped(
    "prbench/TidyBot3D-ground-o7-v0",
    show_viewer=True,
    show_images=True,
    )

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.3f}")
    if terminated or truncated:
        break

env.close()
```

### Example 1: Table random actions
```python
import prbench
prbench.register_all_environments()
env = prbench.make_unwrapped(
    "prbench/TidyBot3D-table-o5-v0",
    show_viewer=True,
    show_images=True)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.3f}")
    print([terminated, truncated])
    if terminated or truncated:
        break

env.close()
```

### Example 2: Random actions in Cupboard
```python
import prbench

prbench.register_all_environments()
env = prbench.make_unwrapped(
    "prbench/TidyBot3D-cupboard-o8-v0",
    show_viewer=True,
    show_images=True)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.3f}")
    print([terminated, truncated])
    if terminated or truncated:
        break

env.close()
```

### Example 3: Random actions in Cabinet
```python
import prbench

prbench.register_all_environments()
env = prbench.make_unwrapped(
    "prbench/TidyBot3D-cabinet-o3-v0",
    show_viewer=True,
    show_images=True)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.3f}")
    if terminated or truncated:
        break

env.close()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all TidyBot dependencies are installed
2. **Model File Errors**: Verify model files are in the correct location
3. **Memory Issues**: TidyBot environments can be memory-intensive; close environments properly

### Getting Help

- Check the main PRBench documentation
- Review the TidyBot original documentation
- Run the integration test to identify issues
- Check the environment metadata for configuration details

## References

- **TidyBot++ Paper**: [Conference on Robot Learning (CoRL), 2024](https://arxiv.org/abs/2412.10447)
- **TidyBot Repository**: [https://github.com/tidybot2/tidybot2](https://github.com/tidybot2/tidybot2)
- **PRBench Documentation**: See main README.md 