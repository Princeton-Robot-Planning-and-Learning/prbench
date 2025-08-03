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
import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import prbench

# Register all environments (including TidyBot)
prbench.register_all_environments()

# Create a TidyBot environment (no policy_type)
env = prbench.make(
    "prbench/TidyBot3D-ground-o5-v0",
    render_images=True,
    show_viewer=False,
    show_images=False
    )

# Standard Gymnasium interface
obs, info = env.reset(seed=123)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
img = env.render()
env.close()
``` 

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
- `scene_type`: `ground`
- `num_objects`: Number of objects in the scene


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
