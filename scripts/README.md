# PRBench Scripts

This directory contains utility scripts for PRBench demonstration collection and video generation.

## Demo Collection with Controller Support

### collect_demos.py

Collect human demonstrations using keyboard/mouse or PS5 DualSense controller.

#### Usage

```bash
python scripts/collect_demos.py <environment_id>
```

Examples:
```bash
python scripts/collect_demos.py prbench/Motion2D-p1-v0
python scripts/collect_demos.py prbench/Obstruction2D-o2-v0
python scripts/collect_demos.py prbench/ClutteredRetrieval2D-o10-v0
```

#### PS5 DualSense Controller Support

The script automatically detects connected PS5 controllers and provides intuitive control mapping:

**Analog Sticks:**
- **Left Stick**: Rotate robot (yaw control)
- **Right Stick**: Move robot base (x, y movement)

**Face Buttons:**
- **X (Cross)**: Toggle vacuum gripper on/off
- **Circle**: Reset environment (start new demo)
- **Square**: Save current demo

**D-Pad:**
- **D-pad Up**: Extend robot arm outward
- **D-pad Down**: Retract robot arm inward

#### Auto-Save Behavior

- ✅ **Goal Reached**: Automatically saves demo and resets for next attempt
- ❌ **Episode Failed/Timeout**: Automatically resets without saving

#### Keyboard/Mouse Fallback

If no controller is detected, the system falls back to keyboard and mouse controls:

- **Mouse**: Click and drag virtual analog sticks on screen
- **W/S**: Extend/retract robot arm
- **Space**: Toggle vacuum gripper
- **R**: Reset environment
- **G**: Save demo
- **Q**: Quit

#### Demo Output

Demos are saved as pickle files in the following structure:
```
demos/
├── EnvironmentName/
│   ├── 0/
│   │   └── timestamp.p
│   ├── 1/
│   │   └── timestamp.p
│   └── ...
```

Each demo file contains:
- Environment ID and seed
- Sequence of observations and actions
- Rewards and termination information

## Video Generation

### generate_demo_video.py

Convert saved demonstration files into GIF videos.

#### Usage

```bash
python scripts/generate_demo_video.py <demo_file.p> [options]
```

#### Options

- `--output`, `-o`: Custom output path (default: auto-generated in docs/envs/assets/demo_gifs/)
- `--fps`: Frames per second (default: environment's render_fps)
- `--loop`: Number of loops for GIF (0 = infinite, default: 0)

#### Examples

```bash
# Basic usage - generates GIF in docs/envs/assets/demo_gifs/
python scripts/generate_demo_video.py demos/Motion2D-p1/0/1752189500.p

# Custom output path and settings
python scripts/generate_demo_video.py demos/Motion2D-p1/0/1752189500.p \
  --output my_demo.gif \
  --fps 30 \
  --loop 0
```

## Environment Documentation

### generate_env_docs.py

Automatically generates markdown documentation for all registered environments. This script is typically run as part of the pre-commit process.

#### Usage

```bash
python scripts/generate_env_docs.py
```

This creates documentation files in `docs/envs/` with environment descriptions, observation/action spaces, and embedded demo videos.
