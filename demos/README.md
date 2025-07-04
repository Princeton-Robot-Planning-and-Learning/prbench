# Human Demonstrations

This directory contains human demonstrations collected using the `scripts/collect_demos.py` script.

## Demo Format

Each demonstration is saved as a JSON file with the following structure:

```json
{
  "env_id": "prbench/Obstruction2D-o2-v0",
  "timestamp": 1234567890,
  "observations": [[...], [...], ...],
  "actions": [[...], [...], ...],
  "rewards": [0.0, -1.0, -1.0, ...],
  "terminated": true,
  "truncated": false
}
```

- `env_id`: The environment identifier
- `timestamp`: Unix timestamp when the demo was created
- `observations`: List of observation arrays (numpy arrays converted to lists)
- `actions`: List of action arrays (numpy arrays converted to lists)
- `rewards`: List of reward values
- `terminated`: Whether the episode terminated naturally
- `truncated`: Whether the episode was truncated (e.g., time limit)

## Usage

To collect demonstrations for a specific environment:

```bash
python scripts/collect_demos.py prbench/Obstruction2D-o2-v0
```

You can also specify a custom demo directory:

```bash
python scripts/collect_demos.py prbench/Obstruction2D-o2-v0 --demo-dir /path/to/demos
```

**Note:** This script requires pygame. If you don't have it installed, you'll get a helpful error message with installation instructions.

## Controls

The controls are defined by each environment. The script will print the available controls at startup, as provided by the environment's `get_human_input_mapping()` method.

## Extending to New Environments

To support human demonstration collection, your environment **must** implement:
- `get_human_input_mapping()`
- `map_human_input_to_action(keys_pressed: set[str])`

See `Obstruction2DEnv` for an example.

## Available Environments

- `prbench/Obstruction2D-o0-v0`: No obstructions
- `prbench/Obstruction2D-o1-v0`: 1 obstruction
- `prbench/Obstruction2D-o2-v0`: 2 obstructions
- `prbench/Obstruction2D-o3-v0`: 3 obstructions
- `prbench/Obstruction2D-o4-v0`: 4 obstructions 