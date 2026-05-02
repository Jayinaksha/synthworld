# SynthWorld - Open-World Simulation Sandbox

A procedurally generated cyberpunk simulation for robotics research and synthetic data generation.

## Features

- **Procedurally Generated World**: Infinite cyberpunk city with buildings, roads, and props
- **Multiple Robot Types**: Wheeled, quadruped, robot arm, and drone simulations
- **AI-Driven NPCs**: Pedestrians and vehicles with behavior trees
- **Synthetic Data Generation**: Export to COCO, KITTI, and YOLO formats
- **Physics Simulation**: PyBullet-based physics with realistic dynamics
- **Sensor Simulation**: RGB/Depth cameras, LiDAR, and IMU
- **HUD**: Cyberpunk-themed heads-up display with FPS, speed, time-of-day, and camera mode

## Requirements

- Python 3.8+
- PyBullet (physics)
- Panda3D (rendering)
- NumPy, SciPy, OpenCV, Pillow
- PyYAML, h5py
- py-trees (NPC behavior trees)
- pygame (audio)
- noise (Perlin noise for terrain)

## Installation

```bash
cd synthworld
pip install -r requirements.txt
```

Or install as a package (also installs the `synthworld` CLI entry point):

```bash
pip install -e .
```

## Quick Start

```python
from synthworld import SynthWorldApp

# Create application
app = SynthWorldApp()

# Spawn a robot (world is generated automatically when run() is called)
app.spawn_robot('wheeled', position=(0, 0, 1))

# Start the simulation
app.run()
```

## Command Line Usage

```bash
# Basic demo
python -m synthworld --demo basic

# Robot showcase (spawns all four robot types)
python -m synthworld --demo robot

# Traffic simulation (max pedestrians & vehicles)
python -m synthworld --demo traffic

# Data generation (captures synthetic data to ./demo_data)
python -m synthworld --demo data --output ./demo_data

# Headless data generation with a fixed seed
python -m synthworld --headless --capture --output ./synthetic_data --seed 42

# Verbose logging
python -m synthworld --verbose
```

### CLI Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--config` | `-c` | `config/settings.yaml` | Path to configuration file |
| `--headless` | | `false` | Run without display |
| `--capture` | | `false` | Enable data capture from start |
| `--output` | `-o` | `./synthetic_data` | Output directory for synthetic data |
| `--seed` | `-s` | random | Random seed for world generation |
| `--verbose` | `-v` | `false` | Enable verbose (DEBUG) logging |
| `--demo` | | | Run a preset demo: `basic`, `robot`, `traffic`, `data` |

## Robot Types

### Wheeled Robot
- Differential drive or Ackermann steering
- Best for ground navigation

### Quadruped Robot
- Four-legged walking robot
- Trot gait implementation

### Robot Arm
- 6-DOF articulated arm
- Forward/inverse kinematics
- Gripper control

### Drone
- Quadrotor with PID flight control
- Attitude and altitude stabilization

## Synthetic Data

### Supported Formats
- **COCO**: Object detection with 2D bounding boxes
- **KITTI**: 3D object detection with calibration
- **YOLO**: Fast object detection training format

### Captured Data
- RGB images
- Depth maps
- LiDAR point clouds
- IMU data
- 2D/3D bounding boxes
- Segmentation masks

## Controls

| Key | Action |
|-----|--------|
| WASD | Move forward / backward / strafe |
| Arrow keys | Look around |
| Space | Move up / fly up |
| Shift | Move down |
| C | Cycle camera mode (Free → Third Person → First Person → Orbit) |
| Mouse wheel | Camera zoom in / out |
| Q / E | Robot arm up / down |
| G | Toggle gripper |
| F | Interact |
| Mouse1 | Use |
| Mouse3 | Grab |
| P | Capture frame |
| Ctrl+R | Toggle recording |
| Escape | Pause / resume simulation |
| Tab | Toggle menu |
| I | Inventory |
| M | Map |
| F1 | Toggle debug info |
| F2 | Toggle physics debug |
| F3 | Debug teleport |

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
display:
  width: 1280
  height: 720
  fullscreen: false
  vsync: true
  target_fps: 60
  title: "SynthWorld Simulator"

physics:
  gravity: [0, 0, -9.81]
  timestep: 0.004166667  # 240 Hz internal physics
  solver_iterations: 50

world:
  seed: 42
  chunk_size: 64
  render_distance: 3  # chunks
  terrain:
    scale: 0.02
    height_multiplier: 20
    octaves: 4
  buildings:
    density: 0.3
    min_height: 10
    max_height: 50

sensors:
  camera:
    width: 640
    height: 480
    fov: 60
  lidar:
    range: 30
    horizontal_resolution: 1.0  # degrees
    vertical_fov: 30
  imu:
    noise_accel: 0.01
    noise_gyro: 0.001

npcs:
  max_count: 20
  spawn_radius: 50
  behavior_update_rate: 10  # Hz

data_export:
  enabled: true
  output_directory: "./synthetic_data"
  capture_interval: 5  # frames
  formats:
    - rgb
    - depth
    - segmentation
    - annotations

audio:
  master_volume: 0.8
  music_volume: 0.5
  sfx_volume: 0.7

debug:
  show_physics: false
  show_fps: true
  show_coordinates: true
  verbose_logging: false
```

## Project Structure

```
synthworld/
├── config/
│   └── settings.yaml
├── engine/
│   ├── core.py        # Main engine loop
│   ├── physics.py     # PyBullet wrapper
│   ├── renderer.py    # Panda3D renderer
│   └── input.py       # Input management (key bindings, action mapping)
├── world/
│   ├── terrain.py     # Procedural terrain
│   ├── buildings.py   # Cyberpunk buildings
│   ├── props.py       # Street props
│   └── generator.py   # World coordination
├── robots/
│   ├── base.py        # Robot base class
│   ├── wheeled.py     # Wheeled robots
│   ├── quadruped.py   # Walking robots
│   ├── arm.py         # Robot arms
│   ├── drone.py       # Quadrotors
│   └── sensors/       # Camera, LiDAR, IMU
├── npcs/
│   ├── base.py        # NPC base class
│   ├── pedestrian.py  # Walking NPCs
│   ├── vehicle.py     # Traffic vehicles
│   └── behavior.py    # Behavior trees
├── data/
│   ├── capture.py     # Data capture
│   ├── annotations.py # Annotation generation
│   └── export.py      # COCO/KITTI/YOLO export
└── synthworld/
    ├── __init__.py
    ├── __main__.py
    ├── app.py         # Main application
    ├── config/        # Bundled default config
    └── ui/
        └── hud.py     # Cyberpunk heads-up display
```

## License

MIT License
