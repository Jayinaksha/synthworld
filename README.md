# SynthWorld - Open-World Simulation Sandbox

A procedurally generated cyberpunk simulation for robotics research and synthetic data generation.

## Features

- **Procedurally Generated World**: Infinite cyberpunk city with buildings, roads, and props
- **Multiple Robot Types**: Wheeled, quadruped, robot arm, and drone simulations
- **AI-Driven NPCs**: Pedestrians and vehicles with behavior trees
- **Synthetic Data Generation**: Export to COCO, KITTI, and YOLO formats
- **Physics Simulation**: PyBullet-based physics with realistic dynamics
- **Sensor Simulation**: RGB/Depth cameras, LiDAR, IMU, GPS

## Requirements

- Python 3.8+
- PyBullet (physics)
- Panda3D (rendering)
- NumPy, OpenCV, PyYAML

## Installation

```bash
cd synthworld
pip install -r requirements.txt
```

## Quick Start

```python
from synthworld import SynthWorldApp

# Create application
app = SynthWorldApp()

# Spawn a robot
app.spawn_robot('wheeled', position=(0, 0, 1))

# Generate world and start
app.generate_initial_world()
app.run()
```

## Command Line Usage

```bash
# Basic demo
python -m synthworld --demo basic

# Robot showcase
python -m synthworld --demo robot

# Traffic simulation
python -m synthworld --demo traffic

# Data generation
python -m synthworld --demo data --output ./synthetic_data
```

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
- IMU/GPS data
- 2D/3D bounding boxes
- Segmentation masks

## Controls

| Key | Action |
|-----|--------|
| WASD | Move robot |
| Arrow keys | Look around |
| Space | Jump/Fly up |
| Shift | Sprint/Crouch |
| Tab | Switch camera mode |
| P | Pause simulation |
| F1 | Toggle debug info |

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
display:
  width: 1280
  height: 720
  fullscreen: false

physics:
  timestep: 0.004167  # 240 Hz
  gravity: -9.81

world:
  seed: 42
  chunk_size: 100
  render_distance: 3

npcs:
  max_pedestrians: 50
  max_vehicles: 30
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
│   └── input.py       # Input management
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
    └── app.py         # Main application
```

## License

MIT License
