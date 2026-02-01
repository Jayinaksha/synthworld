"""
SynthWorld - Open-World Simulation Sandbox

A procedurally generated cyberpunk simulation for robotics research
and synthetic data generation.

Features:
- Procedurally generated cyberpunk city
- Multiple robot types (wheeled, quadruped, arm, drone)
- AI-driven NPCs (pedestrians, vehicles)
- Synthetic data generation (COCO, KITTI, YOLO formats)
- Physics simulation with PyBullet
- 3D rendering with Panda3D

Usage:
    from synthworld import SynthWorldApp
    
    app = SynthWorldApp()
    app.spawn_robot('wheeled')
    app.run()
"""

__version__ = '0.1.0'
__author__ = 'SynthWorld Team'

from .app import SynthWorldApp

# Lazy imports for submodules
def _lazy_import(name):
    """Lazy import of submodules."""
    import importlib
    return importlib.import_module(f'.{name}', __package__)

def __getattr__(name):
    if name in ('engine', 'world', 'robots', 'npcs', 'data'):
        return _lazy_import(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'SynthWorldApp',
    '__version__',
    'engine',
    'world',
    'robots',
    'npcs',
    'data'
]
