"""
Physics utilities for world generation.
"""

import math
import pybullet as p
from typing import Any
from ..engine.physics import RigidBodyConfig, PhysicsWorld
from .buildings import Building
from .props import Prop, PROP_DEFINITIONS, PropDefinition, PropType

def create_building_collider(physics: PhysicsWorld, building: Building):
    """
    Create a static physics collider for a building.
    
    Args:
        physics: PhysicsWorld instance
        building: Building object
    """
    # Dimensions (half extents)
    w = building.width / 2
    d = building.depth / 2
    h = building.height  # Height is full height, but box origin is center
    
    # Position: Building position is at base (z=0 relative to building), but center is (x, y)
    # Box center needs to be at (x, y, z + h/2)
    x, y, z = building.position
    center_pos = (x, y, z + h/2)
    
    # Orientation: Convert degrees to quaternion
    rot_rad = math.radians(building.rotation)
    orientation = p.getQuaternionFromEuler([0, 0, rot_rad])
    
    config = RigidBodyConfig(
        mass=0,  # Static
        position=center_pos,
        orientation=orientation,
        friction=0.8,
        restitution=0.2
    )
    
    # Physics API expects half-extents
    physics.create_box(
        half_extents=(w, d, h/2),
        config=config,
        name=f"building_{id(building)}"
    )

def create_prop_collider(physics: PhysicsWorld, prop: Prop):
    """
    Create a physics collider for a prop.
    
    Args:
        physics: PhysicsWorld instance
        prop: Prop object
    """
    definition = PROP_DEFINITIONS.get(prop.prop_type)
    if not definition:
        # Fallback
        definition = PropDefinition(
            prop.prop_type, 1, 1, 1, (0.5, 0.5, 0.5)
        )
    
    w = definition.width / 2 * prop.scale
    d = definition.depth / 2 * prop.scale
    h = definition.height * prop.scale
    
    # Center position
    # Prop position is at base
    x, y, z = prop.position
    center_pos = (x, y, z + h/2)
    
    # Rotation
    rot_rad = math.radians(prop.rotation)
    orientation = p.getQuaternionFromEuler([0, 0, rot_rad])
    
    config = RigidBodyConfig(
        mass=definition.physics_mass,
        position=center_pos,
        orientation=orientation,
        friction=0.7,
        restitution=0.3
    )
    
    body = physics.create_box(
        half_extents=(w, d, h/2),
        config=config,
        name=f"prop_{id(prop)}"
    )
    
    prop.physics_body = body
