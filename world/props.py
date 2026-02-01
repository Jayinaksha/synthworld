"""
SynthWorld Props Generator

Procedural generation of street furniture, vehicles, and environmental objects.
"""

import random
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class PropType(Enum):
    """Types of props."""
    # Street furniture
    STREET_LIGHT = auto()
    BENCH = auto()
    TRASH_BIN = auto()
    VENDING_MACHINE = auto()
    PHONE_BOOTH = auto()
    
    # Vehicles
    CAR = auto()
    TRUCK = auto()
    MOTORCYCLE = auto()
    DRONE = auto()
    
    # Nature
    TREE = auto()
    BUSH = auto()
    PLANTER = auto()
    
    # Industrial
    CONTAINER = auto()
    BARREL = auto()
    CRATE = auto()
    PIPE = auto()
    
    # Cyberpunk specific
    HOLOGRAM_DISPLAY = auto()
    NEON_SIGN = auto()
    ANTENNA = auto()
    SATELLITE_DISH = auto()
    ROBOT_CHARGING_STATION = auto()


@dataclass
class PropDefinition:
    """Defines properties of a prop type."""
    prop_type: PropType
    width: float
    depth: float
    height: float
    color: Tuple[float, float, float]
    emissive: bool = False
    emissive_color: Tuple[float, float, float] = (0, 0, 0)
    physics_mass: float = 0  # 0 = static
    breakable: bool = False


# Predefined props
PROP_DEFINITIONS = {
    PropType.STREET_LIGHT: PropDefinition(
        PropType.STREET_LIGHT, 0.3, 0.3, 6.0,
        color=(0.2, 0.2, 0.22),
        emissive=True, emissive_color=(1.0, 0.9, 0.7)
    ),
    PropType.BENCH: PropDefinition(
        PropType.BENCH, 2.0, 0.6, 0.8,
        color=(0.3, 0.25, 0.2)
    ),
    PropType.TRASH_BIN: PropDefinition(
        PropType.TRASH_BIN, 0.5, 0.5, 1.0,
        color=(0.25, 0.25, 0.28)
    ),
    PropType.VENDING_MACHINE: PropDefinition(
        PropType.VENDING_MACHINE, 1.0, 0.8, 2.0,
        color=(0.15, 0.15, 0.2),
        emissive=True, emissive_color=(0.0, 0.8, 1.0)
    ),
    PropType.CAR: PropDefinition(
        PropType.CAR, 4.5, 2.0, 1.5,
        color=(0.2, 0.2, 0.25),
        physics_mass=1500
    ),
    PropType.TRUCK: PropDefinition(
        PropType.TRUCK, 8.0, 2.5, 3.0,
        color=(0.25, 0.22, 0.2),
        physics_mass=5000
    ),
    PropType.TREE: PropDefinition(
        PropType.TREE, 3.0, 3.0, 8.0,
        color=(0.1, 0.3, 0.15)
    ),
    PropType.CONTAINER: PropDefinition(
        PropType.CONTAINER, 6.0, 2.4, 2.6,
        color=(0.5, 0.2, 0.1),
        physics_mass=3000
    ),
    PropType.BARREL: PropDefinition(
        PropType.BARREL, 0.6, 0.6, 1.0,
        color=(0.3, 0.15, 0.1),
        physics_mass=50, breakable=True
    ),
    PropType.CRATE: PropDefinition(
        PropType.CRATE, 1.0, 1.0, 1.0,
        color=(0.35, 0.3, 0.2),
        physics_mass=30, breakable=True
    ),
    PropType.HOLOGRAM_DISPLAY: PropDefinition(
        PropType.HOLOGRAM_DISPLAY, 3.0, 0.3, 4.0,
        color=(0.1, 0.1, 0.15),
        emissive=True, emissive_color=(0.0, 0.5, 1.0)
    ),
    PropType.NEON_SIGN: PropDefinition(
        PropType.NEON_SIGN, 2.0, 0.2, 1.0,
        color=(0.8, 0.0, 0.4),
        emissive=True, emissive_color=(1.0, 0.0, 0.5)
    ),
    PropType.ANTENNA: PropDefinition(
        PropType.ANTENNA, 0.5, 0.5, 10.0,
        color=(0.3, 0.3, 0.32),
        emissive=True, emissive_color=(1.0, 0.0, 0.0)
    ),
    PropType.ROBOT_CHARGING_STATION: PropDefinition(
        PropType.ROBOT_CHARGING_STATION, 2.0, 1.5, 2.5,
        color=(0.2, 0.2, 0.25),
        emissive=True, emissive_color=(0.0, 1.0, 0.5)
    ),
}


@dataclass
class Prop:
    """A placed prop instance."""
    prop_type: PropType
    position: Tuple[float, float, float]
    rotation: float = 0.0
    scale: float = 1.0
    color_variation: Tuple[float, float, float] = (0, 0, 0)
    
    # Runtime state
    physics_body: Any = None
    node_path: Any = None


class PropGenerator:
    """
    Generates and manages props in the world.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize prop generator.
        
        Args:
            seed: Random seed
        """
        self.seed = seed
        self.rng = random.Random(seed)
        
        # All placed props
        self._props: List[Prop] = []
        
        logger.info(f"PropGenerator initialized (seed={seed})")
    
    def generate_prop(self, prop_type: PropType,
                     x: float, y: float, z: float = 0.0,
                     rotation: Optional[float] = None,
                     scale: Optional[float] = None) -> Prop:
        """
        Generate a single prop.
        
        Args:
            prop_type: Type of prop to create
            x, y, z: Position
            rotation: Optional rotation in degrees
            scale: Optional scale factor
        
        Returns:
            Prop instance
        """
        if rotation is None:
            rotation = self.rng.uniform(0, 360)
        if scale is None:
            scale = self.rng.uniform(0.9, 1.1)
        
        # Color variation
        variation = (
            self.rng.uniform(-0.05, 0.05),
            self.rng.uniform(-0.05, 0.05),
            self.rng.uniform(-0.05, 0.05)
        )
        
        prop = Prop(
            prop_type=prop_type,
            position=(x, y, z),
            rotation=rotation,
            scale=scale,
            color_variation=variation
        )
        
        self._props.append(prop)
        return prop
    
    def generate_street_props(self, road_points: List[Tuple[float, float]],
                             spacing: float = 10.0) -> List[Prop]:
        """
        Generate street props along a road.
        
        Args:
            road_points: List of (x, y) road centerline points
            spacing: Distance between props
        
        Returns:
            List of generated props
        """
        props = []
        
        # Street lights along road
        accumulated_dist = 0
        for i in range(len(road_points) - 1):
            p1 = road_points[i]
            p2 = road_points[i + 1]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            segment_len = math.sqrt(dx*dx + dy*dy)
            
            if segment_len == 0:
                continue
            
            # Direction perpendicular to road
            perp_x = -dy / segment_len
            perp_y = dx / segment_len
            
            while accumulated_dist < segment_len:
                t = accumulated_dist / segment_len
                x = p1[0] + dx * t
                y = p1[1] + dy * t
                
                # Street light on both sides
                offset = 4.0  # Distance from road center
                
                props.append(self.generate_prop(
                    PropType.STREET_LIGHT,
                    x + perp_x * offset,
                    y + perp_y * offset
                ))
                
                props.append(self.generate_prop(
                    PropType.STREET_LIGHT,
                    x - perp_x * offset,
                    y - perp_y * offset
                ))
                
                accumulated_dist += spacing
            
            accumulated_dist -= segment_len
        
        return props
    
    def generate_parking_lot(self, center_x: float, center_y: float,
                            width: float, depth: float,
                            fill_ratio: float = 0.7) -> List[Prop]:
        """
        Generate a parking lot with vehicles.
        
        Args:
            center_x, center_y: Center position
            width, depth: Size of parking lot
            fill_ratio: How full the lot is (0-1)
        
        Returns:
            List of vehicle props
        """
        props = []
        
        car_width = 2.5
        car_depth = 5.5
        
        rows = int(depth / car_depth)
        cols = int(width / (car_width + 0.5))
        
        for row in range(rows):
            for col in range(cols):
                if self.rng.random() > fill_ratio:
                    continue
                
                x = center_x - width/2 + col * (car_width + 0.5) + car_width/2
                y = center_y - depth/2 + row * car_depth + car_depth/2
                
                # Random vehicle type
                vehicle_type = self.rng.choice([
                    PropType.CAR, PropType.CAR, PropType.CAR,
                    PropType.TRUCK, PropType.MOTORCYCLE
                ])
                
                props.append(self.generate_prop(
                    vehicle_type, x, y,
                    rotation=self.rng.choice([0, 180])  # Facing either direction
                ))
        
        return props
    
    def generate_industrial_area(self, center_x: float, center_y: float,
                                size: float) -> List[Prop]:
        """
        Generate industrial props (containers, barrels, etc.)
        
        Args:
            center_x, center_y: Center position
            size: Area size
        
        Returns:
            List of props
        """
        props = []
        
        # Containers in stacks
        num_stacks = int(size / 15)
        for _ in range(num_stacks):
            stack_x = center_x + self.rng.uniform(-size/2, size/2)
            stack_y = center_y + self.rng.uniform(-size/2, size/2)
            
            # Stack 1-3 containers
            height = 0
            for level in range(self.rng.randint(1, 3)):
                props.append(self.generate_prop(
                    PropType.CONTAINER,
                    stack_x, stack_y, height,
                    rotation=self.rng.choice([0, 90])
                ))
                height += 2.6
        
        # Scattered barrels and crates
        num_scattered = int(size / 5)
        for _ in range(num_scattered):
            x = center_x + self.rng.uniform(-size/2, size/2)
            y = center_y + self.rng.uniform(-size/2, size/2)
            
            prop_type = self.rng.choice([PropType.BARREL, PropType.CRATE])
            props.append(self.generate_prop(prop_type, x, y))
        
        return props
    
    def generate_cyberpunk_decorations(self, center_x: float, center_y: float,
                                      size: float) -> List[Prop]:
        """
        Generate cyberpunk-specific decorations (holograms, neon, etc.)
        
        Args:
            center_x, center_y: Center position
            size: Area size
        
        Returns:
            List of props
        """
        props = []
        
        # Hologram displays
        num_holograms = int(size / 30)
        for _ in range(num_holograms):
            x = center_x + self.rng.uniform(-size/2, size/2)
            y = center_y + self.rng.uniform(-size/2, size/2)
            z = self.rng.uniform(3, 8)  # Elevated
            
            props.append(self.generate_prop(
                PropType.HOLOGRAM_DISPLAY, x, y, z
            ))
        
        # Neon signs
        num_neon = int(size / 20)
        for _ in range(num_neon):
            x = center_x + self.rng.uniform(-size/2, size/2)
            y = center_y + self.rng.uniform(-size/2, size/2)
            z = self.rng.uniform(2, 5)
            
            props.append(self.generate_prop(
                PropType.NEON_SIGN, x, y, z
            ))
        
        # Robot charging stations
        num_stations = int(size / 50)
        for _ in range(num_stations):
            x = center_x + self.rng.uniform(-size/2, size/2)
            y = center_y + self.rng.uniform(-size/2, size/2)
            
            props.append(self.generate_prop(
                PropType.ROBOT_CHARGING_STATION, x, y
            ))
        
        return props
    
    def get_props_in_area(self, center_x: float, center_y: float,
                         radius: float) -> List[Prop]:
        """Get all props within a radius."""
        result = []
        for prop in self._props:
            dx = prop.position[0] - center_x
            dy = prop.position[1] - center_y
            if dx*dx + dy*dy <= radius*radius:
                result.append(prop)
        return result
    
    def remove_prop(self, prop: Prop):
        """Remove a prop from the world."""
        if prop in self._props:
            self._props.remove(prop)
            if prop.node_path:
                prop.node_path.removeNode()
    
    @property
    def all_props(self) -> List[Prop]:
        """Get all props."""
        return self._props.copy()


def create_prop_geometry(prop: Prop) -> Dict[str, Any]:
    """
    Create geometry data for a prop.
    
    Returns:
        Dictionary with vertices, normals, colors, indices
    """
    definition = PROP_DEFINITIONS.get(prop.prop_type)
    if not definition:
        definition = PropDefinition(
            prop.prop_type, 1, 1, 1, (0.5, 0.5, 0.5)
        )
    
    w = definition.width / 2 * prop.scale
    d = definition.depth / 2 * prop.scale
    h = definition.height * prop.scale
    
    # Apply color variation
    color = (
        max(0, min(1, definition.color[0] + prop.color_variation[0])),
        max(0, min(1, definition.color[1] + prop.color_variation[1])),
        max(0, min(1, definition.color[2] + prop.color_variation[2])),
        1.0
    )
    
    # Simple box geometry
    vertices = []
    normals = []
    colors = []
    
    # 6 faces
    faces = [
        ([(-w, -d, 0), (w, -d, 0), (w, -d, h), (-w, -d, h)], (0, -1, 0)),
        ([( w, -d, 0), (w,  d, 0), (w,  d, h), ( w, -d, h)], (1,  0, 0)),
        ([( w,  d, 0), (-w, d, 0), (-w, d, h), ( w,  d, h)], (0,  1, 0)),
        ([(-w,  d, 0), (-w,-d, 0), (-w,-d, h), (-w,  d, h)], (-1, 0, 0)),
        ([(-w, -d, h), ( w,-d, h), ( w, d, h), (-w,  d, h)], (0,  0, 1)),
        ([(-w,  d, 0), ( w, d, 0), ( w,-d, 0), (-w, -d, 0)], (0,  0,-1)),
    ]
    
    for verts, norm in faces:
        for v in verts:
            vertices.append(v)
            normals.append(norm)
            colors.append(color)
    
    indices = []
    for i in range(6):
        base = i * 4
        indices.extend([base, base+1, base+2, base, base+2, base+3])
    
    return {
        'vertices': vertices,
        'normals': normals,
        'colors': colors,
        'indices': indices,
        'emissive': definition.emissive,
        'emissive_color': definition.emissive_color if definition.emissive else None
    }
