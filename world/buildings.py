"""
SynthWorld Building Generator

Procedural cyberpunk building generation using L-systems and parametric rules.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import math
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class BuildingStyle:
    """Defines a building style."""
    name: str
    min_floors: int = 1
    max_floors: int = 10
    min_width: float = 8.0
    max_width: float = 30.0
    min_depth: float = 8.0
    max_depth: float = 30.0
    floor_height: float = 3.0
    window_density: float = 0.7  # 0-1
    has_neon: bool = False
    has_hologram: bool = False
    has_antenna: bool = False
    base_color: Tuple[float, float, float] = (0.3, 0.3, 0.35)
    accent_color: Tuple[float, float, float] = (0.0, 0.8, 1.0)  # Cyan neon


# Predefined cyberpunk building styles
BUILDING_STYLES = {
    'residential': BuildingStyle(
        name='residential',
        min_floors=5, max_floors=20,
        min_width=10, max_width=25,
        min_depth=10, max_depth=25,
        window_density=0.8,
        has_neon=True,
        base_color=(0.25, 0.25, 0.28),
        accent_color=(1.0, 0.3, 0.5)  # Pink
    ),
    'corporate': BuildingStyle(
        name='corporate',
        min_floors=20, max_floors=60,
        min_width=15, max_width=40,
        min_depth=15, max_depth=40,
        floor_height=4.0,
        window_density=0.9,
        has_neon=True,
        has_hologram=True,
        base_color=(0.15, 0.15, 0.2),
        accent_color=(0.0, 0.8, 1.0)  # Cyan
    ),
    'industrial': BuildingStyle(
        name='industrial',
        min_floors=2, max_floors=6,
        min_width=20, max_width=50,
        min_depth=20, max_depth=50,
        floor_height=5.0,
        window_density=0.3,
        has_antenna=True,
        base_color=(0.2, 0.18, 0.15),
        accent_color=(1.0, 0.5, 0.0)  # Orange warning
    ),
    'entertainment': BuildingStyle(
        name='entertainment',
        min_floors=3, max_floors=15,
        min_width=12, max_width=35,
        min_depth=12, max_depth=35,
        window_density=0.5,
        has_neon=True,
        has_hologram=True,
        base_color=(0.1, 0.1, 0.15),
        accent_color=(0.8, 0.0, 1.0)  # Purple
    ),
    'slum': BuildingStyle(
        name='slum',
        min_floors=2, max_floors=8,
        min_width=5, max_width=15,
        min_depth=5, max_depth=15,
        floor_height=2.5,
        window_density=0.4,
        base_color=(0.18, 0.15, 0.12),
        accent_color=(0.7, 0.6, 0.3)  # Dim yellow
    ),
}


@dataclass
class Building:
    """Represents a generated building."""
    position: Tuple[float, float, float]
    width: float
    depth: float
    height: float
    floors: int
    style: BuildingStyle
    rotation: float = 0.0
    
    # Generated geometry data
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)
    colors: List[Tuple[float, float, float, float]] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    
    # Detail elements
    windows: List[Dict[str, Any]] = field(default_factory=list)
    neon_signs: List[Dict[str, Any]] = field(default_factory=list)


class BuildingGenerator:
    """
    Generates procedural cyberpunk buildings.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize building generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
        logger.info(f"BuildingGenerator initialized (seed={seed})")
    
    def generate_building(self, x: float, y: float, 
                         style_name: str = 'corporate',
                         z: float = 0.0) -> Building:
        """
        Generate a single building.
        
        Args:
            x: X position
            y: Y position
            style_name: Building style name
            z: Base Z position (ground level)
        
        Returns:
            Building object with generated geometry
        """
        style = BUILDING_STYLES.get(style_name, BUILDING_STYLES['corporate'])
        
        # Randomize dimensions
        width = self.rng.uniform(style.min_width, style.max_width)
        depth = self.rng.uniform(style.min_depth, style.max_depth)
        floors = self.rng.randint(style.min_floors, style.max_floors)
        height = floors * style.floor_height
        rotation = self.rng.uniform(0, 360)
        
        building = Building(
            position=(x, y, z),
            width=width,
            depth=depth,
            height=height,
            floors=floors,
            style=style,
            rotation=rotation
        )
        
        # Generate geometry
        self._generate_geometry(building)
        
        # Generate windows
        self._generate_windows(building)
        
        # Generate neon signs if applicable
        if style.has_neon:
            self._generate_neon_signs(building)
        
        return building
    
    def _generate_geometry(self, building: Building):
        """Generate building geometry."""
        w, d, h = building.width / 2, building.depth / 2, building.height
        base_color = building.style.base_color
        
        vertices = []
        normals = []
        colors = []
        indices = []
        
        # Generate main building body with setbacks for taller buildings
        setback_floors = [building.floors]
        
        if building.floors > 10:
            # Add setbacks for tall buildings
            setback_floors = [
                building.floors // 3,
                2 * building.floors // 3,
                building.floors
            ]
        
        current_base = 0
        current_w, current_d = w, d
        vertex_offset = 0
        
        for floor_count in setback_floors:
            section_height = floor_count * building.style.floor_height - current_base
            
            # Generate box for this section
            section_verts, section_norms, section_colors, section_indices = self._generate_box(
                current_w, current_d, section_height, current_base, base_color
            )
            
            # Offset indices
            for idx in section_indices:
                indices.append(idx + vertex_offset)
            
            vertices.extend(section_verts)
            normals.extend(section_norms)
            colors.extend(section_colors)
            
            vertex_offset += len(section_verts)
            current_base = floor_count * building.style.floor_height
            
            # Reduce size for next setback
            current_w *= 0.85
            current_d *= 0.85
        
        building.vertices = vertices
        building.normals = normals
        building.colors = colors
        building.indices = indices
    
    def _generate_box(self, half_width: float, half_depth: float, 
                      height: float, base_z: float,
                      color: Tuple[float, float, float]) -> Tuple[List, List, List, List]:
        """Generate a box geometry section."""
        w, d = half_width, half_depth
        
        vertices = []
        normals = []
        colors = []
        
        # Add slight color variation
        def vary_color(c):
            variation = self.rng.uniform(-0.05, 0.05)
            return (
                max(0, min(1, c[0] + variation)),
                max(0, min(1, c[1] + variation)),
                max(0, min(1, c[2] + variation)),
                1.0
            )
        
        # Front face (positive Y)
        face_verts = [(-w, d, base_z), (w, d, base_z), (w, d, base_z + height), (-w, d, base_z + height)]
        face_normal = (0, 1, 0)
        for v in face_verts:
            vertices.append(v)
            normals.append(face_normal)
            colors.append(vary_color(color))
        
        # Back face (negative Y)
        face_verts = [(w, -d, base_z), (-w, -d, base_z), (-w, -d, base_z + height), (w, -d, base_z + height)]
        face_normal = (0, -1, 0)
        for v in face_verts:
            vertices.append(v)
            normals.append(face_normal)
            colors.append(vary_color(color))
        
        # Right face (positive X)
        face_verts = [(w, d, base_z), (w, -d, base_z), (w, -d, base_z + height), (w, d, base_z + height)]
        face_normal = (1, 0, 0)
        for v in face_verts:
            vertices.append(v)
            normals.append(face_normal)
            colors.append(vary_color(color))
        
        # Left face (negative X)
        face_verts = [(-w, -d, base_z), (-w, d, base_z), (-w, d, base_z + height), (-w, -d, base_z + height)]
        face_normal = (-1, 0, 0)
        for v in face_verts:
            vertices.append(v)
            normals.append(face_normal)
            colors.append(vary_color(color))
        
        # Top face
        face_verts = [(-w, d, base_z + height), (w, d, base_z + height), (w, -d, base_z + height), (-w, -d, base_z + height)]
        face_normal = (0, 0, 1)
        for v in face_verts:
            vertices.append(v)
            normals.append(face_normal)
            colors.append(vary_color(color))
        
        # Generate indices (two triangles per face)
        indices = []
        for face in range(5):  # 5 visible faces (no bottom)
            base = face * 4
            indices.extend([base, base+1, base+2, base, base+2, base+3])
        
        return vertices, normals, colors, indices
    
    def _generate_windows(self, building: Building):
        """Generate window positions."""
        style = building.style
        floor_height = style.floor_height
        window_width = 1.5
        window_height = 2.0
        window_spacing = 3.0
        
        for floor in range(building.floors):
            floor_z = floor * floor_height + 1.0  # 1m above floor
            
            # Windows on each side
            for side in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if side[0] != 0:  # X-facing walls
                    wall_length = building.depth
                    wall_x = side[0] * building.width / 2
                    
                    num_windows = int(wall_length / window_spacing)
                    for i in range(num_windows):
                        if self.rng.random() < style.window_density:
                            window_y = -wall_length/2 + (i + 0.5) * window_spacing
                            
                            # Window light color (varies for cyberpunk effect)
                            is_lit = self.rng.random() < 0.7
                            if is_lit:
                                light_colors = [
                                    (1.0, 0.9, 0.7),   # Warm white
                                    (0.7, 0.9, 1.0),   # Cool white
                                    (0.0, 0.8, 1.0),   # Cyan
                                    (1.0, 0.3, 0.5),   # Pink
                                    (0.5, 0.0, 1.0),   # Purple
                                ]
                                light_color = self.rng.choice(light_colors)
                            else:
                                light_color = (0.05, 0.05, 0.08)  # Dark
                            
                            building.windows.append({
                                'position': (wall_x, window_y, floor_z),
                                'size': (0.1, window_width, window_height),
                                'color': light_color,
                                'normal': (side[0], 0, 0)
                            })
                else:  # Y-facing walls
                    wall_length = building.width
                    wall_y = side[1] * building.depth / 2
                    
                    num_windows = int(wall_length / window_spacing)
                    for i in range(num_windows):
                        if self.rng.random() < style.window_density:
                            window_x = -wall_length/2 + (i + 0.5) * window_spacing
                            
                            is_lit = self.rng.random() < 0.7
                            if is_lit:
                                light_colors = [
                                    (1.0, 0.9, 0.7),
                                    (0.7, 0.9, 1.0),
                                    (0.0, 0.8, 1.0),
                                    (1.0, 0.3, 0.5),
                                    (0.5, 0.0, 1.0),
                                ]
                                light_color = self.rng.choice(light_colors)
                            else:
                                light_color = (0.05, 0.05, 0.08)
                            
                            building.windows.append({
                                'position': (window_x, wall_y, floor_z),
                                'size': (window_width, 0.1, window_height),
                                'color': light_color,
                                'normal': (0, side[1], 0)
                            })
    
    def _generate_neon_signs(self, building: Building):
        """Generate neon sign positions."""
        style = building.style
        
        # Add 1-3 neon signs per building
        num_signs = self.rng.randint(1, 3)
        
        neon_colors = [
            style.accent_color,
            (0.0, 1.0, 0.5),   # Green
            (1.0, 0.0, 0.3),   # Red
            (1.0, 0.5, 0.0),   # Orange
            (0.0, 0.5, 1.0),   # Blue
        ]
        
        for _ in range(num_signs):
            # Random position on building face
            side = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            
            sign_z = self.rng.uniform(building.height * 0.3, building.height * 0.8)
            
            if side[0] != 0:
                sign_x = side[0] * (building.width / 2 + 0.2)
                sign_y = self.rng.uniform(-building.depth/3, building.depth/3)
            else:
                sign_x = self.rng.uniform(-building.width/3, building.width/3)
                sign_y = side[1] * (building.depth / 2 + 0.2)
            
            sign_width = self.rng.uniform(3, 8)
            sign_height = self.rng.uniform(1, 3)
            
            building.neon_signs.append({
                'position': (sign_x, sign_y, sign_z),
                'size': (sign_width, sign_height),
                'color': self.rng.choice(neon_colors),
                'normal': (side[0], side[1], 0),
                'animated': self.rng.random() < 0.3  # 30% chance of animation
            })
    
    def generate_city_block(self, center_x: float, center_y: float,
                           block_size: float = 100.0,
                           building_density: float = 0.7) -> List[Building]:
        """
        Generate a city block with multiple buildings.
        
        Args:
            center_x: Block center X
            center_y: Block center Y
            block_size: Size of the block
            building_density: 0-1, how densely packed
        
        Returns:
            List of Building objects
        """
        buildings = []
        
        # Grid of potential building sites
        grid_size = 5  # 5x5 grid
        cell_size = block_size / grid_size
        
        # Leave roads around edges
        inner_size = block_size * 0.85
        offset = (block_size - inner_size) / 2
        
        for gx in range(grid_size):
            for gy in range(grid_size):
                if self.rng.random() > building_density:
                    continue
                
                # Position within cell (with some randomness)
                cell_x = center_x - block_size/2 + offset + gx * (inner_size / grid_size)
                cell_y = center_y - block_size/2 + offset + gy * (inner_size / grid_size)
                
                cell_x += self.rng.uniform(-cell_size * 0.1, cell_size * 0.1)
                cell_y += self.rng.uniform(-cell_size * 0.1, cell_size * 0.1)
                
                # Choose style based on position (zones)
                dist_from_center = math.sqrt(center_x**2 + center_y**2)
                
                if dist_from_center < 100:
                    style = self.rng.choice(['corporate', 'entertainment'])
                elif dist_from_center < 300:
                    style = self.rng.choice(['residential', 'corporate', 'entertainment'])
                elif dist_from_center < 500:
                    style = self.rng.choice(['residential', 'industrial'])
                else:
                    style = self.rng.choice(['slum', 'industrial', 'residential'])
                
                building = self.generate_building(cell_x, cell_y, style)
                buildings.append(building)
        
        return buildings


def create_building_node_path(building: Building, render_node) -> Any:
    """
    Create a Panda3D NodePath from a Building.
    
    Args:
        building: Building object
        render_node: Panda3D render node to attach to
    
    Returns:
        NodePath for the building
    """
    from panda3d.core import (
        GeomVertexData, GeomVertexFormat, GeomVertexWriter,
        Geom, GeomTriangles, GeomNode, NodePath
    )
    
    format = GeomVertexFormat.get_v3n3c4()
    vdata = GeomVertexData('building', format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    color = GeomVertexWriter(vdata, 'color')
    
    for v in building.vertices:
        vertex.addData3(*v)
    for n in building.normals:
        normal.addData3(*n)
    for c in building.colors:
        color.addData4(*c)
    
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(0, len(building.indices), 3):
        tris.addVertices(building.indices[i], building.indices[i+1], building.indices[i+2])
    
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    
    node = GeomNode(f'building_{id(building)}')
    node.addGeom(geom)
    
    node_path = render_node.attachNewNode(node)
    node_path.setPos(*building.position)
    node_path.setH(building.rotation)
    
    return node_path
