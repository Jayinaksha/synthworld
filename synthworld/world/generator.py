"""
SynthWorld World Generator

Main world generation coordinator that combines terrain, buildings, and props.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

from .terrain import TerrainGenerator, ChunkedTerrain
from .buildings import BuildingGenerator, Building, create_building_node_path
from .props import PropGenerator, Prop, PropType

logger = logging.getLogger(__name__)


@dataclass
class WorldChunk:
    """A chunk of the world."""
    chunk_x: int
    chunk_y: int
    terrain_node: Any = None
    buildings: List[Building] = None
    props: List[Prop] = None
    loaded: bool = False


class WorldGenerator:
    """
    Main world generation coordinator.
    Generates and manages the complete game world.
    """
    
    def __init__(self, seed: int = 42, chunk_size: int = 100):
        """
        Initialize world generator.
        
        Args:
            seed: Master random seed
            chunk_size: Size of world chunks
        """
        self.seed = seed
        self.chunk_size = chunk_size
        
        # Sub-generators
        self.terrain_gen = TerrainGenerator(seed=seed)
        self.building_gen = BuildingGenerator(seed=seed)
        self.prop_gen = PropGenerator(seed=seed)
        
        # Chunked terrain manager
        self.chunked_terrain = ChunkedTerrain(
            self.terrain_gen,
            chunk_size=chunk_size,
            height_scale=1.0
        )
        
        # Loaded chunks
        self._chunks: Dict[Tuple[int, int], WorldChunk] = {}
        
        # Reference to renderer (set later)
        self._renderer = None
        self._physics = None
        
        logger.info(f"WorldGenerator initialized (seed={seed}, chunk_size={chunk_size})")
    
    def set_renderer(self, renderer):
        """Set the renderer reference for creating visual nodes."""
        self._renderer = renderer
    
    def set_physics(self, physics):
        """Set the physics world reference."""
        self._physics = physics
    
    def generate_chunk(self, chunk_x: int, chunk_y: int) -> WorldChunk:
        """
        Generate a world chunk.
        
        Args:
            chunk_x: Chunk X coordinate
            chunk_y: Chunk Y coordinate
        
        Returns:
            WorldChunk object
        """
        key = (chunk_x, chunk_y)
        
        if key in self._chunks:
            return self._chunks[key]
        
        chunk = WorldChunk(chunk_x=chunk_x, chunk_y=chunk_y)
        
        # World position of chunk center
        center_x = chunk_x * self.chunk_size + self.chunk_size / 2
        center_y = chunk_y * self.chunk_size + self.chunk_size / 2
        
        # Generate terrain
        heightmap, biome_map = self.chunked_terrain.get_chunk(chunk_x, chunk_y)
        
        # Generate buildings
        chunk.buildings = self._generate_chunk_buildings(center_x, center_y, biome_map)
        
        # Generate props
        chunk.props = self._generate_chunk_props(center_x, center_y, biome_map)
        
        self._chunks[key] = chunk
        
        logger.debug(f"Generated chunk ({chunk_x}, {chunk_y}): "
                    f"{len(chunk.buildings)} buildings, {len(chunk.props)} props")
        
        return chunk
    
    def _generate_chunk_buildings(self, center_x: float, center_y: float,
                                  biome_map) -> List[Building]:
        """Generate buildings for a chunk based on biomes."""
        buildings = []
        
        # Check if this chunk is mostly urban
        urban_ratio = (biome_map == TerrainGenerator.BIOME_URBAN).sum() / biome_map.size
        
        if urban_ratio > 0.3:
            # Generate city block with reduced density for performance
            buildings = self.building_gen.generate_city_block(
                center_x, center_y,
                block_size=self.chunk_size * 0.9,
                building_density=0.15 + urban_ratio * 0.2  # Reduced from 0.4 + 0.4
            )
        
        return buildings
    
    def _generate_chunk_props(self, center_x: float, center_y: float,
                              biome_map) -> List[Prop]:
        """Generate props for a chunk based on biomes."""
        props = []
        
        # Analyze biomes in this chunk
        industrial_ratio = (biome_map == TerrainGenerator.BIOME_INDUSTRIAL).sum() / biome_map.size
        urban_ratio = (biome_map == TerrainGenerator.BIOME_URBAN).sum() / biome_map.size
        
        # Industrial props
        if industrial_ratio > 0.2:
            props.extend(self.prop_gen.generate_industrial_area(
                center_x, center_y, self.chunk_size * industrial_ratio
            ))
        
        # Urban props (street furniture, vehicles)
        if urban_ratio > 0.3:
            # Parking lots
            num_lots = int(urban_ratio * 3)
            for _ in range(num_lots):
                lot_x = center_x + self.prop_gen.rng.uniform(-self.chunk_size/3, self.chunk_size/3)
                lot_y = center_y + self.prop_gen.rng.uniform(-self.chunk_size/3, self.chunk_size/3)
                props.extend(self.prop_gen.generate_parking_lot(
                    lot_x, lot_y, 20, 30, fill_ratio=0.2  # Reduced from 0.5
                ))
        
        # Cyberpunk decorations
        props.extend(self.prop_gen.generate_cyberpunk_decorations(
            center_x, center_y, self.chunk_size
        ))
        
        return props
    
    def load_chunk(self, chunk_x: int, chunk_y: int):
        """
        Load a chunk into the scene (create visual nodes).
        
        Args:
            chunk_x: Chunk X coordinate
            chunk_y: Chunk Y coordinate
        """
        chunk = self.generate_chunk(chunk_x, chunk_y)
        
        if chunk.loaded or not self._renderer:
            return
        
        # Create terrain node
        heightmap, biome_map = self.chunked_terrain.get_chunk(chunk_x, chunk_y)
        
        # Use proper biome coloring
        color_func = self.chunked_terrain.generator.create_color_func(biome_map)
        
        chunk.terrain_node = self._renderer.create_terrain(
            heightmap,
            scale=1.0,
            height_scale=1.0,
            color_func=color_func,
            position=(
                chunk_x * self.chunk_size,
                chunk_y * self.chunk_size,
                0
            ),
            parent=self._renderer.terrain_node
        )
        
        # Create building nodes and physics
        from .buildings import create_building_node_path
        
        # Create a root for this chunk's buildings to allow flattening
        chunk_buildings_root = self._renderer.buildings_node.attachNewNode(f"chunk_{chunk_x}_{chunk_y}_buildings")
        
        for building in chunk.buildings:
            # 1. Visual Node
            if self._renderer:
                node = create_building_node_path(building, chunk_buildings_root)
                building.node_path = node
            
            # 2. Physics Body
            if self._physics:
                # Create static box for building
                from .physics_utils import create_building_collider
                create_building_collider(self._physics, building)
        
        # Flatten buildings for this chunk to reduce draw calls
        if self._renderer:
            chunk_buildings_root.flattenStrong()

        # Create prop nodes and physics
        from .props import create_prop_geometry
        from .props import PropType
        
        # Batch props by type to optimize creation
        props_by_type = {}
        for prop in chunk.props:
            if prop.prop_type not in props_by_type:
                props_by_type[prop.prop_type] = []
            props_by_type[prop.prop_type].append(prop)
            
        chunk_props_root = self._renderer.props_node.attachNewNode(f"chunk_{chunk_x}_{chunk_y}_props")
        
        for prop_type, props in props_by_type.items():
            # Create one master set of geometry for this type
            # Note: This is a simplified optimization. Ideally we'd valid instancing,
            # but flattenStrong() works well for static geometry too.
            if self._renderer and props:
                # Create template geometry
                template_geom = create_prop_geometry(props[0]) # Use first prop's style
                
                for prop in props:
                    # 1. Visual Node
                    # Create unique geom data per prop to allow color variation
                    # effectively, we are just creating them then flattening all at once
                    geom_data = create_prop_geometry(prop)
                    node = self._create_prop_node(geom_data, f"prop_{id(prop)}")
                    node.reparentTo(chunk_props_root)
                    node.setPos(*prop.position)
                    node.setH(prop.rotation)
                    node.setScale(prop.scale)
                    prop.node_path = node
                    
                    # Skip physics for props to improve performance
                    # if self._physics:
                    #     from .physics_utils import create_prop_collider
                    #     create_prop_collider(self._physics, prop)
        
        # Flatten props for this chunk
        if self._renderer:
            chunk_props_root.flattenStrong()
        
        chunk.loaded = True
        logger.debug(f"Loaded chunk ({chunk_x}, {chunk_y})")
    
    
    def _create_prop_node(self, geom_data: Dict[str, Any], name: str) -> Any:
        """Create a Panda3D node for a prop."""
        from panda3d.core import (
            GeomVertexData, GeomVertexFormat, GeomVertexWriter,
            Geom, GeomTriangles, GeomNode, NodePath
        )
        
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData(name, format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        for v in geom_data['vertices']:
            vertex.addData3(*v)
        for n in geom_data['normals']:
            normal.addData3(*n)
        for c in geom_data['colors']:
            color.addData4(*c)
        
        tris = GeomTriangles(Geom.UHStatic)
        indices = geom_data['indices']
        for i in range(0, len(indices), 3):
            tris.addVertices(indices[i], indices[i+1], indices[i+2])
        
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        node = GeomNode(name)
        node.addGeom(geom)
        
        return NodePath(node)

    def unload_chunk(self, chunk_x: int, chunk_y: int):
        """Unload a chunk from the scene."""
        key = (chunk_x, chunk_y)
        
        if key not in self._chunks:
            return
        
        chunk = self._chunks[key]
        
        # Remove nodes
        if chunk.terrain_node:
            chunk.terrain_node.removeNode()
            chunk.terrain_node = None
        
        for building in chunk.buildings or []:
            if hasattr(building, 'node_path') and building.node_path:
                building.node_path.removeNode()
        
        chunk.loaded = False
        
        # Optionally remove from cache
        # del self._chunks[key]
        
        logger.debug(f"Unloaded chunk ({chunk_x}, {chunk_y})")
    
    def update_loaded_chunks(self, player_x: float, player_y: float,
                            render_distance: int = 3):
        """
        Update which chunks are loaded based on player position.
        
        Args:
            player_x: Player X position
            player_y: Player Y position
            render_distance: Number of chunks to load around player
        """
        current_chunk_x = int(player_x // self.chunk_size)
        current_chunk_y = int(player_y // self.chunk_size)
        
        # Determine which chunks should be loaded
        should_be_loaded = set()
        for dx in range(-render_distance, render_distance + 1):
            for dy in range(-render_distance, render_distance + 1):
                should_be_loaded.add((current_chunk_x + dx, current_chunk_y + dy))
        
        # Load new chunks
        for chunk_key in should_be_loaded:
            if chunk_key not in self._chunks or not self._chunks[chunk_key].loaded:
                self.load_chunk(chunk_key[0], chunk_key[1])
        
        # Unload distant chunks
        for chunk_key in list(self._chunks.keys()):
            if chunk_key not in should_be_loaded and self._chunks[chunk_key].loaded:
                self.unload_chunk(chunk_key[0], chunk_key[1])
    
    def get_height_at(self, x: float, y: float) -> float:
        """Get terrain height at world position."""
        return self.chunked_terrain.get_height(x, y)
    
    def get_buildings_near(self, x: float, y: float, radius: float) -> List[Building]:
        """Get buildings near a position."""
        buildings = []
        
        # Check relevant chunks
        min_chunk_x = int((x - radius) // self.chunk_size)
        max_chunk_x = int((x + radius) // self.chunk_size)
        min_chunk_y = int((y - radius) // self.chunk_size)
        max_chunk_y = int((y + radius) // self.chunk_size)
        
        for cx in range(min_chunk_x, max_chunk_x + 1):
            for cy in range(min_chunk_y, max_chunk_y + 1):
                key = (cx, cy)
                if key in self._chunks:
                    for building in self._chunks[key].buildings or []:
                        dx = building.position[0] - x
                        dy = building.position[1] - y
                        if dx*dx + dy*dy <= radius*radius:
                            buildings.append(building)
        
        return buildings
    
    def generate_initial_world(self, center_x: float = 0, center_y: float = 0,
                              radius: int = 2):
        """
        Generate the initial world around a center point.
        
        Args:
            center_x: Center X position
            center_y: Center Y position  
            radius: Number of chunks to generate
        """
        center_chunk_x = int(center_x // self.chunk_size)
        center_chunk_y = int(center_y // self.chunk_size)
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                self.generate_chunk(center_chunk_x + dx, center_chunk_y + dy)
        
        logger.info(f"Generated initial world: {(2*radius+1)**2} chunks")
    
    def get_spawn_position(self) -> Tuple[float, float, float]:
        """Get a suitable spawn position for the player."""
        # Find a relatively flat urban area
        chunk = self.generate_chunk(0, 0)
        
        # Default spawn at center of first chunk
        x = self.chunk_size / 2
        y = self.chunk_size / 2
        z = self.get_height_at(x, y) + 2.0  # Slightly above ground
        
        return (x, y, z)
