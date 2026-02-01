"""
SynthWorld Terrain Generator

Procedural terrain generation using Perlin noise.
"""

import numpy as np
from typing import Tuple, Optional, Callable
import math
import logging

logger = logging.getLogger(__name__)

# Try to use the noise library, fall back to simple implementation
try:
    from noise import pnoise2, snoise2
    HAS_NOISE_LIB = True
except ImportError:
    HAS_NOISE_LIB = False
    logger.warning("noise library not found, using fallback Perlin noise")


def _fallback_perlin(x: float, y: float, octaves: int = 1, 
                     persistence: float = 0.5, lacunarity: float = 2.0) -> float:
    """Simple Perlin noise fallback implementation."""
    # Simple hash-based pseudo-random for reproducibility
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(a, b, t):
        return a + t * (b - a)
    
    def grad(hash_val, x, y):
        h = hash_val & 3
        if h == 0:
            return x + y
        elif h == 1:
            return -x + y
        elif h == 2:
            return x - y
        else:
            return -x - y
    
    # Permutation table
    p = list(range(256))
    import random
    random.seed(42)
    random.shuffle(p)
    p = p + p
    
    def perlin_single(x, y):
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        
        x -= math.floor(x)
        y -= math.floor(y)
        
        u = fade(x)
        v = fade(y)
        
        A = p[X] + Y
        B = p[X + 1] + Y
        
        return lerp(
            lerp(grad(p[A], x, y), grad(p[B], x - 1, y), u),
            lerp(grad(p[A + 1], x, y - 1), grad(p[B + 1], x - 1, y - 1), u),
            v
        )
    
    total = 0
    amplitude = 1
    max_value = 0
    frequency = 1
    
    for _ in range(octaves):
        total += perlin_single(x * frequency, y * frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    
    return total / max_value


class TerrainGenerator:
    """
    Generates terrain heightmaps and biome maps.
    """
    
    # Biome types
    BIOME_URBAN = 0       # City streets (flat)
    BIOME_INDUSTRIAL = 1  # Industrial areas (slightly elevated)
    BIOME_PARK = 2        # Parks (varied terrain with hills)
    BIOME_WATER = 3       # Water bodies
    BIOME_HIGHWAY = 4     # Elevated highways
    
    def __init__(self, seed: int = 42, scale: float = 0.02,
                 octaves: int = 4, persistence: float = 0.5,
                 lacunarity: float = 2.0):
        """
        Initialize terrain generator.
        
        Args:
            seed: Random seed for reproducibility
            scale: Base noise scale (smaller = larger features)
            octaves: Number of noise octaves
            persistence: Amplitude reduction per octave
            lacunarity: Frequency increase per octave
        """
        self.seed = seed
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        
        # Set random seed
        np.random.seed(seed)
        
        logger.info(f"TerrainGenerator initialized (seed={seed})")
    
    def _noise(self, x: float, y: float) -> float:
        """Get noise value at position."""
        if HAS_NOISE_LIB:
            return pnoise2(x, y, 
                          octaves=self.octaves,
                          persistence=self.persistence,
                          lacunarity=self.lacunarity,
                          repeatx=1024, repeaty=1024,
                          base=self.seed)
        else:
            return _fallback_perlin(x + self.seed * 0.1, y + self.seed * 0.1,
                                   self.octaves, self.persistence, self.lacunarity)
    
    def generate_heightmap(self, width: int, height: int,
                          offset_x: float = 0, offset_y: float = 0,
                          height_multiplier: float = 20.0) -> np.ndarray:
        """
        Generate a heightmap.
        
        Args:
            width: Width in samples
            height: Height in samples
            offset_x: World X offset for tiling
            offset_y: World Y offset for tiling
            height_multiplier: Scale factor for height values
        
        Returns:
            2D numpy array of height values
        """
        heightmap = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                nx = (x + offset_x) * self.scale
                ny = (y + offset_y) * self.scale
                
                # Base terrain noise
                value = self._noise(nx, ny)
                
                # Normalize from [-1, 1] to [0, 1]
                value = (value + 1) / 2
                
                heightmap[y, x] = value * height_multiplier
        
        return heightmap
    
    def generate_cyberpunk_terrain(self, width: int, height: int,
                                   offset_x: float = 0, offset_y: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate cyberpunk-style urban terrain with distinct zones.
        
        Returns:
            Tuple of (heightmap, biome_map)
        """
        heightmap = np.zeros((height, width), dtype=np.float32)
        biome_map = np.zeros((height, width), dtype=np.int32)
        
        center_x = width / 2
        center_y = height / 2
        
        for y in range(height):
            for x in range(width):
                nx = (x + offset_x) * self.scale
                ny = (y + offset_y) * self.scale
                
                # Distance from center (normalized)
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2) / max(center_x, center_y)
                
                # Zone noise for biome determination
                zone_noise = self._noise(nx * 2, ny * 2)
                
                # Determine biome based on noise and distance
                if zone_noise < -0.3:
                    biome = self.BIOME_WATER
                    h = -1.0  # Below ground
                elif zone_noise < 0 and dist > 0.6:
                    biome = self.BIOME_PARK
                    h = self._noise(nx, ny) * 3  # Varied terrain
                elif zone_noise > 0.4 and dist < 0.8:
                    biome = self.BIOME_INDUSTRIAL
                    h = 1.0  # Slightly elevated
                else:
                    biome = self.BIOME_URBAN
                    h = 0.0  # Flat streets
                
                # Add highway strips
                highway_noise = math.sin(nx * 0.5) * math.cos(ny * 0.3)
                if abs(highway_noise) < 0.1 and biome == self.BIOME_URBAN:
                    biome = self.BIOME_HIGHWAY
                    h = 3.0  # Elevated highway
                
                heightmap[y, x] = h
                biome_map[y, x] = biome
        
        return heightmap, biome_map
    
    def generate_road_network(self, width: int, height: int,
                             grid_spacing: int = 20) -> np.ndarray:
        """
        Generate a road network mask.
        
        Args:
            width: Width in samples
            height: Height in samples
            grid_spacing: Distance between roads in samples
        
        Returns:
            2D boolean array (True = road)
        """
        roads = np.zeros((height, width), dtype=bool)
        
        # Main grid roads
        for y in range(height):
            for x in range(width):
                # Grid roads
                if x % grid_spacing < 3 or y % grid_spacing < 3:
                    roads[y, x] = True
                
                # Add some diagonal/curved roads using noise
                nx = x * self.scale * 3
                ny = y * self.scale * 3
                road_noise = self._noise(nx, ny)
                
                if abs(road_noise) < 0.05:
                    roads[y, x] = True
        
        return roads
    
    def get_biome_color(self, biome: int, height: float) -> Tuple[float, float, float, float]:
        """Get RGBA color for a biome type (cyberpunk palette)."""
        colors = {
            self.BIOME_URBAN: (0.15, 0.15, 0.18, 1.0),      # Dark concrete
            self.BIOME_INDUSTRIAL: (0.2, 0.18, 0.15, 1.0),  # Rusty industrial
            self.BIOME_PARK: (0.1, 0.25, 0.15, 1.0),        # Neon-tinted green
            self.BIOME_WATER: (0.05, 0.1, 0.2, 0.8),        # Dark polluted water
            self.BIOME_HIGHWAY: (0.1, 0.1, 0.12, 1.0),      # Asphalt
        }
        return colors.get(biome, (0.5, 0.5, 0.5, 1.0))
    
    def create_color_func(self, biome_map: np.ndarray) -> Callable:
        """Create a color function for terrain mesh generation."""
        def color_func(height: float, x: int = 0, y: int = 0) -> Tuple[float, float, float, float]:
            if 0 <= y < biome_map.shape[0] and 0 <= x < biome_map.shape[1]:
                biome = biome_map[y, x]
                return self.get_biome_color(biome, height)
            return (0.5, 0.5, 0.5, 1.0)
        return color_func


class ChunkedTerrain:
    """
    Manages terrain in chunks for large worlds.
    """
    
    def __init__(self, generator: TerrainGenerator, 
                 chunk_size: int = 64,
                 height_scale: float = 1.0):
        """
        Initialize chunked terrain manager.
        
        Args:
            generator: TerrainGenerator instance
            chunk_size: Size of each chunk in samples
            height_scale: Height multiplier
        """
        self.generator = generator
        self.chunk_size = chunk_size
        self.height_scale = height_scale
        
        # Cache of generated chunks
        self._chunks: dict = {}
        self._biome_cache: dict = {}
    
    def get_chunk(self, chunk_x: int, chunk_y: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get or generate a terrain chunk.
        
        Returns:
            Tuple of (heightmap, biome_map) for the chunk
        """
        key = (chunk_x, chunk_y)
        
        if key not in self._chunks:
            offset_x = chunk_x * self.chunk_size
            offset_y = chunk_y * self.chunk_size
            
            heightmap, biome_map = self.generator.generate_cyberpunk_terrain(
                self.chunk_size, self.chunk_size,
                offset_x, offset_y
            )
            
            self._chunks[key] = heightmap
            self._biome_cache[key] = biome_map
        
        return self._chunks[key], self._biome_cache[key]
    
    def get_height(self, world_x: float, world_y: float) -> float:
        """Get interpolated height at world position."""
        chunk_x = int(world_x // self.chunk_size)
        chunk_y = int(world_y // self.chunk_size)
        
        local_x = world_x % self.chunk_size
        local_y = world_y % self.chunk_size
        
        heightmap, _ = self.get_chunk(chunk_x, chunk_y)
        
        # Bilinear interpolation
        x0 = int(local_x)
        y0 = int(local_y)
        x1 = min(x0 + 1, self.chunk_size - 1)
        y1 = min(y0 + 1, self.chunk_size - 1)
        
        fx = local_x - x0
        fy = local_y - y0
        
        h00 = heightmap[y0, x0]
        h10 = heightmap[y0, x1]
        h01 = heightmap[y1, x0]
        h11 = heightmap[y1, x1]
        
        h = (h00 * (1-fx) * (1-fy) + 
             h10 * fx * (1-fy) +
             h01 * (1-fx) * fy +
             h11 * fx * fy)
        
        return h * self.height_scale
    
    def unload_chunk(self, chunk_x: int, chunk_y: int):
        """Unload a chunk from memory."""
        key = (chunk_x, chunk_y)
        if key in self._chunks:
            del self._chunks[key]
        if key in self._biome_cache:
            del self._biome_cache[key]
    
    def clear_cache(self):
        """Clear all cached chunks."""
        self._chunks.clear()
        self._biome_cache.clear()
