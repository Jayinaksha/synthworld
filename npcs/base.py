"""
SynthWorld NPC Base Classes

Base classes for all NPCs in the simulation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class NPCType(Enum):
    """Types of NPCs."""
    PEDESTRIAN = auto()
    VEHICLE = auto()
    ROBOT = auto()
    ANIMAL = auto()
    DRONE = auto()
    STATIC = auto()


class NPCState(Enum):
    """NPC behavioral states."""
    IDLE = auto()
    WALKING = auto()
    RUNNING = auto()
    DRIVING = auto()
    WAITING = auto()
    INTERACTING = auto()
    FLEEING = auto()
    FOLLOWING = auto()
    PATROLLING = auto()


@dataclass
class NPCStats:
    """NPC statistics and attributes."""
    health: float = 100.0
    stamina: float = 100.0
    awareness: float = 1.0  # How alert the NPC is
    aggression: float = 0.0  # 0 = peaceful, 1 = hostile
    fear: float = 0.0
    speed_multiplier: float = 1.0


@dataclass
class NPCMemory:
    """NPC memory of events and entities."""
    known_locations: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    seen_entities: List[str] = field(default_factory=list)
    last_player_position: Optional[np.ndarray] = None
    last_player_seen_time: float = 0.0
    conversation_history: List[str] = field(default_factory=list)


class NPC(ABC):
    """
    Abstract base class for all NPCs.
    """
    
    def __init__(self, name: str, npc_type: NPCType,
                 physics_world=None, renderer=None):
        """
        Initialize NPC.
        
        Args:
            name: NPC name/ID
            npc_type: Type of NPC
            physics_world: Reference to physics simulation
            renderer: Reference to renderer
        """
        self.name = name
        self.npc_type = npc_type
        self._physics = physics_world
        self._renderer = renderer
        
        # Position and movement
        self._position = np.zeros(3)
        self._rotation = np.array([0, 0, 0, 1])  # Quaternion
        self._velocity = np.zeros(3)
        self._target_position: Optional[np.ndarray] = None
        
        # State
        self._state = NPCState.IDLE
        self._stats = NPCStats()
        self._memory = NPCMemory()
        
        # Physics body
        self._physics_body = None
        
        # Visual representation
        self._visual_node = None
        
        # Behavior
        self._behavior_tree = None
        
        # Pathfinding
        self._current_path: List[np.ndarray] = []
        self._path_index = 0
        
        # Is active
        self._active = True
        
        logger.debug(f"NPC '{name}' ({npc_type.name}) created")
    
    @property
    def position(self) -> np.ndarray:
        """Get NPC position."""
        return self._position.copy()
    
    @position.setter
    def position(self, value: Tuple[float, float, float]):
        """Set NPC position."""
        self._position = np.array(value)
        if self._physics_body:
            self._physics_body.position = value
        if self._visual_node:
            self._visual_node.setPos(*value)
    
    @property
    def state(self) -> NPCState:
        """Get current behavioral state."""
        return self._state
    
    @state.setter
    def state(self, value: NPCState):
        """Set behavioral state."""
        if value != self._state:
            logger.debug(f"NPC '{self.name}' state: {self._state.name} -> {value.name}")
            self._state = value
    
    @property
    def heading(self) -> float:
        """Get NPC heading in degrees."""
        q = self._rotation
        siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1])
        cosy_cosp = 1 - 2 * (q[1]**2 + q[2]**2)
        return np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    
    @abstractmethod
    def spawn(self, position: Tuple[float, float, float],
              rotation: float = 0.0):
        """Spawn the NPC at a position."""
        pass
    
    @abstractmethod
    def update(self, dt: float):
        """Update NPC state."""
        pass
    
    def set_behavior_tree(self, behavior_tree):
        """Set the behavior tree for this NPC."""
        self._behavior_tree = behavior_tree
    
    def set_target(self, target: Tuple[float, float, float]):
        """Set movement target."""
        self._target_position = np.array(target)
    
    def set_path(self, path: List[Tuple[float, float, float]]):
        """Set navigation path."""
        self._current_path = [np.array(p) for p in path]
        self._path_index = 0
    
    def clear_path(self):
        """Clear current path."""
        self._current_path = []
        self._path_index = 0
        self._target_position = None
    
    def move_towards(self, target: np.ndarray, speed: float, dt: float):
        """
        Move towards a target position.
        
        Args:
            target: Target position
            speed: Movement speed
            dt: Time delta
        """
        direction = target - self._position
        distance = np.linalg.norm(direction[:2])  # Ignore Z for ground movement
        
        if distance > 0.1:
            # Normalize and move
            direction = direction / distance
            movement = direction * speed * self._stats.speed_multiplier * dt
            
            # Limit movement to remaining distance
            if np.linalg.norm(movement[:2]) > distance:
                movement = direction * distance
            
            self._position += movement
            self._velocity = movement / dt
            
            # Update heading
            self._face_direction(direction)
        else:
            self._velocity = np.zeros(3)
    
    def _face_direction(self, direction: np.ndarray):
        """Rotate to face a direction."""
        angle = np.arctan2(direction[0], direction[1])
        
        # Convert to quaternion (rotation around Z)
        self._rotation = np.array([
            0, 0, np.sin(angle/2), np.cos(angle/2)
        ])
        
        if self._visual_node:
            self._visual_node.setH(np.degrees(angle))
    
    def distance_to(self, position: Tuple[float, float, float]) -> float:
        """Calculate distance to a position."""
        return np.linalg.norm(np.array(position) - self._position)
    
    def can_see(self, position: Tuple[float, float, float],
                fov: float = 120.0, max_distance: float = 20.0) -> bool:
        """
        Check if NPC can see a position.
        
        Args:
            position: Position to check
            fov: Field of view in degrees
            max_distance: Maximum sight distance
        
        Returns:
            True if position is visible
        """
        target = np.array(position)
        to_target = target - self._position
        distance = np.linalg.norm(to_target)
        
        if distance > max_distance:
            return False
        
        # Check angle
        if distance > 0:
            to_target_normalized = to_target / distance
            forward = np.array([np.sin(np.radians(self.heading)),
                              np.cos(np.radians(self.heading)), 0])
            
            dot = np.dot(to_target_normalized[:2], forward[:2])
            angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
            
            if angle > fov / 2:
                return False
        
        # Raycast check (if physics available)
        if self._physics:
            hit = self._physics.ray_cast(
                tuple(self._position + np.array([0, 0, 1.5])),  # Eye level
                tuple(target + np.array([0, 0, 1.5]))
            )
            if hit and hit['body_id'] != -1:
                # Something blocking
                return False
        
        return True
    
    def activate(self):
        """Activate the NPC."""
        self._active = True
    
    def deactivate(self):
        """Deactivate the NPC (pause updates)."""
        self._active = False
        self._velocity = np.zeros(3)
    
    def destroy(self):
        """Remove the NPC from the simulation."""
        if self._physics_body:
            self._physics_body.remove()
            self._physics_body = None
        
        if self._visual_node:
            self._visual_node.removeNode()
            self._visual_node = None
        
        logger.debug(f"NPC '{self.name}' destroyed")


class NPCManager:
    """
    Manages all NPCs in the simulation.
    """
    
    def __init__(self, physics_world=None, renderer=None,
                 max_npcs: int = 100):
        """
        Initialize NPC manager.
        
        Args:
            physics_world: Reference to physics simulation
            renderer: Reference to renderer
            max_npcs: Maximum number of active NPCs
        """
        self._physics = physics_world
        self._renderer = renderer
        self._max_npcs = max_npcs
        
        self._npcs: Dict[str, NPC] = {}
        self._npc_counter = 0
        
        # Spatial partitioning for efficiency
        self._grid_size = 50.0  # Grid cell size
        self._spatial_grid: Dict[Tuple[int, int], List[str]] = {}
        
        logger.info(f"NPCManager initialized (max {max_npcs} NPCs)")
    
    def add_npc(self, npc: NPC) -> bool:
        """Add an NPC to the manager."""
        if len(self._npcs) >= self._max_npcs:
            logger.warning("Max NPCs reached, cannot add more")
            return False
        
        self._npcs[npc.name] = npc
        self._update_spatial_grid(npc)
        return True
    
    def remove_npc(self, name: str):
        """Remove an NPC by name."""
        if name in self._npcs:
            npc = self._npcs[name]
            npc.destroy()
            del self._npcs[name]
    
    def get_npc(self, name: str) -> Optional[NPC]:
        """Get an NPC by name."""
        return self._npcs.get(name)
    
    def get_npcs_in_radius(self, position: Tuple[float, float, float],
                          radius: float) -> List[NPC]:
        """Get all NPCs within a radius of a position."""
        nearby = []
        pos = np.array(position)
        
        for npc in self._npcs.values():
            if np.linalg.norm(npc.position - pos) <= radius:
                nearby.append(npc)
        
        return nearby
    
    def get_nearest_npc(self, position: Tuple[float, float, float],
                       npc_type: Optional[NPCType] = None) -> Optional[NPC]:
        """Get the nearest NPC to a position."""
        pos = np.array(position)
        nearest = None
        min_dist = float('inf')
        
        for npc in self._npcs.values():
            if npc_type and npc.npc_type != npc_type:
                continue
            
            dist = np.linalg.norm(npc.position - pos)
            if dist < min_dist:
                min_dist = dist
                nearest = npc
        
        return nearest
    
    def update(self, dt: float):
        """Update all NPCs."""
        for npc in self._npcs.values():
            if npc._active:
                npc.update(dt)
                self._update_spatial_grid(npc)
    
    def _update_spatial_grid(self, npc: NPC):
        """Update NPC's position in spatial grid."""
        # Calculate grid cell
        cell_x = int(npc.position[0] // self._grid_size)
        cell_y = int(npc.position[1] // self._grid_size)
        cell = (cell_x, cell_y)
        
        # Remove from old cell
        for c, npcs in self._spatial_grid.items():
            if npc.name in npcs:
                npcs.remove(npc.name)
        
        # Add to new cell
        if cell not in self._spatial_grid:
            self._spatial_grid[cell] = []
        self._spatial_grid[cell].append(npc.name)
    
    def spawn_random(self, npc_class, count: int, 
                     area_center: Tuple[float, float],
                     area_size: float = 100.0) -> List[NPC]:
        """
        Spawn multiple random NPCs in an area.
        
        Args:
            npc_class: NPC class to instantiate
            count: Number of NPCs to spawn
            area_center: Center of spawn area
            area_size: Size of spawn area
        
        Returns:
            List of spawned NPCs
        """
        spawned = []
        
        for _ in range(count):
            if len(self._npcs) >= self._max_npcs:
                break
            
            # Random position
            x = area_center[0] + np.random.uniform(-area_size/2, area_size/2)
            y = area_center[1] + np.random.uniform(-area_size/2, area_size/2)
            
            # Create NPC
            self._npc_counter += 1
            name = f"npc_{self._npc_counter}"
            npc = npc_class(name, self._physics, self._renderer)
            npc.spawn((x, y, 0), rotation=np.random.uniform(0, 360))
            
            if self.add_npc(npc):
                spawned.append(npc)
        
        return spawned
    
    def get_all_npcs(self) -> List[NPC]:
        """Get all NPCs."""
        return list(self._npcs.values())
    
    def cleanup(self):
        """Clean up all NPCs."""
        for npc in list(self._npcs.values()):
            npc.destroy()
        self._npcs.clear()
        self._spatial_grid.clear()
