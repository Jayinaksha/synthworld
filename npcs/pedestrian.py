"""
SynthWorld Pedestrian NPC

Walking NPCs - civilians, characters, etc.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum, auto
import random
import logging

from .base import NPC, NPCType, NPCState, NPCStats

logger = logging.getLogger(__name__)


class PedestrianActivity(Enum):
    """Activities for pedestrians."""
    WALKING = auto()
    STANDING = auto()
    SITTING = auto()
    TALKING = auto()
    WORKING = auto()
    SHOPPING = auto()
    WAITING = auto()


@dataclass
class PedestrianAppearance:
    """Visual appearance of a pedestrian."""
    height: float = 1.75
    body_color: Tuple[float, float, float] = (0.3, 0.3, 0.4)
    shirt_color: Tuple[float, float, float] = (0.5, 0.5, 0.6)
    pants_color: Tuple[float, float, float] = (0.2, 0.2, 0.3)
    skin_color: Tuple[float, float, float] = (0.8, 0.6, 0.5)


class Pedestrian(NPC):
    """
    Walking pedestrian NPC.
    """
    
    # Walking speeds (m/s)
    WALK_SPEED = 1.4
    RUN_SPEED = 4.0
    SPRINT_SPEED = 7.0
    
    def __init__(self, name: str, physics_world=None, renderer=None):
        super().__init__(name, NPCType.PEDESTRIAN, physics_world, renderer)
        
        # Appearance
        self.appearance = self._random_appearance()
        
        # Pedestrian-specific state
        self._activity = PedestrianActivity.WALKING
        self._destination: Optional[np.ndarray] = None
        self._waypoints: List[np.ndarray] = []
        self._current_waypoint_index = 0
        
        # Walking parameters
        self._walk_speed = self.WALK_SPEED * (0.8 + random.random() * 0.4)
        self._idle_time = 0.0
        self._max_idle_time = random.uniform(1.0, 5.0)
        
        # Animation state
        self._walk_phase = random.random()  # For leg animation
        
        # Social behavior
        self._is_in_group = False
        self._group_members: List[str] = []
        
        # Avoidance
        self._avoidance_cooldown = 0.0
    
    def _random_appearance(self) -> PedestrianAppearance:
        """Generate random appearance."""
        return PedestrianAppearance(
            height=random.uniform(1.55, 1.95),
            body_color=(random.uniform(0.2, 0.5),
                       random.uniform(0.2, 0.5),
                       random.uniform(0.2, 0.5)),
            shirt_color=(random.random(), random.random(), random.random()),
            pants_color=(random.uniform(0.1, 0.4),
                        random.uniform(0.1, 0.4),
                        random.uniform(0.2, 0.5)),
            skin_color=(random.uniform(0.6, 0.9),
                       random.uniform(0.4, 0.7),
                       random.uniform(0.3, 0.6))
        )
    
    def spawn(self, position: Tuple[float, float, float],
              rotation: float = 0.0):
        """Spawn the pedestrian."""
        self._position = np.array(position)
        
        # Set rotation quaternion from yaw
        angle = np.radians(rotation)
        self._rotation = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])
        
        if self._physics:
            from ..engine.physics import RigidBodyConfig
            
            # Create capsule for pedestrian
            config = RigidBodyConfig(
                mass=70.0,
                position=position,
                friction=0.7,
                kinematic=True  # NPC-controlled movement
            )
            
            self._physics_body = self._physics.create_capsule(
                radius=0.3,
                height=self.appearance.height - 0.6,
                config=config,
                name=f"{self.name}_body"
            )
        
        if self._renderer:
            # Create visual - simplified person shape
            self._visual_node = self._create_visual()
        
        logger.debug(f"Pedestrian '{self.name}' spawned at {position}")
    
    def _create_visual(self):
        """Create visual representation of pedestrian."""
        # Create a simple person shape using cylinders and sphere
        height = self.appearance.height
        
        # Body (capsule-like)
        body = self._renderer.create_capsule(
            0.2, height * 0.4,
            color=self.appearance.shirt_color + (1.0,),
            position=self._position
        )
        
        # Head
        head = self._renderer.create_sphere(
            0.15,
            color=self.appearance.skin_color + (1.0,),
            position=(0, 0, height * 0.45)
        )
        head.reparentTo(body)
        
        # Legs
        leg_l = self._renderer.create_cylinder(
            0.08, height * 0.45,
            color=self.appearance.pants_color + (1.0,),
            position=(-0.1, 0, -height * 0.25)
        )
        leg_l.reparentTo(body)
        
        leg_r = self._renderer.create_cylinder(
            0.08, height * 0.45,
            color=self.appearance.pants_color + (1.0,),
            position=(0.1, 0, -height * 0.25)
        )
        leg_r.reparentTo(body)
        
        return body
    
    def update(self, dt: float):
        """Update pedestrian state."""
        if not self._active:
            return
        
        # Update behavior tree if present
        if self._behavior_tree:
            self._behavior_tree.tick(dt)
        else:
            # Default behavior
            self._default_behavior(dt)
        
        # Update animation
        self._update_animation(dt)
        
        # Sync visuals
        if self._visual_node:
            self._visual_node.setPos(*self._position)
            self._visual_node.setH(self.heading)
    
    def _default_behavior(self, dt: float):
        """Default wandering behavior."""
        if self._state == NPCState.IDLE:
            self._idle_time += dt
            
            if self._idle_time >= self._max_idle_time:
                # Start walking to random destination
                self._pick_random_destination()
                self._state = NPCState.WALKING
                self._idle_time = 0
        
        elif self._state == NPCState.WALKING:
            if self._destination is not None:
                distance = np.linalg.norm(self._destination[:2] - self._position[:2])
                
                if distance < 1.0:
                    # Reached destination
                    self._state = NPCState.IDLE
                    self._max_idle_time = random.uniform(1.0, 5.0)
                else:
                    # Move towards destination
                    self.move_towards(self._destination, self._walk_speed, dt)
                    
                    # Avoid other NPCs
                    self._avoid_collisions(dt)
            else:
                self._pick_random_destination()
    
    def _pick_random_destination(self):
        """Pick a random destination nearby."""
        # Random point within 20-50 meters
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(20, 50)
        
        self._destination = self._position + np.array([
            np.cos(angle) * distance,
            np.sin(angle) * distance,
            0
        ])
    
    def _avoid_collisions(self, dt: float):
        """Avoid other pedestrians and obstacles."""
        self._avoidance_cooldown -= dt
        if self._avoidance_cooldown > 0:
            return
        
        # Check for nearby entities
        avoidance = np.zeros(3)
        
        # If we had access to NPC manager, check nearby NPCs
        # For now, just use physics raycast
        if self._physics:
            # Cast rays in front
            forward = np.array([np.sin(np.radians(self.heading)),
                               np.cos(np.radians(self.heading)), 0])
            
            # Front ray
            start = self._position + np.array([0, 0, 1.0])
            end = start + forward * 3.0
            
            hit = self._physics.ray_cast(tuple(start), tuple(end))
            if hit and hit['body_id'] != -1:
                # Something in front, steer away
                avoidance += np.cross(forward, np.array([0, 0, 1])) * 0.5
                self._avoidance_cooldown = 0.5
        
        if np.linalg.norm(avoidance) > 0:
            self._position += avoidance * dt
    
    def _update_animation(self, dt: float):
        """Update walking animation."""
        if self._state == NPCState.WALKING or self._state == NPCState.RUNNING:
            speed = self._walk_speed if self._state == NPCState.WALKING else self.RUN_SPEED
            self._walk_phase = (self._walk_phase + dt * speed * 2) % 1.0
        else:
            self._walk_phase = 0
    
    def run_to(self, target: Tuple[float, float, float]):
        """Run to a target position."""
        self._destination = np.array(target)
        self._state = NPCState.RUNNING
    
    def walk_to(self, target: Tuple[float, float, float]):
        """Walk to a target position."""
        self._destination = np.array(target)
        self._state = NPCState.WALKING
    
    def stop(self):
        """Stop moving."""
        self._state = NPCState.IDLE
        self._velocity = np.zeros(3)
    
    def flee_from(self, position: Tuple[float, float, float]):
        """Flee from a position."""
        threat_pos = np.array(position)
        direction = self._position - threat_pos
        direction[2] = 0  # Keep on ground
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        self._destination = self._position + direction * 50
        self._state = NPCState.FLEEING
        self._stats.fear = min(1.0, self._stats.fear + 0.5)


class Civilian(Pedestrian):
    """
    Generic civilian NPC with daily routine.
    """
    
    def __init__(self, name: str, physics_world=None, renderer=None,
                 home_location: Optional[Tuple[float, float, float]] = None,
                 work_location: Optional[Tuple[float, float, float]] = None):
        super().__init__(name, physics_world, renderer)
        
        # Daily routine locations
        self.home = np.array(home_location) if home_location else None
        self.work = np.array(work_location) if work_location else None
        
        # Schedule (hour -> activity)
        self.schedule: Dict[int, PedestrianActivity] = {
            6: PedestrianActivity.WALKING,   # Wake up, go to work
            9: PedestrianActivity.WORKING,
            12: PedestrianActivity.WALKING,  # Lunch
            13: PedestrianActivity.WORKING,
            17: PedestrianActivity.SHOPPING,
            19: PedestrianActivity.WALKING,  # Go home
            22: PedestrianActivity.STANDING  # At home
        }
        
        self._current_schedule_hour = -1
    
    def update(self, dt: float):
        """Update with schedule-based behavior."""
        # Get current simulation hour
        current_hour = self._get_simulation_hour()
        
        if current_hour != self._current_schedule_hour:
            self._current_schedule_hour = current_hour
            self._update_schedule()
        
        super().update(dt)
    
    def _get_simulation_hour(self) -> int:
        """Get current simulation hour (0-23)."""
        # Would be hooked up to game time system
        import time
        return int(time.time() / 3600) % 24
    
    def _update_schedule(self):
        """Update behavior based on schedule."""
        hour = self._current_schedule_hour
        
        # Find applicable schedule entry
        applicable_hour = max([h for h in self.schedule.keys() if h <= hour],
                             default=min(self.schedule.keys()))
        
        activity = self.schedule.get(applicable_hour, PedestrianActivity.STANDING)
        self._activity = activity
        
        # Set destination based on activity
        if activity == PedestrianActivity.WORKING and self.work is not None:
            self._destination = self.work.copy()
            self._state = NPCState.WALKING
        elif activity == PedestrianActivity.STANDING and self.home is not None:
            self._destination = self.home.copy()
            self._state = NPCState.WALKING


class CyberpunkCitizen(Pedestrian):
    """
    Cyberpunk-themed citizen with enhanced appearance.
    """
    
    def __init__(self, name: str, physics_world=None, renderer=None):
        super().__init__(name, physics_world, renderer)
        
        # Override appearance with cyberpunk style
        self.appearance = self._cyberpunk_appearance()
        
        # Cyberpunk-specific attributes
        self.has_cybernetics = random.random() > 0.6
        self.faction = random.choice(['corpo', 'street', 'nomad', 'civilian'])
    
    def _cyberpunk_appearance(self) -> PedestrianAppearance:
        """Generate cyberpunk-themed appearance."""
        # Neon-accented colors
        neon_colors = [
            (1.0, 0.0, 0.5),   # Magenta
            (0.0, 1.0, 1.0),   # Cyan
            (1.0, 0.5, 0.0),   # Orange
            (0.5, 0.0, 1.0),   # Purple
            (0.0, 1.0, 0.5),   # Green
        ]
        
        accent = random.choice(neon_colors)
        
        return PedestrianAppearance(
            height=random.uniform(1.6, 2.0),
            body_color=(0.1, 0.1, 0.15),  # Dark base
            shirt_color=accent if random.random() > 0.5 else (0.2, 0.2, 0.25),
            pants_color=(0.05, 0.05, 0.08),  # Very dark
            skin_color=(random.uniform(0.5, 0.9),
                       random.uniform(0.4, 0.7),
                       random.uniform(0.3, 0.6))
        )
