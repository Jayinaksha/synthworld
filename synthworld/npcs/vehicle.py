"""
SynthWorld Vehicle NPCs

AI-controlled vehicles for traffic simulation.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum, auto
import random
import logging

from .base import NPC, NPCType, NPCState

logger = logging.getLogger(__name__)


class VehicleType(Enum):
    """Types of vehicles."""
    SEDAN = auto()
    SPORTS = auto()
    SUV = auto()
    TRUCK = auto()
    MOTORCYCLE = auto()
    VAN = auto()
    BUS = auto()
    POLICE = auto()
    TAXI = auto()
    CYBERBIKE = auto()


@dataclass
class VehicleSpecs:
    """Vehicle specifications."""
    length: float
    width: float
    height: float
    max_speed: float  # m/s
    acceleration: float  # m/s^2
    braking: float  # m/s^2
    turn_radius: float  # minimum turn radius


# Vehicle specifications by type
VEHICLE_SPECS = {
    VehicleType.SEDAN: VehicleSpecs(4.5, 1.8, 1.4, 40.0, 4.0, 8.0, 5.0),
    VehicleType.SPORTS: VehicleSpecs(4.3, 1.9, 1.2, 60.0, 7.0, 10.0, 4.5),
    VehicleType.SUV: VehicleSpecs(4.8, 2.0, 1.8, 35.0, 3.5, 7.0, 6.0),
    VehicleType.TRUCK: VehicleSpecs(6.0, 2.2, 2.5, 30.0, 2.5, 6.0, 8.0),
    VehicleType.MOTORCYCLE: VehicleSpecs(2.0, 0.8, 1.1, 50.0, 6.0, 9.0, 2.0),
    VehicleType.VAN: VehicleSpecs(5.5, 2.0, 2.2, 30.0, 3.0, 6.5, 7.0),
    VehicleType.BUS: VehicleSpecs(12.0, 2.5, 3.0, 25.0, 2.0, 5.0, 12.0),
    VehicleType.POLICE: VehicleSpecs(4.8, 1.9, 1.5, 50.0, 5.0, 9.0, 5.5),
    VehicleType.TAXI: VehicleSpecs(4.5, 1.8, 1.5, 35.0, 4.0, 8.0, 5.0),
    VehicleType.CYBERBIKE: VehicleSpecs(2.5, 0.9, 1.2, 55.0, 8.0, 10.0, 1.5),
}


class VehicleNPC(NPC):
    """
    AI-controlled vehicle.
    """
    
    def __init__(self, name: str, vehicle_type: VehicleType = VehicleType.SEDAN,
                 physics_world=None, renderer=None):
        super().__init__(name, NPCType.VEHICLE, physics_world, renderer)
        
        self.vehicle_type = vehicle_type
        self.specs = VEHICLE_SPECS[vehicle_type]
        
        # Vehicle state
        self._speed = 0.0
        self._steering_angle = 0.0
        self._max_steering = 0.5  # radians
        
        # Traffic behavior
        self._following_distance = 5.0
        self._lane_offset = 0.0
        self._target_speed = self.specs.max_speed * 0.6  # Cruise at 60%
        
        # Path following
        self._road_path: List[np.ndarray] = []
        self._path_index = 0
        
        # Visual customization
        self.color = self._random_vehicle_color()
        
        # Emergency vehicle
        self.is_emergency = vehicle_type in [VehicleType.POLICE]
        self._siren_on = False
    
    def _random_vehicle_color(self) -> Tuple[float, float, float]:
        """Generate random vehicle color."""
        colors = [
            (0.1, 0.1, 0.1),   # Black
            (0.9, 0.9, 0.9),   # White
            (0.5, 0.0, 0.0),   # Red
            (0.0, 0.0, 0.5),   # Blue
            (0.3, 0.3, 0.3),   # Gray
            (0.8, 0.6, 0.0),   # Yellow/Gold
            (0.0, 0.4, 0.2),   # Green
        ]
        return random.choice(colors)
    
    def spawn(self, position: Tuple[float, float, float],
              rotation: float = 0.0):
        """Spawn the vehicle."""
        self._position = np.array(position)
        
        angle = np.radians(rotation)
        self._rotation = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])
        
        if self._physics:
            from ..engine.physics import RigidBodyConfig
            
            config = RigidBodyConfig(
                mass=0.0,  # Static/kinematic - controlled by NPC logic
                position=position,
                friction=0.8
            )
            
            self._physics_body = self._physics.create_box(
                half_extents=(
                    self.specs.width / 2,
                    self.specs.length / 2,
                    self.specs.height / 2
                ),
                config=config,
                name=f"{self.name}_body"
            )
        
        if self._renderer:
            self._visual_node = self._create_visual()
        
        logger.debug(f"Vehicle '{self.name}' ({self.vehicle_type.name}) spawned")
    
    def _create_visual(self):
        """Create vehicle visual."""
        if not self._renderer:
            return None
            
        s = self.specs
        
        try:
            # Main body box
            body = self._renderer.create_box(
                s.width, s.length, s.height * 0.6,
                color=self.color + (1.0,),
                position=tuple(self._position)
            )
            return body
        except Exception as e:
            logger.warning(f"Could not create vehicle visual: {e}")
            return None
    
    def update(self, dt: float):
        """Update vehicle state."""
        if not self._active:
            return
        
        # Follow path or behavior tree
        if self._behavior_tree:
            self._behavior_tree.tick(dt)
        else:
            self._follow_path(dt)
        
        # Apply vehicle dynamics
        self._apply_dynamics(dt)
        
        # Sync visuals
        if self._visual_node:
            self._visual_node.setPos(*self._position)
            self._visual_node.setH(self.heading)
    
    def _follow_path(self, dt: float):
        """Follow the road path."""
        if not self._road_path:
            return
        
        if self._path_index >= len(self._road_path):
            # End of path
            self._speed = max(0, self._speed - self.specs.braking * dt)
            return
        
        # Get target waypoint
        target = self._road_path[self._path_index]
        to_target = target - self._position
        distance = np.linalg.norm(to_target[:2])
        
        if distance < 3.0:
            # Move to next waypoint
            self._path_index += 1
            return
        
        # Calculate steering
        target_heading = np.arctan2(to_target[0], to_target[1])
        current_heading = np.radians(self.heading)
        
        heading_error = target_heading - current_heading
        # Normalize angle
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        self._steering_angle = np.clip(heading_error, -self._max_steering, self._max_steering)
        
        # Speed control
        if self._speed < self._target_speed:
            self._speed += self.specs.acceleration * dt
        elif self._speed > self._target_speed:
            self._speed -= self.specs.braking * dt
        
        self._speed = np.clip(self._speed, 0, self.specs.max_speed)
    
    def _apply_dynamics(self, dt: float):
        """Apply vehicle dynamics (simplified Ackermann)."""
        if self._speed < 0.01:
            return
        
        # Bicycle model
        heading = np.radians(self.heading)
        
        # Angular velocity from steering
        if abs(self._steering_angle) > 0.01:
            turn_radius = self.specs.length / np.tan(self._steering_angle)
            angular_velocity = self._speed / turn_radius
        else:
            angular_velocity = 0
        
        # Update heading
        new_heading = heading + angular_velocity * dt
        self._rotation = np.array([
            0, 0, np.sin(new_heading/2), np.cos(new_heading/2)
        ])
        
        # Update position
        forward = np.array([np.sin(new_heading), np.cos(new_heading), 0])
        self._position += forward * self._speed * dt
        self._velocity = forward * self._speed
    
    def set_road_path(self, path: List[Tuple[float, float, float]]):
        """Set the road path to follow."""
        self._road_path = [np.array(p) for p in path]
        self._path_index = 0
    
    def set_target_speed(self, speed: float):
        """Set target cruising speed."""
        self._target_speed = np.clip(speed, 0, self.specs.max_speed)
    
    def brake(self):
        """Apply brakes."""
        self._target_speed = 0
    
    def accelerate(self):
        """Accelerate to max speed."""
        self._target_speed = self.specs.max_speed
    
    def toggle_siren(self):
        """Toggle siren (for emergency vehicles)."""
        if self.is_emergency:
            self._siren_on = not self._siren_on


class TrafficCar(VehicleNPC):
    """
    Basic traffic participant.
    """
    
    def __init__(self, name: str, physics_world=None, renderer=None):
        # Random vehicle type
        vehicle_type = random.choice([
            VehicleType.SEDAN, VehicleType.SEDAN,  # More common
            VehicleType.SUV, VehicleType.SUV,
            VehicleType.SPORTS,
            VehicleType.TRUCK,
            VehicleType.VAN,
            VehicleType.TAXI,
        ])
        
        super().__init__(name, vehicle_type, physics_world, renderer)
        
        # Traffic-specific parameters
        self._reaction_time = random.uniform(0.3, 0.8)
        self._aggression = random.uniform(0, 0.5)
        self._following_distance = 5.0 + random.uniform(-1, 3)
        
        # Traffic rules
        self._respects_lights = random.random() > 0.1  # 90% respect lights
        self._uses_turn_signals = random.random() > 0.2  # 80% use signals
    
    def update(self, dt: float):
        """Update with traffic-aware behavior."""
        # Check for vehicles ahead
        self._check_traffic_ahead()
        
        super().update(dt)
    
    def _check_traffic_ahead(self):
        """Check for vehicles in front and adjust speed."""
        if not self._physics:
            return
        
        # Cast ray forward
        heading = np.radians(self.heading)
        forward = np.array([np.sin(heading), np.cos(heading), 0])
        
        start = self._position + np.array([0, 0, 0.5])
        end = start + forward * self._following_distance
        
        hit = self._physics.ray_cast(tuple(start), tuple(end))
        if hit and hit['body_id'] != -1:
            # Something ahead, slow down
            distance = hit['hit_fraction'] * self._following_distance
            
            if distance < self._following_distance * 0.5:
                # Too close, brake
                self._target_speed = 0
            else:
                # Adjust speed to match
                self._target_speed = self.specs.max_speed * (distance / self._following_distance)


class CyberpunkVehicle(VehicleNPC):
    """
    Futuristic cyberpunk vehicle.
    """
    
    def __init__(self, name: str, physics_world=None, renderer=None):
        # Cyberpunk vehicle types
        vehicle_type = random.choice([
            VehicleType.SPORTS,
            VehicleType.CYBERBIKE,
            VehicleType.SEDAN,
        ])
        
        super().__init__(name, vehicle_type, physics_world, renderer)
        
        # Cyberpunk visual enhancements
        self._neon_color = random.choice([
            (1.0, 0.0, 0.5),   # Magenta
            (0.0, 1.0, 1.0),   # Cyan
            (1.0, 0.5, 0.0),   # Orange
        ])
        
        self._has_neon_lights = random.random() > 0.3
        self._is_autonomous = random.random() > 0.6
    
    def _create_visual(self):
        """Create cyberpunk vehicle visual with neon accents."""
        base_visual = super()._create_visual()
        
        if base_visual is None:
            return None
        
        if self._has_neon_lights and self._renderer:
            try:
                s = self.specs
                
                # Underglow
                underglow = self._renderer.create_box(
                    s.width * 0.8, s.length * 0.8, 0.02,
                    color=self._neon_color + (0.5,),
                    position=(0, 0, -s.height/2 - 0.1)
                )
                underglow.reparentTo(base_visual)
            except Exception:
                pass  # Skip underglow if creation fails
        
        return base_visual
