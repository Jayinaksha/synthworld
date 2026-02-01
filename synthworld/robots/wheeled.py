"""
SynthWorld Wheeled Robot

Differential drive and other wheeled robot implementations.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import logging

from .base import Robot, RobotType, RobotState

logger = logging.getLogger(__name__)


@dataclass
class WheelConfig:
    """Configuration for a wheel."""
    name: str
    position: Tuple[float, float, float]  # Relative to robot base
    radius: float
    width: float
    max_torque: float = 10.0
    max_velocity: float = 20.0  # rad/s


class WheeledRobot(Robot):
    """
    Generic wheeled robot base class.
    """
    
    def __init__(self, name: str, physics_world=None, renderer=None):
        super().__init__(name, RobotType.WHEELED, physics_world, renderer)
        
        # Wheel configuration
        self._wheels: Dict[str, WheelConfig] = {}
        self._wheel_velocities: Dict[str, float] = {}
        
        # Robot parameters
        self.wheel_base = 0.5  # Distance between front and back wheels
        self.track_width = 0.4  # Distance between left and right wheels
        self.wheel_radius = 0.1
        
        # Control limits
        self.max_linear_velocity = 5.0  # m/s
        self.max_angular_velocity = 3.0  # rad/s
    
    def spawn(self, position: Tuple[float, float, float],
              orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        """Spawn the wheeled robot."""
        self._state.position = np.array(position)
        self._state.orientation = np.array(orientation)
        
        if self._physics:
            # Create physics body - simple box for chassis
            from ..engine.physics import RigidBodyConfig
            
            config = RigidBodyConfig(
                mass=10.0,
                position=position,
                orientation=orientation,
                friction=0.8
            )
            
            self._physics_body = self._physics.create_box(
                half_extents=(self.wheel_base/2, self.track_width/2, 0.15),
                config=config,
                name=f"{self.name}_chassis"
            )
            self._physics_body.set_color((0.2, 0.3, 0.8, 1.0))
        
        if self._renderer:
            # Create visual representation
            self._visual_node = self._renderer.create_box(
                self.wheel_base, self.track_width, 0.3,
                color=(0.2, 0.3, 0.8, 1.0),
                position=position,
                name=f"{self.name}_visual"
            )
        
        logger.info(f"Spawned wheeled robot '{self.name}' at {position}")
    
    def update(self, dt: float):
        """Update robot state."""
        # Sync from physics
        self.sync_state_from_physics()
        
        # Update visuals
        self.sync_visuals_from_physics()
        
        # Update battery (simulate drain)
        if self.speed > 0.1:
            self._state.battery_level = max(0, self._state.battery_level - 0.0001 * dt)
    
    def apply_control(self, control: Dict[str, float]):
        """
        Apply control commands.
        
        Args:
            control: Dict with 'linear' and 'angular' velocities
        """
        if not self._control_enabled:
            return
        
        linear = control.get('linear', 0) * self.max_linear_velocity
        angular = control.get('angular', 0) * self.max_angular_velocity
        
        # Convert to wheel velocities (differential drive kinematics)
        left_vel = linear - angular * self.track_width / 2
        right_vel = linear + angular * self.track_width / 2
        
        self._wheel_velocities['left'] = left_vel / self.wheel_radius
        self._wheel_velocities['right'] = right_vel / self.wheel_radius
        
        # Apply forces to physics body
        if self._physics_body:
            # Get forward direction
            heading = np.radians(self.heading)
            forward = np.array([np.sin(heading), np.cos(heading), 0])
            
            # Apply linear force
            force = forward * linear * 100  # Scale for mass
            self._physics_body.apply_force(tuple(force))
            
            # Apply torque for turning
            self._physics_body.apply_torque((0, 0, angular * 50))


class DifferentialDriveRobot(WheeledRobot):
    """
    Two-wheeled differential drive robot (like a TurtleBot).
    """
    
    def __init__(self, name: str, physics_world=None, renderer=None,
                 wheel_radius: float = 0.1, wheel_separation: float = 0.3):
        super().__init__(name, physics_world, renderer)
        
        self.wheel_radius = wheel_radius
        self.track_width = wheel_separation
        self.wheel_base = 0.2
        
        # Two-wheeled specific parameters
        self.base_height = 0.1
        self.has_caster = True  # Support wheel at back
        
        # Configure wheels
        self._wheels = {
            'left': WheelConfig(
                name='left',
                position=(-wheel_separation/2, 0, wheel_radius),
                radius=wheel_radius,
                width=0.03
            ),
            'right': WheelConfig(
                name='right',
                position=(wheel_separation/2, 0, wheel_radius),
                radius=wheel_radius,
                width=0.03
            )
        }
    
    def spawn(self, position: Tuple[float, float, float],
              orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        """Spawn differential drive robot."""
        self._state.position = np.array(position)
        self._state.orientation = np.array(orientation)
        
        if self._physics:
            from ..engine.physics import RigidBodyConfig
            
            # Cylindrical base
            config = RigidBodyConfig(
                mass=5.0,
                position=position,
                orientation=orientation,
                friction=0.9
            )
            
            self._physics_body = self._physics.create_cylinder(
                radius=0.2,
                height=self.base_height * 2,
                config=config,
                name=f"{self.name}_base"
            )
            self._physics_body.set_color((0.8, 0.3, 0.1, 1.0))  # Orange
        
        if self._renderer:
            # Create visual - cylindrical base
            self._visual_node = self._renderer.create_cylinder(
                0.2, self.base_height * 2,
                color=(0.8, 0.3, 0.1, 1.0),
                position=position,
                name=f"{self.name}_visual"
            )
            
            # Add wheel visuals
            for wheel_name, wheel_config in self._wheels.items():
                wheel_pos = (
                    position[0] + wheel_config.position[0],
                    position[1] + wheel_config.position[1],
                    position[2] + wheel_config.position[2]
                )
                wheel_node = self._renderer.create_cylinder(
                    wheel_config.radius, wheel_config.width,
                    color=(0.2, 0.2, 0.2, 1.0),
                    position=wheel_pos,
                    rotation=(90, 0, 0)  # Rotate to be wheel-like
                )
                wheel_node.reparentTo(self._visual_node)
        
        logger.info(f"Spawned differential drive robot '{self.name}'")
    
    def set_wheel_velocities(self, left_vel: float, right_vel: float):
        """
        Directly set wheel velocities.
        
        Args:
            left_vel: Left wheel velocity in rad/s
            right_vel: Right wheel velocity in rad/s
        """
        self._wheel_velocities['left'] = np.clip(left_vel, -20, 20)
        self._wheel_velocities['right'] = np.clip(right_vel, -20, 20)
        
        # Convert to robot motion
        v_l = left_vel * self.wheel_radius
        v_r = right_vel * self.wheel_radius
        
        linear = (v_l + v_r) / 2
        angular = (v_r - v_l) / self.track_width
        
        if self._physics_body:
            heading = np.radians(self.heading)
            forward = np.array([np.sin(heading), np.cos(heading), 0])
            
            force = forward * linear * 50
            self._physics_body.apply_force(tuple(force))
            self._physics_body.apply_torque((0, 0, angular * 30))
    
    def get_wheel_encoders(self) -> Dict[str, float]:
        """Get wheel encoder readings (cumulative rotation)."""
        # In a real implementation, this would track total rotation
        return {
            'left': self._wheel_velocities.get('left', 0),
            'right': self._wheel_velocities.get('right', 0)
        }
    
    def get_odometry(self) -> Dict[str, float]:
        """Get odometry estimate."""
        return {
            'x': self._state.position[0],
            'y': self._state.position[1],
            'theta': np.radians(self.heading),
            'linear_velocity': self.speed,
            'angular_velocity': self._state.angular_velocity[2]
        }


class AckermannRobot(WheeledRobot):
    """
    Car-like robot with Ackermann steering.
    """
    
    def __init__(self, name: str, physics_world=None, renderer=None,
                 wheelbase: float = 0.5, track_width: float = 0.4):
        super().__init__(name, physics_world, renderer)
        
        self.wheel_base = wheelbase
        self.track_width = track_width
        self.wheel_radius = 0.08
        
        # Ackermann specific
        self.max_steering_angle = 0.5  # radians
        self.current_steering = 0.0
        
        # Configure wheels
        self._wheels = {
            'front_left': WheelConfig('front_left', (-track_width/2, wheelbase/2, self.wheel_radius), self.wheel_radius, 0.04),
            'front_right': WheelConfig('front_right', (track_width/2, wheelbase/2, self.wheel_radius), self.wheel_radius, 0.04),
            'rear_left': WheelConfig('rear_left', (-track_width/2, -wheelbase/2, self.wheel_radius), self.wheel_radius, 0.04),
            'rear_right': WheelConfig('rear_right', (track_width/2, -wheelbase/2, self.wheel_radius), self.wheel_radius, 0.04),
        }
    
    def spawn(self, position: Tuple[float, float, float],
              orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        """Spawn Ackermann robot (car-like)."""
        self._state.position = np.array(position)
        self._state.orientation = np.array(orientation)
        
        if self._physics:
            from ..engine.physics import RigidBodyConfig
            
            config = RigidBodyConfig(
                mass=20.0,
                position=position,
                orientation=orientation,
                friction=0.7
            )
            
            self._physics_body = self._physics.create_box(
                half_extents=(self.track_width/2, self.wheel_base/2, 0.1),
                config=config,
                name=f"{self.name}_chassis"
            )
            self._physics_body.set_color((0.1, 0.1, 0.8, 1.0))  # Blue
        
        if self._renderer:
            self._visual_node = self._renderer.create_box(
                self.track_width, self.wheel_base, 0.2,
                color=(0.1, 0.1, 0.8, 1.0),
                position=position,
                name=f"{self.name}_visual"
            )
        
        logger.info(f"Spawned Ackermann robot '{self.name}'")
    
    def apply_control(self, control: Dict[str, float]):
        """
        Apply Ackermann control.
        
        Args:
            control: Dict with 'throttle' (-1 to 1) and 'steering' (-1 to 1)
        """
        if not self._control_enabled:
            return
        
        throttle = np.clip(control.get('throttle', 0), -1, 1)
        steering = np.clip(control.get('steering', 0), -1, 1)
        
        # Update steering angle
        self.current_steering = steering * self.max_steering_angle
        
        # Apply forces
        if self._physics_body:
            heading = np.radians(self.heading)
            forward = np.array([np.sin(heading), np.cos(heading), 0])
            
            # Driving force
            drive_force = forward * throttle * self.max_linear_velocity * 100
            self._physics_body.apply_force(tuple(drive_force))
            
            # Steering - apply torque based on speed and steering angle
            speed = self.speed
            if speed > 0.1:
                turn_rate = speed * math.tan(self.current_steering) / self.wheel_base
                self._physics_body.apply_torque((0, 0, turn_rate * 50))
    
    def get_steering_angle(self) -> float:
        """Get current steering angle in radians."""
        return self.current_steering
