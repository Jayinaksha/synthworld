"""
SynthWorld Robot Base Class

Base class for all robot types with common functionality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class RobotType(Enum):
    """Types of robots."""
    WHEELED = auto()
    QUADRUPED = auto()
    ARM = auto()
    DRONE = auto()
    HUMANOID = auto()


@dataclass
class RobotState:
    """Robot state information."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # Quaternion
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)
    battery_level: float = 1.0  # 0-1
    is_active: bool = True


@dataclass
class SensorReading:
    """Generic sensor reading."""
    sensor_name: str
    timestamp: float
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class Robot(ABC):
    """
    Abstract base class for all robots.
    
    Provides common interface for robot control, state management,
    and sensor integration.
    """
    
    def __init__(self, name: str, robot_type: RobotType,
                 physics_world=None, renderer=None):
        """
        Initialize robot.
        
        Args:
            name: Robot name/ID
            robot_type: Type of robot
            physics_world: Reference to physics simulation
            renderer: Reference to renderer for visualization
        """
        self.name = name
        self.robot_type = robot_type
        self._physics = physics_world
        self._renderer = renderer
        
        # State
        self._state = RobotState()
        
        # Physics body reference
        self._physics_body = None
        self._joints: Dict[str, Any] = {}
        
        # Visual node
        self._visual_node = None
        
        # Sensors
        self._sensors: Dict[str, 'Sensor'] = {}
        
        # Control mode
        self._control_enabled = True
        
        logger.info(f"Robot '{name}' ({robot_type.name}) created")
    
    @property
    def state(self) -> RobotState:
        """Get current robot state."""
        return self._state
    
    @property
    def position(self) -> np.ndarray:
        """Get robot position."""
        return self._state.position.copy()
    
    @position.setter
    def position(self, value: Tuple[float, float, float]):
        """Set robot position."""
        self._state.position = np.array(value)
        if self._physics_body:
            self._physics_body.position = value
        if self._visual_node:
            self._visual_node.setPos(*value)
    
    @property
    def orientation(self) -> np.ndarray:
        """Get robot orientation (quaternion)."""
        return self._state.orientation.copy()
    
    @property
    def heading(self) -> float:
        """Get robot heading (yaw) in degrees."""
        # Extract yaw from quaternion
        q = self._state.orientation
        siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1])
        cosy_cosp = 1 - 2 * (q[1]**2 + q[2]**2)
        return np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    
    @property
    def velocity(self) -> np.ndarray:
        """Get linear velocity."""
        return self._state.linear_velocity.copy()
    
    @property
    def speed(self) -> float:
        """Get speed (magnitude of velocity)."""
        return np.linalg.norm(self._state.linear_velocity)
    
    def add_sensor(self, name: str, sensor: 'Sensor'):
        """Add a sensor to the robot."""
        sensor.attach_to_robot(self)
        self._sensors[name] = sensor
        logger.debug(f"Added sensor '{name}' to robot '{self.name}'")
    
    def get_sensor(self, name: str) -> Optional['Sensor']:
        """Get a sensor by name."""
        return self._sensors.get(name)
    
    def remove_sensor(self, name: str):
        """Remove a sensor."""
        if name in self._sensors:
            del self._sensors[name]
    
    def read_sensor(self, name: str) -> Optional[SensorReading]:
        """Read data from a sensor."""
        sensor = self._sensors.get(name)
        if sensor:
            return sensor.read()
        return None
    
    def read_all_sensors(self) -> Dict[str, SensorReading]:
        """Read data from all sensors."""
        readings = {}
        for name, sensor in self._sensors.items():
            readings[name] = sensor.read()
        return readings
    
    @abstractmethod
    def spawn(self, position: Tuple[float, float, float],
              orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        """
        Spawn the robot in the world.
        
        Args:
            position: Initial position (x, y, z)
            orientation: Initial orientation (quaternion)
        """
        pass
    
    @abstractmethod
    def update(self, dt: float):
        """
        Update robot state.
        
        Args:
            dt: Time since last update in seconds
        """
        pass
    
    @abstractmethod
    def apply_control(self, control: Dict[str, float]):
        """
        Apply control commands to the robot.
        
        Args:
            control: Dictionary of control values (specific to robot type)
        """
        pass
    
    def sync_state_from_physics(self):
        """Sync state from physics simulation."""
        if self._physics_body:
            self._state.position = self._physics_body.position
            self._state.orientation = self._physics_body.orientation
            self._state.linear_velocity = self._physics_body.linear_velocity
            self._state.angular_velocity = self._physics_body.angular_velocity
        
        # Sync joints
        for name, joint in self._joints.items():
            self._state.joint_positions[name] = joint.position
            self._state.joint_velocities[name] = joint.velocity
    
    def sync_visuals_from_physics(self):
        """Sync visual representation from physics."""
        if self._visual_node and self._physics_body:
            pos = self._physics_body.position
            orn = self._physics_body.orientation
            
            self._visual_node.setPos(*pos)
            # Convert quaternion to HPR for Panda3D
            # Simplified - may need proper conversion
            euler = self._physics_body.euler_angles
            self._visual_node.setHpr(
                np.degrees(euler[2]),  # Heading (yaw)
                np.degrees(euler[1]),  # Pitch
                np.degrees(euler[0])   # Roll
            )
    
    def enable_control(self):
        """Enable robot control."""
        self._control_enabled = True
    
    def disable_control(self):
        """Disable robot control (robot will coast)."""
        self._control_enabled = False
    
    def reset(self, position: Optional[Tuple[float, float, float]] = None):
        """
        Reset robot to initial state.
        
        Args:
            position: Optional new starting position
        """
        if position:
            self.position = position
        
        # Reset velocities
        if self._physics_body:
            self._physics_body.linear_velocity = (0, 0, 0)
            self._physics_body.angular_velocity = (0, 0, 0)
        
        # Reset joints
        for joint in self._joints.values():
            joint.reset(0.0, 0.0)
        
        # Reset state
        self._state = RobotState(position=np.array(position) if position else np.zeros(3))
    
    def destroy(self):
        """Remove the robot from the simulation."""
        if self._physics_body:
            self._physics_body.remove()
            self._physics_body = None
        
        if self._visual_node:
            self._visual_node.removeNode()
            self._visual_node = None
        
        self._sensors.clear()
        self._joints.clear()
        
        logger.info(f"Robot '{self.name}' destroyed")
    
    def get_transform_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        pos = self._state.position
        q = self._state.orientation
        
        # Quaternion to rotation matrix
        x, y, z, w = q
        rot = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        # Build 4x4 transform
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        
        return transform


class Sensor(ABC):
    """
    Abstract base class for robot sensors.
    """
    
    def __init__(self, name: str, update_rate: float = 30.0):
        """
        Initialize sensor.
        
        Args:
            name: Sensor name
            update_rate: Sensor update rate in Hz
        """
        self.name = name
        self.update_rate = update_rate
        self._robot: Optional[Robot] = None
        self._last_update: float = 0
        self._last_reading: Optional[SensorReading] = None
    
    def attach_to_robot(self, robot: Robot):
        """Attach sensor to a robot."""
        self._robot = robot
    
    @abstractmethod
    def read(self) -> SensorReading:
        """
        Read sensor data.
        
        Returns:
            SensorReading with current data
        """
        pass
    
    @property
    def attached_robot(self) -> Optional[Robot]:
        """Get the robot this sensor is attached to."""
        return self._robot


class RobotManager:
    """
    Manages all robots in the simulation.
    """
    
    def __init__(self, physics_world=None, renderer=None):
        """
        Initialize robot manager.
        
        Args:
            physics_world: Reference to physics simulation
            renderer: Reference to renderer
        """
        self._physics = physics_world
        self._renderer = renderer
        self._robots: Dict[str, Robot] = {}
        self._active_robot: Optional[Robot] = None
        
        logger.info("RobotManager initialized")
    
    def add_robot(self, robot: Robot):
        """Add a robot to the manager."""
        self._robots[robot.name] = robot
        if self._active_robot is None:
            self._active_robot = robot
    
    def remove_robot(self, name: str):
        """Remove a robot from the manager."""
        if name in self._robots:
            robot = self._robots[name]
            robot.destroy()
            del self._robots[name]
            
            if self._active_robot == robot:
                self._active_robot = next(iter(self._robots.values()), None)
    
    def get_robot(self, name: str) -> Optional[Robot]:
        """Get a robot by name."""
        return self._robots.get(name)
    
    @property
    def active_robot(self) -> Optional[Robot]:
        """Get the currently active (controlled) robot."""
        return self._active_robot
    
    def set_active_robot(self, name: str):
        """Set the active robot by name."""
        if name in self._robots:
            self._active_robot = self._robots[name]
    
    def update(self, dt: float):
        """Update all robots."""
        for robot in self._robots.values():
            robot.update(dt)
    
    def get_all_robots(self) -> List[Robot]:
        """Get all robots."""
        return list(self._robots.values())
    
    def cleanup(self):
        """Clean up all robots."""
        for robot in list(self._robots.values()):
            robot.destroy()
        self._robots.clear()
        self._active_robot = None
