"""
SynthWorld LiDAR Sensor

LiDAR (Light Detection and Ranging) sensor simulation.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import time
import math
import logging

from ..base import Sensor, SensorReading

logger = logging.getLogger(__name__)


@dataclass
class LidarPoint:
    """A single LiDAR point."""
    x: float
    y: float
    z: float
    intensity: float = 1.0
    ring: int = 0  # Vertical ring index
    time_offset: float = 0.0  # Time offset within scan


class LidarSensor(Sensor):
    """
    LiDAR sensor simulating ray-based range finding.
    
    Supports both 2D (planar) and 3D scanning patterns.
    """
    
    def __init__(self, name: str,
                 horizontal_fov: float = 360.0,
                 vertical_fov: float = 30.0,
                 horizontal_resolution: float = 1.0,
                 vertical_resolution: float = 2.0,
                 min_range: float = 0.1,
                 max_range: float = 30.0,
                 position: Tuple[float, float, float] = (0, 0, 0.5),
                 orientation: Tuple[float, float, float] = (0, 0, 0),
                 update_rate: float = 10.0,
                 noise_stddev: float = 0.01):
        """
        Initialize LiDAR sensor.
        
        Args:
            name: Sensor name
            horizontal_fov: Horizontal field of view in degrees
            vertical_fov: Vertical field of view in degrees
            horizontal_resolution: Horizontal angular resolution in degrees
            vertical_resolution: Vertical angular resolution in degrees
            min_range: Minimum detection range in meters
            max_range: Maximum detection range in meters
            position: Position offset from robot base
            orientation: Orientation offset (roll, pitch, yaw) in degrees
            update_rate: Update rate in Hz
            noise_stddev: Range noise standard deviation in meters
        """
        super().__init__(name, update_rate)
        
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.horizontal_resolution = horizontal_resolution
        self.vertical_resolution = vertical_resolution
        self.min_range = min_range
        self.max_range = max_range
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.noise_stddev = noise_stddev
        
        # Calculate number of rays
        self.num_horizontal = int(horizontal_fov / horizontal_resolution)
        self.num_vertical = max(1, int(vertical_fov / vertical_resolution))
        
        # Pre-compute ray directions
        self._ray_directions = self._compute_ray_directions()
        
        # Last scan data
        self._last_points: Optional[np.ndarray] = None
        self._last_ranges: Optional[np.ndarray] = None
        self._last_capture_time: float = 0
        
        # Physics world reference
        self._physics_world = None
        
        logger.info(f"LidarSensor '{name}' initialized: "
                   f"{self.num_horizontal}x{self.num_vertical} rays, "
                   f"range {min_range}-{max_range}m")
    
    def _compute_ray_directions(self) -> np.ndarray:
        """Pre-compute ray directions in sensor frame."""
        directions = []
        
        # Horizontal angles
        h_start = -self.horizontal_fov / 2
        h_angles = np.linspace(h_start, h_start + self.horizontal_fov, 
                              self.num_horizontal, endpoint=False)
        
        # Vertical angles
        if self.num_vertical > 1:
            v_start = -self.vertical_fov / 2
            v_angles = np.linspace(v_start, v_start + self.vertical_fov,
                                  self.num_vertical)
        else:
            v_angles = [0.0]
        
        for v_angle in v_angles:
            v_rad = np.radians(v_angle)
            for h_angle in h_angles:
                h_rad = np.radians(h_angle)
                
                # Direction vector
                x = np.cos(v_rad) * np.sin(h_rad)
                y = np.cos(v_rad) * np.cos(h_rad)
                z = np.sin(v_rad)
                
                directions.append([x, y, z])
        
        return np.array(directions)
    
    def set_physics_world(self, physics_world):
        """Set physics world reference for ray casting."""
        self._physics_world = physics_world
    
    def read(self) -> SensorReading:
        """Perform LiDAR scan and return point cloud."""
        timestamp = time.time()
        
        # Check update rate
        if timestamp - self._last_capture_time < 1.0 / self.update_rate:
            if self._last_reading:
                return self._last_reading
        
        # Perform scan
        points, ranges, intensities = self._perform_scan()
        
        # Add noise
        if self.noise_stddev > 0 and len(ranges) > 0:
            noise = np.random.normal(0, self.noise_stddev, ranges.shape)
            ranges = np.clip(ranges + noise, self.min_range, self.max_range)
            # Recompute points with noisy ranges
            points = self._ranges_to_points(ranges)
        
        self._last_points = points
        self._last_ranges = ranges
        self._last_capture_time = timestamp
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=timestamp,
            data={
                'points': points,
                'ranges': ranges,
                'intensities': intensities
            },
            metadata={
                'num_points': len(points),
                'horizontal_fov': self.horizontal_fov,
                'vertical_fov': self.vertical_fov,
                'min_range': self.min_range,
                'max_range': self.max_range,
                'frame_id': 'lidar_frame'
            }
        )
        
        self._last_reading = reading
        return reading
    
    def _perform_scan(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform ray casting scan."""
        if not self._robot or not self._physics_world:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
        
        # Get sensor pose in world frame
        robot_pos = self._robot.position
        robot_heading = np.radians(self._robot.heading)
        
        sensor_pos = robot_pos + self.position
        
        # Rotate ray directions to world frame
        # Simplified rotation (just yaw)
        cos_h = np.cos(robot_heading + np.radians(self.orientation[2]))
        sin_h = np.sin(robot_heading + np.radians(self.orientation[2]))
        
        rotation = np.array([
            [cos_h, -sin_h, 0],
            [sin_h, cos_h, 0],
            [0, 0, 1]
        ])
        
        world_directions = self._ray_directions @ rotation.T
        
        # Compute ray endpoints
        from_positions = np.tile(sensor_pos, (len(world_directions), 1))
        to_positions = from_positions + world_directions * self.max_range
        
        # Batch ray cast
        hits = self._physics_world.ray_cast_batch(
            from_positions.tolist(),
            to_positions.tolist()
        )
        
        # Process results
        points = []
        ranges = []
        intensities = []
        
        for i, hit in enumerate(hits):
            if hit is not None:
                hit_pos = np.array(hit['hit_position'])
                distance = np.linalg.norm(hit_pos - sensor_pos)
                
                if self.min_range <= distance <= self.max_range:
                    # Transform to sensor frame
                    local_point = rotation.T @ (hit_pos - sensor_pos)
                    points.append(local_point)
                    ranges.append(distance)
                    intensities.append(1.0 - hit['hit_fraction'])  # Simplified intensity
        
        return np.array(points), np.array(ranges), np.array(intensities)
    
    def _ranges_to_points(self, ranges: np.ndarray) -> np.ndarray:
        """Convert ranges back to points using ray directions."""
        valid_count = min(len(ranges), len(self._ray_directions))
        points = self._ray_directions[:valid_count] * ranges[:, np.newaxis]
        return points
    
    def get_point_cloud(self) -> np.ndarray:
        """Get last point cloud."""
        return self._last_points if self._last_points is not None else np.zeros((0, 3))
    
    def get_2d_scan(self, ring: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 2D planar scan data.
        
        Args:
            ring: Which vertical ring to extract (-1 = middle)
        
        Returns:
            Tuple of (angles in radians, ranges in meters)
        """
        if self._last_ranges is None:
            return np.zeros(0), np.zeros(0)
        
        if ring < 0:
            ring = self.num_vertical // 2
        
        # Extract ring
        start_idx = ring * self.num_horizontal
        end_idx = start_idx + self.num_horizontal
        
        if end_idx > len(self._last_ranges):
            return np.zeros(0), np.zeros(0)
        
        ranges = self._last_ranges[start_idx:end_idx]
        
        h_start = -self.horizontal_fov / 2
        angles = np.linspace(h_start, h_start + self.horizontal_fov,
                            self.num_horizontal, endpoint=False)
        angles = np.radians(angles)
        
        return angles, ranges


class Lidar2D(LidarSensor):
    """
    2D/planar LiDAR sensor.
    """
    
    def __init__(self, name: str,
                 fov: float = 360.0,
                 resolution: float = 1.0,
                 min_range: float = 0.1,
                 max_range: float = 30.0,
                 **kwargs):
        super().__init__(
            name,
            horizontal_fov=fov,
            vertical_fov=0,
            horizontal_resolution=resolution,
            vertical_resolution=1,
            min_range=min_range,
            max_range=max_range,
            **kwargs
        )


class VelodyneLidar(LidarSensor):
    """
    Velodyne-style rotating 3D LiDAR.
    """
    
    def __init__(self, name: str, 
                 model: str = 'VLP-16',
                 **kwargs):
        """
        Initialize Velodyne-style LiDAR.
        
        Args:
            name: Sensor name
            model: Model name ('VLP-16', 'VLP-32', 'HDL-64E')
        """
        # Preset configurations for common models
        configs = {
            'VLP-16': {
                'horizontal_fov': 360.0,
                'vertical_fov': 30.0,
                'horizontal_resolution': 0.2,
                'vertical_resolution': 2.0,
                'max_range': 100.0,
                'update_rate': 10.0
            },
            'VLP-32': {
                'horizontal_fov': 360.0,
                'vertical_fov': 40.0,
                'horizontal_resolution': 0.2,
                'vertical_resolution': 1.25,
                'max_range': 200.0,
                'update_rate': 10.0
            },
            'HDL-64E': {
                'horizontal_fov': 360.0,
                'vertical_fov': 26.8,
                'horizontal_resolution': 0.08,
                'vertical_resolution': 0.4,
                'max_range': 120.0,
                'update_rate': 10.0
            }
        }
        
        config = configs.get(model, configs['VLP-16'])
        config.update(kwargs)
        
        super().__init__(name, **config)
        
        self.model = model
        logger.info(f"VelodyneLidar '{name}' ({model}) initialized")
