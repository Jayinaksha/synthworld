"""
SynthWorld Camera Sensors

RGB and depth camera sensors for robots.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time
import logging

from ..base import Sensor, SensorReading

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    width: int
    height: int
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    
    def to_matrix(self) -> np.ndarray:
        """Get 3x3 camera matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    @classmethod
    def from_fov(cls, width: int, height: int, fov_degrees: float) -> 'CameraIntrinsics':
        """Create intrinsics from field of view."""
        fov_rad = np.radians(fov_degrees)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2
        return cls(width, height, fx, fy, cx, cy)


class CameraSensor(Sensor):
    """
    RGB camera sensor.
    """
    
    def __init__(self, name: str, 
                 width: int = 640, height: int = 480,
                 fov: float = 60.0,
                 near: float = 0.1, far: float = 100.0,
                 position: Tuple[float, float, float] = (0, 0, 0),
                 orientation: Tuple[float, float, float] = (0, 0, 0),
                 update_rate: float = 30.0,
                 noise_stddev: float = 0.0):
        """
        Initialize camera sensor.
        
        Args:
            name: Sensor name
            width: Image width in pixels
            height: Image height in pixels
            fov: Field of view in degrees
            near: Near clipping plane
            far: Far clipping plane
            position: Position offset from robot base
            orientation: Orientation offset (roll, pitch, yaw) in degrees
            update_rate: Update rate in Hz
            noise_stddev: Gaussian noise standard deviation (0-255)
        """
        super().__init__(name, update_rate)
        
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.noise_stddev = noise_stddev
        
        # Compute intrinsics
        self.intrinsics = CameraIntrinsics.from_fov(width, height, fov)
        
        # Last captured image
        self._last_image: Optional[np.ndarray] = None
        self._last_capture_time: float = 0
        
        # Reference to physics world for rendering
        self._physics_world = None
    
    def set_physics_world(self, physics_world):
        """Set reference to physics world for image rendering."""
        self._physics_world = physics_world
    
    def read(self) -> SensorReading:
        """Capture and return RGB image."""
        timestamp = time.time()
        
        # Check update rate
        if timestamp - self._last_capture_time < 1.0 / self.update_rate:
            # Return cached reading
            if self._last_reading:
                return self._last_reading
        
        # Capture image
        image = self._capture_image()
        
        # Add noise if configured
        if self.noise_stddev > 0:
            noise = np.random.normal(0, self.noise_stddev, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        self._last_image = image
        self._last_capture_time = timestamp
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=timestamp,
            data=image,
            metadata={
                'width': self.width,
                'height': self.height,
                'fov': self.fov,
                'intrinsics': self.intrinsics.to_matrix().tolist(),
                'encoding': 'rgb8'
            }
        )
        
        self._last_reading = reading
        return reading
    
    def _capture_image(self) -> np.ndarray:
        """Capture image from physics simulation."""
        if not self._robot or not self._physics_world:
            # Return placeholder
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Get camera pose in world frame
        robot_pos = self._robot.position
        robot_orn = self._robot.orientation
        
        # Camera position (simplified - should use proper transform)
        cam_pos = robot_pos + self.position
        
        # Look direction (forward from robot + camera orientation)
        heading = np.radians(self._robot.heading + self.orientation[2])
        pitch = np.radians(self.orientation[1])
        
        look_dist = 10.0
        target_pos = (
            cam_pos[0] + look_dist * np.sin(heading) * np.cos(pitch),
            cam_pos[1] + look_dist * np.cos(heading) * np.cos(pitch),
            cam_pos[2] + look_dist * np.sin(pitch)
        )
        
        # Compute view and projection matrices
        view_matrix = self._physics_world.compute_view_matrix(
            tuple(cam_pos), target_pos, (0, 0, 1)
        )
        
        projection_matrix = self._physics_world.compute_projection_matrix(
            self.fov, self.width / self.height, self.near, self.far
        )
        
        # Render image
        rgb, depth, seg = self._physics_world.get_camera_image(
            self.width, self.height, view_matrix, projection_matrix
        )
        
        return rgb
    
    def get_last_image(self) -> Optional[np.ndarray]:
        """Get last captured image."""
        return self._last_image
    
    def get_intrinsics(self) -> CameraIntrinsics:
        """Get camera intrinsics."""
        return self.intrinsics


class DepthCamera(CameraSensor):
    """
    Depth camera sensor.
    """
    
    def __init__(self, name: str, 
                 width: int = 640, height: int = 480,
                 fov: float = 60.0,
                 near: float = 0.1, far: float = 10.0,
                 **kwargs):
        super().__init__(name, width, height, fov, near, far, **kwargs)
        
        self._last_depth: Optional[np.ndarray] = None
    
    def read(self) -> SensorReading:
        """Capture and return depth image."""
        timestamp = time.time()
        
        if timestamp - self._last_capture_time < 1.0 / self.update_rate:
            if self._last_reading:
                return self._last_reading
        
        depth = self._capture_depth()
        
        self._last_depth = depth
        self._last_capture_time = timestamp
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=timestamp,
            data=depth,
            metadata={
                'width': self.width,
                'height': self.height,
                'near': self.near,
                'far': self.far,
                'encoding': 'depth32f',
                'unit': 'meters'
            }
        )
        
        self._last_reading = reading
        return reading
    
    def _capture_depth(self) -> np.ndarray:
        """Capture depth image from physics simulation."""
        if not self._robot or not self._physics_world:
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        robot_pos = self._robot.position
        cam_pos = robot_pos + self.position
        
        heading = np.radians(self._robot.heading + self.orientation[2])
        pitch = np.radians(self.orientation[1])
        
        look_dist = 10.0
        target_pos = (
            cam_pos[0] + look_dist * np.sin(heading) * np.cos(pitch),
            cam_pos[1] + look_dist * np.cos(heading) * np.cos(pitch),
            cam_pos[2] + look_dist * np.sin(pitch)
        )
        
        view_matrix = self._physics_world.compute_view_matrix(
            tuple(cam_pos), target_pos, (0, 0, 1)
        )
        
        projection_matrix = self._physics_world.compute_projection_matrix(
            self.fov, self.width / self.height, self.near, self.far
        )
        
        rgb, depth, seg = self._physics_world.get_camera_image(
            self.width, self.height, view_matrix, projection_matrix
        )
        
        # Convert depth buffer to meters
        # PyBullet returns normalized depth, convert to actual distance
        depth_meters = self.far * self.near / (self.far - (self.far - self.near) * depth)
        
        return depth_meters.astype(np.float32)
    
    def get_point_cloud(self) -> np.ndarray:
        """
        Convert last depth image to point cloud.
        
        Returns:
            Point cloud as Nx3 array
        """
        if self._last_depth is None:
            return np.zeros((0, 3))
        
        depth = self._last_depth
        K = self.intrinsics
        
        # Create meshgrid of pixel coordinates
        u = np.arange(K.width)
        v = np.arange(K.height)
        u, v = np.meshgrid(u, v)
        
        # Convert to 3D points
        z = depth
        x = (u - K.cx) * z / K.fx
        y = (v - K.cy) * z / K.fy
        
        # Stack and reshape
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)
        
        # Filter invalid points
        valid = (points[:, 2] > 0) & (points[:, 2] < self.far)
        points = points[valid]
        
        return points


class RGBDCamera(Sensor):
    """
    Combined RGB-D camera sensor.
    """
    
    def __init__(self, name: str,
                 width: int = 640, height: int = 480,
                 fov: float = 60.0,
                 near: float = 0.1, far: float = 10.0,
                 **kwargs):
        super().__init__(name, kwargs.get('update_rate', 30.0))
        
        # Create component sensors
        self.rgb = CameraSensor(f"{name}_rgb", width, height, fov, near, far, **kwargs)
        self.depth = DepthCamera(f"{name}_depth", width, height, fov, near, far, **kwargs)
        
        self.width = width
        self.height = height
    
    def attach_to_robot(self, robot):
        """Attach sensor to robot."""
        super().attach_to_robot(robot)
        self.rgb.attach_to_robot(robot)
        self.depth.attach_to_robot(robot)
    
    def set_physics_world(self, physics_world):
        """Set physics world reference."""
        self.rgb.set_physics_world(physics_world)
        self.depth.set_physics_world(physics_world)
    
    def read(self) -> SensorReading:
        """Capture aligned RGB-D data."""
        timestamp = time.time()
        
        rgb_reading = self.rgb.read()
        depth_reading = self.depth.read()
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=timestamp,
            data={
                'rgb': rgb_reading.data,
                'depth': depth_reading.data
            },
            metadata={
                'width': self.width,
                'height': self.height,
                'rgb_encoding': 'rgb8',
                'depth_encoding': 'depth32f',
                'intrinsics': self.rgb.intrinsics.to_matrix().tolist()
            }
        )
        
        self._last_reading = reading
        return reading
    
    def get_colored_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get colored point cloud from last capture.
        
        Returns:
            Tuple of (points Nx3, colors Nx3)
        """
        if not self.rgb._last_image or self.depth._last_depth is None:
            return np.zeros((0, 3)), np.zeros((0, 3))
        
        depth = self.depth._last_depth
        rgb = self.rgb._last_image
        K = self.rgb.intrinsics
        
        u = np.arange(K.width)
        v = np.arange(K.height)
        u, v = np.meshgrid(u, v)
        
        z = depth
        x = (u - K.cx) * z / K.fx
        y = (v - K.cy) * z / K.fy
        
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb.reshape(-1, 3) / 255.0
        
        valid = (points[:, 2] > 0) & (points[:, 2] < self.depth.far)
        
        return points[valid], colors[valid]


class SemanticCamera(CameraSensor):
    """
    Semantic segmentation camera.
    """
    
    def __init__(self, name: str, 
                 width: int = 640, height: int = 480,
                 fov: float = 60.0,
                 **kwargs):
        super().__init__(name, width, height, fov, **kwargs)
        
        # Class labels
        self.class_labels = {
            0: 'background',
            1: 'ground',
            2: 'building',
            3: 'vehicle',
            4: 'robot',
            5: 'person',
            6: 'vegetation',
            7: 'road',
            8: 'sign',
            9: 'other'
        }
        
        # Class colors for visualization
        self.class_colors = {
            0: (0, 0, 0),
            1: (128, 128, 128),
            2: (70, 70, 70),
            3: (0, 0, 142),
            4: (220, 20, 60),
            5: (255, 0, 0),
            6: (107, 142, 35),
            7: (128, 64, 128),
            8: (220, 220, 0),
            9: (119, 11, 32)
        }
        
        self._last_segmentation: Optional[np.ndarray] = None
    
    def read(self) -> SensorReading:
        """Capture semantic segmentation."""
        timestamp = time.time()
        
        if timestamp - self._last_capture_time < 1.0 / self.update_rate:
            if self._last_reading:
                return self._last_reading
        
        segmentation = self._capture_segmentation()
        
        self._last_segmentation = segmentation
        self._last_capture_time = timestamp
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=timestamp,
            data=segmentation,
            metadata={
                'width': self.width,
                'height': self.height,
                'encoding': 'semantic_id',
                'class_labels': self.class_labels
            }
        )
        
        self._last_reading = reading
        return reading
    
    def _capture_segmentation(self) -> np.ndarray:
        """Capture segmentation mask from physics simulation."""
        if not self._robot or not self._physics_world:
            return np.zeros((self.height, self.width), dtype=np.int32)
        
        robot_pos = self._robot.position
        cam_pos = robot_pos + self.position
        
        heading = np.radians(self._robot.heading + self.orientation[2])
        look_dist = 10.0
        target_pos = (
            cam_pos[0] + look_dist * np.sin(heading),
            cam_pos[1] + look_dist * np.cos(heading),
            cam_pos[2]
        )
        
        view_matrix = self._physics_world.compute_view_matrix(
            tuple(cam_pos), target_pos, (0, 0, 1)
        )
        
        projection_matrix = self._physics_world.compute_projection_matrix(
            self.fov, self.width / self.height, self.near, self.far
        )
        
        rgb, depth, seg = self._physics_world.get_camera_image(
            self.width, self.height, view_matrix, projection_matrix
        )
        
        return seg
    
    def get_colored_segmentation(self) -> np.ndarray:
        """Get colored visualization of segmentation."""
        if self._last_segmentation is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        colored = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            mask = self._last_segmentation == class_id
            colored[mask] = color
        
        return colored
