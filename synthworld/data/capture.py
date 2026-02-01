"""
SynthWorld Data Capture

Capture simulation data for synthetic dataset generation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Data captured for a single frame."""
    frame_id: int
    timestamp: float
    
    # Images
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    segmentation_image: Optional[np.ndarray] = None
    instance_image: Optional[np.ndarray] = None
    
    # Camera info
    camera_position: Optional[np.ndarray] = None
    camera_orientation: Optional[np.ndarray] = None
    camera_intrinsics: Optional[np.ndarray] = None
    camera_extrinsics: Optional[np.ndarray] = None
    
    # Point cloud
    point_cloud: Optional[np.ndarray] = None
    point_colors: Optional[np.ndarray] = None
    
    # Annotations
    bounding_boxes_2d: List[Dict] = field(default_factory=list)
    bounding_boxes_3d: List[Dict] = field(default_factory=list)
    segmentation_labels: Dict[int, str] = field(default_factory=dict)
    
    # Robot data
    robot_position: Optional[np.ndarray] = None
    robot_orientation: Optional[np.ndarray] = None
    robot_velocity: Optional[np.ndarray] = None
    robot_joint_states: Dict[str, float] = field(default_factory=dict)
    
    # Sensor data
    lidar_points: Optional[np.ndarray] = None
    imu_data: Optional[Dict] = None
    
    # Environment
    time_of_day: float = 12.0  # Hour
    weather: str = "clear"
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureConfig:
    """Configuration for data capture."""
    capture_rgb: bool = True
    capture_depth: bool = True
    capture_segmentation: bool = True
    capture_instance: bool = False
    capture_lidar: bool = True
    capture_imu: bool = True
    capture_pointcloud: bool = False
    
    image_width: int = 640
    image_height: int = 480
    
    auto_annotate: bool = True
    save_to_disk: bool = True
    output_path: str = "./synthetic_data"
    
    capture_interval: float = 0.1  # Seconds between captures
    max_frames: int = -1  # -1 = unlimited


class DataCapture:
    """
    Captures simulation data for synthetic dataset generation.
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        """
        Initialize data capture.
        
        Args:
            config: Capture configuration
        """
        self.config = config or CaptureConfig()
        
        # Frame counter
        self._frame_count = 0
        self._last_capture_time = 0.0
        
        # Captured data buffer
        self._buffer: List[FrameData] = []
        self._buffer_size = 100
        
        # Session info
        self._session_id = f"session_{int(time.time())}"
        self._session_start = time.time()
        
        # References to simulation components
        self._physics_world = None
        self._renderer = None
        self._robot = None
        self._npc_manager = None
        
        # Annotation generator
        self._annotator: Optional['AnnotationGenerator'] = None
        
        # Is capturing
        self._is_capturing = False
        
        # Create output directory
        if self.config.save_to_disk:
            self._setup_output_directory()
        
        logger.info(f"DataCapture initialized (session: {self._session_id})")
    
    def _setup_output_directory(self):
        """Create output directory structure."""
        base_path = Path(self.config.output_path) / self._session_id
        
        (base_path / "rgb").mkdir(parents=True, exist_ok=True)
        (base_path / "depth").mkdir(parents=True, exist_ok=True)
        (base_path / "segmentation").mkdir(parents=True, exist_ok=True)
        (base_path / "annotations").mkdir(parents=True, exist_ok=True)
        (base_path / "lidar").mkdir(parents=True, exist_ok=True)
        (base_path / "metadata").mkdir(parents=True, exist_ok=True)
        
        self._output_path = base_path
        logger.info(f"Output directory: {base_path}")
    
    def set_references(self, physics_world=None, renderer=None,
                      robot=None, npc_manager=None):
        """Set references to simulation components."""
        self._physics_world = physics_world
        self._renderer = renderer
        self._robot = robot
        self._npc_manager = npc_manager
        
        # Create annotator
        from .annotations import AnnotationGenerator
        self._annotator = AnnotationGenerator(physics_world, npc_manager)
    
    def start_capture(self):
        """Start data capture."""
        self._is_capturing = True
        self._session_start = time.time()
        logger.info("Data capture started")
    
    def stop_capture(self):
        """Stop data capture."""
        self._is_capturing = False
        logger.info(f"Data capture stopped ({self._frame_count} frames captured)")
    
    def update(self, dt: float):
        """
        Called each frame to potentially capture data.
        
        Args:
            dt: Time since last frame
        """
        if not self._is_capturing:
            return
        
        current_time = time.time() - self._session_start
        
        # Check capture interval
        if current_time - self._last_capture_time < self.config.capture_interval:
            return
        
        # Check max frames
        if self.config.max_frames > 0 and self._frame_count >= self.config.max_frames:
            self.stop_capture()
            return
        
        # Capture frame
        frame_data = self.capture_frame()
        
        if frame_data:
            self._buffer.append(frame_data)
            self._frame_count += 1
            self._last_capture_time = current_time
            
            # Flush buffer if full
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()
    
    def capture_frame(self) -> Optional[FrameData]:
        """Capture a single frame of data."""
        timestamp = time.time() - self._session_start
        
        frame = FrameData(
            frame_id=self._frame_count,
            timestamp=timestamp
        )
        
        # Capture images from camera sensors
        if self._robot:
            for sensor_name in ['camera', 'rgb_camera', 'main_camera']:
                sensor = self._robot.get_sensor(sensor_name)
                if sensor:
                    self._capture_from_camera(frame, sensor)
                    break
            
            # Capture depth
            for sensor_name in ['depth_camera', 'rgbd_camera']:
                sensor = self._robot.get_sensor(sensor_name)
                if sensor and self.config.capture_depth:
                    self._capture_depth(frame, sensor)
                    break
            
            # Capture LiDAR
            if self.config.capture_lidar:
                lidar = self._robot.get_sensor('lidar')
                if lidar:
                    self._capture_lidar(frame, lidar)
            
            # Capture IMU
            if self.config.capture_imu:
                imu = self._robot.get_sensor('imu')
                if imu:
                    self._capture_imu(frame, imu)
            
            # Robot state
            frame.robot_position = self._robot.position
            frame.robot_orientation = self._robot.orientation
            frame.robot_velocity = self._robot.velocity
            frame.robot_joint_states = self._robot.state.joint_positions.copy()
        
        # Capture from renderer if no robot camera
        if frame.rgb_image is None and self._renderer:
            self._capture_from_renderer(frame)
        
        # Generate annotations
        if self.config.auto_annotate and self._annotator:
            self._generate_annotations(frame)
        
        # Add metadata
        frame.metadata = {
            'session_id': self._session_id,
            'simulation_time': timestamp,
            'real_time': time.time()
        }
        
        # Save to disk
        if self.config.save_to_disk:
            self._save_frame(frame)
        
        return frame
    
    def _capture_from_camera(self, frame: FrameData, sensor):
        """Capture RGB from camera sensor."""
        reading = sensor.read()
        if reading and reading.data is not None:
            if isinstance(reading.data, dict):
                # RGBD camera
                frame.rgb_image = reading.data.get('rgb')
                frame.depth_image = reading.data.get('depth')
            else:
                frame.rgb_image = reading.data
            
            if 'intrinsics' in reading.metadata:
                frame.camera_intrinsics = np.array(reading.metadata['intrinsics'])
    
    def _capture_depth(self, frame: FrameData, sensor):
        """Capture depth data."""
        reading = sensor.read()
        if reading and reading.data is not None:
            if isinstance(reading.data, dict):
                frame.depth_image = reading.data.get('depth')
            else:
                frame.depth_image = reading.data
    
    def _capture_lidar(self, frame: FrameData, sensor):
        """Capture LiDAR data."""
        reading = sensor.read()
        if reading and reading.data is not None:
            frame.lidar_points = reading.data.get('points')
    
    def _capture_imu(self, frame: FrameData, sensor):
        """Capture IMU data."""
        reading = sensor.read()
        if reading:
            frame.imu_data = {
                'linear_acceleration': reading.data.linear_acceleration.tolist(),
                'angular_velocity': reading.data.angular_velocity.tolist(),
                'orientation': reading.data.orientation.tolist()
            }
    
    def _capture_from_renderer(self, frame: FrameData):
        """Capture image from renderer."""
        if self._renderer:
            rgb = self._renderer.capture_frame(
                self.config.image_width,
                self.config.image_height
            )
            if rgb is not None:
                frame.rgb_image = rgb
    
    def _generate_annotations(self, frame: FrameData):
        """Generate annotations for the frame."""
        if not self._annotator:
            return
        
        # 2D bounding boxes
        if frame.rgb_image is not None:
            boxes_2d = self._annotator.generate_2d_boxes(
                frame.camera_position,
                frame.camera_orientation,
                frame.camera_intrinsics,
                (self.config.image_width, self.config.image_height)
            )
            frame.bounding_boxes_2d = boxes_2d
        
        # 3D bounding boxes
        boxes_3d = self._annotator.generate_3d_boxes()
        frame.bounding_boxes_3d = boxes_3d
        
        # Segmentation labels
        if frame.segmentation_image is not None:
            frame.segmentation_labels = self._annotator.get_segmentation_labels()
    
    def _save_frame(self, frame: FrameData):
        """Save frame data to disk."""
        if not hasattr(self, '_output_path'):
            return
        
        frame_id = f"{frame.frame_id:06d}"
        
        try:
            # Save RGB
            if frame.rgb_image is not None:
                import cv2
                rgb_path = self._output_path / "rgb" / f"{frame_id}.png"
                cv2.imwrite(str(rgb_path), cv2.cvtColor(frame.rgb_image, cv2.COLOR_RGB2BGR))
            
            # Save depth
            if frame.depth_image is not None:
                depth_path = self._output_path / "depth" / f"{frame_id}.npy"
                np.save(str(depth_path), frame.depth_image)
            
            # Save segmentation
            if frame.segmentation_image is not None:
                seg_path = self._output_path / "segmentation" / f"{frame_id}.png"
                import cv2
                cv2.imwrite(str(seg_path), frame.segmentation_image)
            
            # Save LiDAR
            if frame.lidar_points is not None:
                lidar_path = self._output_path / "lidar" / f"{frame_id}.npy"
                np.save(str(lidar_path), frame.lidar_points)
            
            # Save annotations
            annotations = {
                'frame_id': frame.frame_id,
                'timestamp': frame.timestamp,
                'bounding_boxes_2d': frame.bounding_boxes_2d,
                'bounding_boxes_3d': frame.bounding_boxes_3d,
                'robot_position': frame.robot_position.tolist() if frame.robot_position is not None else None,
                'robot_orientation': frame.robot_orientation.tolist() if frame.robot_orientation is not None else None,
            }
            
            ann_path = self._output_path / "annotations" / f"{frame_id}.json"
            with open(ann_path, 'w') as f:
                json.dump(annotations, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving frame {frame_id}: {e}")
    
    def _flush_buffer(self):
        """Flush capture buffer to disk."""
        self._buffer.clear()
    
    def get_captured_frames(self) -> List[FrameData]:
        """Get all captured frames in buffer."""
        return self._buffer.copy()
    
    def get_frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count
    
    def reset(self):
        """Reset capture state."""
        self._frame_count = 0
        self._last_capture_time = 0
        self._buffer.clear()
        self._session_id = f"session_{int(time.time())}"
        
        if self.config.save_to_disk:
            self._setup_output_directory()
