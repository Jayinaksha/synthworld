"""
SynthWorld Data Export

Export synthetic data to various formats (COCO, KITTI, etc.).
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataExporter:
    """
    Base class for data exporters.
    """
    
    def __init__(self, output_path: str):
        """
        Initialize exporter.
        
        Args:
            output_path: Base output directory
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def export(self, data: List[Any], filename: str = None) -> str:
        """
        Export data.
        
        Args:
            data: Data to export
            filename: Output filename
        
        Returns:
            Path to exported file
        """
        raise NotImplementedError


class COCOExporter(DataExporter):
    """
    Export data in COCO format.
    
    COCO format structure:
    {
        "info": {...},
        "licenses": [...],
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }
    """
    
    CATEGORIES = [
        {"id": 1, "name": "person", "supercategory": "human"},
        {"id": 2, "name": "vehicle", "supercategory": "vehicle"},
        {"id": 3, "name": "robot", "supercategory": "machine"},
        {"id": 4, "name": "building", "supercategory": "structure"},
        {"id": 5, "name": "vegetation", "supercategory": "nature"},
        {"id": 6, "name": "road", "supercategory": "infrastructure"},
        {"id": 7, "name": "sign", "supercategory": "object"},
        {"id": 8, "name": "traffic_light", "supercategory": "object"},
        {"id": 9, "name": "obstacle", "supercategory": "object"},
        {"id": 10, "name": "other", "supercategory": "misc"},
    ]
    
    def __init__(self, output_path: str, dataset_name: str = "synthworld"):
        super().__init__(output_path)
        
        self.dataset_name = dataset_name
        
        # Create subdirectories
        (self.output_path / "images").mkdir(exist_ok=True)
        (self.output_path / "annotations").mkdir(exist_ok=True)
        
        # Counters
        self._image_id = 0
        self._annotation_id = 0
        
        # Accumulated data
        self._images = []
        self._annotations = []
    
    def add_frame(self, frame_data: 'FrameData', image_filename: str):
        """
        Add a frame to the COCO dataset.
        
        Args:
            frame_data: Frame data object
            image_filename: Filename of the saved image
        """
        self._image_id += 1
        
        # Add image entry
        image_entry = {
            "id": self._image_id,
            "file_name": image_filename,
            "width": frame_data.rgb_image.shape[1] if frame_data.rgb_image is not None else 640,
            "height": frame_data.rgb_image.shape[0] if frame_data.rgb_image is not None else 480,
            "date_captured": datetime.now().isoformat(),
            "license": 1
        }
        self._images.append(image_entry)
        
        # Add annotations
        for box in frame_data.bounding_boxes_2d:
            self._annotation_id += 1
            
            x_min = box.get('x_min', 0)
            y_min = box.get('y_min', 0)
            width = box.get('width', 0)
            height = box.get('height', 0)
            
            annotation = {
                "id": self._annotation_id,
                "image_id": self._image_id,
                "category_id": box.get('class_id', 10),
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []  # Could add polygon segmentation
            }
            self._annotations.append(annotation)
    
    def export(self, data: List[Any] = None, filename: str = "annotations.json") -> str:
        """Export accumulated data to COCO format JSON."""
        
        coco_data = {
            "info": {
                "description": f"SynthWorld Synthetic Dataset - {self.dataset_name}",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "SynthWorld Generator",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Synthetic Data License",
                    "url": ""
                }
            ],
            "images": self._images,
            "annotations": self._annotations,
            "categories": self.CATEGORIES
        }
        
        output_file = self.output_path / "annotations" / filename
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"Exported COCO annotations to {output_file}")
        logger.info(f"  Images: {len(self._images)}, Annotations: {len(self._annotations)}")
        
        return str(output_file)
    
    def reset(self):
        """Reset accumulated data."""
        self._images = []
        self._annotations = []
        self._image_id = 0
        self._annotation_id = 0


class KITTIExporter(DataExporter):
    """
    Export data in KITTI format.
    
    KITTI format:
    - images: PNG files in image_2/
    - labels: TXT files in label_2/
    - calibration: TXT files in calib/
    
    Label format (per line):
    type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y [score]
    """
    
    def __init__(self, output_path: str):
        super().__init__(output_path)
        
        # Create KITTI directory structure
        (self.output_path / "image_2").mkdir(exist_ok=True)
        (self.output_path / "label_2").mkdir(exist_ok=True)
        (self.output_path / "velodyne").mkdir(exist_ok=True)
        (self.output_path / "calib").mkdir(exist_ok=True)
        
        self._frame_count = 0
    
    def add_frame(self, frame_data: 'FrameData'):
        """
        Add a frame in KITTI format.
        
        Args:
            frame_data: Frame data object
        """
        frame_id = f"{self._frame_count:06d}"
        self._frame_count += 1
        
        # Save image
        if frame_data.rgb_image is not None:
            try:
                import cv2
                img_path = self.output_path / "image_2" / f"{frame_id}.png"
                cv2.imwrite(str(img_path), cv2.cvtColor(frame_data.rgb_image, cv2.COLOR_RGB2BGR))
            except ImportError:
                logger.warning("OpenCV not available, skipping image save")
        
        # Save labels
        self._save_labels(frame_id, frame_data)
        
        # Save velodyne (LiDAR)
        if frame_data.lidar_points is not None:
            self._save_velodyne(frame_id, frame_data.lidar_points)
        
        # Save calibration
        self._save_calibration(frame_id, frame_data)
    
    def _save_labels(self, frame_id: str, frame_data: 'FrameData'):
        """Save labels in KITTI format."""
        label_path = self.output_path / "label_2" / f"{frame_id}.txt"
        
        lines = []
        
        for box_3d in frame_data.bounding_boxes_3d:
            # Get 2D box if available
            box_2d = None
            for b2d in frame_data.bounding_boxes_2d:
                if b2d.get('instance_id') == box_3d.get('instance_id'):
                    box_2d = b2d
                    break
            
            # Type
            class_name = box_3d.get('class_name', 'DontCare')
            if class_name == 'person':
                kitti_type = 'Pedestrian'
            elif class_name == 'vehicle':
                kitti_type = 'Car'
            else:
                kitti_type = 'DontCare'
            
            # Truncation and occlusion
            truncated = box_2d.get('truncated', False) if box_2d else False
            truncated = 1.0 if truncated else 0.0
            occluded = 0  # 0=fully visible, 1=partly occluded, 2=largely occluded, 3=unknown
            
            # Alpha (observation angle)
            alpha = 0.0
            
            # 2D bbox
            if box_2d:
                bbox = [box_2d.get('x_min', 0), box_2d.get('y_min', 0),
                       box_2d.get('x_max', 0), box_2d.get('y_max', 0)]
            else:
                bbox = [0, 0, 0, 0]
            
            # 3D dimensions (h, w, l in KITTI)
            dims = box_3d.get('dimensions', [1, 1, 1])
            height, width, length = dims[2], dims[1], dims[0]
            
            # 3D location (camera coordinates)
            center = box_3d.get('center', [0, 0, 0])
            x, y, z = center[0], center[1], center[2]
            
            # Rotation around Y-axis
            rotation = box_3d.get('rotation', [0, 0, 0, 1])
            rotation_y = 0.0  # Would need to extract from quaternion
            
            # Format line
            line = f"{kitti_type} {truncated:.2f} {occluded} {alpha:.2f} "
            line += f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "
            line += f"{height:.2f} {width:.2f} {length:.2f} "
            line += f"{x:.2f} {y:.2f} {z:.2f} {rotation_y:.2f}"
            
            lines.append(line)
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def _save_velodyne(self, frame_id: str, points: np.ndarray):
        """Save LiDAR points in KITTI velodyne format."""
        # KITTI velodyne format: N x 4 (x, y, z, intensity) float32
        velodyne_path = self.output_path / "velodyne" / f"{frame_id}.bin"
        
        if points.shape[1] == 3:
            # Add intensity column
            intensity = np.ones((len(points), 1), dtype=np.float32)
            points = np.hstack([points, intensity])
        
        points.astype(np.float32).tofile(str(velodyne_path))
    
    def _save_calibration(self, frame_id: str, frame_data: 'FrameData'):
        """Save camera calibration in KITTI format."""
        calib_path = self.output_path / "calib" / f"{frame_id}.txt"
        
        # Get intrinsics
        if frame_data.camera_intrinsics is not None:
            K = frame_data.camera_intrinsics
        else:
            # Default intrinsics
            K = np.array([
                [721.5377, 0, 320],
                [0, 721.5377, 240],
                [0, 0, 1]
            ])
        
        # Create projection matrices
        # P0-P3: 3x4 projection matrices for cameras
        P = np.zeros((3, 4))
        P[:3, :3] = K
        
        # R0_rect: Rectification matrix (identity for synthetic)
        R0 = np.eye(3)
        
        # Tr_velo_to_cam: Velodyne to camera transformation
        Tr = np.eye(4)[:3, :]
        
        # Write calibration file
        with open(calib_path, 'w') as f:
            f.write(f"P0: {' '.join(map(str, P.flatten()))}\n")
            f.write(f"P1: {' '.join(map(str, P.flatten()))}\n")
            f.write(f"P2: {' '.join(map(str, P.flatten()))}\n")
            f.write(f"P3: {' '.join(map(str, P.flatten()))}\n")
            f.write(f"R0_rect: {' '.join(map(str, R0.flatten()))}\n")
            f.write(f"Tr_velo_to_cam: {' '.join(map(str, Tr.flatten()))}\n")
            f.write(f"Tr_imu_to_velo: {' '.join(map(str, Tr.flatten()))}\n")
    
    def export(self, data: List[Any] = None, filename: str = None) -> str:
        """Export is handled per-frame in add_frame."""
        logger.info(f"KITTI dataset exported to {self.output_path}")
        logger.info(f"  Frames: {self._frame_count}")
        return str(self.output_path)


class YOLOExporter(DataExporter):
    """
    Export data in YOLO format.
    
    YOLO format:
    - images in images/train/ or images/val/
    - labels in labels/train/ or labels/val/
    - data.yaml with class information
    
    Label format (per line):
    class_id x_center y_center width height (all normalized 0-1)
    """
    
    def __init__(self, output_path: str, class_names: List[str] = None):
        super().__init__(output_path)
        
        self.class_names = class_names or [
            'person', 'vehicle', 'robot', 'building', 'vegetation',
            'road', 'sign', 'traffic_light', 'obstacle', 'other'
        ]
        
        # Create directory structure
        for split in ['train', 'val']:
            (self.output_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        self._frame_count = {'train': 0, 'val': 0}
    
    def add_frame(self, frame_data: 'FrameData', split: str = 'train'):
        """
        Add a frame in YOLO format.
        
        Args:
            frame_data: Frame data object
            split: 'train' or 'val'
        """
        frame_id = f"{self._frame_count[split]:06d}"
        self._frame_count[split] += 1
        
        # Save image
        if frame_data.rgb_image is not None:
            try:
                import cv2
                img_path = self.output_path / "images" / split / f"{frame_id}.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(frame_data.rgb_image, cv2.COLOR_RGB2BGR))
            except ImportError:
                pass
        
        # Save labels
        self._save_labels(frame_id, split, frame_data)
    
    def _save_labels(self, frame_id: str, split: str, frame_data: 'FrameData'):
        """Save labels in YOLO format."""
        label_path = self.output_path / "labels" / split / f"{frame_id}.txt"
        
        lines = []
        
        img_h = frame_data.rgb_image.shape[0] if frame_data.rgb_image is not None else 480
        img_w = frame_data.rgb_image.shape[1] if frame_data.rgb_image is not None else 640
        
        for box in frame_data.bounding_boxes_2d:
            class_id = box.get('class_id', 10) - 1  # YOLO classes are 0-indexed
            
            x_min = box.get('x_min', 0)
            y_min = box.get('y_min', 0)
            x_max = box.get('x_max', 0)
            y_max = box.get('y_max', 0)
            
            # Convert to YOLO format (normalized center, width, height)
            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h
            
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def export(self, data: List[Any] = None, filename: str = "data.yaml") -> str:
        """Export dataset configuration."""
        config = {
            'path': str(self.output_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        # Write YAML
        config_path = self.output_path / filename
        with open(config_path, 'w') as f:
            for key, value in config.items():
                if isinstance(value, list):
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"YOLO dataset exported to {self.output_path}")
        logger.info(f"  Train: {self._frame_count['train']}, Val: {self._frame_count['val']}")
        
        return str(config_path)
