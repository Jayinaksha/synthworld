"""
SynthWorld Annotation Generation

Generate ground truth annotations for synthetic data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class ObjectCategory(Enum):
    """Categories for object detection."""
    PERSON = 1
    VEHICLE = 2
    ROBOT = 3
    BUILDING = 4
    VEGETATION = 5
    ROAD = 6
    SIGN = 7
    TRAFFIC_LIGHT = 8
    OBSTACLE = 9
    OTHER = 10


@dataclass
class BoundingBox:
    """Base bounding box."""
    class_id: int
    class_name: str
    instance_id: int
    confidence: float = 1.0  # Ground truth = 1.0


@dataclass
class BoundingBox2D(BoundingBox):
    """2D bounding box in image coordinates."""
    x_min: int = 0
    y_min: int = 0
    x_max: int = 0
    y_max: int = 0
    
    @property
    def width(self) -> int:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> int:
        return self.y_max - self.y_min
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_dict(self) -> Dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'instance_id': self.instance_id,
            'x_min': self.x_min,
            'y_min': self.y_min,
            'x_max': self.x_max,
            'y_max': self.y_max,
            'width': self.width,
            'height': self.height
        }
    
    def to_yolo(self, image_width: int, image_height: int) -> Tuple[int, float, float, float, float]:
        """Convert to YOLO format (class, x_center, y_center, width, height) normalized."""
        x_center = (self.x_min + self.x_max) / 2 / image_width
        y_center = (self.y_min + self.y_max) / 2 / image_height
        w = self.width / image_width
        h = self.height / image_height
        return (self.class_id, x_center, y_center, w, h)


@dataclass
class BoundingBox3D(BoundingBox):
    """3D bounding box in world coordinates."""
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dimensions: np.ndarray = field(default_factory=lambda: np.ones(3))  # length, width, height
    rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # quaternion
    
    @property
    def corners(self) -> np.ndarray:
        """Get 8 corner points of the box."""
        l, w, h = self.dimensions
        
        # Local corners
        corners_local = np.array([
            [-l/2, -w/2, -h/2],
            [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [l/2, -w/2, h/2],
            [l/2, w/2, h/2],
            [-l/2, w/2, h/2],
        ])
        
        # Rotate and translate
        # Simplified - should use proper quaternion rotation
        return corners_local + self.center
    
    def to_dict(self) -> Dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'instance_id': self.instance_id,
            'center': self.center.tolist(),
            'dimensions': self.dimensions.tolist(),
            'rotation': self.rotation.tolist()
        }


@dataclass
class Annotation:
    """Complete annotation for a frame."""
    frame_id: int
    timestamp: float
    
    boxes_2d: List[BoundingBox2D] = field(default_factory=list)
    boxes_3d: List[BoundingBox3D] = field(default_factory=list)
    
    # Semantic segmentation (class ID per pixel)
    semantic_mask: Optional[np.ndarray] = None
    
    # Instance segmentation (instance ID per pixel)
    instance_mask: Optional[np.ndarray] = None
    
    # Depth
    depth_map: Optional[np.ndarray] = None
    
    # Keypoints (e.g., for pose estimation)
    keypoints: List[Dict] = field(default_factory=list)
    
    # Scene flow (motion vectors)
    scene_flow: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'boxes_2d': [b.to_dict() for b in self.boxes_2d],
            'boxes_3d': [b.to_dict() for b in self.boxes_3d],
            'keypoints': self.keypoints
        }


class AnnotationGenerator:
    """
    Generates ground truth annotations from simulation state.
    """
    
    # Class ID mapping
    CLASS_MAPPING = {
        'person': ObjectCategory.PERSON.value,
        'pedestrian': ObjectCategory.PERSON.value,
        'civilian': ObjectCategory.PERSON.value,
        'vehicle': ObjectCategory.VEHICLE.value,
        'car': ObjectCategory.VEHICLE.value,
        'truck': ObjectCategory.VEHICLE.value,
        'robot': ObjectCategory.ROBOT.value,
        'building': ObjectCategory.BUILDING.value,
        'tree': ObjectCategory.VEGETATION.value,
        'road': ObjectCategory.ROAD.value,
        'sign': ObjectCategory.SIGN.value,
    }
    
    def __init__(self, physics_world=None, npc_manager=None, robot_manager=None):
        """
        Initialize annotation generator.
        
        Args:
            physics_world: Reference to physics simulation
            npc_manager: Reference to NPC manager
            robot_manager: Reference to robot manager
        """
        self._physics = physics_world
        self._npc_manager = npc_manager
        self._robot_manager = robot_manager
        
        # Instance ID counter
        self._instance_counter = 0
        self._instance_map: Dict[str, int] = {}  # name -> instance_id
    
    def generate_2d_boxes(self, camera_position: np.ndarray,
                         camera_orientation: np.ndarray,
                         camera_intrinsics: np.ndarray,
                         image_size: Tuple[int, int]) -> List[Dict]:
        """
        Generate 2D bounding boxes for all visible objects.
        
        Args:
            camera_position: Camera position in world coordinates
            camera_orientation: Camera orientation (quaternion)
            camera_intrinsics: 3x3 camera matrix
            image_size: (width, height) of image
        
        Returns:
            List of 2D bounding box dictionaries
        """
        boxes = []
        
        if camera_position is None or camera_intrinsics is None:
            return boxes
        
        width, height = image_size
        
        # Get all entities to annotate
        entities = self._get_all_entities()
        
        for entity in entities:
            # Project 3D bounding box to 2D
            box_3d = entity.get('box_3d')
            if box_3d is None:
                continue
            
            # Get 3D corners
            corners_3d = self._get_box_corners(box_3d)
            
            # Project corners to image plane
            corners_2d = self._project_points(
                corners_3d, camera_position, camera_orientation, camera_intrinsics
            )
            
            if corners_2d is None or len(corners_2d) == 0:
                continue
            
            # Get 2D bounding box from projected corners
            x_coords = corners_2d[:, 0]
            y_coords = corners_2d[:, 1]
            
            x_min = max(0, int(np.min(x_coords)))
            y_min = max(0, int(np.min(y_coords)))
            x_max = min(width, int(np.max(x_coords)))
            y_max = min(height, int(np.max(y_coords)))
            
            # Check if box is valid and visible
            if x_max <= x_min or y_max <= y_min:
                continue
            
            if x_max < 0 or y_max < 0 or x_min >= width or y_min >= height:
                continue
            
            # Check visibility (occlusion would require ray casting)
            visibility = self._compute_visibility(entity, camera_position)
            
            if visibility < 0.1:
                continue
            
            # Get or create instance ID
            entity_name = entity.get('name', 'unknown')
            if entity_name not in self._instance_map:
                self._instance_map[entity_name] = self._instance_counter
                self._instance_counter += 1
            
            box = BoundingBox2D(
                class_id=entity.get('class_id', ObjectCategory.OTHER.value),
                class_name=entity.get('class_name', 'other'),
                instance_id=self._instance_map[entity_name],
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max
            )
            
            box_dict = box.to_dict()
            box_dict['visibility'] = visibility
            box_dict['truncated'] = x_min == 0 or y_min == 0 or x_max == width or y_max == height
            boxes.append(box_dict)
        
        return boxes
    
    def generate_3d_boxes(self) -> List[Dict]:
        """
        Generate 3D bounding boxes for all objects.
        
        Returns:
            List of 3D bounding box dictionaries
        """
        boxes = []
        
        entities = self._get_all_entities()
        
        for entity in entities:
            box_3d = entity.get('box_3d')
            if box_3d is None:
                continue
            
            entity_name = entity.get('name', 'unknown')
            if entity_name not in self._instance_map:
                self._instance_map[entity_name] = self._instance_counter
                self._instance_counter += 1
            
            box = BoundingBox3D(
                class_id=entity.get('class_id', ObjectCategory.OTHER.value),
                class_name=entity.get('class_name', 'other'),
                instance_id=self._instance_map[entity_name],
                center=np.array(box_3d['center']),
                dimensions=np.array(box_3d['dimensions']),
                rotation=np.array(box_3d.get('rotation', [0, 0, 0, 1]))
            )
            
            box_dict = box.to_dict()
            box_dict['velocity'] = entity.get('velocity', [0, 0, 0])
            boxes.append(box_dict)
        
        return boxes
    
    def _get_all_entities(self) -> List[Dict]:
        """Get all annotatable entities from simulation."""
        entities = []
        
        # Get NPCs
        if self._npc_manager:
            for npc in self._npc_manager.get_all_npcs():
                # Determine class
                class_name = 'other'
                if hasattr(npc, 'npc_type'):
                    from ..npcs.base import NPCType
                    if npc.npc_type == NPCType.PEDESTRIAN:
                        class_name = 'person'
                    elif npc.npc_type == NPCType.VEHICLE:
                        class_name = 'vehicle'
                
                class_id = self.CLASS_MAPPING.get(class_name, ObjectCategory.OTHER.value)
                
                # Get bounding box
                position = npc.position
                
                # Estimate dimensions based on type
                if class_name == 'person':
                    dimensions = [0.5, 0.5, 1.8]
                elif class_name == 'vehicle':
                    dimensions = [4.0, 2.0, 1.5]
                else:
                    dimensions = [1.0, 1.0, 1.0]
                
                entities.append({
                    'name': npc.name,
                    'class_id': class_id,
                    'class_name': class_name,
                    'position': position,
                    'velocity': npc._velocity.tolist() if hasattr(npc, '_velocity') else [0, 0, 0],
                    'box_3d': {
                        'center': position.tolist(),
                        'dimensions': dimensions,
                        'rotation': npc._rotation.tolist() if hasattr(npc, '_rotation') else [0, 0, 0, 1]
                    }
                })
        
        # Get robots (other than the observing robot)
        if self._robot_manager:
            for robot in self._robot_manager.get_all_robots():
                class_id = ObjectCategory.ROBOT.value
                
                position = robot.position
                dimensions = [1.0, 1.0, 0.5]  # Default robot size
                
                entities.append({
                    'name': robot.name,
                    'class_id': class_id,
                    'class_name': 'robot',
                    'position': position,
                    'velocity': robot.velocity.tolist(),
                    'box_3d': {
                        'center': position.tolist(),
                        'dimensions': dimensions,
                        'rotation': robot.orientation.tolist()
                    }
                })
        
        return entities
    
    def _get_box_corners(self, box_3d: Dict) -> np.ndarray:
        """Get 8 corners of a 3D bounding box."""
        center = np.array(box_3d['center'])
        l, w, h = box_3d['dimensions']
        
        corners = np.array([
            [-l/2, -w/2, -h/2],
            [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [l/2, -w/2, h/2],
            [l/2, w/2, h/2],
            [-l/2, w/2, h/2],
        ])
        
        # TODO: Apply rotation
        
        return corners + center
    
    def _project_points(self, points_3d: np.ndarray,
                       camera_pos: np.ndarray,
                       camera_orn: np.ndarray,
                       K: np.ndarray) -> Optional[np.ndarray]:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: Nx3 array of 3D points
            camera_pos: Camera position
            camera_orn: Camera orientation (quaternion)
            K: 3x3 camera intrinsics matrix
        
        Returns:
            Nx2 array of 2D pixel coordinates
        """
        if points_3d is None or len(points_3d) == 0:
            return None
        
        # Transform to camera frame
        points_cam = points_3d - camera_pos
        
        # TODO: Apply camera rotation (quaternion to rotation matrix)
        
        # Filter points behind camera
        valid = points_cam[:, 2] > 0.1
        if not np.any(valid):
            return None
        
        points_cam = points_cam[valid]
        
        # Project
        points_2d = np.zeros((len(points_cam), 2))
        
        for i, p in enumerate(points_cam):
            if p[2] > 0:
                points_2d[i, 0] = K[0, 0] * p[0] / p[2] + K[0, 2]
                points_2d[i, 1] = K[1, 1] * p[1] / p[2] + K[1, 2]
        
        return points_2d
    
    def _compute_visibility(self, entity: Dict, camera_pos: np.ndarray) -> float:
        """
        Compute visibility score for an entity.
        
        Args:
            entity: Entity dictionary
            camera_pos: Camera position
        
        Returns:
            Visibility score (0-1)
        """
        if self._physics is None:
            return 1.0
        
        # Cast ray from camera to entity center
        entity_pos = np.array(entity.get('position', [0, 0, 0]))
        
        hit = self._physics.ray_cast(
            tuple(camera_pos + np.array([0, 0, 0.5])),
            tuple(entity_pos + np.array([0, 0, 0.5]))
        )
        
        if hit and hit['body_id'] != -1:
            # Check if hit object is the target
            return 0.5  # Partially occluded
        
        return 1.0  # Fully visible
    
    def get_segmentation_labels(self) -> Dict[int, str]:
        """Get mapping of segmentation IDs to class names."""
        labels = {}
        for name, class_id in self.CLASS_MAPPING.items():
            cat = ObjectCategory(class_id)
            labels[class_id] = cat.name.lower()
        return labels
