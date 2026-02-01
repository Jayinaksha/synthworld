"""
SynthWorld Renderer

Panda3D-based rendering system with CPU fallback support.
Provides scene management, camera control, lighting, and procedural geometry.
"""

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    WindowProperties, FrameBufferProperties, GraphicsPipe,
    NodePath, GeomNode, Geom, GeomVertexData, GeomVertexFormat,
    GeomVertexWriter, GeomTriangles, GeomLines,
    Vec3, Vec4, Point3, LColor, LVector3,
    AmbientLight, DirectionalLight, PointLight, Spotlight,
    Texture, TextureStage, PNMImage,
    Camera, Lens, PerspectiveLens, OrthographicLens,
    CollisionTraverser, CollisionNode, CollisionHandlerQueue,
    CollisionRay, CollisionSphere,
    loadPrcFileData
)
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

# Configure Panda3D for CPU rendering if no GPU
loadPrcFileData('', 'load-display pandagl')  # Try OpenGL first
loadPrcFileData('', 'aux-display p3tinydisplay')  # Fallback to software
loadPrcFileData('', 'win-size 1280 720')
loadPrcFileData('', 'window-title SynthWorld Simulator')
loadPrcFileData('', 'sync-video #t')
loadPrcFileData('', 'show-frame-rate-meter #t')


@dataclass
class CameraMode:
    """Camera mode configuration."""
    FIRST_PERSON = "first_person"
    THIRD_PERSON = "third_person"
    FREE_CAM = "free_cam"
    ORBIT = "orbit"


class ProceduralMesh:
    """
    Helper class for creating procedural geometry.
    """
    
    @staticmethod
    def create_box(width: float, height: float, depth: float,
                   color: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)) -> GeomNode:
        """Create a box geometry."""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('box', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_writer = GeomVertexWriter(vdata, 'color')
        
        hw, hh, hd = width/2, height/2, depth/2
        
        # Define vertices for each face
        faces = [
            # Front face
            [(-hw, -hh, hd), (hw, -hh, hd), (hw, hh, hd), (-hw, hh, hd), (0, 0, 1)],
            # Back face  
            [(hw, -hh, -hd), (-hw, -hh, -hd), (-hw, hh, -hd), (hw, hh, -hd), (0, 0, -1)],
            # Top face
            [(-hw, hh, hd), (hw, hh, hd), (hw, hh, -hd), (-hw, hh, -hd), (0, 1, 0)],
            # Bottom face
            [(-hw, -hh, -hd), (hw, -hh, -hd), (hw, -hh, hd), (-hw, -hh, hd), (0, -1, 0)],
            # Right face
            [(hw, -hh, hd), (hw, -hh, -hd), (hw, hh, -hd), (hw, hh, hd), (1, 0, 0)],
            # Left face
            [(-hw, -hh, -hd), (-hw, -hh, hd), (-hw, hh, hd), (-hw, hh, -hd), (-1, 0, 0)],
        ]
        
        for face in faces:
            norm = face[4]
            for i in range(4):
                v = face[i]
                vertex.addData3(*v)
                normal.addData3(*norm)
                color_writer.addData4(*color)
        
        # Create triangles
        tris = GeomTriangles(Geom.UHStatic)
        for face_idx in range(6):
            base = face_idx * 4
            tris.addVertices(base, base + 1, base + 2)
            tris.addVertices(base, base + 2, base + 3)
        
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        node = GeomNode('box')
        node.addGeom(geom)
        return node
    
    @staticmethod
    def create_sphere(radius: float, segments: int = 16, rings: int = 16,
                      color: Tuple[float, float, float, float] = (0.7, 0.3, 0.3, 1.0)) -> GeomNode:
        """Create a sphere geometry."""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('sphere', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_writer = GeomVertexWriter(vdata, 'color')
        
        # Generate vertices
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            for seg in range(segments + 1):
                theta = 2 * math.pi * seg / segments
                
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.sin(phi) * math.sin(theta)
                z = radius * math.cos(phi)
                
                nx, ny, nz = x/radius, y/radius, z/radius
                
                vertex.addData3(x, y, z)
                normal.addData3(nx, ny, nz)
                color_writer.addData4(*color)
        
        # Create triangles
        tris = GeomTriangles(Geom.UHStatic)
        for ring in range(rings):
            for seg in range(segments):
                curr = ring * (segments + 1) + seg
                next_row = curr + segments + 1
                
                tris.addVertices(curr, next_row, curr + 1)
                tris.addVertices(curr + 1, next_row, next_row + 1)
        
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        node = GeomNode('sphere')
        node.addGeom(geom)
        return node
    
    @staticmethod
    def create_cylinder(radius: float, height: float, segments: int = 16,
                        color: Tuple[float, float, float, float] = (0.3, 0.7, 0.3, 1.0)) -> GeomNode:
        """Create a cylinder geometry."""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('cylinder', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_writer = GeomVertexWriter(vdata, 'color')
        
        hh = height / 2
        
        # Side vertices
        for i in range(segments + 1):
            theta = 2 * math.pi * i / segments
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            nx, ny = math.cos(theta), math.sin(theta)
            
            # Bottom vertex
            vertex.addData3(x, y, -hh)
            normal.addData3(nx, ny, 0)
            color_writer.addData4(*color)
            
            # Top vertex
            vertex.addData3(x, y, hh)
            normal.addData3(nx, ny, 0)
            color_writer.addData4(*color)
        
        # Cap centers
        bottom_center_idx = (segments + 1) * 2
        vertex.addData3(0, 0, -hh)
        normal.addData3(0, 0, -1)
        color_writer.addData4(*color)
        
        top_center_idx = bottom_center_idx + 1
        vertex.addData3(0, 0, hh)
        normal.addData3(0, 0, 1)
        color_writer.addData4(*color)
        
        # Cap edge vertices
        for i in range(segments + 1):
            theta = 2 * math.pi * i / segments
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            
            # Bottom cap
            vertex.addData3(x, y, -hh)
            normal.addData3(0, 0, -1)
            color_writer.addData4(*color)
            
            # Top cap
            vertex.addData3(x, y, hh)
            normal.addData3(0, 0, 1)
            color_writer.addData4(*color)
        
        tris = GeomTriangles(Geom.UHStatic)
        
        # Side triangles
        for i in range(segments):
            base = i * 2
            tris.addVertices(base, base + 2, base + 1)
            tris.addVertices(base + 1, base + 2, base + 3)
        
        # Cap triangles
        cap_start = top_center_idx + 1
        for i in range(segments):
            # Bottom cap
            tris.addVertices(bottom_center_idx, cap_start + i * 2 + 2, cap_start + i * 2)
            # Top cap
            tris.addVertices(top_center_idx, cap_start + i * 2 + 1, cap_start + i * 2 + 3)
        
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        node = GeomNode('cylinder')
        node.addGeom(geom)
        return node
    
    @staticmethod
    def create_plane(width: float, depth: float, 
                     segments_x: int = 1, segments_z: int = 1,
                     color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)) -> GeomNode:
        """Create a flat plane geometry."""
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('plane', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_writer = GeomVertexWriter(vdata, 'color')
        
        hw, hd = width / 2, depth / 2
        
        for z in range(segments_z + 1):
            for x in range(segments_x + 1):
                px = -hw + (width * x / segments_x)
                pz = -hd + (depth * z / segments_z)
                
                vertex.addData3(px, pz, 0)
                normal.addData3(0, 0, 1)
                color_writer.addData4(*color)
        
        tris = GeomTriangles(Geom.UHStatic)
        for z in range(segments_z):
            for x in range(segments_x):
                curr = z * (segments_x + 1) + x
                tris.addVertices(curr, curr + segments_x + 1, curr + 1)
                tris.addVertices(curr + 1, curr + segments_x + 1, curr + segments_x + 2)
        
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        node = GeomNode('plane')
        node.addGeom(geom)
        return node
    
    @staticmethod
    def create_heightmap_mesh(heightmap: np.ndarray, scale: float = 1.0, 
                              height_scale: float = 1.0,
                              color_func=None) -> GeomNode:
        """
        Create a mesh from a 2D heightmap array.
        
        Args:
            heightmap: 2D numpy array of heights
            scale: Horizontal scale factor
            height_scale: Vertical scale factor
            color_func: Optional function(height) -> (r, g, b, a) for coloring
        """
        format = GeomVertexFormat.get_v3n3c4()
        vdata = GeomVertexData('terrain', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color_writer = GeomVertexWriter(vdata, 'color')
        
        rows, cols = heightmap.shape
        
        # Calculate normals by averaging adjacent face normals
        normals = np.zeros((rows, cols, 3))
        for z in range(rows):
            for x in range(cols):
                # Get neighboring heights
                h = heightmap[z, x]
                hL = heightmap[z, max(0, x-1)]
                hR = heightmap[z, min(cols-1, x+1)]
                hD = heightmap[max(0, z-1), x]
                hU = heightmap[min(rows-1, z+1), x]
                
                # Compute normal from central differences
                nx = (hL - hR) * height_scale
                ny = 2.0 * scale
                nz = (hD - hU) * height_scale
                
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                normals[z, x] = [nx/length, ny/length, nz/length]
        
        # Create vertices
        for z in range(rows):
            for x in range(cols):
                h = heightmap[z, x] * height_scale
                px = (x - cols/2) * scale
                py = (z - rows/2) * scale
                
                vertex.addData3(px, py, h)
                normal.addData3(*normals[z, x])
                
                if color_func:
                    color = color_func(h)
                else:
                    # Default gradient from green to brown based on height
                    t = min(1.0, max(0.0, h / (height_scale * 10)))
                    color = (0.3 + 0.4*t, 0.6 - 0.3*t, 0.2, 1.0)
                
                color_writer.addData4(*color)
        
        # Create triangles
        tris = GeomTriangles(Geom.UHStatic)
        for z in range(rows - 1):
            for x in range(cols - 1):
                curr = z * cols + x
                tris.addVertices(curr, curr + cols, curr + 1)
                tris.addVertices(curr + 1, curr + cols, curr + cols + 1)
        
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        node = GeomNode('terrain')
        node.addGeom(geom)
        return node


class CameraController:
    """
    Camera controller with multiple modes.
    """
    
    def __init__(self, camera: NodePath, render: NodePath):
        self.camera = camera
        self.render = render
        self.mode = CameraMode.FREE_CAM
        
        # Camera state
        self.position = Vec3(0, -20, 10)
        self.heading = 0  # Yaw
        self.pitch = -20  # Look down slightly
        
        # Target for third-person/orbit
        self.target: Optional[NodePath] = None
        self.target_offset = Vec3(0, 0, 2)  # Offset above target
        self.orbit_distance = 10
        self.orbit_height = 5
        
        # Movement settings
        self.move_speed = 20.0
        self.rotate_speed = 100.0
        self.zoom_speed = 5.0
        
        # Apply initial transform
        self._update_camera()
    
    def set_mode(self, mode: str):
        """Change camera mode."""
        self.mode = mode
        logger.info(f"Camera mode: {mode}")
    
    def set_target(self, target: NodePath):
        """Set target for third-person/orbit modes."""
        self.target = target
    
    def _update_camera(self):
        """Update camera position/orientation based on current mode."""
        if self.mode == CameraMode.FREE_CAM:
            self.camera.setPos(self.position)
            self.camera.setHpr(self.heading, self.pitch, 0)
            
        elif self.mode == CameraMode.THIRD_PERSON and self.target:
            target_pos = self.target.getPos() + self.target_offset
            
            # Position behind target
            heading_rad = math.radians(self.target.getH())
            offset = Vec3(
                -self.orbit_distance * math.sin(heading_rad),
                -self.orbit_distance * math.cos(heading_rad),
                self.orbit_height
            )
            
            self.camera.setPos(target_pos + offset)
            self.camera.lookAt(target_pos)
            
        elif self.mode == CameraMode.ORBIT and self.target:
            target_pos = self.target.getPos() + self.target_offset
            
            heading_rad = math.radians(self.heading)
            pitch_rad = math.radians(self.pitch)
            
            offset = Vec3(
                self.orbit_distance * math.cos(pitch_rad) * math.sin(heading_rad),
                -self.orbit_distance * math.cos(pitch_rad) * math.cos(heading_rad),
                self.orbit_distance * math.sin(pitch_rad)
            )
            
            self.camera.setPos(target_pos + offset)
            self.camera.lookAt(target_pos)
            
        elif self.mode == CameraMode.FIRST_PERSON and self.target:
            # Camera at target position + eye offset
            eye_offset = Vec3(0, 0, 1.7)  # Eye height
            self.camera.setPos(self.target.getPos() + eye_offset)
            self.camera.setHpr(self.target.getH(), self.pitch, 0)
    
    def move(self, forward: float, right: float, up: float, dt: float):
        """Move camera (free cam mode only)."""
        if self.mode != CameraMode.FREE_CAM:
            return
        
        heading_rad = math.radians(self.heading)
        
        # Calculate movement direction
        dx = (forward * math.sin(heading_rad) + right * math.cos(heading_rad)) * self.move_speed * dt
        dy = (forward * math.cos(heading_rad) - right * math.sin(heading_rad)) * self.move_speed * dt
        dz = up * self.move_speed * dt
        
        self.position += Vec3(dx, dy, dz)
        self._update_camera()
    
    def rotate(self, d_heading: float, d_pitch: float, dt: float):
        """Rotate camera."""
        self.heading += d_heading * self.rotate_speed * dt
        self.pitch = max(-89, min(89, self.pitch + d_pitch * self.rotate_speed * dt))
        self._update_camera()
    
    def zoom(self, amount: float, dt: float):
        """Zoom in/out (third-person/orbit modes)."""
        if self.mode in [CameraMode.THIRD_PERSON, CameraMode.ORBIT]:
            self.orbit_distance = max(2, self.orbit_distance - amount * self.zoom_speed * dt)
            self._update_camera()
    
    def update(self, dt: float):
        """Update camera (call every frame for modes that track targets)."""
        if self.mode in [CameraMode.THIRD_PERSON, CameraMode.FIRST_PERSON, CameraMode.ORBIT]:
            self._update_camera()


class Renderer(ShowBase):
    """
    Main rendering manager using Panda3D.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize renderer.
        
        Args:
            config: Display configuration dictionary
        """
        # Set window properties before init
        loadPrcFileData('', f'win-size {config.get("width", 1280)} {config.get("height", 720)}')
        loadPrcFileData('', f'fullscreen {"#t" if config.get("fullscreen", False) else "#f"}')
        loadPrcFileData('', f'sync-video {"#t" if config.get("vsync", True) else "#f"}')
        loadPrcFileData('', f'window-title {config.get("title", "SynthWorld")}')
        
        super().__init__()
        
        self.config = config
        self.target_fps = config.get('target_fps', 60)
        
        # Scene graph organization
        self.world_node = self.render.attachNewNode("world")
        self.entities_node = self.world_node.attachNewNode("entities")
        self.terrain_node = self.world_node.attachNewNode("terrain")
        self.buildings_node = self.world_node.attachNewNode("buildings")
        self.props_node = self.world_node.attachNewNode("props")
        
        # Camera setup
        self.camera_controller = CameraController(self.camera, self.render)
        
        # Lighting
        self._setup_lighting()
        
        # Day/night cycle
        self.time_of_day = 12.0  # 0-24 hours
        self.day_cycle_speed = 0.0  # Hours per second (0 = disabled)
        
        # Entity tracking
        self._entities: Dict[str, NodePath] = {}
        self._entity_counter = 0
        
        # Frame counter for data capture
        self.frame_count = 0
        
        # Disable default mouse control
        self.disableMouse()
        
        logger.info("Renderer initialized")
    
    def _setup_lighting(self):
        """Set up scene lighting."""
        # Ambient light
        self.ambient_light = AmbientLight('ambient')
        self.ambient_light.setColor(Vec4(0.3, 0.3, 0.35, 1))
        self.ambient_light_np = self.render.attachNewNode(self.ambient_light)
        self.render.setLight(self.ambient_light_np)
        
        # Sun (directional light)
        self.sun_light = DirectionalLight('sun')
        self.sun_light.setColor(Vec4(1.0, 0.95, 0.8, 1))
        self.sun_light_np = self.render.attachNewNode(self.sun_light)
        self.sun_light_np.setHpr(45, -45, 0)
        self.render.setLight(self.sun_light_np)
        
        # Enable shadows (basic)
        self.sun_light.setShadowCaster(True, 1024, 1024)
        self.render.setShaderAuto()
    
    def update_day_night(self, dt: float):
        """Update day/night cycle lighting."""
        if self.day_cycle_speed > 0:
            self.time_of_day = (self.time_of_day + self.day_cycle_speed * dt) % 24
        
        # Calculate sun position based on time
        sun_angle = (self.time_of_day / 24.0) * 360 - 90  # -90 at midnight, 90 at noon
        self.sun_light_np.setP(sun_angle)
        
        # Adjust colors based on time
        if 6 <= self.time_of_day <= 18:  # Day
            day_factor = 1.0 - abs(self.time_of_day - 12) / 6.0
            sun_intensity = 0.6 + 0.4 * day_factor
            ambient_intensity = 0.3 + 0.2 * day_factor
            
            # Warmer colors at sunrise/sunset
            if self.time_of_day < 8 or self.time_of_day > 16:
                self.sun_light.setColor(Vec4(1.0, 0.7, 0.4, 1) * sun_intensity)
            else:
                self.sun_light.setColor(Vec4(1.0, 0.95, 0.8, 1) * sun_intensity)
            
            self.ambient_light.setColor(Vec4(0.3, 0.3, 0.35, 1) * ambient_intensity)
        else:  # Night
            self.sun_light.setColor(Vec4(0.1, 0.1, 0.2, 1))
            self.ambient_light.setColor(Vec4(0.05, 0.05, 0.1, 1))
    
    def set_time_of_day(self, hour: float):
        """Set time of day (0-24)."""
        self.time_of_day = hour % 24
        self.update_day_night(0)
    
    def create_entity(self, geometry: GeomNode, name: str = "",
                      position: Tuple[float, float, float] = (0, 0, 0),
                      rotation: Tuple[float, float, float] = (0, 0, 0),
                      scale: Tuple[float, float, float] = (1, 1, 1),
                      parent: Optional[NodePath] = None) -> NodePath:
        """
        Create an entity in the scene.
        
        Args:
            geometry: GeomNode containing the geometry
            name: Optional name for the entity
            position: Initial position (x, y, z)
            rotation: Initial rotation in degrees (heading, pitch, roll)
            scale: Scale factors (x, y, z)
            parent: Parent node (default: entities_node)
        
        Returns:
            NodePath to the created entity
        """
        if not name:
            name = f"entity_{self._entity_counter}"
            self._entity_counter += 1
        
        parent = parent or self.entities_node
        node_path = parent.attachNewNode(geometry)
        node_path.setPos(*position)
        node_path.setHpr(*rotation)
        node_path.setScale(*scale)
        
        self._entities[name] = node_path
        return node_path
    
    def create_box(self, width: float, height: float, depth: float,
                   color: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
                   **kwargs) -> NodePath:
        """Create a box entity."""
        geom = ProceduralMesh.create_box(width, height, depth, color)
        return self.create_entity(geom, **kwargs)
    
    def create_sphere(self, radius: float, segments: int = 16,
                      color: Tuple[float, float, float, float] = (0.7, 0.3, 0.3, 1.0),
                      **kwargs) -> NodePath:
        """Create a sphere entity."""
        geom = ProceduralMesh.create_sphere(radius, segments, segments, color)
        return self.create_entity(geom, **kwargs)
    
    def create_cylinder(self, radius: float, height: float, segments: int = 16,
                        color: Tuple[float, float, float, float] = (0.3, 0.7, 0.3, 1.0),
                        **kwargs) -> NodePath:
        """Create a cylinder entity."""
        geom = ProceduralMesh.create_cylinder(radius, height, segments, color)
        return self.create_entity(geom, **kwargs)
    
    def create_plane(self, width: float, depth: float,
                     color: Tuple[float, float, float, float] = (0.4, 0.6, 0.4, 1.0),
                     **kwargs) -> NodePath:
        """Create a ground plane entity."""
        geom = ProceduralMesh.create_plane(width, depth, 1, 1, color)
        kwargs['parent'] = kwargs.get('parent', self.terrain_node)
        return self.create_entity(geom, **kwargs)
    
    def create_terrain(self, heightmap: np.ndarray, scale: float = 1.0,
                       height_scale: float = 1.0, color_func=None,
                       **kwargs) -> NodePath:
        """Create terrain from heightmap."""
        geom = ProceduralMesh.create_heightmap_mesh(heightmap, scale, height_scale, color_func)
        kwargs['parent'] = kwargs.get('parent', self.terrain_node)
        return self.create_entity(geom, **kwargs)
    
    def remove_entity(self, name_or_nodepath):
        """Remove an entity from the scene."""
        if isinstance(name_or_nodepath, str):
            if name_or_nodepath in self._entities:
                self._entities[name_or_nodepath].removeNode()
                del self._entities[name_or_nodepath]
        else:
            name_or_nodepath.removeNode()
            # Remove from dict if present
            for name, np in list(self._entities.items()):
                if np == name_or_nodepath:
                    del self._entities[name]
                    break
    
    def get_entity(self, name: str) -> Optional[NodePath]:
        """Get entity by name."""
        return self._entities.get(name)
    
    def capture_frame(self, width: Optional[int] = None, 
                      height: Optional[int] = None) -> np.ndarray:
        """
        Capture current frame as numpy array.
        
        Returns:
            RGB image as numpy array (height, width, 3)
        """
        width = width or self.win.getXSize()
        height = height or self.win.getYSize()
        
        # Get the display region texture
        tex = self.win.getScreenshot()
        if tex:
            # Convert to numpy
            img = PNMImage()
            tex.store(img)
            
            data = np.zeros((img.getYSize(), img.getXSize(), 3), dtype=np.uint8)
            for y in range(img.getYSize()):
                for x in range(img.getXSize()):
                    data[y, x, 0] = int(img.getRed(x, y) * 255)
                    data[y, x, 1] = int(img.getGreen(x, y) * 255)
                    data[y, x, 2] = int(img.getBlue(x, y) * 255)
            
            return data
        
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def render_to_texture(self, camera_pos: Tuple[float, float, float],
                          camera_target: Tuple[float, float, float],
                          width: int = 640, height: int = 480,
                          fov: float = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render scene from a specific camera position.
        
        Returns:
            Tuple of (RGB image, depth image)
        """
        # Create offscreen buffer
        fb_props = FrameBufferProperties()
        fb_props.setRgbColor(True)
        fb_props.setDepthBits(24)
        
        buffer = self.graphicsEngine.makeOutput(
            self.pipe, "offscreen", 0,
            fb_props, WindowProperties.size(width, height),
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(), self.win
        )
        
        if not buffer:
            logger.error("Failed to create offscreen buffer")
            return np.zeros((height, width, 3)), np.zeros((height, width))
        
        # Create camera for this buffer
        cam_lens = PerspectiveLens()
        cam_lens.setFov(fov)
        cam_lens.setAspectRatio(width / height)
        
        cam = Camera('sensor_cam')
        cam.setLens(cam_lens)
        cam_np = NodePath(cam)
        cam_np.reparentTo(self.render)
        cam_np.setPos(*camera_pos)
        cam_np.lookAt(Point3(*camera_target))
        
        # Create display region
        dr = buffer.makeDisplayRegion()
        dr.setCamera(cam_np)
        
        # Render one frame
        self.graphicsEngine.renderFrame()
        
        # Get textures
        rgb_tex = buffer.getTexture()
        
        # Extract RGB
        rgb_img = PNMImage()
        rgb_tex.store(rgb_img)
        
        rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                rgb_data[y, x, 0] = int(rgb_img.getRed(x, y) * 255)
                rgb_data[y, x, 1] = int(rgb_img.getGreen(x, y) * 255)
                rgb_data[y, x, 2] = int(rgb_img.getBlue(x, y) * 255)
        
        # Depth is more complex - for now return placeholder
        depth_data = np.zeros((height, width), dtype=np.float32)
        
        # Cleanup
        cam_np.removeNode()
        self.graphicsEngine.removeWindow(buffer)
        
        return rgb_data, depth_data
    
    def set_background_color(self, r: float, g: float, b: float):
        """Set the background/sky color."""
        self.setBackgroundColor(r, g, b, 1.0)
    
    def run_frame(self) -> bool:
        """
        Run a single frame.
        
        Returns:
            False if window was closed, True otherwise
        """
        self.frame_count += 1
        self.taskMgr.step()
        return not self.win.isClosed()
