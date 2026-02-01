"""
SynthWorld Physics Engine

PyBullet-based physics simulation with game-friendly wrapper classes.
Provides rigid body dynamics, collision detection, and robot simulation.
"""

import pybullet as p
import pybullet_data
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CollisionInfo:
    """Information about a collision between two bodies."""
    body_a: int
    body_b: int
    contact_point: Tuple[float, float, float]
    contact_normal: Tuple[float, float, float]
    contact_distance: float
    normal_force: float


@dataclass
class RigidBodyConfig:
    """Configuration for creating a rigid body."""
    mass: float = 1.0
    position: Tuple[float, float, float] = (0, 0, 0)
    orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)  # quaternion
    linear_damping: float = 0.04
    angular_damping: float = 0.04
    friction: float = 0.5
    restitution: float = 0.0
    collision_group: int = 1
    collision_mask: int = -1  # collide with all


class RigidBody:
    """
    Wrapper for a PyBullet rigid body with convenient interface.
    """
    
    def __init__(self, physics_world: 'PhysicsWorld', body_id: int, 
                 name: str = "", is_static: bool = False):
        self._world = physics_world
        self._body_id = body_id
        self._name = name
        self._is_static = is_static
        self._user_data: Dict[str, Any] = {}
    
    @property
    def body_id(self) -> int:
        return self._body_id
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_static(self) -> bool:
        return self._is_static
    
    @property
    def position(self) -> np.ndarray:
        """Get current position as numpy array [x, y, z]."""
        pos, _ = p.getBasePositionAndOrientation(
            self._body_id, physicsClientId=self._world.client_id
        )
        return np.array(pos)
    
    @position.setter
    def position(self, value: Tuple[float, float, float]):
        """Set position."""
        _, orn = p.getBasePositionAndOrientation(
            self._body_id, physicsClientId=self._world.client_id
        )
        p.resetBasePositionAndOrientation(
            self._body_id, value, orn,
            physicsClientId=self._world.client_id
        )
    
    @property
    def orientation(self) -> np.ndarray:
        """Get current orientation as quaternion [x, y, z, w]."""
        _, orn = p.getBasePositionAndOrientation(
            self._body_id, physicsClientId=self._world.client_id
        )
        return np.array(orn)
    
    @orientation.setter
    def orientation(self, value: Tuple[float, float, float, float]):
        """Set orientation quaternion."""
        pos, _ = p.getBasePositionAndOrientation(
            self._body_id, physicsClientId=self._world.client_id
        )
        p.resetBasePositionAndOrientation(
            self._body_id, pos, value,
            physicsClientId=self._world.client_id
        )
    
    @property
    def euler_angles(self) -> np.ndarray:
        """Get orientation as Euler angles [roll, pitch, yaw] in radians."""
        return np.array(p.getEulerFromQuaternion(self.orientation))
    
    @property
    def linear_velocity(self) -> np.ndarray:
        """Get linear velocity [vx, vy, vz]."""
        vel, _ = p.getBaseVelocity(
            self._body_id, physicsClientId=self._world.client_id
        )
        return np.array(vel)
    
    @linear_velocity.setter
    def linear_velocity(self, value: Tuple[float, float, float]):
        """Set linear velocity."""
        _, ang_vel = p.getBaseVelocity(
            self._body_id, physicsClientId=self._world.client_id
        )
        p.resetBaseVelocity(
            self._body_id, value, ang_vel,
            physicsClientId=self._world.client_id
        )
    
    @property
    def angular_velocity(self) -> np.ndarray:
        """Get angular velocity [wx, wy, wz]."""
        _, ang_vel = p.getBaseVelocity(
            self._body_id, physicsClientId=self._world.client_id
        )
        return np.array(ang_vel)
    
    @angular_velocity.setter
    def angular_velocity(self, value: Tuple[float, float, float]):
        """Set angular velocity."""
        lin_vel, _ = p.getBaseVelocity(
            self._body_id, physicsClientId=self._world.client_id
        )
        p.resetBaseVelocity(
            self._body_id, lin_vel, value,
            physicsClientId=self._world.client_id
        )
    
    def apply_force(self, force: Tuple[float, float, float], 
                    position: Optional[Tuple[float, float, float]] = None):
        """Apply force at a position (world coordinates). If position is None, applies at center of mass."""
        if position is None:
            position = self.position
        p.applyExternalForce(
            self._body_id, -1, force, position, p.WORLD_FRAME,
            physicsClientId=self._world.client_id
        )
    
    def apply_torque(self, torque: Tuple[float, float, float]):
        """Apply torque [tx, ty, tz]."""
        p.applyExternalTorque(
            self._body_id, -1, torque, p.WORLD_FRAME,
            physicsClientId=self._world.client_id
        )
    
    def get_contacts(self) -> List[CollisionInfo]:
        """Get all contact points for this body."""
        contacts = []
        contact_points = p.getContactPoints(
            bodyA=self._body_id,
            physicsClientId=self._world.client_id
        )
        for cp in contact_points:
            contacts.append(CollisionInfo(
                body_a=cp[1],
                body_b=cp[2],
                contact_point=cp[5],
                contact_normal=cp[7],
                contact_distance=cp[8],
                normal_force=cp[9]
            ))
        return contacts
    
    def set_color(self, rgba: Tuple[float, float, float, float]):
        """Set visual color of the body."""
        p.changeVisualShape(
            self._body_id, -1, rgbaColor=rgba,
            physicsClientId=self._world.client_id
        )
    
    def remove(self):
        """Remove this body from the simulation."""
        self._world.remove_body(self)


class Joint:
    """
    Wrapper for a PyBullet joint with convenient interface.
    """
    
    def __init__(self, physics_world: 'PhysicsWorld', body_id: int, 
                 joint_index: int, joint_info: tuple):
        self._world = physics_world
        self._body_id = body_id
        self._joint_index = joint_index
        self._info = joint_info
        
        # Parse joint info
        self._name = joint_info[1].decode('utf-8')
        self._type = joint_info[2]
        self._lower_limit = joint_info[8]
        self._upper_limit = joint_info[9]
        self._max_force = joint_info[10]
        self._max_velocity = joint_info[11]
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def joint_index(self) -> int:
        return self._joint_index
    
    @property
    def joint_type(self) -> int:
        return self._type
    
    @property
    def limits(self) -> Tuple[float, float]:
        return (self._lower_limit, self._upper_limit)
    
    @property
    def position(self) -> float:
        """Get current joint position."""
        state = p.getJointState(
            self._body_id, self._joint_index,
            physicsClientId=self._world.client_id
        )
        return state[0]
    
    @property
    def velocity(self) -> float:
        """Get current joint velocity."""
        state = p.getJointState(
            self._body_id, self._joint_index,
            physicsClientId=self._world.client_id
        )
        return state[1]
    
    def set_position(self, position: float, max_force: Optional[float] = None):
        """Set target position (position control mode)."""
        force = max_force if max_force else self._max_force
        p.setJointMotorControl2(
            self._body_id, self._joint_index,
            p.POSITION_CONTROL,
            targetPosition=position,
            force=force,
            physicsClientId=self._world.client_id
        )
    
    def set_velocity(self, velocity: float, max_force: Optional[float] = None):
        """Set target velocity (velocity control mode)."""
        force = max_force if max_force else self._max_force
        p.setJointMotorControl2(
            self._body_id, self._joint_index,
            p.VELOCITY_CONTROL,
            targetVelocity=velocity,
            force=force,
            physicsClientId=self._world.client_id
        )
    
    def set_torque(self, torque: float):
        """Apply torque directly (torque control mode)."""
        p.setJointMotorControl2(
            self._body_id, self._joint_index,
            p.TORQUE_CONTROL,
            force=torque,
            physicsClientId=self._world.client_id
        )
    
    def reset(self, position: float = 0.0, velocity: float = 0.0):
        """Reset joint state."""
        p.resetJointState(
            self._body_id, self._joint_index,
            targetValue=position,
            targetVelocity=velocity,
            physicsClientId=self._world.client_id
        )


class PhysicsWorld:
    """
    Main physics simulation manager.
    Wraps PyBullet and provides a clean interface for game use.
    """
    
    def __init__(self, gravity: Tuple[float, float, float] = (0, 0, -9.81),
                 timestep: float = 1/240, 
                 solver_iterations: int = 50,
                 use_gui: bool = False):
        """
        Initialize physics world.
        
        Args:
            gravity: Gravity vector [x, y, z]
            timestep: Physics simulation timestep
            solver_iterations: Number of constraint solver iterations
            use_gui: If True, use PyBullet GUI (for debugging)
        """
        self._timestep = timestep
        self._gravity = gravity
        self._solver_iterations = solver_iterations
        
        # Initialize PyBullet
        if use_gui:
            self._client_id = p.connect(p.GUI)
        else:
            self._client_id = p.connect(p.DIRECT)
        
        # Configure physics
        p.setGravity(*gravity, physicsClientId=self._client_id)
        p.setTimeStep(timestep, physicsClientId=self._client_id)
        p.setPhysicsEngineParameter(
            numSolverIterations=solver_iterations,
            physicsClientId=self._client_id
        )
        
        # Add PyBullet data path for built-in models
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Track bodies
        self._bodies: Dict[int, RigidBody] = {}
        self._body_counter = 0
        
        # Collision callbacks
        self._collision_callbacks: List[callable] = []
        
        # Simulation time
        self._sim_time = 0.0
        
        logger.info(f"PhysicsWorld initialized (client_id={self._client_id})")
    
    @property
    def client_id(self) -> int:
        return self._client_id
    
    @property
    def timestep(self) -> float:
        return self._timestep
    
    @property
    def sim_time(self) -> float:
        return self._sim_time
    
    @property
    def bodies(self) -> Dict[int, RigidBody]:
        return self._bodies.copy()
    
    def step(self):
        """Advance simulation by one timestep."""
        p.stepSimulation(physicsClientId=self._client_id)
        self._sim_time += self._timestep
        
        # Process collision callbacks
        if self._collision_callbacks:
            self._process_collisions()
    
    def step_seconds(self, seconds: float):
        """Advance simulation by a number of seconds (multiple steps)."""
        steps = int(seconds / self._timestep)
        for _ in range(steps):
            self.step()
    
    def _process_collisions(self):
        """Process collision callbacks."""
        contact_points = p.getContactPoints(physicsClientId=self._client_id)
        for cp in contact_points:
            collision = CollisionInfo(
                body_a=cp[1],
                body_b=cp[2],
                contact_point=cp[5],
                contact_normal=cp[7],
                contact_distance=cp[8],
                normal_force=cp[9]
            )
            for callback in self._collision_callbacks:
                callback(collision)
    
    def add_collision_callback(self, callback: callable):
        """Add a function to be called on collisions."""
        self._collision_callbacks.append(callback)
    
    def create_ground_plane(self) -> RigidBody:
        """Create a ground plane at z=0."""
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client_id)
        body = RigidBody(self, plane_id, "ground_plane", is_static=True)
        self._bodies[plane_id] = body
        return body
    
    def create_box(self, half_extents: Tuple[float, float, float],
                   config: Optional[RigidBodyConfig] = None,
                   name: str = "") -> RigidBody:
        """Create a box rigid body."""
        config = config or RigidBodyConfig()
        
        # Create collision and visual shapes
        col_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents,
            physicsClientId=self._client_id
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=[0.7, 0.7, 0.7, 1],
            physicsClientId=self._client_id
        )
        
        # Create body
        body_id = p.createMultiBody(
            baseMass=config.mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=config.position,
            baseOrientation=config.orientation,
            physicsClientId=self._client_id
        )
        
        # Set dynamics properties
        p.changeDynamics(
            body_id, -1,
            linearDamping=config.linear_damping,
            angularDamping=config.angular_damping,
            lateralFriction=config.friction,
            restitution=config.restitution,
            physicsClientId=self._client_id
        )
        
        is_static = config.mass == 0
        name = name or f"box_{self._body_counter}"
        self._body_counter += 1
        
        body = RigidBody(self, body_id, name, is_static)
        self._bodies[body_id] = body
        return body
    
    def create_sphere(self, radius: float,
                      config: Optional[RigidBodyConfig] = None,
                      name: str = "") -> RigidBody:
        """Create a sphere rigid body."""
        config = config or RigidBodyConfig()
        
        col_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=radius,
            physicsClientId=self._client_id
        )
        vis_shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius,
            rgbaColor=[0.7, 0.3, 0.3, 1],
            physicsClientId=self._client_id
        )
        
        body_id = p.createMultiBody(
            baseMass=config.mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=config.position,
            baseOrientation=config.orientation,
            physicsClientId=self._client_id
        )
        
        p.changeDynamics(
            body_id, -1,
            linearDamping=config.linear_damping,
            angularDamping=config.angular_damping,
            lateralFriction=config.friction,
            restitution=config.restitution,
            physicsClientId=self._client_id
        )
        
        is_static = config.mass == 0
        name = name or f"sphere_{self._body_counter}"
        self._body_counter += 1
        
        body = RigidBody(self, body_id, name, is_static)
        self._bodies[body_id] = body
        return body
    
    def create_cylinder(self, radius: float, height: float,
                        config: Optional[RigidBodyConfig] = None,
                        name: str = "") -> RigidBody:
        """Create a cylinder rigid body."""
        config = config or RigidBodyConfig()
        
        col_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height,
            physicsClientId=self._client_id
        )
        vis_shape = p.createVisualShape(
            p.GEOM_CYLINDER, radius=radius, length=height,
            rgbaColor=[0.3, 0.7, 0.3, 1],
            physicsClientId=self._client_id
        )
        
        body_id = p.createMultiBody(
            baseMass=config.mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=config.position,
            baseOrientation=config.orientation,
            physicsClientId=self._client_id
        )
        
        p.changeDynamics(
            body_id, -1,
            linearDamping=config.linear_damping,
            angularDamping=config.angular_damping,
            lateralFriction=config.friction,
            restitution=config.restitution,
            physicsClientId=self._client_id
        )
        
        is_static = config.mass == 0
        name = name or f"cylinder_{self._body_counter}"
        self._body_counter += 1
        
        body = RigidBody(self, body_id, name, is_static)
        self._bodies[body_id] = body
        return body
    
    def create_mesh(self, filename: str,
                    scale: Tuple[float, float, float] = (1, 1, 1),
                    config: Optional[RigidBodyConfig] = None,
                    name: str = "") -> RigidBody:
        """Create a rigid body from a mesh file (.obj, .stl)."""
        config = config or RigidBodyConfig()
        
        col_shape = p.createCollisionShape(
            p.GEOM_MESH, fileName=filename, meshScale=scale,
            physicsClientId=self._client_id
        )
        vis_shape = p.createVisualShape(
            p.GEOM_MESH, fileName=filename, meshScale=scale,
            rgbaColor=[0.5, 0.5, 0.5, 1],
            physicsClientId=self._client_id
        )
        
        body_id = p.createMultiBody(
            baseMass=config.mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=config.position,
            baseOrientation=config.orientation,
            physicsClientId=self._client_id
        )
        
        is_static = config.mass == 0
        name = name or f"mesh_{self._body_counter}"
        self._body_counter += 1
        
        body = RigidBody(self, body_id, name, is_static)
        self._bodies[body_id] = body
        return body
    
    def load_urdf(self, filename: str,
                  position: Tuple[float, float, float] = (0, 0, 0),
                  orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
                  use_fixed_base: bool = False,
                  name: str = "") -> Tuple[RigidBody, Dict[str, Joint]]:
        """
        Load a URDF model (robot, complex object).
        
        Returns:
            Tuple of (RigidBody for base, dict of Joint objects by name)
        """
        body_id = p.loadURDF(
            filename,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=use_fixed_base,
            physicsClientId=self._client_id
        )
        
        name = name or f"urdf_{self._body_counter}"
        self._body_counter += 1
        
        body = RigidBody(self, body_id, name, is_static=use_fixed_base)
        self._bodies[body_id] = body
        
        # Get joints
        joints = {}
        num_joints = p.getNumJoints(body_id, physicsClientId=self._client_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(body_id, i, physicsClientId=self._client_id)
            joint = Joint(self, body_id, i, joint_info)
            joints[joint.name] = joint
        
        return body, joints
    
    def remove_body(self, body: RigidBody):
        """Remove a body from the simulation."""
        if body.body_id in self._bodies:
            p.removeBody(body.body_id, physicsClientId=self._client_id)
            del self._bodies[body.body_id]
    
    def ray_cast(self, from_pos: Tuple[float, float, float],
                 to_pos: Tuple[float, float, float]) -> Optional[Dict]:
        """
        Cast a ray and return hit information.
        
        Returns:
            Dict with hit_body, hit_position, hit_normal, hit_fraction
            or None if no hit
        """
        result = p.rayTest(from_pos, to_pos, physicsClientId=self._client_id)[0]
        
        if result[0] == -1:  # No hit
            return None
        
        return {
            'hit_body_id': result[0],
            'hit_link_index': result[1],
            'hit_fraction': result[2],
            'hit_position': result[3],
            'hit_normal': result[4]
        }
    
    def ray_cast_batch(self, from_positions: List[Tuple[float, float, float]],
                       to_positions: List[Tuple[float, float, float]]) -> List[Optional[Dict]]:
        """Cast multiple rays efficiently."""
        results = p.rayTestBatch(
            from_positions, to_positions,
            physicsClientId=self._client_id
        )
        
        hits = []
        for result in results:
            if result[0] == -1:
                hits.append(None)
            else:
                hits.append({
                    'hit_body_id': result[0],
                    'hit_link_index': result[1],
                    'hit_fraction': result[2],
                    'hit_position': result[3],
                    'hit_normal': result[4]
                })
        return hits
    
    def get_camera_image(self, width: int, height: int,
                         view_matrix: List[float],
                         projection_matrix: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render a camera image from the physics simulation.
        
        Returns:
            Tuple of (rgb_image, depth_image, segmentation_mask)
        """
        _, _, rgb, depth, seg = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,  # CPU renderer
            physicsClientId=self._client_id
        )
        
        # Convert to numpy arrays
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        depth_array = np.array(depth, dtype=np.float32).reshape(height, width)
        seg_array = np.array(seg, dtype=np.int32).reshape(height, width)
        
        return rgb_array, depth_array, seg_array
    
    def compute_view_matrix(self, eye_position: Tuple[float, float, float],
                            target_position: Tuple[float, float, float],
                            up_vector: Tuple[float, float, float] = (0, 0, 1)) -> List[float]:
        """Compute a view matrix for camera rendering."""
        return p.computeViewMatrix(
            cameraEyePosition=eye_position,
            cameraTargetPosition=target_position,
            cameraUpVector=up_vector
        )
    
    def compute_projection_matrix(self, fov: float, aspect: float,
                                  near: float, far: float) -> List[float]:
        """Compute a projection matrix for camera rendering."""
        return p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=near, farVal=far
        )
    
    def reset(self):
        """Reset the simulation to initial state."""
        p.resetSimulation(physicsClientId=self._client_id)
        p.setGravity(*self._gravity, physicsClientId=self._client_id)
        p.setTimeStep(self._timestep, physicsClientId=self._client_id)
        p.setPhysicsEngineParameter(
            numSolverIterations=self._solver_iterations,
            physicsClientId=self._client_id
        )
        self._bodies.clear()
        self._sim_time = 0.0
    
    def close(self):
        """Disconnect from physics simulation."""
        p.disconnect(physicsClientId=self._client_id)
        logger.info("PhysicsWorld closed")


# Utility functions
def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert Euler angles to quaternion."""
    return p.getQuaternionFromEuler([roll, pitch, yaw])


def euler_from_quaternion(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles."""
    return p.getEulerFromQuaternion(quat)
