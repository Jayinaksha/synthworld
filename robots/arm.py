"""
SynthWorld Robot Arm

Articulated robot arm/manipulator implementation.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import math
import logging

from .base import Robot, RobotType

logger = logging.getLogger(__name__)


@dataclass
class JointConfig:
    """Configuration for a robot joint."""
    name: str
    axis: Tuple[float, float, float]  # Rotation axis
    offset: Tuple[float, float, float]  # Position offset from parent
    limits: Tuple[float, float]  # Min, max angle in radians
    max_velocity: float = 2.0  # rad/s
    max_torque: float = 50.0  # Nm


@dataclass
class JointState:
    """Current state of a joint."""
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    target_position: Optional[float] = None


class RobotArm:
    """
    Articulated robot arm with multiple joints.
    Can be attached to a mobile base or used standalone.
    """
    
    def __init__(self, name: str, num_joints: int = 6,
                 link_lengths: Optional[List[float]] = None):
        """
        Initialize robot arm.
        
        Args:
            name: Arm name
            num_joints: Number of joints
            link_lengths: Length of each link (default: equal lengths)
        """
        self.name = name
        self.num_joints = num_joints
        
        if link_lengths is None:
            link_lengths = [0.15] * num_joints
        self.link_lengths = link_lengths
        
        # Joint configuration (typical 6-DOF arm)
        self._joint_configs: List[JointConfig] = []
        self._setup_joints()
        
        # Joint states
        self._joint_states: List[JointState] = [
            JointState() for _ in range(num_joints)
        ]
        
        # End effector
        self.gripper_open = True
        self.gripper_position = 0.0  # 0=closed, 1=open
        
        # Physics bodies
        self._link_bodies: List[Any] = []
        self._joint_physics: List[Any] = []
        
        # Visual nodes
        self._visual_nodes: List[Any] = []
        
        logger.info(f"RobotArm '{name}' created with {num_joints} joints")
    
    def _setup_joints(self):
        """Set up default joint configuration."""
        # Typical 6-DOF industrial arm configuration:
        # J1: Base rotation (Z-axis)
        # J2: Shoulder (Y-axis)
        # J3: Elbow (Y-axis)
        # J4: Wrist roll (X-axis)
        # J5: Wrist pitch (Y-axis)
        # J6: End effector roll (X-axis)
        
        axes = [
            (0, 0, 1),  # Base
            (0, 1, 0),  # Shoulder
            (0, 1, 0),  # Elbow
            (1, 0, 0),  # Wrist roll
            (0, 1, 0),  # Wrist pitch
            (1, 0, 0),  # EE roll
        ]
        
        limits = [
            (-3.14, 3.14),   # Base: full rotation
            (-2.0, 2.0),     # Shoulder
            (-2.4, 2.4),     # Elbow
            (-3.14, 3.14),   # Wrist roll
            (-2.0, 2.0),     # Wrist pitch
            (-3.14, 3.14),   # EE roll
        ]
        
        z_offset = 0
        for i in range(min(self.num_joints, len(axes))):
            self._joint_configs.append(JointConfig(
                name=f"joint_{i+1}",
                axis=axes[i],
                offset=(0, 0, self.link_lengths[i] if i > 0 else 0.05),
                limits=limits[i]
            ))
            z_offset += self.link_lengths[i] if i > 0 else 0.05
    
    def get_joint_positions(self) -> List[float]:
        """Get current joint positions."""
        return [state.position for state in self._joint_states]
    
    def set_joint_positions(self, positions: List[float]):
        """Set target joint positions."""
        for i, pos in enumerate(positions[:self.num_joints]):
            limits = self._joint_configs[i].limits
            clamped = np.clip(pos, limits[0], limits[1])
            self._joint_states[i].target_position = clamped
    
    def set_joint_position(self, joint_index: int, position: float):
        """Set a single joint target position."""
        if 0 <= joint_index < self.num_joints:
            limits = self._joint_configs[joint_index].limits
            clamped = np.clip(position, limits[0], limits[1])
            self._joint_states[joint_index].target_position = clamped
    
    def update(self, dt: float):
        """Update arm state."""
        # Simple position control with velocity limits
        for i, state in enumerate(self._joint_states):
            if state.target_position is not None:
                error = state.target_position - state.position
                max_vel = self._joint_configs[i].max_velocity
                
                velocity = np.clip(error * 5.0, -max_vel, max_vel)
                state.velocity = velocity
                state.position += velocity * dt
                
                # Check if target reached
                if abs(error) < 0.01:
                    state.velocity = 0
        
        # Update gripper
        target_gripper = 1.0 if self.gripper_open else 0.0
        gripper_diff = target_gripper - self.gripper_position
        self.gripper_position += np.clip(gripper_diff * 5.0, -2, 2) * dt
    
    def forward_kinematics(self) -> np.ndarray:
        """
        Compute end effector pose using forward kinematics.
        
        Returns:
            4x4 transformation matrix for end effector
        """
        transform = np.eye(4)
        
        for i, (config, state) in enumerate(zip(self._joint_configs, self._joint_states)):
            # Translation
            T = np.eye(4)
            T[:3, 3] = config.offset
            
            # Rotation
            R = np.eye(4)
            angle = state.position
            axis = np.array(config.axis)
            
            # Rodrigues rotation
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R[:3, :3] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            
            transform = transform @ T @ R
        
        return transform
    
    def get_end_effector_position(self) -> Tuple[float, float, float]:
        """Get end effector position in arm base frame."""
        transform = self.forward_kinematics()
        return tuple(transform[:3, 3])
    
    def inverse_kinematics(self, target_pos: Tuple[float, float, float],
                          target_orn: Optional[Tuple] = None) -> Optional[List[float]]:
        """
        Compute joint angles for target end effector pose.
        Uses iterative Jacobian-based IK.
        
        Args:
            target_pos: Target position (x, y, z)
            target_orn: Optional target orientation (quaternion)
        
        Returns:
            List of joint angles, or None if unreachable
        """
        # Simple iterative IK
        max_iterations = 100
        tolerance = 0.01
        
        target = np.array(target_pos)
        current_joints = [state.position for state in self._joint_states]
        
        for _ in range(max_iterations):
            # Get current EE position
            current_pos = np.array(self.get_end_effector_position())
            error = target - current_pos
            
            if np.linalg.norm(error) < tolerance:
                return current_joints
            
            # Compute Jacobian numerically
            J = self._compute_jacobian()
            
            # Damped least squares
            damping = 0.1
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(3))
            
            # Joint velocity
            dq = J_pinv @ error
            
            # Update joints
            for i in range(min(len(dq), self.num_joints)):
                limits = self._joint_configs[i].limits
                current_joints[i] = np.clip(
                    current_joints[i] + dq[i] * 0.5,
                    limits[0], limits[1]
                )
                self._joint_states[i].position = current_joints[i]
        
        logger.warning(f"IK did not converge for target {target_pos}")
        return None
    
    def _compute_jacobian(self, delta: float = 0.001) -> np.ndarray:
        """Compute position Jacobian numerically."""
        J = np.zeros((3, self.num_joints))
        
        base_pos = np.array(self.get_end_effector_position())
        
        for i in range(self.num_joints):
            # Perturb joint
            original = self._joint_states[i].position
            self._joint_states[i].position = original + delta
            
            perturbed_pos = np.array(self.get_end_effector_position())
            J[:, i] = (perturbed_pos - base_pos) / delta
            
            self._joint_states[i].position = original
        
        return J
    
    def open_gripper(self):
        """Open the gripper."""
        self.gripper_open = True
    
    def close_gripper(self):
        """Close the gripper."""
        self.gripper_open = False
    
    def toggle_gripper(self):
        """Toggle gripper state."""
        self.gripper_open = not self.gripper_open
    
    def go_home(self):
        """Move arm to home position."""
        home_positions = [0.0] * self.num_joints
        self.set_joint_positions(home_positions)


class ArmRobot(Robot):
    """
    Mobile robot with an attached arm.
    """
    
    def __init__(self, name: str, physics_world=None, renderer=None,
                 arm_joints: int = 6):
        super().__init__(name, RobotType.ARM, physics_world, renderer)
        
        # Base dimensions
        self.base_width = 0.4
        self.base_depth = 0.4
        self.base_height = 0.3
        
        # Create arm
        self.arm = RobotArm(f"{name}_arm", num_joints=arm_joints)
        
        # Arm mounting point
        self.arm_mount_offset = (0, 0, self.base_height)
    
    def spawn(self, position: Tuple[float, float, float],
              orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        """Spawn the arm robot."""
        self._state.position = np.array(position)
        self._state.orientation = np.array(orientation)
        
        if self._physics:
            from ..engine.physics import RigidBodyConfig
            
            # Create base
            config = RigidBodyConfig(
                mass=30.0,
                position=position,
                orientation=orientation,
                friction=0.9
            )
            
            self._physics_body = self._physics.create_box(
                half_extents=(self.base_width/2, self.base_depth/2, self.base_height/2),
                config=config,
                name=f"{self.name}_base"
            )
            self._physics_body.set_color((0.3, 0.3, 0.8, 1.0))
        
        if self._renderer:
            # Create base visual
            self._visual_node = self._renderer.create_box(
                self.base_width, self.base_depth, self.base_height,
                color=(0.3, 0.3, 0.8, 1.0),
                position=position,
                name=f"{self.name}_base_visual"
            )
            
            # Create arm visuals
            self._create_arm_visuals()
        
        logger.info(f"Spawned arm robot '{self.name}'")
    
    def _create_arm_visuals(self):
        """Create visual representation of the arm."""
        if not self._renderer or not self._visual_node:
            return
        
        arm_base_pos = self.arm_mount_offset
        
        # Create link visuals
        current_pos = list(arm_base_pos)
        
        for i, length in enumerate(self.arm.link_lengths):
            # Link cylinder
            link = self._renderer.create_cylinder(
                0.03, length,
                color=(0.5, 0.5, 0.55, 1.0),
                position=tuple(current_pos)
            )
            link.reparentTo(self._visual_node)
            self.arm._visual_nodes.append(link)
            
            # Update position for next link
            current_pos[2] += length
        
        # Gripper
        gripper = self._renderer.create_box(
            0.08, 0.04, 0.06,
            color=(0.2, 0.2, 0.25, 1.0),
            position=tuple(current_pos)
        )
        gripper.reparentTo(self._visual_node)
    
    def update(self, dt: float):
        """Update arm robot."""
        # Update arm
        self.arm.update(dt)
        
        # Sync from physics
        self.sync_state_from_physics()
        self.sync_visuals_from_physics()
        
        # Update arm visuals based on joint positions
        self._update_arm_visuals()
    
    def _update_arm_visuals(self):
        """Update arm visual positions based on joint states."""
        # This would animate the arm links based on joint positions
        # Simplified for now
        pass
    
    def apply_control(self, control: Dict[str, float]):
        """
        Apply control commands.
        
        Args:
            control: Dict with:
                - Base movement: 'linear', 'angular'
                - Arm control: 'joint_0' through 'joint_N', or 'ee_x', 'ee_y', 'ee_z'
                - Gripper: 'gripper' (1=open, -1=close)
        """
        if not self._control_enabled:
            return
        
        # Base movement (if mobile)
        linear = control.get('linear', 0)
        angular = control.get('angular', 0)
        
        if self._physics_body and (abs(linear) > 0.01 or abs(angular) > 0.01):
            heading = np.radians(self.heading)
            forward = np.array([np.sin(heading), np.cos(heading), 0])
            
            force = forward * linear * 100
            self._physics_body.apply_force(tuple(force))
            self._physics_body.apply_torque((0, 0, angular * 30))
        
        # Joint control
        for i in range(self.arm.num_joints):
            key = f'joint_{i}'
            if key in control:
                current = self.arm._joint_states[i].position
                velocity = control[key]  # -1 to 1
                new_pos = current + velocity * 0.1
                self.arm.set_joint_position(i, new_pos)
        
        # End effector control (Cartesian)
        if 'ee_x' in control or 'ee_y' in control or 'ee_z' in control:
            current_ee = self.arm.get_end_effector_position()
            target = (
                current_ee[0] + control.get('ee_x', 0) * 0.01,
                current_ee[1] + control.get('ee_y', 0) * 0.01,
                current_ee[2] + control.get('ee_z', 0) * 0.01
            )
            self.arm.inverse_kinematics(target)
        
        # Gripper
        gripper_cmd = control.get('gripper', 0)
        if gripper_cmd > 0.5:
            self.arm.open_gripper()
        elif gripper_cmd < -0.5:
            self.arm.close_gripper()
    
    def get_ee_position_world(self) -> Tuple[float, float, float]:
        """Get end effector position in world frame."""
        ee_local = self.arm.get_end_effector_position()
        
        # Transform to world frame
        base_pos = self._state.position
        return (
            base_pos[0] + ee_local[0] + self.arm_mount_offset[0],
            base_pos[1] + ee_local[1] + self.arm_mount_offset[1],
            base_pos[2] + ee_local[2] + self.arm_mount_offset[2]
        )
