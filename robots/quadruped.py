"""
SynthWorld Quadruped Robot

Four-legged walking robot implementation.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import math
import logging

from .base import Robot, RobotType

logger = logging.getLogger(__name__)


@dataclass
class LegConfig:
    """Configuration for a robot leg."""
    name: str
    hip_offset: Tuple[float, float, float]  # Position relative to body center
    upper_length: float
    lower_length: float
    hip_limits: Tuple[float, float] = (-0.5, 0.5)  # Hip rotation limits
    thigh_limits: Tuple[float, float] = (-1.5, 1.5)  # Thigh limits
    knee_limits: Tuple[float, float] = (-2.5, 0.0)  # Knee limits


@dataclass
class LegState:
    """Current state of a leg."""
    hip_angle: float = 0.0
    thigh_angle: float = 0.0
    knee_angle: float = 0.0
    foot_position: Tuple[float, float, float] = (0, 0, 0)
    is_stance: bool = True  # True if foot is on ground


class QuadrupedRobot(Robot):
    """
    Four-legged walking robot (like Spot, ANYmal).
    """
    
    def __init__(self, name: str, physics_world=None, renderer=None,
                 body_length: float = 0.5, body_width: float = 0.3,
                 body_height: float = 0.1, leg_length: float = 0.3):
        super().__init__(name, RobotType.QUADRUPED, physics_world, renderer)
        
        # Body dimensions
        self.body_length = body_length
        self.body_width = body_width
        self.body_height = body_height
        
        # Leg parameters
        self.upper_leg_length = leg_length * 0.5
        self.lower_leg_length = leg_length * 0.5
        self.default_height = leg_length * 0.7  # Standing height
        
        # Configure legs
        self._leg_configs = {
            'front_left': LegConfig(
                name='front_left',
                hip_offset=(-body_width/2, body_length/2, 0),
                upper_length=self.upper_leg_length,
                lower_length=self.lower_leg_length
            ),
            'front_right': LegConfig(
                name='front_right',
                hip_offset=(body_width/2, body_length/2, 0),
                upper_length=self.upper_leg_length,
                lower_length=self.lower_leg_length
            ),
            'rear_left': LegConfig(
                name='rear_left',
                hip_offset=(-body_width/2, -body_length/2, 0),
                upper_length=self.upper_leg_length,
                lower_length=self.lower_leg_length
            ),
            'rear_right': LegConfig(
                name='rear_right',
                hip_offset=(body_width/2, -body_length/2, 0),
                upper_length=self.upper_leg_length,
                lower_length=self.lower_leg_length
            ),
        }
        
        # Leg states
        self._leg_states: Dict[str, LegState] = {
            name: LegState() for name in self._leg_configs
        }
        
        # Gait parameters
        self.gait_phase = 0.0
        self.gait_frequency = 1.0  # Hz
        self.stride_length = 0.15
        self.step_height = 0.05
        
        # Control mode
        self.is_standing = True
        self.is_walking = False
    
    def spawn(self, position: Tuple[float, float, float],
              orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        """Spawn the quadruped robot."""
        # Adjust position to account for leg height
        spawn_pos = (position[0], position[1], position[2] + self.default_height)
        
        self._state.position = np.array(spawn_pos)
        self._state.orientation = np.array(orientation)
        
        if self._physics:
            from ..engine.physics import RigidBodyConfig
            
            # Create body
            config = RigidBodyConfig(
                mass=15.0,
                position=spawn_pos,
                orientation=orientation,
                friction=0.8
            )
            
            self._physics_body = self._physics.create_box(
                half_extents=(self.body_width/2, self.body_length/2, self.body_height/2),
                config=config,
                name=f"{self.name}_body"
            )
            self._physics_body.set_color((0.9, 0.7, 0.1, 1.0))  # Yellow/gold
        
        if self._renderer:
            # Create body visual
            self._visual_node = self._renderer.create_box(
                self.body_width, self.body_length, self.body_height,
                color=(0.9, 0.7, 0.1, 1.0),
                position=spawn_pos,
                name=f"{self.name}_body_visual"
            )
            
            # Create leg visuals
            for leg_name, leg_config in self._leg_configs.items():
                self._create_leg_visual(leg_name, leg_config)
        
        # Initialize leg positions to standing pose
        self._set_standing_pose()
        
        logger.info(f"Spawned quadruped robot '{self.name}'")
    
    def _create_leg_visual(self, leg_name: str, config: LegConfig):
        """Create visual representation of a leg."""
        if not self._renderer or not self._visual_node:
            return
        
        # Upper leg
        upper_leg = self._renderer.create_cylinder(
            0.02, config.upper_length,
            color=(0.3, 0.3, 0.35, 1.0),
            position=(0, 0, -config.upper_length/2)
        )
        upper_leg.reparentTo(self._visual_node)
        upper_leg.setPos(*config.hip_offset)
        
        # Lower leg
        lower_leg = self._renderer.create_cylinder(
            0.015, config.lower_length,
            color=(0.25, 0.25, 0.3, 1.0),
            position=(0, 0, -config.lower_length/2)
        )
        lower_leg.reparentTo(upper_leg)
        lower_leg.setPos(0, 0, -config.upper_length)
    
    def _set_standing_pose(self):
        """Set legs to standing pose."""
        for leg_name, config in self._leg_configs.items():
            # Simple standing - legs straight down
            self._leg_states[leg_name] = LegState(
                hip_angle=0.0,
                thigh_angle=0.3,  # Slight bend
                knee_angle=-0.6,  # Bent to achieve standing height
                foot_position=config.hip_offset,
                is_stance=True
            )
    
    def update(self, dt: float):
        """Update quadruped state."""
        if self.is_walking:
            self._update_walking_gait(dt)
        
        # Sync from physics
        self.sync_state_from_physics()
        
        # Update visuals
        self.sync_visuals_from_physics()
    
    def _update_walking_gait(self, dt: float):
        """Update walking gait."""
        self.gait_phase = (self.gait_phase + dt * self.gait_frequency) % 1.0
        
        # Trot gait - diagonal pairs move together
        # Front-left and rear-right, front-right and rear-left
        
        leg_phases = {
            'front_left': self.gait_phase,
            'rear_right': self.gait_phase,
            'front_right': (self.gait_phase + 0.5) % 1.0,
            'rear_left': (self.gait_phase + 0.5) % 1.0,
        }
        
        for leg_name, phase in leg_phases.items():
            self._update_leg_trajectory(leg_name, phase)
    
    def _update_leg_trajectory(self, leg_name: str, phase: float):
        """Update leg position based on gait phase."""
        state = self._leg_states[leg_name]
        config = self._leg_configs[leg_name]
        
        # Swing phase (0-0.5) vs stance phase (0.5-1.0)
        if phase < 0.5:
            # Swing phase - leg in air
            swing_progress = phase / 0.5
            
            # Parabolic foot trajectory
            x_offset = self.stride_length * (swing_progress - 0.5)
            z_offset = self.step_height * math.sin(swing_progress * math.pi)
            
            state.foot_position = (
                config.hip_offset[0] + x_offset,
                config.hip_offset[1],
                -self.default_height + z_offset
            )
            state.is_stance = False
        else:
            # Stance phase - leg on ground, pushes back
            stance_progress = (phase - 0.5) / 0.5
            
            x_offset = self.stride_length * (0.5 - stance_progress)
            
            state.foot_position = (
                config.hip_offset[0] + x_offset,
                config.hip_offset[1],
                -self.default_height
            )
            state.is_stance = True
        
        # Apply forces when in stance
        if state.is_stance and self._physics_body:
            heading = np.radians(self.heading)
            forward = np.array([np.sin(heading), np.cos(heading), 0])
            
            # Each leg in stance contributes to forward motion
            force = forward * 5.0  # Leg push force
            self._physics_body.apply_force(tuple(force))
    
    def apply_control(self, control: Dict[str, float]):
        """
        Apply control to quadruped.
        
        Args:
            control: Dict with:
                - 'forward': Forward/backward command (-1 to 1)
                - 'lateral': Left/right strafe (-1 to 1)
                - 'turn': Turning command (-1 to 1)
                - 'height': Body height adjustment
        """
        if not self._control_enabled:
            return
        
        forward = control.get('forward', 0)
        lateral = control.get('lateral', 0)
        turn = control.get('turn', 0)
        
        # Start/stop walking based on commands
        if abs(forward) > 0.1 or abs(lateral) > 0.1 or abs(turn) > 0.1:
            self.is_walking = True
            
            # Adjust gait frequency based on speed
            self.gait_frequency = 1.0 + abs(forward) * 0.5
            
            # Adjust stride length
            self.stride_length = 0.1 + abs(forward) * 0.1
        else:
            self.is_walking = False
        
        # Apply turning
        if self._physics_body and abs(turn) > 0.1:
            self._physics_body.apply_torque((0, 0, turn * 5))
    
    def stand(self):
        """Command robot to stand in place."""
        self.is_walking = False
        self._set_standing_pose()
    
    def sit(self):
        """Command robot to sit (lower body)."""
        self.is_walking = False
        # Lower the body height
        for leg_name in self._leg_states:
            state = self._leg_states[leg_name]
            state.thigh_angle = 0.8
            state.knee_angle = -1.5
    
    def get_leg_states(self) -> Dict[str, LegState]:
        """Get current state of all legs."""
        return self._leg_states.copy()
    
    def get_foot_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Get foot positions in world frame."""
        foot_positions = {}
        body_pos = self._state.position
        
        for leg_name, state in self._leg_states.items():
            # Transform foot position to world frame
            foot_pos = (
                body_pos[0] + state.foot_position[0],
                body_pos[1] + state.foot_position[1],
                body_pos[2] + state.foot_position[2]
            )
            foot_positions[leg_name] = foot_pos
        
        return foot_positions
    
    def get_ground_contacts(self) -> Dict[str, bool]:
        """Check which feet are in contact with ground."""
        return {name: state.is_stance for name, state in self._leg_states.items()}
