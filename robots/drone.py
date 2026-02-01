"""
SynthWorld Drone Robot

Quadrotor/multicopter drone implementation.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import math
import logging

from .base import Robot, RobotType

logger = logging.getLogger(__name__)


@dataclass
class RotorConfig:
    """Configuration for a rotor."""
    name: str
    position: Tuple[float, float, float]  # Relative to drone center
    direction: int  # 1 for CW, -1 for CCW
    thrust_coefficient: float = 1.0
    torque_coefficient: float = 0.01


class DroneRobot(Robot):
    """
    Quadrotor drone (UAV).
    """
    
    def __init__(self, name: str, physics_world=None, renderer=None,
                 arm_length: float = 0.2, mass: float = 1.5):
        super().__init__(name, RobotType.DRONE, physics_world, renderer)
        
        # Drone parameters
        self.arm_length = arm_length
        self.mass = mass
        self.body_height = 0.05
        
        # Rotor configuration (X configuration)
        self._rotors: Dict[str, RotorConfig] = {
            'front_right': RotorConfig(
                name='front_right',
                position=(arm_length/math.sqrt(2), arm_length/math.sqrt(2), 0),
                direction=-1  # CCW
            ),
            'front_left': RotorConfig(
                name='front_left',
                position=(-arm_length/math.sqrt(2), arm_length/math.sqrt(2), 0),
                direction=1  # CW
            ),
            'rear_left': RotorConfig(
                name='rear_left',
                position=(-arm_length/math.sqrt(2), -arm_length/math.sqrt(2), 0),
                direction=-1  # CCW
            ),
            'rear_right': RotorConfig(
                name='rear_right',
                position=(arm_length/math.sqrt(2), -arm_length/math.sqrt(2), 0),
                direction=1  # CW
            ),
        }
        
        # Rotor speeds (0-1 normalized)
        self._rotor_speeds: Dict[str, float] = {name: 0 for name in self._rotors}
        
        # Flight controller parameters
        self.hover_thrust = 0.5  # Thrust needed to hover
        self.max_thrust = 20.0  # N per rotor
        self.max_tilt = 0.5  # radians
        
        # PID controllers for attitude
        self._roll_pid = PIDController(5.0, 0.5, 1.0)
        self._pitch_pid = PIDController(5.0, 0.5, 1.0)
        self._yaw_pid = PIDController(2.0, 0.1, 0.5)
        self._altitude_pid = PIDController(2.0, 0.1, 1.0)
        
        # Target setpoints
        self.target_altitude = 0.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw_rate = 0.0
        
        # Flight modes
        self.is_armed = False
        self.is_flying = False
    
    def spawn(self, position: Tuple[float, float, float],
              orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        """Spawn the drone."""
        self._state.position = np.array(position)
        self._state.orientation = np.array(orientation)
        
        if self._physics:
            from ..engine.physics import RigidBodyConfig
            
            config = RigidBodyConfig(
                mass=self.mass,
                position=position,
                orientation=orientation,
                linear_damping=0.5,
                angular_damping=0.8
            )
            
            # Central body
            self._physics_body = self._physics.create_box(
                half_extents=(0.05, 0.05, self.body_height/2),
                config=config,
                name=f"{self.name}_body"
            )
            self._physics_body.set_color((0.8, 0.2, 0.2, 1.0))  # Red
        
        if self._renderer:
            # Create body visual
            self._visual_node = self._renderer.create_box(
                0.1, 0.1, self.body_height,
                color=(0.8, 0.2, 0.2, 1.0),
                position=position,
                name=f"{self.name}_body_visual"
            )
            
            # Create arms and rotors
            for rotor_name, rotor in self._rotors.items():
                # Arm
                arm_length = np.linalg.norm(rotor.position[:2])
                arm = self._renderer.create_box(
                    arm_length, 0.02, 0.01,
                    color=(0.3, 0.3, 0.35, 1.0),
                    position=(rotor.position[0]/2, rotor.position[1]/2, 0)
                )
                arm.reparentTo(self._visual_node)
                
                # Rotor disk
                rotor_vis = self._renderer.create_cylinder(
                    0.08, 0.01,
                    color=(0.2, 0.2, 0.22, 0.5),
                    position=rotor.position
                )
                rotor_vis.reparentTo(self._visual_node)
        
        logger.info(f"Spawned drone '{self.name}'")
    
    def arm(self):
        """Arm the drone (enable motors)."""
        self.is_armed = True
        logger.info(f"Drone '{self.name}' armed")
    
    def disarm(self):
        """Disarm the drone (disable motors)."""
        self.is_armed = False
        self._rotor_speeds = {name: 0 for name in self._rotors}
        logger.info(f"Drone '{self.name}' disarmed")
    
    def takeoff(self, altitude: float = 2.0):
        """Command drone to take off to specified altitude."""
        if not self.is_armed:
            self.arm()
        
        self.target_altitude = altitude
        self.is_flying = True
        logger.info(f"Drone '{self.name}' taking off to {altitude}m")
    
    def land(self):
        """Command drone to land."""
        self.target_altitude = 0.1  # Just above ground
        # Will disarm when landed
        logger.info(f"Drone '{self.name}' landing")
    
    def update(self, dt: float):
        """Update drone state."""
        # Sync from physics
        self.sync_state_from_physics()
        
        if self.is_armed and self.is_flying:
            self._update_flight_controller(dt)
            self._apply_rotor_forces()
        
        # Update visuals
        self.sync_visuals_from_physics()
        
        # Check for landing
        if self.is_flying and self._state.position[2] < 0.15:
            # Landed
            if abs(self.target_altitude) < 0.2:
                self.is_flying = False
                self.disarm()
    
    def _update_flight_controller(self, dt: float):
        """Update flight controller (attitude stabilization)."""
        # Get current attitude (roll, pitch, yaw from quaternion)
        euler = self._state.orientation  # Would need proper conversion
        
        # Simplified: extract approximate roll/pitch from orientation
        roll = np.arctan2(2*(euler[3]*euler[0] + euler[1]*euler[2]),
                          1 - 2*(euler[0]**2 + euler[1]**2))
        pitch = np.arcsin(np.clip(2*(euler[3]*euler[1] - euler[2]*euler[0]), -1, 1))
        
        # Altitude control
        altitude_error = self.target_altitude - self._state.position[2]
        thrust_adjustment = self._altitude_pid.update(altitude_error, dt)
        base_thrust = self.hover_thrust + thrust_adjustment
        
        # Attitude control
        roll_output = self._roll_pid.update(self.target_roll - roll, dt)
        pitch_output = self._pitch_pid.update(self.target_pitch - pitch, dt)
        yaw_output = self._yaw_pid.update(self.target_yaw_rate - self._state.angular_velocity[2], dt)
        
        # Mix to rotor speeds
        self._rotor_speeds['front_right'] = np.clip(
            base_thrust - roll_output - pitch_output + yaw_output, 0, 1)
        self._rotor_speeds['front_left'] = np.clip(
            base_thrust + roll_output - pitch_output - yaw_output, 0, 1)
        self._rotor_speeds['rear_left'] = np.clip(
            base_thrust + roll_output + pitch_output + yaw_output, 0, 1)
        self._rotor_speeds['rear_right'] = np.clip(
            base_thrust - roll_output + pitch_output - yaw_output, 0, 1)
    
    def _apply_rotor_forces(self):
        """Apply rotor thrust forces to physics body."""
        if not self._physics_body:
            return
        
        total_thrust = np.array([0.0, 0.0, 0.0])
        total_torque = np.array([0.0, 0.0, 0.0])
        
        for rotor_name, rotor in self._rotors.items():
            speed = self._rotor_speeds[rotor_name]
            thrust = speed * self.max_thrust
            
            # Thrust vector (in body frame, upward)
            thrust_vec = np.array([0, 0, thrust])
            total_thrust += thrust_vec
            
            # Torque from thrust offset
            rotor_pos = np.array(rotor.position)
            thrust_torque = np.cross(rotor_pos, thrust_vec)
            total_torque += thrust_torque
            
            # Reaction torque from rotor spin
            reaction_torque = rotor.direction * speed * rotor.torque_coefficient * self.max_thrust
            total_torque[2] += reaction_torque
        
        # Transform to world frame and apply
        # (Simplified - should use proper rotation)
        self._physics_body.apply_force(tuple(total_thrust))
        self._physics_body.apply_torque(tuple(total_torque))
    
    def apply_control(self, control: Dict[str, float]):
        """
        Apply control commands.
        
        Args:
            control: Dict with:
                - 'throttle': Altitude rate (-1 to 1)
                - 'roll': Roll command (-1 to 1)
                - 'pitch': Pitch command (-1 to 1)
                - 'yaw': Yaw rate command (-1 to 1)
        """
        if not self._control_enabled or not self.is_armed:
            return
        
        throttle = control.get('throttle', 0)
        roll = control.get('roll', 0)
        pitch = control.get('pitch', 0)
        yaw = control.get('yaw', 0)
        
        # Update setpoints
        self.target_altitude += throttle * 0.1  # Altitude rate
        self.target_altitude = max(0.1, self.target_altitude)  # Don't go underground
        
        self.target_roll = roll * self.max_tilt
        self.target_pitch = pitch * self.max_tilt
        self.target_yaw_rate = yaw * 2.0  # rad/s
        
        if not self.is_flying and throttle > 0.3:
            self.takeoff(self._state.position[2] + 2.0)
    
    def get_rotor_speeds(self) -> Dict[str, float]:
        """Get current rotor speeds."""
        return self._rotor_speeds.copy()
    
    def get_battery_voltage(self) -> float:
        """Get simulated battery voltage."""
        return 11.1 + 1.5 * self._state.battery_level  # 11.1V to 12.6V
    
    def hover(self):
        """Hold current position."""
        self.target_roll = 0
        self.target_pitch = 0
        self.target_yaw_rate = 0
    
    def set_velocity(self, vx: float, vy: float, vz: float):
        """
        Set desired velocity (velocity control mode).
        
        Args:
            vx: Forward velocity (m/s)
            vy: Right velocity (m/s)  
            vz: Up velocity (m/s)
        """
        # Convert velocity to attitude commands
        # Simplified - real flight controller would be more complex
        self.target_pitch = -np.clip(vx * 0.1, -self.max_tilt, self.max_tilt)
        self.target_roll = np.clip(vy * 0.1, -self.max_tilt, self.max_tilt)
        self.target_altitude += vz * 0.02


class PIDController:
    """Simple PID controller."""
    
    def __init__(self, kp: float, ki: float, kd: float,
                 integral_limit: float = 10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        
        self._integral = 0.0
        self._last_error = 0.0
    
    def update(self, error: float, dt: float) -> float:
        """Update PID controller."""
        # Proportional
        p = self.kp * error
        
        # Integral
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self._integral
        
        # Derivative
        derivative = (error - self._last_error) / dt if dt > 0 else 0
        d = self.kd * derivative
        
        self._last_error = error
        
        return p + i + d
    
    def reset(self):
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = 0.0
