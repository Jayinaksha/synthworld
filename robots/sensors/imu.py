"""
SynthWorld IMU Sensor

Inertial Measurement Unit sensor simulation.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import time
import logging

from ..base import Sensor, SensorReading

logger = logging.getLogger(__name__)


@dataclass
class IMUData:
    """IMU measurement data."""
    linear_acceleration: np.ndarray  # m/s^2
    angular_velocity: np.ndarray      # rad/s
    orientation: np.ndarray           # quaternion (x, y, z, w)


class IMUSensor(Sensor):
    """
    Inertial Measurement Unit sensor.
    
    Provides:
    - 3-axis accelerometer
    - 3-axis gyroscope
    - Orientation estimate (optional)
    """
    
    GRAVITY = 9.81  # m/s^2
    
    def __init__(self, name: str,
                 position: Tuple[float, float, float] = (0, 0, 0),
                 orientation: Tuple[float, float, float] = (0, 0, 0),
                 update_rate: float = 100.0,
                 accel_noise_stddev: float = 0.01,
                 gyro_noise_stddev: float = 0.001,
                 accel_bias: Tuple[float, float, float] = (0, 0, 0),
                 gyro_bias: Tuple[float, float, float] = (0, 0, 0),
                 gravity_compensation: bool = True):
        """
        Initialize IMU sensor.
        
        Args:
            name: Sensor name
            position: Position offset from robot base
            orientation: Orientation offset (roll, pitch, yaw) in degrees
            update_rate: Update rate in Hz
            accel_noise_stddev: Accelerometer noise standard deviation (m/s^2)
            gyro_noise_stddev: Gyroscope noise standard deviation (rad/s)
            accel_bias: Accelerometer bias (m/s^2)
            gyro_bias: Gyroscope bias (rad/s)
            gravity_compensation: Whether to include gravity in acceleration
        """
        super().__init__(name, update_rate)
        
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.accel_noise_stddev = accel_noise_stddev
        self.gyro_noise_stddev = gyro_noise_stddev
        self.accel_bias = np.array(accel_bias)
        self.gyro_bias = np.array(gyro_bias)
        self.gravity_compensation = gravity_compensation
        
        # State for differentiation
        self._last_velocity = np.zeros(3)
        self._last_timestamp = 0.0
        
        # Orientation estimator state
        self._estimated_orientation = np.array([0, 0, 0, 1])  # Quaternion
        
        logger.info(f"IMUSensor '{name}' initialized at {update_rate}Hz")
    
    def read(self) -> SensorReading:
        """Read IMU data."""
        timestamp = time.time()
        
        if not self._robot:
            # Return identity/zero measurement
            return self._create_zero_reading(timestamp)
        
        # Get robot state
        linear_velocity = self._robot.velocity
        angular_velocity = self._robot.state.angular_velocity
        orientation = self._robot.orientation
        
        # Calculate acceleration from velocity change
        dt = timestamp - self._last_timestamp if self._last_timestamp > 0 else 0.01
        dt = max(dt, 0.001)  # Prevent division by zero
        
        linear_acceleration = (linear_velocity - self._last_velocity) / dt
        
        # Transform to body frame
        # Simplified - should use proper rotation
        accel_body = linear_acceleration.copy()
        gyro_body = angular_velocity.copy()
        
        # Add gravity in body frame if not compensated
        if self.gravity_compensation:
            # Gravity vector rotated to body frame
            # Simplified: assume mostly upright
            roll = self._robot.state.orientation[0]
            pitch = self._robot.state.orientation[1]
            
            gravity_body = np.array([
                -self.GRAVITY * np.sin(pitch),
                self.GRAVITY * np.sin(roll) * np.cos(pitch),
                self.GRAVITY * np.cos(roll) * np.cos(pitch)
            ])
            accel_body += gravity_body
        
        # Add noise
        accel_noise = np.random.normal(0, self.accel_noise_stddev, 3)
        gyro_noise = np.random.normal(0, self.gyro_noise_stddev, 3)
        
        accel_body += accel_noise + self.accel_bias
        gyro_body += gyro_noise + self.gyro_bias
        
        # Store for next iteration
        self._last_velocity = linear_velocity.copy()
        self._last_timestamp = timestamp
        
        # Create reading
        imu_data = IMUData(
            linear_acceleration=accel_body,
            angular_velocity=gyro_body,
            orientation=orientation
        )
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=timestamp,
            data=imu_data,
            metadata={
                'frame_id': 'imu_frame',
                'has_orientation': True,
                'accel_covariance': [self.accel_noise_stddev**2] * 9,
                'gyro_covariance': [self.gyro_noise_stddev**2] * 9
            }
        )
        
        self._last_reading = reading
        return reading
    
    def _create_zero_reading(self, timestamp: float) -> SensorReading:
        """Create a zero/identity reading."""
        # When at rest, accelerometer should read gravity (upward = positive z)
        accel = np.array([0, 0, self.GRAVITY]) if self.gravity_compensation else np.zeros(3)
        
        imu_data = IMUData(
            linear_acceleration=accel,
            angular_velocity=np.zeros(3),
            orientation=np.array([0, 0, 0, 1])
        )
        
        return SensorReading(
            sensor_name=self.name,
            timestamp=timestamp,
            data=imu_data,
            metadata={'frame_id': 'imu_frame'}
        )
    
    def get_acceleration(self) -> np.ndarray:
        """Get last acceleration reading."""
        if self._last_reading:
            return self._last_reading.data.linear_acceleration
        return np.array([0, 0, self.GRAVITY])
    
    def get_angular_velocity(self) -> np.ndarray:
        """Get last angular velocity reading."""
        if self._last_reading:
            return self._last_reading.data.angular_velocity
        return np.zeros(3)
    
    def get_orientation(self) -> np.ndarray:
        """Get orientation estimate (quaternion)."""
        if self._last_reading:
            return self._last_reading.data.orientation
        return np.array([0, 0, 0, 1])
    
    def get_euler_angles(self) -> np.ndarray:
        """Get orientation as Euler angles (roll, pitch, yaw) in radians."""
        q = self.get_orientation()
        
        # Convert quaternion to Euler
        x, y, z, w = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def calibrate(self, num_samples: int = 100):
        """
        Calibrate sensor (estimate biases).
        
        Should be called when robot is stationary.
        
        Args:
            num_samples: Number of samples to collect
        """
        if not self._robot:
            logger.warning("Cannot calibrate: no robot attached")
            return
        
        accel_samples = []
        gyro_samples = []
        
        for _ in range(num_samples):
            reading = self.read()
            accel_samples.append(reading.data.linear_acceleration)
            gyro_samples.append(reading.data.angular_velocity)
            time.sleep(1.0 / self.update_rate)
        
        # Estimate biases
        accel_mean = np.mean(accel_samples, axis=0)
        gyro_mean = np.mean(gyro_samples, axis=0)
        
        # At rest, acceleration should be [0, 0, g]
        expected_accel = np.array([0, 0, self.GRAVITY])
        self.accel_bias = accel_mean - expected_accel
        
        # At rest, gyro should be zero
        self.gyro_bias = gyro_mean
        
        logger.info(f"IMU calibrated: accel_bias={self.accel_bias}, gyro_bias={self.gyro_bias}")


class GPS(Sensor):
    """
    GPS sensor simulation.
    """
    
    def __init__(self, name: str,
                 update_rate: float = 10.0,
                 position_noise_stddev: float = 1.0,
                 velocity_noise_stddev: float = 0.1,
                 reference_lat: float = 0.0,
                 reference_lon: float = 0.0):
        """
        Initialize GPS sensor.
        
        Args:
            name: Sensor name
            update_rate: Update rate in Hz
            position_noise_stddev: Position noise in meters
            velocity_noise_stddev: Velocity noise in m/s
            reference_lat: Reference latitude for local coordinate conversion
            reference_lon: Reference longitude for local coordinate conversion
        """
        super().__init__(name, update_rate)
        
        self.position_noise_stddev = position_noise_stddev
        self.velocity_noise_stddev = velocity_noise_stddev
        self.reference_lat = reference_lat
        self.reference_lon = reference_lon
        
        # Meters per degree (approximate)
        self.meters_per_deg_lat = 111320
        self.meters_per_deg_lon = 111320 * np.cos(np.radians(reference_lat))
    
    def read(self) -> SensorReading:
        """Read GPS data."""
        timestamp = time.time()
        
        if not self._robot:
            return SensorReading(
                sensor_name=self.name,
                timestamp=timestamp,
                data={'fix': False},
                metadata={'status': 'no_fix'}
            )
        
        # Get true position
        pos = self._robot.position
        vel = self._robot.velocity
        
        # Add noise
        pos_noise = np.random.normal(0, self.position_noise_stddev, 3)
        vel_noise = np.random.normal(0, self.velocity_noise_stddev, 3)
        
        noisy_pos = pos + pos_noise
        noisy_vel = vel + vel_noise
        
        # Convert to lat/lon (simple planar approximation)
        lat = self.reference_lat + noisy_pos[1] / self.meters_per_deg_lat
        lon = self.reference_lon + noisy_pos[0] / self.meters_per_deg_lon
        alt = noisy_pos[2]
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=timestamp,
            data={
                'fix': True,
                'latitude': lat,
                'longitude': lon,
                'altitude': alt,
                'velocity': noisy_vel,
                'local_x': noisy_pos[0],
                'local_y': noisy_pos[1],
                'local_z': noisy_pos[2]
            },
            metadata={
                'status': 'fix',
                'num_satellites': np.random.randint(6, 12),
                'hdop': 1.0 + np.random.random(),
                'reference_lat': self.reference_lat,
                'reference_lon': self.reference_lon
            }
        )
        
        self._last_reading = reading
        return reading


class Compass(Sensor):
    """
    Magnetometer/compass sensor simulation.
    """
    
    def __init__(self, name: str,
                 update_rate: float = 20.0,
                 noise_stddev: float = 0.02,
                 declination: float = 0.0):
        """
        Initialize compass sensor.
        
        Args:
            name: Sensor name
            update_rate: Update rate in Hz
            noise_stddev: Heading noise in radians
            declination: Magnetic declination in radians
        """
        super().__init__(name, update_rate)
        
        self.noise_stddev = noise_stddev
        self.declination = declination
    
    def read(self) -> SensorReading:
        """Read compass heading."""
        timestamp = time.time()
        
        if not self._robot:
            return SensorReading(
                sensor_name=self.name,
                timestamp=timestamp,
                data={'heading': 0.0},
                metadata={}
            )
        
        # Get true heading
        true_heading = np.radians(self._robot.heading)
        
        # Add noise and declination
        noise = np.random.normal(0, self.noise_stddev)
        magnetic_heading = true_heading + self.declination + noise
        
        # Normalize to [0, 2π]
        magnetic_heading = magnetic_heading % (2 * np.pi)
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=timestamp,
            data={
                'heading': magnetic_heading,
                'heading_degrees': np.degrees(magnetic_heading)
            },
            metadata={
                'declination': self.declination
            }
        )
        
        self._last_reading = reading
        return reading
