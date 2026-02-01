"""
SynthWorld Core Engine

Main game loop and system coordination.
"""

import yaml
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from .physics import PhysicsWorld
from .renderer import Renderer, CameraMode
from .input import InputManager, InputAction

logger = logging.getLogger(__name__)


@dataclass
class EngineStats:
    """Engine performance statistics."""
    fps: float = 0
    frame_time: float = 0
    physics_time: float = 0
    render_time: float = 0
    frame_count: int = 0
    sim_time: float = 0


class Engine:
    """
    Main game engine that coordinates all systems.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the engine.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize logging
        log_level = logging.DEBUG if self.config.get('debug', {}).get('verbose_logging', False) else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Performance tracking
        self.stats = EngineStats()
        self._last_frame_time = time.time()
        self._fps_samples: List[float] = []
        
        # Initialize renderer (Panda3D) - must be first
        logger.info("Initializing renderer...")
        self.renderer = Renderer(self.config.get('display', {}))
        
        # Initialize physics
        logger.info("Initializing physics...")
        physics_config = self.config.get('physics', {})
        self.physics = PhysicsWorld(
            gravity=tuple(physics_config.get('gravity', [0, 0, -9.81])),
            timestep=physics_config.get('timestep', 1/240),
            solver_iterations=physics_config.get('solver_iterations', 50),
            use_gui=False  # We use Panda3D for rendering
        )
        
        # Initialize input
        logger.info("Initializing input...")
        self.input = InputManager(self.renderer)
        
        # Set up input callbacks
        self._setup_input_callbacks()
        
        # Game state
        self._running = False
        self._paused = False
        
        # Systems registry
        self._systems: Dict[str, Any] = {}
        self._update_callbacks: List[Callable[[float], None]] = []
        
        # Physics accumulator for fixed timestep
        self._physics_accumulator = 0.0
        self._physics_timestep = physics_config.get('timestep', 1/240)
        
        # Target frame time
        self._target_fps = self.config.get('display', {}).get('target_fps', 60)
        self._target_frame_time = 1.0 / self._target_fps
        
        logger.info("Engine initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Try default path
        default_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
        if default_path.exists():
            with open(default_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Return minimal default config
        return {
            'display': {'width': 1280, 'height': 720, 'target_fps': 60},
            'physics': {'gravity': [0, 0, -9.81], 'timestep': 1/240}
        }
    
    def _setup_input_callbacks(self):
        """Set up input action callbacks."""
        # Pause
        self.input.register_callback(InputAction.PAUSE, self._toggle_pause)
        
        # Camera mode
        self.input.register_callback(InputAction.CAMERA_MODE_NEXT, self._cycle_camera_mode)
        
        # Debug
        self.input.register_callback(InputAction.DEBUG_TOGGLE, self._toggle_debug)
        self.input.register_callback(InputAction.DEBUG_PHYSICS, self._toggle_physics_debug)
    
    def _toggle_pause(self):
        """Toggle pause state."""
        self._paused = not self._paused
        logger.info(f"Game {'paused' if self._paused else 'resumed'}")
    
    def _cycle_camera_mode(self):
        """Cycle through camera modes."""
        modes = [CameraMode.FREE_CAM, CameraMode.THIRD_PERSON, 
                 CameraMode.FIRST_PERSON, CameraMode.ORBIT]
        current = self.renderer.camera_controller.mode
        try:
            idx = modes.index(current)
            next_mode = modes[(idx + 1) % len(modes)]
        except ValueError:
            next_mode = modes[0]
        
        self.renderer.camera_controller.set_mode(next_mode)
    
    def _toggle_debug(self):
        """Toggle debug display."""
        # Toggle FPS meter
        if hasattr(self.renderer, 'frameRateMeter'):
            if self.renderer.frameRateMeter:
                self.renderer.setFrameRateMeter(False)
            else:
                self.renderer.setFrameRateMeter(True)
    
    def _toggle_physics_debug(self):
        """Toggle physics debug visualization."""
        # This would enable PyBullet debug rendering
        # For now, just log
        logger.info("Physics debug toggle (not yet implemented)")
    
    def register_system(self, name: str, system: Any):
        """Register a game system."""
        self._systems[name] = system
        logger.info(f"Registered system: {name}")
    
    def get_system(self, name: str) -> Optional[Any]:
        """Get a registered system."""
        return self._systems.get(name)
    
    def register_update_callback(self, callback: Callable[[float], None]):
        """Register a callback to be called every frame with delta time."""
        self._update_callbacks.append(callback)
    
    def _update(self, dt: float):
        """Main update loop."""
        if self._paused:
            return
        
        # Update physics with fixed timestep
        physics_start = time.time()
        self._physics_accumulator += dt
        
        while self._physics_accumulator >= self._physics_timestep:
            self.physics.step()
            self._physics_accumulator -= self._physics_timestep
        
        self.stats.physics_time = time.time() - physics_start
        
        # Update camera
        movement = self.input.get_movement_input()
        look = self.input.get_look_input()
        
        self.renderer.camera_controller.move(movement[0], movement[1], movement[2], dt)
        self.renderer.camera_controller.rotate(-look[0], look[1], dt)
        self.renderer.camera_controller.update(dt)
        
        # Update day/night
        self.renderer.update_day_night(dt)
        
        # Call registered update callbacks
        for callback in self._update_callbacks:
            try:
                callback(dt)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
        
        # Update simulation time
        self.stats.sim_time += dt
    
    def _calculate_stats(self, dt: float):
        """Calculate performance statistics."""
        self.stats.frame_time = dt
        self.stats.frame_count += 1
        
        # Rolling FPS average
        self._fps_samples.append(1.0 / dt if dt > 0 else 0)
        if len(self._fps_samples) > 60:
            self._fps_samples.pop(0)
        
        self.stats.fps = sum(self._fps_samples) / len(self._fps_samples)
    
    def run(self):
        """Run the main game loop."""
        logger.info("Starting engine...")
        self._running = True
        
        # Lock mouse for FPS-style control
        self.input.lock_mouse()
        
        try:
            while self._running:
                # Calculate delta time
                current_time = time.time()
                dt = current_time - self._last_frame_time
                self._last_frame_time = current_time
                
                # Cap delta time to prevent spiral of death
                dt = min(dt, 0.1)
                
                # Update
                self._update(dt)
                
                # Render
                render_start = time.time()
                if not self.renderer.run_frame():
                    self._running = False
                    break
                self.stats.render_time = time.time() - render_start
                
                # Calculate stats
                self._calculate_stats(dt)
                
                # Frame rate limiting (if running faster than target)
                elapsed = time.time() - current_time
                if elapsed < self._target_frame_time:
                    time.sleep(self._target_frame_time - elapsed)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()
    
    def run_frame(self) -> bool:
        """
        Run a single frame update and render.
        
        Returns:
            True if window is still open, False if closed
        """
        # Calculate delta time
        current_time = time.time()
        dt = current_time - self._last_frame_time
        self._last_frame_time = current_time
        
        # Cap delta time
        dt = min(dt, 0.1)
        
        # Update (if not paused)
        if not self._paused:
            self._update(dt)
        
        # Render
        render_start = time.time()
        if not self.renderer.run_frame():
            return False
        self.stats.render_time = time.time() - render_start
        
        # Calculate stats
        self._calculate_stats(dt)
        
        return True
    
    def stop(self):
        """Stop the engine."""
        self._running = False
    
    def shutdown(self):
        """Clean up and shut down."""
        logger.info("Shutting down engine...")
        
        # Clean up systems
        for name, system in self._systems.items():
            if hasattr(system, 'cleanup'):
                system.cleanup()
        
        # Clean up input
        self.input.cleanup()
        
        # Close physics
        self.physics.close()
        
        logger.info("Engine shutdown complete")


def create_engine(config_path: Optional[str] = None) -> Engine:
    """
    Create and return an engine instance.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        Initialized Engine instance
    """
    return Engine(config_path)
