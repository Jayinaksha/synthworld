"""
SynthWorld Application

Main application class that ties all systems together.
"""

import logging
import time
import random
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class SynthWorldApp:
    """
    Main SynthWorld application.
    
    Integrates all subsystems:
    - Engine (physics, rendering, input)
    - World generation
    - Robot simulation
    - NPC system
    - Data capture
    """
    
    def __init__(self, config_path: str = None,
                 headless: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize SynthWorld application.
        
        Args:
            config_path: Path to configuration file
            headless: Run without display
            seed: Random seed for reproducibility
        """
        self._headless = headless
        self._seed = seed or random.randint(0, 999999)
        
        # Set random seeds
        random.seed(self._seed)
        np.random.seed(self._seed)
        
        logger.info(f"Initializing SynthWorld (seed: {self._seed})")
        
        # Store and load configuration
        self._config_path = config_path
        self._config = self._load_config(config_path)
        
        # Initialize subsystems
        self._engine = None
        self._world_gen = None
        self._robot_manager = None
        self._npc_manager = None
        self._data_capture = None
        
        # Player/robot reference
        self._player_robot = None
        
        # Game state
        self._is_running = False
        self._is_paused = False
        self._simulation_time = 0.0
        
        # Initialize
        self._initialize()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml
        
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
        
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Config not found at {config_path}, using defaults")
            config = self._default_config()
        
        return config
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'display': {
                'width': 1280,
                'height': 720,
                'fullscreen': False,
                'target_fps': 60
            },
            'physics': {
                'timestep': 1/240,
                'gravity': -9.81
            },
            'world': {
                'seed': self._seed,
                'chunk_size': 100,
                'render_distance': 3
            },
            'robot': {
                'default_type': 'wheeled'
            },
            'npcs': {
                'max_pedestrians': 50,
                'max_vehicles': 30
            }
        }
    
    def _initialize(self):
        """Initialize all subsystems."""
        # Initialize engine
        from .engine import Engine
        self._engine = Engine(config_path=self._config_path)
        
        # Initialize world generator
        from .world import WorldGenerator
        world_config = self._config.get('world', {})
        self._world_gen = WorldGenerator(
            seed=world_config.get('seed', self._seed),
            chunk_size=world_config.get('chunk_size', 100)
        )
        # Set references for world generator
        self._world_gen.set_renderer(self._engine.renderer)
        self._world_gen.set_physics(self._engine.physics)
        
        # Initialize robot manager
        from .robots import RobotManager
        self._robot_manager = RobotManager(
            physics_world=self._engine.physics,
            renderer=self._engine.renderer
        )
        
        # Initialize NPC manager
        from .npcs import NPCManager
        npc_config = self._config.get('npcs', {})
        self._npc_manager = NPCManager(
            physics_world=self._engine.physics,
            renderer=self._engine.renderer,
            max_npcs=npc_config.get('max_pedestrians', 50) + npc_config.get('max_vehicles', 30)
        )
        
        # Register systems with engine
        self._engine.register_system('world', self._world_gen)
        self._engine.register_system('robots', self._robot_manager)
        self._engine.register_system('npcs', self._npc_manager)
        
        # Setup visible scene
        self._setup_scene()
        
        # Initialize UI
        from .ui.hud import HUD
        self._hud = HUD()
        
        # Setup input callbacks
        self._setup_input()
        
        logger.info("All subsystems initialized")
    
    def _setup_scene(self):
        """Setup initial visible scene elements."""
        renderer = self._engine.renderer
        
        # Set a nice cyberpunk sky color (dark blue/purple)
        renderer.set_background_color(0.05, 0.02, 0.1)
        
        # Create a large ground plane so there's something visible
        ground = renderer.create_plane(
            500, 500,
            color=(0.15, 0.15, 0.18, 1.0),  # Dark concrete
            name="ground"
        )
        ground.setPos(0, 0, 0)
        
        # Create a physics ground plane
        self._engine.physics.create_ground_plane()
        
        # Position camera to see the action
        cam = renderer.camera_controller
        cam.position.x = 0
        cam.position.y = -30
        cam.position.z = 15
        cam.heading = 0
        cam.pitch = -20
        
        # Set time to noon for good lighting
        renderer.set_time_of_day(12)
        
        # Create some initial test objects so scene isn't empty
        renderer.create_box(
            2, 2, 2,
            color=(0.8, 0.2, 0.2, 1.0),  # Red box
            name="test_box_1",
            position=(3, 0, 1)
        )
        renderer.create_sphere(
            1.5,
            color=(0.2, 0.8, 0.2, 1.0),  # Green sphere
            name="test_sphere_1",
            position=(-3, 0, 1.5)
        )
        renderer.create_cylinder(
            0.5, 4,
            color=(0.2, 0.2, 0.8, 1.0),  # Blue cylinder
            name="test_cylinder_1",
            position=(0, 5, 2)
        )
        
        logger.info("Scene setup complete")
    
    def _setup_input(self):
        """Setup input callbacks."""
        input_mgr = self._engine.input
        
        # Pause toggle
        input_mgr.register_callback('pause', self._toggle_pause)
        
        # Camera mode cycling
        input_mgr.register_callback('camera_mode', self._cycle_camera)
        
        # Robot spawn
        input_mgr.register_callback('spawn_robot', self._spawn_next_robot)
    
    def _toggle_pause(self):
        """Toggle pause state."""
        self._is_paused = not self._is_paused
        logger.info(f"Simulation {'paused' if self._is_paused else 'resumed'}")
    
    def _cycle_camera(self):
        """Cycle through camera modes."""
        if self._engine and self._engine.renderer:
            mode = self._engine.renderer.cycle_camera_mode()
            if self._hud:
                self._hud.set_mode_text(f"MODE: {mode.replace('_', ' ')}")
    
    def _spawn_next_robot(self):
        """Spawn the next robot type."""
        robot_types = ['wheeled', 'quadruped', 'arm', 'drone']
        current_count = len(self._robot_manager.get_all_robots())
        robot_type = robot_types[current_count % len(robot_types)]
        
        self.spawn_robot(robot_type)
    
    def spawn_robot(self, robot_type: str = 'wheeled',
                   position: tuple = None) -> Optional[object]:
        """
        Spawn a robot in the world.
        
        Args:
            robot_type: Type of robot ('wheeled', 'quadruped', 'arm', 'drone')
            position: Spawn position (default: near current camera)
        
        Returns:
            The spawned robot
        """
        if position is None:
            # Spawn near center
            position = (0, 0, 1)
        
        robot = None
        name = f"{robot_type}_{len(self._robot_manager.get_all_robots())}"
        
        if robot_type == 'wheeled':
            from .robots import DifferentialDriveRobot
            robot = DifferentialDriveRobot(
                name, self._engine.physics, self._engine.renderer
            )
        elif robot_type == 'quadruped':
            from .robots import QuadrupedRobot
            robot = QuadrupedRobot(
                name, self._engine.physics, self._engine.renderer
            )
        elif robot_type == 'arm':
            from .robots import ArmRobot
            robot = ArmRobot(
                name, self._engine.physics, self._engine.renderer
            )
        elif robot_type == 'drone':
            from .robots import DroneRobot
            robot = DroneRobot(
                name, self._engine.physics, self._engine.renderer
            )
        
        if robot:
            robot.spawn(position)
            self._robot_manager.add_robot(robot)
            
            if self._player_robot is None:
                self._player_robot = robot
                self._robot_manager.set_active_robot(name)
            
            logger.info(f"Spawned {robot_type} robot '{name}' at {position}")
        
        return robot
    
    def spawn_npcs(self, num_pedestrians: int = 20, num_vehicles: int = 10):
        """
        Spawn NPCs in the world.
        
        Args:
            num_pedestrians: Number of pedestrians to spawn
            num_vehicles: Number of vehicles to spawn
        """
        from .npcs import CyberpunkCitizen, TrafficCar
        
        # Spawn pedestrians
        self._npc_manager.spawn_random(
            CyberpunkCitizen, num_pedestrians,
            area_center=(0, 0),
            area_size=200
        )
        
        # Spawn vehicles
        self._npc_manager.spawn_random(
            TrafficCar, num_vehicles,
            area_center=(0, 0),
            area_size=300
        )
        
        logger.info(f"Spawned {num_pedestrians} pedestrians and {num_vehicles} vehicles")
    
    def generate_initial_world(self, center: tuple = (0, 0)):
        """
        Generate the initial world around a center point.
        
        Args:
            center: Center coordinates (x, y)
        """
        logger.info("Generating initial world...")
        
        render_distance = self._config.get('world', {}).get('render_distance', 3)
        
        # Generate initial chunks
        self._world_gen.update_loaded_chunks(
            center[0], center[1],
            render_distance=render_distance
        )
        
        logger.info("Initial world generated")
    
    def enable_capture(self, output_path: str = './synthetic_data'):
        """
        Enable data capture.
        
        Args:
            output_path: Output directory for captured data
        """
        from .data import DataCapture, CaptureConfig
        
        config = CaptureConfig(
            output_path=output_path,
            capture_rgb=True,
            capture_depth=True,
            capture_lidar=True,
            auto_annotate=True
        )
        
        self._data_capture = DataCapture(config)
        self._data_capture.set_references(
            physics_world=self._engine.physics,
            renderer=self._engine.renderer,
            robot=self._player_robot,
            npc_manager=self._npc_manager
        )
        
        logger.info(f"Data capture enabled (output: {output_path})")
    
    def run(self):
        """Run the main application loop."""
        logger.info("Starting SynthWorld...")
        
        # Generate initial world
        self.generate_initial_world()
        
        # Lock mouse for camera control
        if self._engine and self._engine.input:
            self._engine.input.lock_mouse()
        
        # Spawn default robot
        self.spawn_robot('wheeled', (0, 0, 1))
        
        # Spawn some NPCs
        self.spawn_npcs(num_pedestrians=20, num_vehicles=10)
        
        # Start data capture if enabled
        if self._data_capture:
            self._data_capture.start_capture()
        
        self._is_running = True
        last_time = time.time()
        
        try:
            while self._is_running:
                # Calculate delta time
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # Cap delta time
                dt = min(dt, 0.1)
                
                if not self._is_paused:
                    # Update simulation
                    self._update(dt)
                    self._simulation_time += dt
                
                # Process engine frame
                if not self._engine.run_frame():
                    self._is_running = False
        
        finally:
            self._cleanup()
    
    def _update(self, dt: float):
        """Update all systems."""
        # Update robot control from input
        if self._player_robot:
            self._update_robot_control()
        
        # Update world chunks based on player position
        if self._player_robot and self._world_gen:
            pos = self._player_robot.position
            self._world_gen.update_loaded_chunks(
                pos[0], pos[1]
            )
        
        # Update UI
        if self._hud:
            stats = self._engine.stats.__dict__
            stats['sim_time'] = self._simulation_time
            self._hud.update(dt, stats, self._player_robot)
            self._hud.set_time_text(self._engine.renderer.time_of_day)
        
        # Update data capture
        if self._data_capture:
            self._data_capture.update(dt)
    
    def _update_robot_control(self):
        """Update robot control from player input."""
        if not self._player_robot or not self._engine.input:
            return
        
        robot_input = self._engine.input.get_robot_input()
        self._player_robot.apply_control(robot_input)
    
    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        if self._data_capture:
            self._data_capture.stop_capture()
        
        if self._npc_manager:
            self._npc_manager.cleanup()
        
        if self._robot_manager:
            self._robot_manager.cleanup()
        
        if self._engine:
            self._engine.shutdown()
    
    def run_demo(self, demo_name: str):
        """
        Run a specific demo.
        
        Args:
            demo_name: Name of demo ('basic', 'robot', 'traffic', 'data')
        """
        logger.info(f"Running demo: {demo_name}")
        
        if demo_name == 'basic':
            self._demo_basic()
        elif demo_name == 'robot':
            self._demo_robot()
        elif demo_name == 'traffic':
            self._demo_traffic()
        elif demo_name == 'data':
            self._demo_data_generation()
    
    def _demo_basic(self):
        """Basic demo - world exploration."""
        self.generate_initial_world()
        self.spawn_robot('wheeled')
        self.run()
    
    def _demo_robot(self):
        """Robot demo - showcase different robot types."""
        self.generate_initial_world()
        
        # Spawn different robot types
        self.spawn_robot('wheeled', (0, 0, 1))
        self.spawn_robot('quadruped', (5, 0, 1))
        self.spawn_robot('drone', (0, 5, 3))
        self.spawn_robot('arm', (-5, 0, 1))
        
        self.run()
    
    def _demo_traffic(self):
        """Traffic demo - busy city simulation."""
        self.generate_initial_world()
        self.spawn_robot('wheeled')
        self.spawn_npcs(num_pedestrians=50, num_vehicles=30)
        self.run()
    
    def _demo_data_generation(self):
        """Data generation demo - capture synthetic data."""
        self.generate_initial_world()
        self.spawn_robot('wheeled')
        self.spawn_npcs(num_pedestrians=30, num_vehicles=20)
        self.enable_capture('./demo_data')
        self.run()
