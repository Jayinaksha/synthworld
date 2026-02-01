"""
SynthWorld Input Manager

Handles keyboard, mouse, and gamepad input with action mapping.
"""

from direct.showbase.DirectObject import DirectObject
from panda3d.core import KeyboardButton, MouseButton, ModifierButtons
from typing import Dict, Set, Callable, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class InputAction(Enum):
    """Predefined input actions."""
    # Movement
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    
    # Looking
    LOOK_LEFT = auto()
    LOOK_RIGHT = auto()
    LOOK_UP = auto()
    LOOK_DOWN = auto()
    
    # Camera
    CAMERA_ZOOM_IN = auto()
    CAMERA_ZOOM_OUT = auto()
    CAMERA_MODE_NEXT = auto()
    
    # Robot Control
    ROBOT_ACCELERATE = auto()
    ROBOT_BRAKE = auto()
    ROBOT_TURN_LEFT = auto()
    ROBOT_TURN_RIGHT = auto()
    ROBOT_ARM_UP = auto()
    ROBOT_ARM_DOWN = auto()
    ROBOT_GRIPPER_TOGGLE = auto()
    
    # Interaction
    INTERACT = auto()
    USE = auto()
    GRAB = auto()
    DROP = auto()
    
    # UI
    PAUSE = auto()
    MENU = auto()
    INVENTORY = auto()
    MAP = auto()
    
    # Data Export
    CAPTURE_FRAME = auto()
    TOGGLE_RECORDING = auto()
    
    # Debug
    DEBUG_TOGGLE = auto()
    DEBUG_PHYSICS = auto()
    DEBUG_TELEPORT = auto()


@dataclass
class InputBinding:
    """A binding between a key/button and an action."""
    action: InputAction
    key: str
    modifiers: List[str] = field(default_factory=list)
    on_press: bool = True  # Trigger on press vs release
    repeat: bool = False   # Allow key repeat


class InputManager(DirectObject):
    """
    Manages input with action mapping and callbacks.
    """
    
    # Default key bindings
    DEFAULT_BINDINGS = [
        # Movement (WASD)
        InputBinding(InputAction.MOVE_FORWARD, 'w', repeat=True),
        InputBinding(InputAction.MOVE_BACKWARD, 's', repeat=True),
        InputBinding(InputAction.MOVE_LEFT, 'a', repeat=True),
        InputBinding(InputAction.MOVE_RIGHT, 'd', repeat=True),
        InputBinding(InputAction.MOVE_UP, 'space', repeat=True),
        InputBinding(InputAction.MOVE_DOWN, 'shift', repeat=True),
        
        # Arrow keys alternative
        InputBinding(InputAction.LOOK_LEFT, 'arrow_left', repeat=True),
        InputBinding(InputAction.LOOK_RIGHT, 'arrow_right', repeat=True),
        InputBinding(InputAction.LOOK_UP, 'arrow_up', repeat=True),
        InputBinding(InputAction.LOOK_DOWN, 'arrow_down', repeat=True),
        
        # Camera
        InputBinding(InputAction.CAMERA_ZOOM_IN, 'wheel_up'),
        InputBinding(InputAction.CAMERA_ZOOM_OUT, 'wheel_down'),
        InputBinding(InputAction.CAMERA_MODE_NEXT, 'c'),
        
        # Robot
        InputBinding(InputAction.ROBOT_ACCELERATE, 'w', repeat=True),
        InputBinding(InputAction.ROBOT_BRAKE, 's', repeat=True),
        InputBinding(InputAction.ROBOT_TURN_LEFT, 'a', repeat=True),
        InputBinding(InputAction.ROBOT_TURN_RIGHT, 'd', repeat=True),
        InputBinding(InputAction.ROBOT_ARM_UP, 'q', repeat=True),
        InputBinding(InputAction.ROBOT_ARM_DOWN, 'e', repeat=True),
        InputBinding(InputAction.ROBOT_GRIPPER_TOGGLE, 'g'),
        
        # Interaction
        InputBinding(InputAction.INTERACT, 'f'),
        InputBinding(InputAction.USE, 'mouse1'),
        InputBinding(InputAction.GRAB, 'mouse3'),
        
        # UI
        InputBinding(InputAction.PAUSE, 'escape'),
        InputBinding(InputAction.MENU, 'tab'),
        InputBinding(InputAction.INVENTORY, 'i'),
        InputBinding(InputAction.MAP, 'm'),
        
        # Data export
        InputBinding(InputAction.CAPTURE_FRAME, 'p'),
        InputBinding(InputAction.TOGGLE_RECORDING, 'r', modifiers=['control']),
        
        # Debug
        InputBinding(InputAction.DEBUG_TOGGLE, 'f1'),
        InputBinding(InputAction.DEBUG_PHYSICS, 'f2'),
        InputBinding(InputAction.DEBUG_TELEPORT, 'f3'),
    ]
    
    def __init__(self, base):
        """
        Initialize input manager.
        
        Args:
            base: Panda3D ShowBase instance
        """
        super().__init__()
        self.base = base
        
        # Active bindings
        self._bindings: Dict[str, List[InputBinding]] = {}
        
        # Currently pressed keys
        self._pressed: Set[str] = set()
        
        # Action callbacks
        self._action_callbacks: Dict[InputAction, List[Callable]] = {}
        
        # Mouse state
        self._mouse_pos: Tuple[float, float] = (0, 0)
        self._mouse_delta: Tuple[float, float] = (0, 0)
        self._last_mouse_pos: Optional[Tuple[float, float]] = None
        self._mouse_locked = False
        
        # Set up default bindings
        self._setup_default_bindings()
        
        # Register mouse movement tracking
        self.base.taskMgr.add(self._update_mouse, 'input_mouse_update')
        
        logger.info("InputManager initialized")
    
    def _setup_default_bindings(self):
        """Set up default key bindings."""
        for binding in self.DEFAULT_BINDINGS:
            self.add_binding(binding)
    
    def add_binding(self, binding: InputBinding):
        """Add an input binding."""
        key = binding.key
        
        if key not in self._bindings:
            self._bindings[key] = []
            
            # Register with Panda3D
            if binding.on_press:
                self.accept(key, self._on_key_press, [key])
                self.accept(f'{key}-up', self._on_key_release, [key])
            else:
                self.accept(f'{key}-up', self._on_key_release, [key])
        
        self._bindings[key].append(binding)
    
    def remove_binding(self, action: InputAction, key: str):
        """Remove a binding."""
        if key in self._bindings:
            self._bindings[key] = [b for b in self._bindings[key] if b.action != action]
            if not self._bindings[key]:
                self.ignore(key)
                self.ignore(f'{key}-up')
                del self._bindings[key]
    
    def register_callback(self, action: InputAction, callback: Callable):
        """Register a callback for an action."""
        if action not in self._action_callbacks:
            self._action_callbacks[action] = []
        self._action_callbacks[action].append(callback)
    
    def unregister_callback(self, action: InputAction, callback: Callable):
        """Unregister a callback."""
        if action in self._action_callbacks:
            self._action_callbacks[action] = [
                c for c in self._action_callbacks[action] if c != callback
            ]
    
    def _on_key_press(self, key: str):
        """Handle key press."""
        self._pressed.add(key)
        
        if key in self._bindings:
            for binding in self._bindings[key]:
                if binding.on_press:
                    self._trigger_action(binding.action)
    
    def _on_key_release(self, key: str):
        """Handle key release."""
        self._pressed.discard(key)
        
        if key in self._bindings:
            for binding in self._bindings[key]:
                if not binding.on_press:
                    self._trigger_action(binding.action)
    
    def _trigger_action(self, action: InputAction):
        """Trigger all callbacks for an action."""
        if action in self._action_callbacks:
            for callback in self._action_callbacks[action]:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in input callback for {action}: {e}")
    
    def _update_mouse(self, task):
        """Update mouse position and delta."""
        if self.base.mouseWatcherNode.hasMouse():
            x = self.base.mouseWatcherNode.getMouseX()
            y = self.base.mouseWatcherNode.getMouseY()
            
            self._mouse_pos = (x, y)
            
            if self._last_mouse_pos is not None:
                self._mouse_delta = (
                    x - self._last_mouse_pos[0],
                    y - self._last_mouse_pos[1]
                )
            else:
                self._mouse_delta = (0, 0)
            
            self._last_mouse_pos = (x, y)
            
            # Re-center mouse if locked
            if self._mouse_locked:
                self.base.win.movePointer(0, 
                    int(self.base.win.getXSize() / 2),
                    int(self.base.win.getYSize() / 2)
                )
                self._last_mouse_pos = None
        
        return task.cont
    
    def is_pressed(self, key: str) -> bool:
        """Check if a key is currently pressed."""
        return key in self._pressed
    
    def is_action_active(self, action: InputAction) -> bool:
        """Check if any key bound to an action is pressed."""
        for key, bindings in self._bindings.items():
            for binding in bindings:
                if binding.action == action and binding.repeat:
                    if key in self._pressed:
                        return True
        return False
    
    @property
    def mouse_position(self) -> Tuple[float, float]:
        """Get normalized mouse position (-1 to 1)."""
        return self._mouse_pos
    
    @property
    def mouse_delta(self) -> Tuple[float, float]:
        """Get mouse movement since last frame."""
        return self._mouse_delta
    
    def lock_mouse(self):
        """Lock mouse to window center (FPS-style control)."""
        props = self.base.win.getProperties()
        props.setCursorHidden(True)
        self.base.win.requestProperties(props)
        self._mouse_locked = True
    
    def unlock_mouse(self):
        """Unlock mouse."""
        props = self.base.win.getProperties()
        props.setCursorHidden(False)
        self.base.win.requestProperties(props)
        self._mouse_locked = False
        self._last_mouse_pos = None
    
    def get_movement_input(self) -> Tuple[float, float, float]:
        """
        Get movement input as a vector (forward, right, up).
        Values are -1, 0, or 1.
        """
        forward = 0
        right = 0
        up = 0
        
        if self.is_action_active(InputAction.MOVE_FORWARD):
            forward += 1
        if self.is_action_active(InputAction.MOVE_BACKWARD):
            forward -= 1
        if self.is_action_active(InputAction.MOVE_RIGHT):
            right += 1
        if self.is_action_active(InputAction.MOVE_LEFT):
            right -= 1
        if self.is_action_active(InputAction.MOVE_UP):
            up += 1
        if self.is_action_active(InputAction.MOVE_DOWN):
            up -= 1
        
        return (forward, right, up)
    
    def get_look_input(self) -> Tuple[float, float]:
        """
        Get look input as (horizontal, vertical).
        Combines keyboard arrows and mouse delta.
        """
        h = 0
        v = 0
        
        # Keyboard
        if self.is_action_active(InputAction.LOOK_LEFT):
            h -= 1
        if self.is_action_active(InputAction.LOOK_RIGHT):
            h += 1
        if self.is_action_active(InputAction.LOOK_UP):
            v += 1
        if self.is_action_active(InputAction.LOOK_DOWN):
            v -= 1
        
        # Add mouse delta (scaled)
        if self._mouse_locked:
            h += self._mouse_delta[0] * 10
            v += self._mouse_delta[1] * 10
        
        return (h, v)
    
    def get_robot_input(self) -> Dict[str, float]:
        """Get robot control input."""
        result = {
            'throttle': 0,
            'steering': 0,
            'arm_vertical': 0,
            'gripper': False
        }
        
        if self.is_action_active(InputAction.ROBOT_ACCELERATE):
            result['throttle'] += 1
        if self.is_action_active(InputAction.ROBOT_BRAKE):
            result['throttle'] -= 1
        if self.is_action_active(InputAction.ROBOT_TURN_LEFT):
            result['steering'] -= 1
        if self.is_action_active(InputAction.ROBOT_TURN_RIGHT):
            result['steering'] += 1
        if self.is_action_active(InputAction.ROBOT_ARM_UP):
            result['arm_vertical'] += 1
        if self.is_action_active(InputAction.ROBOT_ARM_DOWN):
            result['arm_vertical'] -= 1
        
        return result
    
    def cleanup(self):
        """Clean up input bindings."""
        self.ignoreAll()
        self.base.taskMgr.remove('input_mouse_update')
