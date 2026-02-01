"""
SynthWorld HUD System

Heads-up display for the simulation.
"""

from direct.gui.OnscreenText import OnscreenText
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import TextNode, Vec3, Vec4, TransparencyAttrib
import logging

logger = logging.getLogger(__name__)


class HUD:
    """
    Heads-Up Display manager.
    Displays vital statistics like FPS, speed, time, and mode.
    """
    
    def __init__(self):
        self.elements = {}
        self.visible = True
        
        # Style constants - Cyberpunk theme
        self.font_color = (0.2, 0.9, 1.0, 1.0)  # Cyan
        self.accent_color = (1.0, 0.2, 0.6, 1.0) # Magenta
        self.bg_color = (0.05, 0.05, 0.1, 0.8)   # Dark semi-transparent
        
        self._setup_ui()
        logger.info("HUD initialized")
    
    def _setup_ui(self):
        """Create basic UI elements."""
        # Stats (Top Left)
        self.elements['stats_bg'] = self._create_label(
            "", (-0.95, 0.90), scale=0.04, align=TextNode.ALeft
        )
        
        # Time (Top Center)
        self.elements['time'] = self._create_label(
            "12:00", (0, 0.90), scale=0.06, align=TextNode.ACenter,
            fg=self.font_color
        )
        
        # Mode (Bottom Left)
        self.elements['mode'] = self._create_label(
            "MODE: FREE CAM", (-0.95, -0.90), scale=0.05, align=TextNode.ALeft,
            fg=self.accent_color
        )
        
        # Vehicle Info (Bottom Right)
        self.elements['speed'] = self._create_label(
            "0 km/h", (0.95, -0.90), scale=0.06, align=TextNode.ARight,
            fg=self.font_color
        )
        
        # Crosshair (Center)
        self.elements['crosshair'] = OnscreenText(
            text="+",
            pos=(0, 0),
            scale=0.05,
            fg=(1, 1, 1, 0.5),
            align=TextNode.ACenter,
            mayChange=False
        )
    
    def _create_label(self, text, pos, scale=0.05, align=TextNode.ALeft, fg=None):
        """Helper to create text labels."""
        if fg is None:
            fg = self.font_color
            
        return OnscreenText(
            text=text,
            pos=pos,
            scale=scale,
            fg=fg,
            align=align,
            shadow=(0, 0, 0, 1),
            shadowOffset=(0.05, 0.05),
            mayChange=True
        )
    
    def update(self, dt: float, engine_stats: dict, player_robot=None):
        """
        Update HUD elements.
        
        Args:
            dt: Delta time
            engine_stats: Dictionary containing engine stats (fps, time, etc.)
            player_robot: Reference to player-controlled robot (optional)
        """
        if not self.visible:
            return
            
        # Update stats
        fps = engine_stats.get('fps', 0.0)
        sim_time = engine_stats.get('sim_time', 0.0)
        entities = engine_stats.get('info', '')
        
        self.elements['stats_bg'].setText(
            f"FPS: {fps:.1f}\n"
            f"Time: {sim_time:.1f}s"
        )
        
        # Update robot info
        if player_robot:
            vel = player_robot.velocity
            # Ensure velocity is a scalar (speed)
            if hasattr(vel, '__len__'):
                import numpy as np
                speed_mps = np.linalg.norm(vel)
            else:
                speed_mps = float(vel)
                
            speed_kph = speed_mps * 3.6
            self.elements['speed'].setText(f"{speed_kph:.1f} km/h")
        else:
            self.elements['speed'].setText("")
            
    def set_mode_text(self, text: str):
        """Update the mode text."""
        if 'mode' in self.elements:
            self.elements['mode'].setText(text)
            
    def set_time_text(self, hour: float):
        """Update time display."""
        h = int(hour)
        m = int((hour % 1) * 60)
        self.elements['time'].setText(f"{h:02d}:{m:02d}")

    def show(self):
        self.visible = True
        for el in self.elements.values():
            el.show()
            
    def hide(self):
        self.visible = False
        for el in self.elements.values():
            el.hide()
