"""
SynthWorld Behavior Trees

Behavior tree implementation for NPC AI.
"""

from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import logging

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status returned by behavior tree nodes."""
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class BehaviorNode(ABC):
    """
    Base class for behavior tree nodes.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.parent: Optional['BehaviorNode'] = None
        self.blackboard: Dict[str, Any] = {}
    
    def set_blackboard(self, blackboard: Dict[str, Any]):
        """Set the blackboard (shared memory) for this node."""
        self.blackboard = blackboard
    
    @abstractmethod
    def tick(self, dt: float) -> NodeStatus:
        """
        Execute this node.
        
        Args:
            dt: Time delta in seconds
        
        Returns:
            NodeStatus indicating result
        """
        pass
    
    def reset(self):
        """Reset node state."""
        pass


class CompositeNode(BehaviorNode):
    """Base class for nodes with children."""
    
    def __init__(self, name: str = "", children: List[BehaviorNode] = None):
        super().__init__(name)
        self.children = children or []
        
        for child in self.children:
            child.parent = self
    
    def add_child(self, child: BehaviorNode):
        """Add a child node."""
        child.parent = self
        self.children.append(child)
    
    def set_blackboard(self, blackboard: Dict[str, Any]):
        super().set_blackboard(blackboard)
        for child in self.children:
            child.set_blackboard(blackboard)
    
    def reset(self):
        for child in self.children:
            child.reset()


class Sequence(CompositeNode):
    """
    Executes children in sequence.
    Returns FAILURE if any child fails.
    Returns SUCCESS when all children succeed.
    """
    
    def __init__(self, name: str = "Sequence", children: List[BehaviorNode] = None):
        super().__init__(name, children)
        self._current_child_index = 0
    
    def tick(self, dt: float) -> NodeStatus:
        while self._current_child_index < len(self.children):
            child = self.children[self._current_child_index]
            status = child.tick(dt)
            
            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.FAILURE:
                self._current_child_index = 0
                return NodeStatus.FAILURE
            
            self._current_child_index += 1
        
        self._current_child_index = 0
        return NodeStatus.SUCCESS
    
    def reset(self):
        super().reset()
        self._current_child_index = 0


class Selector(CompositeNode):
    """
    Executes children until one succeeds.
    Returns SUCCESS if any child succeeds.
    Returns FAILURE when all children fail.
    """
    
    def __init__(self, name: str = "Selector", children: List[BehaviorNode] = None):
        super().__init__(name, children)
        self._current_child_index = 0
    
    def tick(self, dt: float) -> NodeStatus:
        while self._current_child_index < len(self.children):
            child = self.children[self._current_child_index]
            status = child.tick(dt)
            
            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.SUCCESS:
                self._current_child_index = 0
                return NodeStatus.SUCCESS
            
            self._current_child_index += 1
        
        self._current_child_index = 0
        return NodeStatus.FAILURE
    
    def reset(self):
        super().reset()
        self._current_child_index = 0


class Parallel(CompositeNode):
    """
    Executes all children in parallel.
    Policy determines when to return SUCCESS/FAILURE.
    """
    
    def __init__(self, name: str = "Parallel", 
                 children: List[BehaviorNode] = None,
                 success_threshold: int = -1,
                 failure_threshold: int = 1):
        super().__init__(name, children)
        # -1 means all must succeed
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
    
    def tick(self, dt: float) -> NodeStatus:
        success_count = 0
        failure_count = 0
        
        for child in self.children:
            status = child.tick(dt)
            
            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1
        
        # Check thresholds
        required_success = len(self.children) if self.success_threshold < 0 else self.success_threshold
        
        if success_count >= required_success:
            return NodeStatus.SUCCESS
        if failure_count >= self.failure_threshold:
            return NodeStatus.FAILURE
        
        return NodeStatus.RUNNING


class Decorator(BehaviorNode):
    """Base class for decorator nodes (single child)."""
    
    def __init__(self, name: str = "", child: BehaviorNode = None):
        super().__init__(name)
        self.child = child
        if child:
            child.parent = self
    
    def set_child(self, child: BehaviorNode):
        child.parent = self
        self.child = child
    
    def set_blackboard(self, blackboard: Dict[str, Any]):
        super().set_blackboard(blackboard)
        if self.child:
            self.child.set_blackboard(blackboard)
    
    def reset(self):
        if self.child:
            self.child.reset()


class Inverter(Decorator):
    """Inverts the result of the child node."""
    
    def tick(self, dt: float) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE
        
        status = self.child.tick(dt)
        
        if status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS
        
        return NodeStatus.RUNNING


class Succeeder(Decorator):
    """Always returns SUCCESS."""
    
    def tick(self, dt: float) -> NodeStatus:
        if self.child:
            self.child.tick(dt)
        return NodeStatus.SUCCESS


class Repeater(Decorator):
    """Repeats child node a number of times."""
    
    def __init__(self, name: str = "Repeater", child: BehaviorNode = None,
                 times: int = -1):
        super().__init__(name, child)
        self.times = times  # -1 = infinite
        self._count = 0
    
    def tick(self, dt: float) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE
        
        status = self.child.tick(dt)
        
        if status == NodeStatus.RUNNING:
            return NodeStatus.RUNNING
        
        self._count += 1
        
        if self.times > 0 and self._count >= self.times:
            self._count = 0
            return NodeStatus.SUCCESS
        
        self.child.reset()
        return NodeStatus.RUNNING
    
    def reset(self):
        super().reset()
        self._count = 0


class Cooldown(Decorator):
    """Prevents child from running until cooldown expires."""
    
    def __init__(self, name: str = "Cooldown", child: BehaviorNode = None,
                 duration: float = 1.0):
        super().__init__(name, child)
        self.duration = duration
        self._time_remaining = 0.0
    
    def tick(self, dt: float) -> NodeStatus:
        if self._time_remaining > 0:
            self._time_remaining -= dt
            return NodeStatus.FAILURE
        
        if not self.child:
            return NodeStatus.FAILURE
        
        status = self.child.tick(dt)
        
        if status != NodeStatus.RUNNING:
            self._time_remaining = self.duration
        
        return status
    
    def reset(self):
        super().reset()
        self._time_remaining = 0


class Condition(BehaviorNode):
    """Checks a condition and returns SUCCESS or FAILURE."""
    
    def __init__(self, name: str = "Condition",
                 condition: Callable[[Dict[str, Any]], bool] = None):
        super().__init__(name)
        self._condition = condition
    
    def tick(self, dt: float) -> NodeStatus:
        if self._condition and self._condition(self.blackboard):
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class Action(BehaviorNode):
    """Executes an action."""
    
    def __init__(self, name: str = "Action",
                 action: Callable[[Dict[str, Any], float], NodeStatus] = None):
        super().__init__(name)
        self._action = action
    
    def tick(self, dt: float) -> NodeStatus:
        if self._action:
            return self._action(self.blackboard, dt)
        return NodeStatus.FAILURE


class Wait(BehaviorNode):
    """Waits for a specified duration."""
    
    def __init__(self, name: str = "Wait", duration: float = 1.0):
        super().__init__(name)
        self.duration = duration
        self._elapsed = 0.0
    
    def tick(self, dt: float) -> NodeStatus:
        self._elapsed += dt
        
        if self._elapsed >= self.duration:
            self._elapsed = 0
            return NodeStatus.SUCCESS
        
        return NodeStatus.RUNNING
    
    def reset(self):
        self._elapsed = 0


class RandomSelector(CompositeNode):
    """Selects a random child to execute."""
    
    def __init__(self, name: str = "RandomSelector",
                 children: List[BehaviorNode] = None):
        super().__init__(name, children)
        self._selected_index = -1
    
    def tick(self, dt: float) -> NodeStatus:
        if not self.children:
            return NodeStatus.FAILURE
        
        if self._selected_index < 0:
            self._selected_index = random.randint(0, len(self.children) - 1)
        
        status = self.children[self._selected_index].tick(dt)
        
        if status != NodeStatus.RUNNING:
            self._selected_index = -1
        
        return status
    
    def reset(self):
        super().reset()
        self._selected_index = -1


class BehaviorTree:
    """
    Complete behavior tree.
    """
    
    def __init__(self, root: BehaviorNode = None, name: str = "BehaviorTree"):
        self.name = name
        self.root = root
        self.blackboard: Dict[str, Any] = {}
        
        if root:
            root.set_blackboard(self.blackboard)
    
    def set_root(self, root: BehaviorNode):
        """Set the root node."""
        self.root = root
        root.set_blackboard(self.blackboard)
    
    def tick(self, dt: float) -> NodeStatus:
        """Execute one tick of the behavior tree."""
        if not self.root:
            return NodeStatus.FAILURE
        
        return self.root.tick(dt)
    
    def reset(self):
        """Reset the tree."""
        if self.root:
            self.root.reset()


# Pre-built NPC behaviors

class NPCBehavior:
    """Factory for common NPC behavior trees."""
    
    @staticmethod
    def create_wanderer() -> BehaviorTree:
        """Create a simple wandering behavior."""
        
        def pick_destination(bb: Dict, dt: float) -> NodeStatus:
            import numpy as np
            npc = bb.get('npc')
            if npc:
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(10, 30)
                target = npc.position + np.array([
                    np.cos(angle) * dist,
                    np.sin(angle) * dist,
                    0
                ])
                bb['target'] = target
                return NodeStatus.SUCCESS
            return NodeStatus.FAILURE
        
        def move_to_target(bb: Dict, dt: float) -> NodeStatus:
            import numpy as np
            npc = bb.get('npc')
            target = bb.get('target')
            
            if npc and target is not None:
                distance = np.linalg.norm(target[:2] - npc.position[:2])
                
                if distance < 2.0:
                    return NodeStatus.SUCCESS
                
                npc.move_towards(target, 1.4, dt)
                return NodeStatus.RUNNING
            
            return NodeStatus.FAILURE
        
        tree = BehaviorTree(
            root=Repeater(
                child=Sequence(children=[
                    Action("PickDestination", pick_destination),
                    Action("MoveToTarget", move_to_target),
                    Wait("Rest", duration=random.uniform(1, 4))
                ]),
                times=-1  # Infinite
            ),
            name="Wanderer"
        )
        
        return tree
    
    @staticmethod
    def create_patrol(waypoints: List) -> BehaviorTree:
        """Create a patrol behavior along waypoints."""
        
        def get_next_waypoint(bb: Dict, dt: float) -> NodeStatus:
            waypoints_list = bb.get('waypoints', [])
            index = bb.get('waypoint_index', 0)
            
            if not waypoints_list:
                return NodeStatus.FAILURE
            
            bb['target'] = waypoints_list[index]
            bb['waypoint_index'] = (index + 1) % len(waypoints_list)
            return NodeStatus.SUCCESS
        
        def move_to_waypoint(bb: Dict, dt: float) -> NodeStatus:
            import numpy as np
            npc = bb.get('npc')
            target = bb.get('target')
            
            if npc and target is not None:
                target_np = np.array(target)
                distance = np.linalg.norm(target_np[:2] - npc.position[:2])
                
                if distance < 1.0:
                    return NodeStatus.SUCCESS
                
                npc.move_towards(target_np, 1.2, dt)
                return NodeStatus.RUNNING
            
            return NodeStatus.FAILURE
        
        tree = BehaviorTree(
            root=Repeater(
                child=Sequence(children=[
                    Action("GetWaypoint", get_next_waypoint),
                    Action("MoveToWaypoint", move_to_waypoint),
                    Wait("Pause", duration=1.0)
                ]),
                times=-1
            ),
            name="Patrol"
        )
        
        tree.blackboard['waypoints'] = waypoints
        tree.blackboard['waypoint_index'] = 0
        
        return tree
    
    @staticmethod
    def create_flee_behavior(threat_key: str = 'threat') -> BehaviorTree:
        """Create a flee behavior from a threat."""
        
        def has_threat(bb: Dict) -> bool:
            return bb.get(threat_key) is not None
        
        def flee(bb: Dict, dt: float) -> NodeStatus:
            import numpy as np
            npc = bb.get('npc')
            threat = bb.get(threat_key)
            
            if npc and threat is not None:
                threat_pos = np.array(threat)
                direction = npc.position - threat_pos
                direction[2] = 0
                
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                
                target = npc.position + direction * 5
                npc.move_towards(target, 5.0, dt)  # Run
                
                # Check if safe
                distance = np.linalg.norm(npc.position - threat_pos)
                if distance > 30:
                    bb[threat_key] = None
                    return NodeStatus.SUCCESS
                
                return NodeStatus.RUNNING
            
            return NodeStatus.FAILURE
        
        tree = BehaviorTree(
            root=Selector(children=[
                Sequence(children=[
                    Condition("HasThreat", has_threat),
                    Action("Flee", flee)
                ]),
                # Fallback: just stand
                Wait("Idle", 1.0)
            ]),
            name="Flee"
        )
        
        return tree
