import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pandas as pd
from datetime import datetime

@dataclass
class Node :
    """Node information"""
    id: str
    x: float  # mm
    y: float  # mm
    node_type: Optional[str] = None
    
@dataclass
class Member :
    """Member information"""
    id: str
    node_start: str
    node_end: str
    member_type: Optional[str] = None  # 'strut' or 'tie'
    force: Optional[float] = None  # kN
    angle: Optional[float] = None  # degrees
    
@dataclass
class Material :
    """Material properties"""
    fck: float  # MPa
    fy: float   # MPa

class STMDesignChecker :
    def __init__(self, material: Material, beam_width: float, beam_height: float, beam_length: float) :
        self.material = material
        self.beam_width = beam_width  # mm
        self.beam_height = beam_height  # mm
        self.beam_length = beam_length  # mm
        self.nodes: Dict[str, Node] = {}
        self.members: Dict[str, Member] = {}
        self.loads: List[Tuple[str, float, float]] = []  # (node_id, Fx, Fy)
        self.supports: List[Tuple[str, str]] = []  # (node_id, support_type)

    def __init__(self, material: Material, beam_width: float, beam_height: float, beam_length: float):
        """
        Args:
            material: Material properties
            beam_width: Beam width (mm)
            beam_height: Beam height (mm)
            beam_length: Beam length (mm)
        """
        self.material = material
        self.b = beam_width
        self.h = beam_height
        self.L = beam_length
        self.nodes: Dict[str, Node] = {}
        self.members: Dict[str, Member] = {}
        self.loads: List[Tuple[str, float, float]] = []
        self.supports: Dict[str, str] = {}
        self.reactions: Dict[str, Dict[str, float]] = {}
        self.results = {
            'struts': {},
            'ties': {},
            'nodes': {},
            'nodes_detailed': {},
            'overall': True,
            'messages': []
        }
        
    def add_node(self, node: Node):
        """Add node"""
        self.nodes[node.id] = node
        
    def add_member(self, member: Member):
        """Add member"""
        self.members[member.id] = member
        
    def add_load(self, node_id: str, Fx: float, Fy: float):
        """Add load at node"""
        self.loads.append((node_id, Fx, Fy))
        
    def add_support(self, node_id: str, support_type: str):
        """Add support (pin or roller)"""
        self.supports[node_id] = support_type
    
    def get_member_length(self, member: Member) -> float:
        """Calculate member length"""
        # 자동화 방법 고려
        n1 = self.nodes[member.node_start]
        n2 = self.nodes[member.node_end]
        return np.sqrt((n2.x - n1.x)**2 + (n2.y - n1.y)**2)
    
    def get_member_angle(self, member: Member, in_degrees: bool = False) -> float:
        """Calculate member angle from horizontal"""
        n1 = self.nodes[member.node_start]
        n2 = self.nodes[member.node_end]
        
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        
        angle_rad = np.arctan2(abs(dy), abs(dx))
        
        if in_degrees:
            return np.degrees(angle_rad)
        return angle_rad

    def classify_member_type(self, member: Member) -> str:
        """Classify member as strut or tie based on angle"""
        angle_deg = self.get_member_angle(member, in_degrees=True)
        if angle_deg < 45:
            return 'tie'
        else:
            return 'strut'