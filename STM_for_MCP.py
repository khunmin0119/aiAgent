# this file is written for connecting mcp. It is based on STM_code.py
# subject is creating member, setting steel, design automation

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

class Stm_design_checker :
    