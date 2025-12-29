import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pandas as pd
from datetime import datetime

@dataclass
class Node:
    """Node information"""
    id: str
    x: float  # mm
    y: float  # mm
    node_type: Optional[str] = None
    
@dataclass
class Member:
    """Member information"""
    id: str
    node_start: str
    node_end: str
    member_type: Optional[str] = None  # 'strut' or 'tie'
    force: Optional[float] = None  # kN
    angle: Optional[float] = None  # degrees
    
@dataclass
class Material:
    """Material properties"""
    fck: float  # MPa
    fy: float   # MPa

class STMDesignChecker:
    """
    STM design verification following KDS 14 20 52
    Based on Example 10.2 from the standard
    """
    
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
    
    def get_max_connections(self, node_id: str) -> int:
        """Get maximum allowed connections for a node"""
        # C, D, F, G, H는 4개 연결 가능, 나머지는 3개
        if node_id in ['C', 'D', 'F', 'G', 'H']:
            return 4
        else:
            return 3
    
    def count_node_connections(self, node_id: str) -> int:
        """Count current connections for a node (including support)"""
        count = 0
        
        # Count member connections
        for m in self.members.values():
            if m.node_start == node_id or m.node_end == node_id:
                count += 1
        
        # Count support as 1 connection
        if node_id in self.supports:
            count += 1
        
        return count
    
    def is_member_connected(self, node1: str, node2: str) -> bool:
        """Check if two nodes are already connected"""
        for m in self.members.values():
            if (m.node_start == node1 and m.node_end == node2) or \
               (m.node_start == node2 and m.node_end == node1):
                return True
        return False
    
    def auto_connect_members(self):
        """
        Automatically connect members following rules:
        1. Most nodes connect to 3 members (support counts as 1)
        2. C, D, F, G, H can connect to 4 members
        3. Priority: horizontal, then vertical, then diagonal (nearest)
        """
        node_list = list(self.nodes.keys())
        nodes_data = [(nid, self.nodes[nid].x, self.nodes[nid].y) for nid in node_list]
        
        # Separate upper and lower layers
        y_coords = [n[2] for n in nodes_data]
        y_mid = (max(y_coords) + min(y_coords)) / 2
        
        upper_nodes = sorted([n for n in nodes_data if n[2] > y_mid], key=lambda x: x[1])
        lower_nodes = sorted([n for n in nodes_data if n[2] <= y_mid], key=lambda x: x[1])
        
        print(f"\n[INFO] Auto-connecting members")
        print(f"       Upper layer: {[n[0] for n in upper_nodes]}")
        print(f"       Lower layer: {[n[0] for n in lower_nodes]}")
        print(f"       Special nodes (4 connections allowed): C, D, F, G, H")
        
        # Step 1: Connect horizontal members (upper and lower chords)
        print("\n[Step 1] Horizontal connections:")
        for i in range(len(upper_nodes) - 1):
            node1 = upper_nodes[i][0]
            node2 = upper_nodes[i + 1][0]
            
            max1 = self.get_max_connections(node1)
            max2 = self.get_max_connections(node2)
            
            if self.count_node_connections(node1) < max1 and \
            self.count_node_connections(node2) < max2:
                member_id = f"{node1}{node2}"
                self.add_member(Member(member_id, node1, node2, None, None))
                print(f"       {member_id}: {node1}({self.count_node_connections(node1)}/{max1}) - {node2}({self.count_node_connections(node2)}/{max2})")
        
        for i in range(len(lower_nodes) - 1):
            node1 = lower_nodes[i][0]
            node2 = lower_nodes[i + 1][0]
            
            max1 = self.get_max_connections(node1)
            max2 = self.get_max_connections(node2)
            
            if self.count_node_connections(node1) < max1 and \
            self.count_node_connections(node2) < max2:
                member_id = f"{node1}{node2}"
                self.add_member(Member(member_id, node1, node2, None, None))
                print(f"       {member_id}: {node1}({self.count_node_connections(node1)}/{max1}) - {node2}({self.count_node_connections(node2)}/{max2})")
        
        # Step 2: Connect vertical members (directly above/below)
        print("\n[Step 2] Vertical connections:")
        for upper_id, upper_x, upper_y in upper_nodes:
            for lower_id, lower_x, lower_y in lower_nodes:
                if abs(upper_x - lower_x) < 50:  # Same x-coordinate
                    max_upper = self.get_max_connections(upper_id)
                    max_lower = self.get_max_connections(lower_id)
                    
                    if self.count_node_connections(upper_id) < max_upper and \
                    self.count_node_connections(lower_id) < max_lower and \
                    not self.is_member_connected(upper_id, lower_id):
                        member_id = ''.join(sorted([upper_id, lower_id]))
                        self.add_member(Member(member_id, upper_id, lower_id, None, None))
                        print(f"       {member_id}: {upper_id}({self.count_node_connections(upper_id)}/{max_upper}) - {lower_id}({self.count_node_connections(lower_id)}/{max_lower})")
        
        # Step 3: Connect diagonal members to complete connections
        print("\n[Step 3] Diagonal connections (completing to max):")
        for node_id in node_list:
            current_count = self.count_node_connections(node_id)
            max_conn = self.get_max_connections(node_id)
            
            if current_count < max_conn:
                needed = max_conn - current_count
                node = self.nodes[node_id]
                
                print(f"       Node {node_id} needs {needed} more connection(s) (max={max_conn})")
                
                # Find nearest unconnected nodes
                distances = []
                for other_id, other_node in self.nodes.items():
                    if other_id != node_id:
                        # ★★★ A-C, D-F 연결 금지 ★★★
                        if (node_id == 'A' and other_id == 'C') or (node_id == 'C' and other_id == 'A'):
                            continue
                        if (node_id == 'D' and other_id == 'F') or (node_id == 'F' and other_id == 'D'):
                            continue
                        
                        if not self.is_member_connected(node_id, other_id):
                            # Check if other node also needs connection
                            other_count = self.count_node_connections(other_id)
                            other_max = self.get_max_connections(other_id)
                            if other_count < other_max:
                                dist = np.sqrt((other_node.x - node.x)**2 + (other_node.y - node.y)**2)
                                distances.append((other_id, dist))
                
                distances.sort(key=lambda x: x[1])
                
                added = 0
                for other_id, dist in distances:
                    if added >= needed:
                        break
                    
                    # Double check counts before adding
                    max_node = self.get_max_connections(node_id)
                    max_other = self.get_max_connections(other_id)
                    
                    if self.count_node_connections(node_id) < max_node and \
                    self.count_node_connections(other_id) < max_other:
                        member_id = ''.join(sorted([node_id, other_id]))
                        self.add_member(Member(member_id, node_id, other_id, None, None))
                        print(f"       {member_id}: {node_id}({self.count_node_connections(node_id)}/{max_node}) - {other_id}({self.count_node_connections(other_id)}/{max_other}) [dist={dist:.0f}mm]")
                        added += 1
        
        # Report final connections
        print(f"\n[INFO] Final connection status:")
        for node_id in sorted(node_list):
            count = self.count_node_connections(node_id)
            max_conn = self.get_max_connections(node_id)
            
            connected_to = []
            for m in self.members.values():
                if m.node_start == node_id:
                    connected_to.append(m.node_end)
                elif m.node_end == node_id:
                    connected_to.append(m.node_start)
            
            support_str = ""
            if node_id in self.supports:
                support_str = " + support"
            
            status = "OK" if count == max_conn else f"WARNING({count}/{max_conn})"
            print(f"       Node {node_id}: {count}/{max_conn} connections {connected_to}{support_str} [{status}]")
    
    def get_member_length(self, member: Member) -> float:
        """Calculate member length"""
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
        """
        Classify member as strut or tie based on position:
        - Tie: bottom chord (lower horizontal) or vertical
        - Strut: top chord (upper horizontal) or diagonal
        """
        n1 = self.nodes[member.node_start]
        n2 = self.nodes[member.node_end]
        
        dx = abs(n2.x - n1.x)
        dy = abs(n2.y - n1.y)
        
        # Horizontal member
        if dy < 50:
            # Bottom chord is tie, top chord is strut
            avg_y = (n1.y + n2.y) / 2
            if avg_y < self.h * 0.3:
                return 'tie'  # Bottom horizontal = tie
            else:
                return 'strut'  # Top horizontal = strut
        
        # Vertical member
        elif dx < 50:
            return 'tie'  # Vertical = tie
        
        # Diagonal member
        else:
            return 'strut'  # Diagonal = strut
    
    def solve_member_forces(self) -> Dict[str, float]:
        """
        Solve member forces using static equilibrium
        Following the calculation method in KDS Example 10.2
        """
        n_nodes = len(self.nodes)
        n_members = len(self.members)
        
        # Count reactions
        n_reactions = 0
        reaction_map = {}
        for node_id, support_type in self.supports.items():
            if support_type == 'pin':
                reaction_map[f'Rx_{node_id}'] = n_reactions
                reaction_map[f'Ry_{node_id}'] = n_reactions + 1
                n_reactions += 2
            elif support_type == 'roller':
                reaction_map[f'Ry_{node_id}'] = n_reactions
                n_reactions += 1
        
        n_unknowns = n_members + n_reactions
        n_equations = n_nodes * 2
        
        print(f"\n[INFO] Solving equilibrium equations")
        print(f"       Unknowns: {n_unknowns} (Members: {n_members}, Reactions: {n_reactions})")
        print(f"       Equations: {n_equations}")
        
        # Build equilibrium matrix
        A = np.zeros((n_equations, n_unknowns))
        b = np.zeros(n_equations)
        
        node_list = list(self.nodes.keys())
        
        for i, node_id in enumerate(node_list):
            row_x = i * 2
            row_y = i * 2 + 1
            
            # Member contributions
            for j, (member_id, member) in enumerate(self.members.items()):
                n1 = self.nodes[member.node_start]
                n2 = self.nodes[member.node_end]
                
                length = self.get_member_length(member)
                cos_val = (n2.x - n1.x) / length
                sin_val = (n2.y - n1.y) / length
                
                if member.node_start == node_id:
                    A[row_x, j] = cos_val
                    A[row_y, j] = sin_val
                elif member.node_end == node_id:
                    A[row_x, j] = -cos_val
                    A[row_y, j] = -sin_val
            
            # Reaction contributions
            if node_id in self.supports:
                if f'Rx_{node_id}' in reaction_map:
                    col = n_members + reaction_map[f'Rx_{node_id}']
                    A[row_x, col] = 1.0
                if f'Ry_{node_id}' in reaction_map:
                    col = n_members + reaction_map[f'Ry_{node_id}']
                    A[row_y, col] = 1.0
            
            # Load contributions
            for load_node_id, Fx, Fy in self.loads:
                if load_node_id == node_id:
                    b[row_x] -= Fx
                    b[row_y] -= Fy
        
        # Solve using least squares
        try:
            solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Extract member forces
            member_forces = {}
            for j, (member_id, member) in enumerate(self.members.items()):
                force = solution[j]
                member_forces[member_id] = force
                self.members[member_id].force = force
                self.members[member_id].angle = self.get_member_angle(member, in_degrees=True)
                self.members[member_id].member_type = self.classify_member_type(member)
            
            # Extract reactions
            self.reactions = {}
            for node_id in self.supports.keys():
                self.reactions[node_id] = {}
                if f'Rx_{node_id}' in reaction_map:
                    col = n_members + reaction_map[f'Rx_{node_id}']
                    self.reactions[node_id]['Rx'] = solution[col]
                if f'Ry_{node_id}' in reaction_map:
                    col = n_members + reaction_map[f'Ry_{node_id}']
                    self.reactions[node_id]['Ry'] = solution[col]
            
            print(f"       Solution converged (rank: {rank})")
            return member_forces
            
        except np.linalg.LinAlgError:
            print("[ERROR] Failed to solve equilibrium")
            return {}
    
    def infer_node_types(self):
        """
        Determine node types based on connected members
        Following KDS Table 10.2: CCC, CCT, CTT, TTT
        """
        for node_id, node in self.nodes.items():
            n_struts = 0
            n_ties = 0
            
            for member in self.members.values():
                if member.node_start == node_id or member.node_end == node_id:
                    if member.member_type == 'strut':
                        n_struts += 1
                    elif member.member_type == 'tie':
                        n_ties += 1
            
            total = n_struts + n_ties
            
            # Assign node type following KDS standard
            if n_struts == 3 and n_ties == 0:
                node.node_type = 'CCC'
            elif n_struts == 2 and n_ties == 1:
                node.node_type = 'CCT'
            elif n_struts == 1 and n_ties == 2:
                node.node_type = 'CTT'
            elif n_struts == 0 and n_ties == 3:
                node.node_type = 'TTT'
            else:
                node.node_type = f'C{n_struts}T{n_ties}'
    
    def calculate_strut_width(self, member: Member, l_b: float = 450, 
                             cover: float = 40, d_s: float = 16, 
                             d_b: float = 32, clearance: float = 40) -> float:
        """
        Calculate strut width following KDS Example 10.2.3
        
        w_s = l_b * sin(theta) + w_t * cos(theta)
        
        where:
        - l_b: bearing plate length (mm)
        - theta: member angle from horizontal
        - w_t: tie width = 2(cover + d_s + d_b + clearance/2)
        """
        angle_rad = self.get_member_angle(member)
        
        # Tie width calculation
        w_t = 2 * (cover + d_s + d_b + clearance / 2)
        
        # Strut width
        w_s = l_b * np.sin(angle_rad) + w_t * np.cos(angle_rad)
        
        return w_s
    
    def calculate_required_strut_width(self, force: float, beta_s: float) -> float:
        """
        Calculate required strut width following KDS Eq 10.1, 10.3
        
        w_s,req = F_u / (phi * f_ce * b_w)
        where f_ce = 0.85 * beta_s * f_ck
        """
        phi = 0.75
        f_ce = 0.85 * beta_s * self.material.fck
        
        w_req = (abs(force) * 1000) / (phi * f_ce * self.b)
        
        return w_req
    
    def get_beta_s(self, member_id: str) -> float:
        """
        Get beta_s value following KDS Table 10.1
        Bottle-shaped struts (AB, CG, EF, DH): beta_s = 0.6
        Other struts: beta_s = 1.0
        """
        # Bottle-shaped struts have beta_s = 0.6
        bottle_shaped = ['AB', 'CG', 'EF', 'DH', 'BA', 'GC', 'FE', 'HD']
        
        if member_id in bottle_shaped:
            return 0.6
        else:
            return 1.0
    
    def get_beta_n(self, node_type: str) -> float:
        """Get beta_n value following KDS Table 10.2"""
        beta_n_table = {
            'CCC': 1.0,
            'CCT': 0.80,
            'CTT': 0.60,
            'TTT': 0.60
        }
        return beta_n_table.get(node_type, 0.60)
    
    def check_strut_strength(self, member: Member, l_b: float = 450) -> Tuple[bool, Dict]:
        """
        Check strut strength following KDS Section 10.1
        
        phi * F_ns >= F_u
        F_ns = f_ce * A_cs = (0.85 * beta_s * f_ck) * (w_s * b_w)
        """
        phi = 0.75
        
        # Get beta_s
        beta_s = self.get_beta_s(member.id)
        
        # Calculate strut width
        w_s = self.calculate_strut_width(member, l_b)
        
        # Calculate required width
        w_req = self.calculate_required_strut_width(member.force, beta_s)
        
        # Effective strength
        f_ce = 0.85 * beta_s * self.material.fck
        
        # Cross-sectional area
        A_cs = w_s * self.b
        
        # Nominal strength
        F_ns = f_ce * A_cs / 1000  # kN
        
        # Design strength
        phi_Fns = phi * F_ns
        
        # Required strength
        F_u = abs(member.force)
        
        is_ok = phi_Fns >= F_u
        
        result = {
            'member_id': member.id,
            'type': 'strut',
            'beta_s': beta_s,
            'f_ce': f_ce,
            'w_req': w_req,
            'w_s': w_s,
            'A_cs': A_cs,
            'F_ns': F_ns,
            'phi': phi,
            'phi_Fns': phi_Fns,
            'F_u': F_u,
            'ratio': F_u / phi_Fns if phi_Fns > 0 else float('inf'),
            'is_ok': is_ok
        }
        
        return is_ok, result
    
    def check_tie_strength(self, member: Member) -> Tuple[bool, Dict]:
        """
        Check tie strength following KDS Section 10.1
        
        phi * F_nt >= F_u
        F_nt = A_ts * f_y
        """
        phi = 0.85
        
        F_u = abs(member.force)
        
        # Required steel area
        A_s_req = (F_u * 1000) / (phi * self.material.fy)
        
        # Provided steel area (with 10% margin)
        A_s = A_s_req * 1.1
        
        # Nominal strength
        F_nt = A_s * self.material.fy / 1000  # kN
        
        # Design strength
        phi_Fnt = phi * F_nt
        
        is_ok = phi_Fnt >= F_u
        
        result = {
            'member_id': member.id,
            'type': 'tie',
            'A_s_req': A_s_req,
            'A_s': A_s,
            'F_nt': F_nt,
            'phi': phi,
            'phi_Fnt': phi_Fnt,
            'F_u': F_u,
            'ratio': F_u / phi_Fnt if phi_Fnt > 0 else float('inf'),
            'is_ok': is_ok
        }
        
        return is_ok, result
    
    def check_node_strength(self, node: Node) -> Tuple[bool, Dict]:
        """
        Check node strength following KDS Section 10.3
        
        phi * F_nn >= F_u
        F_nn = f_cn * A_n = (beta_n * f_ck) * A_n
        """
        phi = 0.75
        
        # Get beta_n
        beta_n = self.get_beta_n(node.node_type)
        
        # Effective concrete strength at node
        f_cn = beta_n * self.material.fck
        
        # Node area (simplified as square)
        A_n = self.b * self.b
        
        # Nominal strength
        F_nn = f_cn * A_n / 1000  # kN
        
        # Design strength
        phi_Fnn = phi * F_nn
        
        # Maximum compression force at node
        max_force = 0
        for m in self.members.values():
            if m.node_start == node.id or m.node_end == node.id:
                if m.force and m.member_type == 'strut':
                    max_force = max(max_force, abs(m.force))
        
        F_u = max_force
        is_ok = phi_Fnn >= F_u
        
        result = {
            'node_id': node.id,
            'node_type': node.node_type,
            'beta_n': beta_n,
            'f_cn': f_cn,
            'A_n': A_n,
            'F_nn': F_nn,
            'phi': phi,
            'phi_Fnn': phi_Fnn,
            'F_u': F_u,
            'ratio': F_u / phi_Fnn if phi_Fnn > 0 else float('inf'),
            'is_ok': is_ok
        }
        
        return is_ok, result
    
    def verify_stm(self, l_b: float = 450) -> Dict:
        """
        Complete STM verification following KDS 14 20 52
        """
        print("\n" + "="*80)
        print("STM Design Verification (KDS 14 20 52)")
        print("="*80)
        
        # Step 1: Solve member forces
        print("\n[Step 1] Solving member forces")
        member_forces = self.solve_member_forces()
        
        if not member_forces:
            print("[ERROR] Failed to solve")
            return self.results
        
        # Step 2: Determine node types
        print("\n[Step 2] Determining node types")
        self.infer_node_types()
        
        # Step 3: Verify strengths
        print("\n[Step 3] Verifying strengths")
        
        self.results = {
            'struts': {},
            'ties': {},
            'nodes': {},
            'overall': True,
            'messages': []
        }
        
        # Check struts
        n_strut_ok = 0
        n_strut_total = 0
        for member_id, member in self.members.items():
            if member.member_type == 'strut':
                n_strut_total += 1
                is_ok, result = self.check_strut_strength(member, l_b)
                self.results['struts'][member_id] = result
                if is_ok:
                    n_strut_ok += 1
                else:
                    self.results['overall'] = False
        
        # Check ties
        n_tie_ok = 0
        n_tie_total = 0
        for member_id, member in self.members.items():
            if member.member_type == 'tie':
                n_tie_total += 1
                is_ok, result = self.check_tie_strength(member)
                self.results['ties'][member_id] = result
                if is_ok:
                    n_tie_ok += 1
                else:
                    self.results['overall'] = False
        
        # Check nodes
        n_node_ok = 0
        n_node_total = 0
        for node_id, node in self.nodes.items():
            n_node_total += 1
            is_ok, result = self.check_node_strength(node)
            self.results['nodes'][node_id] = result
            if is_ok:
                n_node_ok += 1
            else:
                self.results['overall'] = False
        
        # Summary
        print(f"\n       Struts: {n_strut_ok}/{n_strut_total} OK")
        print(f"       Ties: {n_tie_ok}/{n_tie_total} OK")
        print(f"       Nodes: {n_node_ok}/{n_node_total} OK")
        
        if self.results['overall']:
            self.results['messages'].append("[PASS] All checks satisfied")
        else:
            self.results['messages'].append("[FAIL] Some checks failed")
        
        return self.results
    
    def export_to_excel(self, filename: str = None):
        """Export verification results to Excel file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"STM_Verification_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Project Info
            info_data = {
                'Item': ['Project', 'Date', 'Beam Width (mm)', 'Beam Height (mm)', 
                        'Beam Length (mm)', 'fck (MPa)', 'fy (MPa)', 'Overall Result'],
                'Value': ['STM Verification', datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         self.b, self.h, self.L, self.material.fck, self.material.fy,
                         'PASS' if self.results['overall'] else 'FAIL']
            }
            df_info = pd.DataFrame(info_data)
            df_info.to_excel(writer, sheet_name='Project Info', index=False)
            
            # Sheet 2: Member Forces
            member_data = []
            for mid, m in sorted(self.members.items()):
                member_data.append({
                    'Member ID': mid,
                    'Start Node': m.node_start,
                    'End Node': m.node_end,
                    'Type': m.member_type if m.member_type else 'N/A',
                    'Force (kN)': round(m.force, 2) if m.force else 0,
                    'Angle (deg)': round(m.angle, 2) if m.angle else 0,
                    'beta_s': self.get_beta_s(mid) if m.member_type == 'strut' else '-'
                })
            df_members = pd.DataFrame(member_data)
            df_members.to_excel(writer, sheet_name='Member Forces', index=False)
            
            # Sheet 3: Node Classifications
            node_data = []
            for nid, n in sorted(self.nodes.items()):
                struts = sum(1 for m in self.members.values() 
                           if (m.node_start == nid or m.node_end == nid) and m.member_type == 'strut')
                ties = sum(1 for m in self.members.values() 
                          if (m.node_start == nid or m.node_end == nid) and m.member_type == 'tie')
                
                node_data.append({
                    'Node ID': nid,
                    'X (mm)': n.x,
                    'Y (mm)': n.y,
                    'Node Type': n.node_type if n.node_type else 'N/A',
                    'beta_n': self.get_beta_n(n.node_type) if n.node_type else 0,
                    'Struts': struts,
                    'Ties': ties
                })
            df_nodes = pd.DataFrame(node_data)
            df_nodes.to_excel(writer, sheet_name='Node Classifications', index=False)
            
            # Sheet 4: Strut Verification
            if self.results['struts']:
                strut_data = []
                for mid in sorted(self.results['struts'].keys()):
                    res = self.results['struts'][mid]
                    strut_data.append({
                        'Member ID': res['member_id'],
                        'beta_s': res['beta_s'],
                        'f_ce (MPa)': round(res['f_ce'], 2),
                        'Required Width (mm)': round(res['w_req'], 2),
                        'Provided Width (mm)': round(res['w_s'], 2),
                        'Force (kN)': round(res['F_u'], 2),
                        'phi*F_ns (kN)': round(res['phi_Fns'], 2),
                        'Ratio': round(res['ratio'], 3),
                        'Status': 'OK' if res['is_ok'] else 'NG'
                    })
                df_struts = pd.DataFrame(strut_data)
                df_struts.to_excel(writer, sheet_name='Strut Verification', index=False)
            
            # Sheet 5: Tie Verification
            if self.results['ties']:
                tie_data = []
                for mid in sorted(self.results['ties'].keys()):
                    res = self.results['ties'][mid]
                    tie_data.append({
                        'Member ID': res['member_id'],
                        'Force (kN)': round(res['F_u'], 2),
                        'Required A_s (mm^2)': round(res['A_s_req'], 2),
                        'Provided A_s (mm^2)': round(res['A_s'], 2),
                        'phi*F_nt (kN)': round(res['phi_Fnt'], 2),
                        'Ratio': round(res['ratio'], 3),
                        'Status': 'OK' if res['is_ok'] else 'NG'
                    })
                df_ties = pd.DataFrame(tie_data)
                df_ties.to_excel(writer, sheet_name='Tie Verification', index=False)
            
            # Sheet 6: Node Verification
            if self.results['nodes']:
                node_verify_data = []
                for nid in sorted(self.results['nodes'].keys()):
                    res = self.results['nodes'][nid]
                    node_verify_data.append({
                        'Node ID': res['node_id'],
                        'Node Type': res['node_type'],
                        'beta_n': res['beta_n'],
                        'f_cn (MPa)': round(res['f_cn'], 2),
                        'Force (kN)': round(res['F_u'], 2),
                        'phi*F_nn (kN)': round(res['phi_Fnn'], 2),
                        'Ratio': round(res['ratio'], 3),
                        'Status': 'OK' if res['is_ok'] else 'NG'
                    })
                df_node_verify = pd.DataFrame(node_verify_data)
                df_node_verify.to_excel(writer, sheet_name='Node Verification', index=False)
            
            # Sheet 7: Reactions
            reaction_data = []
            for node_id in sorted(self.reactions.keys()):
                r = self.reactions[node_id]
                reaction_data.append({
                    'Node ID': node_id,
                    'Support Type': self.supports[node_id],
                    'Rx (kN)': round(r.get('Rx', 0), 2),
                    'Ry (kN)': round(r.get('Ry', 0), 2)
                })
            df_reactions = pd.DataFrame(reaction_data)
            df_reactions.to_excel(writer, sheet_name='Reactions', index=False)
        
        print(f"\n[INFO] Results exported to: {filename}")
        return filename
    
    def print_results(self):
        """Print detailed verification results"""
        print("\n" + "="*80)
        print("STM VERIFICATION RESULTS (KDS 14 20 52)")
        print("="*80)
        
        # Table 1: Member forces
        print("\n[Table 1] Member Forces and Classifications")
        print("-"*80)
        print(f"{'Member':<8} {'Type':<8} {'Force(kN)':<12} {'Angle(deg)':<12} {'beta_s':<8}")
        print("-"*80)
        for mid, m in sorted(self.members.items()):
            mtype = m.member_type if m.member_type else "?"
            force_str = f"{m.force:.1f}" if m.force else "N/A"
            angle_str = f"{m.angle:.1f}" if m.angle else "N/A"
            
            beta_s_str = "-"
            if mtype == 'strut':
                beta_s_str = f"{self.get_beta_s(mid):.1f}"
            
            print(f"{mid:<8} {mtype:<8} {force_str:<12} {angle_str:<12} {beta_s_str:<8}")
        
        # Table 2: Node types
        print("\n[Table 2] Node Classifications")
        print("-"*80)
        print(f"{'Node':<8} {'Type':<8} {'beta_n':<10} {'Description'}")
        print("-"*80)
        for nid, n in sorted(self.nodes.items()):
            ntype = n.node_type if n.node_type else "?"
            beta_n = self.get_beta_n(ntype)
            
            struts = sum(1 for m in self.members.values() 
                        if (m.node_start == nid or m.node_end == nid) and m.member_type == 'strut')
            ties = sum(1 for m in self.members.values() 
                      if (m.node_start == nid or m.node_end == nid) and m.member_type == 'tie')
            
            desc = f"{struts} struts + {ties} ties"
            print(f"{nid:<8} {ntype:<8} {beta_n:<10.2f} {desc}")
        
        # Table 3: Strut verification
        if self.results['struts']:
            print("\n[Table 3] Strut Strength Verification")
            print("-"*80)
            print(f"{'Member':<8} {'beta_s':<8} {'Force':<10} {'w_req':<10} {'w_s':<10} {'Status'}")
            print(f"{'':8} {'':8} {'(kN)':<10} {'(mm)':<10} {'(mm)':<10} {''}")
            print("-"*80)
            for mid in sorted(self.results['struts'].keys()):
                res = self.results['struts'][mid]
                status = "OK" if res['is_ok'] else "NG"
                print(f"{mid:<8} {res['beta_s']:<8.1f} {res['F_u']:<10.1f} {res['w_req']:<10.1f} {res['w_s']:<10.1f} {status}")
        
        # Table 4: Tie verification
        if self.results['ties']:
            print("\n[Table 4] Tie Strength Verification")
            print("-"*80)
            print(f"{'Member':<8} {'Force':<12} {'A_s,req':<12} {'Status'}")
            print(f"{'':8} {'(kN)':<12} {'(mm^2)':<12} {''}")
            print("-"*80)
            for mid in sorted(self.results['ties'].keys()):
                res = self.results['ties'][mid]
                status = "OK" if res['is_ok'] else "NG"
                print(f"{mid:<8} {res['F_u']:<12.1f} {res['A_s_req']:<12.1f} {status}")
        
        # Table 5: Node verification
        if self.results['nodes']:
            print("\n[Table 5] Node Strength Verification")
            print("-"*80)
            print(f"{'Node':<8} {'Type':<8} {'beta_n':<10} {'Force':<10} {'phi*Fnn':<12} {'Status'}")
            print(f"{'':8} {'':8} {'':10} {'(kN)':<10} {'(kN)':<12} {''}")
            print("-"*80)
            for nid in sorted(self.results['nodes'].keys()):
                res = self.results['nodes'][nid]
                status = "OK" if res['is_ok'] else "NG"
                print(f"{nid:<8} {res['node_type']:<8} {res['beta_n']:<10.2f} {res['F_u']:<10.1f} {res['phi_Fnn']:<12.1f} {status}")
        
        # Reactions
        print("\n[Reactions]")
        print("-"*80)
        for node_id in sorted(self.reactions.keys()):
            r = self.reactions[node_id]
            Rx = r.get('Rx', 0)
            Ry = r.get('Ry', 0)
            print(f"Node {node_id}: Rx = {Rx:.1f} kN, Ry = {Ry:.1f} kN")
        
        # Final result
        print("\n" + "="*80)
        for msg in self.results['messages']:
            print(msg)
        print("="*80 + "\n")
    
    def plot_stm(self, save_path: str = None, show_forces: bool = True, fontsize: int = 14):
        """Plot STM model with adjustable font size"""
        # Set font sizes
        title_size = fontsize + 4
        label_size = fontsize + 2
        node_label_size = fontsize + 1
        node_type_size = fontsize - 1
        force_label_size = fontsize - 2
        legend_size = fontsize
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Draw beam outline
        ax.add_patch(plt.Rectangle((0, 0), self.L, self.h, 
                                   fill=False, edgecolor='black', linewidth=2, linestyle='--'))
        
        # Draw members
        for mid, m in self.members.items():
            n1 = self.nodes[m.node_start]
            n2 = self.nodes[m.node_end]
            
            if m.member_type == 'strut':
                color = 'blue'
                linestyle = '-'
                linewidth = 3
            else:
                color = 'red'
                linestyle = '--'
                linewidth = 2
            
            ax.plot([n1.x, n2.x], [n1.y, n2.y], 
                   color=color, linestyle=linestyle, linewidth=linewidth)
            
            if show_forces and m.force:
                mid_x = (n1.x + n2.x) / 2
                mid_y = (n1.y + n2.y) / 2
                ax.text(mid_x, mid_y, f"{m.force:.0f}", 
                       fontsize=force_label_size, ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Draw nodes
        for nid, n in self.nodes.items():
            ax.plot(n.x, n.y, 'ko', markersize=10)
            ax.text(n.x, n.y + 80, nid, 
                   ha='center', fontsize=node_label_size, fontweight='bold')
            if n.node_type:
                ax.text(n.x, n.y - 80, n.node_type, 
                       ha='center', fontsize=node_type_size, color='green')
        
        # Draw supports
        for nid, support_type in self.supports.items():
            n = self.nodes[nid]
            if support_type == 'pin':
                ax.plot(n.x, n.y - 100, 'v', markersize=15, color='black')
            elif support_type == 'roller':
                ax.plot(n.x, n.y - 100, 'o', markersize=12, color='black')
        
        # Draw loads
        for load_nid, Fx, Fy in self.loads:
            n = self.nodes[load_nid]
            ax.arrow(n.x, n.y + 200, 0, -150, 
                    head_width=50, head_length=30, fc='darkgreen', ec='darkgreen')
            ax.text(n.x, n.y + 250, f"{abs(Fy):.0f}kN", 
                   ha='center', fontsize=label_size, color='darkgreen')
        
        ax.set_xlabel('Length (mm)', fontsize=label_size)
        ax.set_ylabel('Height (mm)', fontsize=label_size)
        ax.set_title('Strut-Tie Model (STM) - KDS 14 20 52', fontsize=title_size, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=3, label='Strut (Compression)'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Tie (Tension)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=legend_size)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Plot saved to {save_path}")
        
        return fig, ax


# Example: KDS Example 10.2
if __name__ == "__main__":
    # Material properties
    material = Material(fck=27, fy=400)
    
    # Beam dimensions
    checker = STMDesignChecker(
        material=material,
        beam_width=500,      # mm
        beam_height=2000,    # mm (1725 + 125 + 150)
        beam_length=6900     # mm
    )
    
    # Add nodes (coordinates from KDS Example 10.2.3)
    checker.add_node(Node('A', 225, 125, None))
    checker.add_node(Node('B', 1225, 1850, None))
    checker.add_node(Node('C', 2225, 1850, None))
    checker.add_node(Node('D', 4675, 1850, None))
    checker.add_node(Node('E', 5675, 1850, None))
    checker.add_node(Node('F', 6675, 125, None))
    checker.add_node(Node('G', 1225, 125, None))
    checker.add_node(Node('H', 5675, 125, None))
    
    # Auto-connect members (C, D, F, G, H can have 4 connections)
    checker.auto_connect_members()
    
    # Add loads (2,000 kN at B and D)
    checker.add_load('B', 0, -2000)
    checker.add_load('D', 0, -2000)
    
    # Add supports
    checker.add_support('A', 'pin')
    checker.add_support('F', 'roller')
    
    # Verify STM
    results = checker.verify_stm(l_b=450)
    
    # Print results
    checker.print_results()
    
    # Export to Excel
    checker.export_to_excel()
    
    # Plot with larger font
    checker.plot_stm(save_path='stm_verification.png', show_forces=True, fontsize=16)
    plt.show()