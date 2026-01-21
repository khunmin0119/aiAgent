import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    
    [수정 완료]
    1. f_cn = 0.85 × beta_n × f_ck (KDS 식 10.11)
    2. 타이 폭: 수평=250mm, 수직=1000mm
    3. w_t: 수평 스트럿=300mm, 사재=250mm
    4. l_b: CG/DH=1000mm, 나머지=450mm
    
    [추가]
    5. 철근 배근 설계 (교재 477페이지)
    6. 배근도 시각화 (교재 479페이지)
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
            'nodes_detailed': {},
            'overall': True,
            'messages': []
        }
        # 철근 배근 결과
        self.rebar_design = {}
        
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
            elif n_struts == 2 and n_ties == 1 or n_struts == 1 and n_ties == 1:
                node.node_type = 'CCT'
            elif n_struts == 1 and n_ties == 2 or n_struts == 1 and n_ties == 3:
                node.node_type = 'CTT'
            elif n_struts == 0 and n_ties == 3:
                node.node_type = 'TTT'
            else:
                node.node_type = f'C{n_struts}T{n_ties}'
    
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
    
    def get_l_b(self, member_id: str) -> float:
        """
        Get bearing plate length following KDS Example 10.2.3
        CG, DH (middle bottle-shaped struts): l_b = 1,000 mm
        Others: l_b = 450 mm
        """
        # Middle bottle-shaped struts have larger bearing plates
        large_bearing = ['CG', 'DH', 'GC', 'HD']
        
        if member_id in large_bearing:
            return 1000  # mm
        else:
            return 450   # mm
    
    def get_beta_n(self, node_type: str) -> float:
        """Get beta_n value following KDS Table 10.2"""
        beta_n_table = {
            'CCC': 1.0,
            'CCT': 0.80,
            'CTT': 0.60,
            'TTT': 0.60
        }
        return beta_n_table.get(node_type, 0.60)
    
    def calculate_strut_width(self, member: Member, l_b: float = 450, 
                             cover: float = 40, d_s: float = 16, 
                             d_b: float = 32, clearance: float = 40) -> float:
        """
        Calculate strut width following KDS Example 10.2.3
        """
        angle_rad = self.get_member_angle(member)
        angle_deg = np.degrees(angle_rad)
        
        if angle_deg < 5:
            w_t = 300  # mm (수평 스트럿: BC, CD, DE)
        else:
            w_t = 250  # mm (사재: AB, CG, DH, EF)
        
        # Strut width
        w_s = l_b * np.sin(angle_rad) + w_t * np.cos(angle_rad)
        
        return w_s
    
    def check_strut_strength(self, member: Member, l_b: float = 450) -> Tuple[bool, Dict]:
        """Check strut strength following KDS Section 10.1"""
        phi = 0.75
        
        # Get beta_s
        beta_s = self.get_beta_s(member.id)
        
        # Get bearing plate length (member-specific)
        l_b_member = self.get_l_b(member.id)
        
        # Calculate strut width
        w_s = self.calculate_strut_width(member, l_b_member)
        
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
            'F_u': F_u,
            'phi_Fns': phi_Fns,
            'is_ok': is_ok
        }
        
        return is_ok, result
    
    def check_tie_strength(self, member: Member) -> Tuple[bool, Dict]:
        """Check tie strength following KDS Section 10.1"""
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
        """Check node strength following KDS Section 10.3"""
        phi = 0.75
        
        # Get beta_n
        beta_n = self.get_beta_n(node.node_type)
        
        # Effective concrete strength at node (KDS 식 10.11)
        f_cn = 0.85 * beta_n * self.material.fck
        
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
            'F_u': F_u,
            'phi_Fnn': phi_Fnn,
            'is_ok': is_ok
        }
        
        return is_ok, result
    
    def verify_stm(self, l_b: float = 450) -> Dict:
        """Complete STM verification following KDS 14 20 52"""
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
            'nodes_detailed': {},
            'overall': True,
            'messages': []
        }
        
        # Check struts
        for member_id, member in self.members.items():
            if member.member_type == 'strut':
                is_ok, result = self.check_strut_strength(member, l_b)
                self.results['struts'][member_id] = result
                if not is_ok:
                    self.results['overall'] = False
        
        # Check ties
        for member_id, member in self.members.items():
            if member.member_type == 'tie':
                is_ok, result = self.check_tie_strength(member)
                self.results['ties'][member_id] = result
                if not is_ok:
                    self.results['overall'] = False
        
        # Check nodes
        for node_id, node in self.nodes.items():
            is_ok, result = self.check_node_strength(node)
            self.results['nodes'][node_id] = result
            if not is_ok:
                self.results['overall'] = False
        
        if self.results['overall']:
            self.results['messages'].append("[PASS] All checks satisfied")
        else:
            self.results['messages'].append("[FAIL] Some checks failed")
        
        return self.results
    
    # ========================================================================
    # 철근 배근 설계 (교재 477페이지)
    # ========================================================================
    
    def design_tie_reinforcement(self, member_id: str) -> Dict:
        """교재 477페이지 철근 배근 설계"""
        if member_id not in self.results['ties']:
            return {}
        
        A_s_req = self.results['ties'][member_id]['A_s_req']
        
        # 수직 타이 판정
        n1 = self.nodes[self.members[member_id].node_start]
        n2 = self.nodes[self.members[member_id].node_end]
        dx = abs(n2.x - n1.x)
        is_vertical = (dx < 50)
        
        if is_vertical:
            # 교재: D16@120 양면 배근
            A_bar = 198.6  # D16
            n = max(int(np.ceil(A_s_req / (2 * A_bar))), 15)
            spacing = 120
            A_s_provided = 2 * n * A_bar
            
            design = {
                'member_id': member_id,
                'diameter': 16,
                'n_bars': n,
                'spacing': spacing,
                'A_s_req': A_s_req,
                'A_s_provided': A_s_provided,
                'arrangement': 'double',
                'note': f'D16@{spacing} 양면'
            }
        else:
            # 수평 타이
            A_bar = 126.7  # D13
            n = int(np.ceil(A_s_req / A_bar))
            spacing = 300
            A_s_provided = n * A_bar
            
            design = {
                'member_id': member_id,
                'diameter': 13,
                'n_bars': n,
                'spacing': spacing,
                'A_s_req': A_s_req,
                'A_s_provided': A_s_provided,
                'arrangement': 'single',
                'note': f'D13@{spacing}'
            }
        
        self.rebar_design[member_id] = design
        return design
    
    def design_all_ties(self) -> Dict:
        """모든 타이 철근 배근 설계"""
        print("\n" + "="*80)
        print("철근 배근 설계 (교재 477페이지)")
        print("="*80)
        
        for member_id in self.results['ties'].keys():
            design = self.design_tie_reinforcement(member_id)
            if design:
                print(f"\n[{member_id}]")
                print(f"  A_s_req: {design['A_s_req']:.1f} mm²")
                print(f"  A_s: {design['A_s_provided']:.1f} mm²")
                print(f"  배근: {design['note']}")
        
        return self.rebar_design
    
    def plot_rebar_section(self, member_id: str, save_path: str = None):
        """철근 배근 단면도 (교재 479페이지)"""
        if member_id not in self.rebar_design:
            print(f"[ERROR] {member_id} 배근 설계 없음")
            return
        
        design = self.rebar_design[member_id]
        db = design['diameter']
        n_bars = design['n_bars']
        spacing = design['spacing']
        
        cover = 40
        d_s = 16
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 단면 윤곽
        rect = patches.Rectangle((0, 0), self.b, self.h,
                                 linewidth=2, edgecolor='black',
                                 facecolor='lightgray', alpha=0.3)
        ax.add_patch(rect)
        
        # 철근
        if design['arrangement'] == 'double':
            # 상부
            y_top = self.h - (cover + d_s + db/2)
            for i in range(n_bars):
                x = cover + d_s + db/2 + i * spacing
                if x < self.b - (cover + d_s + db/2):
                    circle = plt.Circle((x, y_top), db/2, color='blue', ec='black', lw=1.5)
                    ax.add_patch(circle)
            
            # 하부
            y_bottom = cover + d_s + db/2
            for i in range(n_bars):
                x = cover + d_s + db/2 + i * spacing
                if x < self.b - (cover + d_s + db/2):
                    circle = plt.Circle((x, y_bottom), db/2, color='red', ec='black', lw=1.5)
                    ax.add_patch(circle)
        else:
            # 하부만
            y_bottom = cover + d_s + db/2
            for i in range(n_bars):
                x = cover + d_s + db/2 + i * spacing
                if x < self.b - (cover + d_s + db/2):
                    circle = plt.Circle((x, y_bottom), db/2, color='red', ec='black', lw=1.5)
                    ax.add_patch(circle)
        
        # 치수선
        ax.annotate('', xy=(self.b, -50), xytext=(0, -50),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(self.b/2, -80, f'b = {self.b:.0f}mm', ha='center', fontsize=12, fontweight='bold')
        
        ax.annotate('', xy=(self.b + 50, self.h), xytext=(self.b + 50, 0),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(self.b + 80, self.h/2, f'h = {self.h:.0f}mm',
               ha='left', va='center', fontsize=12, fontweight='bold', rotation=90)
        
        ax.set_title(f'{member_id} - {design["note"]}',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlim(-100, self.b + 150)
        ax.set_ylim(-150, self.h + 100)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[INFO] 배근도 저장: {save_path}")
        
        return fig, ax
    
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*80)
        print("검증 결과")
        print("="*80)
        
        # 타이 검증
        print("\n[타이]")
        for mid, res in sorted(self.results['ties'].items()):
            status = "OK" if res['is_ok'] else "NG"
            print(f"{mid:<8} A_s_req = {res['A_s_req']:>8.1f} mm²  {status}")
        
        # 철근 배근
        if self.rebar_design:
            print("\n[철근 배근]")
            for mid, design in sorted(self.rebar_design.items()):
                print(f"{mid:<8} {design['note']:<20} (A_s = {design['A_s_provided']:.1f} mm²)")
        
        print("\n" + "="*80)
        for msg in self.results['messages']:
            print(msg)
        print("="*80)


# 실행
if __name__ == "__main__":
    material = Material(fck=27, fy=400)
    
    checker = STMDesignChecker(material, 500, 2000, 6900)
    
    # 절점
    checker.add_node(Node('A', 225, 125))
    checker.add_node(Node('B', 1225, 1850))
    checker.add_node(Node('C', 2225, 1850))
    checker.add_node(Node('D', 4675, 1850))
    checker.add_node(Node('E', 5675, 1850))
    checker.add_node(Node('F', 6675, 125))
    checker.add_node(Node('G', 1225, 125))
    checker.add_node(Node('H', 5675, 125))
    
    # 부재
    checker.add_member(Member('BC', 'B', 'C'))
    checker.add_member(Member('CD', 'C', 'D'))
    checker.add_member(Member('DE', 'D', 'E'))
    checker.add_member(Member('AG', 'A', 'G'))
    checker.add_member(Member('GH', 'G', 'H'))
    checker.add_member(Member('HF', 'H', 'F'))
    checker.add_member(Member('BG', 'B', 'G'))
    checker.add_member(Member('EH', 'E', 'H'))
    checker.add_member(Member('AB', 'A', 'B'))
    checker.add_member(Member('CG', 'C', 'G'))
    checker.add_member(Member('DH', 'D', 'H'))
    checker.add_member(Member('EF', 'E', 'F'))
    
    # 하중
    checker.add_load('C', 0, -2000)
    checker.add_load('D', 0, -2000)
    
    # 지지점
    checker.add_support('A', 'pin')
    checker.add_support('F', 'roller')
    
    # STM 검증
    results = checker.verify_stm()
    
    # 철근 배근 설계
    rebar_results = checker.design_all_ties()
    
    # 결과 출력
    checker.print_results()
    
    # 배근도
    checker.plot_rebar_section('BG', 'Rebar_BG.png')
    checker.plot_rebar_section('EH', 'Rebar_EH.png')
    
    plt.show()