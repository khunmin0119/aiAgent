"""
STM 정역학 자동 해석 모듈
절점과 부재로부터 부재력을 자동으로 계산합니다.
"""

import numpy as np
from scipy.linalg import lstsq
from dataclasses import dataclass
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


@dataclass
class Node:
    """절점 클래스"""
    id: str
    x: float
    y: float
    node_type: Optional[str] = None
    is_support: bool = False
    support_type: Optional[str] = None  # 'pin', 'roller', 'fixed'


@dataclass
class Member:
    """부재 클래스"""
    id: str
    node_start: str
    node_end: str
    member_type: Optional[str] = None  # 'strut' or 'tie'


@dataclass  
class Load:
    """하중 클래스"""
    node_id: str
    Fx: float = 0.0
    Fy: float = 0.0


class STMStaticAnalyzer:
    """STM 정역학 해석기"""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.members: Dict[str, Member] = {}
        self.loads: List[Load] = []
        self.reactions: Dict[str, Dict[str, float]] = {}
        
    def add_node(self, node: Node):
        """절점 추가"""
        self.nodes[node.id] = node
        
    def add_member(self, member: Member):
        """부재 추가"""
        self.members[member.id] = member
        
    def add_load(self, load: Load):
        """하중 추가"""
        self.loads.append(load)
        
    def solve_member_forces(self) -> Dict[str, float]:
        """부재력 자동 계산 (정역학 평형방정식)"""
        
        print("\n" + "="*70, flush=True)
        print("부재력 자동 계산 시작", flush=True)
        print("="*70, flush=True)
        
        n_nodes = len(self.nodes)
        n_members = len(self.members)
        
        # 미지수 개수 계산
        n_support_reactions = sum(
            2 if node.support_type == 'pin' else 
            1 if node.support_type == 'roller' else 
            3 if node.support_type == 'fixed' else 0
            for node in self.nodes.values()
        )
        
        n_unknowns = n_members + n_support_reactions
        n_equations = 2 * n_nodes  # 각 절점마다 ΣFx=0, ΣFy=0
        
        print(f"미지수 개수: {n_unknowns}", flush=True)
        print(f"  - 부재력: {n_members}개", flush=True)
        print(f"  - 반력: {n_support_reactions}개", flush=True)
        print(f"방정식 개수: {n_equations} (절점 {n_nodes}개 × 2)\n", flush=True)
        
        # 계수 행렬 A와 상수 벡터 b 구성
        A = np.zeros((n_equations, n_unknowns))
        b = np.zeros(n_equations)
        
        member_list = list(self.members.values())
        member_idx_map = {m.id: i for i, m in enumerate(member_list)}
        
        # 지점 반력 인덱스 매핑
        reaction_idx = n_members
        support_reaction_map = {}
        
        for node in self.nodes.values():
            if node.is_support:
                if node.support_type == 'pin':
                    support_reaction_map[(node.id, 'Rx')] = reaction_idx
                    support_reaction_map[(node.id, 'Ry')] = reaction_idx + 1
                    reaction_idx += 2
                elif node.support_type == 'roller':
                    support_reaction_map[(node.id, 'Ry')] = reaction_idx
                    reaction_idx += 1
        
        # 평형방정식 구성
        for node_idx, (node_id, node) in enumerate(self.nodes.items()):
            eq_x = 2 * node_idx  # ΣFx = 0
            eq_y = 2 * node_idx + 1  # ΣFy = 0
            
            # 부재력 기여
            for member in self.members.values():
                if member.node_start == node_id or member.node_end == node_id:
                    start_node = self.nodes[member.node_start]
                    end_node = self.nodes[member.node_end]
                    
                    dx = end_node.x - start_node.x
                    dy = end_node.y - start_node.y
                    L = np.sqrt(dx**2 + dy**2)
                    
                    if L < 1e-6:
                        continue
                    
                    cos_theta = dx / L
                    sin_theta = dy / L
                    
                    member_idx = member_idx_map[member.id]
                    
                    # 부재력 방향 (start → end가 양수)
                    if member.node_start == node_id:
                        A[eq_x, member_idx] = -cos_theta
                        A[eq_y, member_idx] = -sin_theta
                    else:
                        A[eq_x, member_idx] = cos_theta
                        A[eq_y, member_idx] = sin_theta
            
            # 지점 반력 기여
            if node.is_support:
                if (node_id, 'Rx') in support_reaction_map:
                    A[eq_x, support_reaction_map[(node_id, 'Rx')]] = 1
                if (node_id, 'Ry') in support_reaction_map:
                    A[eq_y, support_reaction_map[(node_id, 'Ry')]] = 1
            
            # 외력
            for load in self.loads:
                if load.node_id == node_id:
                    b[eq_x] -= load.Fx
                    b[eq_y] -= load.Fy
        
        # 연립방정식 풀기
        print(f"계수 행렬 A: {A.shape}", flush=True)
        print(f"상수 벡터 b: {b.shape}\n", flush=True)
        
        if n_equations > n_unknowns:
            print(f"⚠️ 부정정 구조 (방정식 수 > 미지수 수)", flush=True)
            print(f"   최소제곱법 사용", flush=True)
            x, residuals, rank, s = lstsq(A, b)
            print(f"✅ 최소제곱해 계산 완료 (rank: {rank})\n", flush=True)
        else:
            x = np.linalg.solve(A, b)
            print(f"✅ 연립방정식 해 계산 완료\n", flush=True)
        
        # 결과 저장
        member_forces = {}
        for i, member in enumerate(member_list):
            member_forces[member.id] = x[i]
        
        # 반력 저장
        for (node_id, react_type), idx in support_reaction_map.items():
            if node_id not in self.reactions:
                self.reactions[node_id] = {}
            self.reactions[node_id][react_type] = x[idx]
        
        print("="*70, flush=True)
        print("✅ 부재력 계산 완료", flush=True)
        print("="*70, flush=True)
        
        return member_forces
    
    def infer_member_types(self):
        """부재 종류 자동 판단 (Strut/Tie)"""
        print("\n" + "="*70, flush=True)
        print("부재 종류 자동 판단", flush=True)
        print("="*70, flush=True)
        
        member_forces = self.solve_member_forces() if not hasattr(self, '_member_forces') else self._member_forces
        self._member_forces = member_forces
        
        for member_id, force in member_forces.items():
            member = self.members[member_id]
            if force < -1e-6:
                member.member_type = 'strut'
                print(f"  {member_id}: {force:10.1f} kN → Strut (압축재)", flush=True)
            else:
                member.member_type = 'tie'
                print(f"  {member_id}: {force:10.1f} kN → Tie (인장재)", flush=True)
        
        print("="*70, flush=True)
    
    def infer_node_types(self):
        """절점 형식 자동 판단"""
        print("\n" + "="*70, flush=True)
        print("절점 형식 자동 판단", flush=True)
        print("="*70, flush=True)
        
        for node_id, node in self.nodes.items():
            connected_members = [m for m in self.members.values() 
                                if m.node_start == node_id or m.node_end == node_id]
            
            n_struts = sum(1 for m in connected_members if m.member_type == 'strut')
            n_ties = sum(1 for m in connected_members if m.member_type == 'tie')
            
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
            
            print(f"  절점 {node_id}: {n_struts}개 Strut + {n_ties}개 Tie → {node.node_type}", flush=True)
        
        print("="*70, flush=True)
    
    def visualize_results(self, save_path='stm_result.png'):
        """결과 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 부재 그리기
        member_forces = self._member_forces if hasattr(self, '_member_forces') else {}
        
        for member in self.members.values():
            start_node = self.nodes[member.node_start]
            end_node = self.nodes[member.node_end]
            
            force = member_forces.get(member.id, 0)
            color = 'blue' if member.member_type == 'strut' else 'red'
            style = '-' if member.member_type == 'strut' else '--'
            
            ax.plot([start_node.x, end_node.x], [start_node.y, end_node.y],
                   color=color, linestyle=style, linewidth=2, label=member.member_type)
            
            # 부재력 표시
            mid_x = (start_node.x + end_node.x) / 2
            mid_y = (start_node.y + end_node.y) / 2
            ax.text(mid_x, mid_y, f'{force:.0f}', fontsize=8, ha='center')
        
        # 절점 그리기
        for node in self.nodes.values():
            marker = 's' if node.is_support else 'o'
            ax.plot(node.x, node.y, marker, markersize=10, color='black')
            ax.text(node.x, node.y+20, node.id, fontsize=12, ha='center', fontweight='bold')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        print(f"\n✅ 시각화 저장: {save_path}", flush=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
