"""
Method 1: Gradient-based Position Optimization
================================================
사용자가 노드 위치 + 부재 연결 제공
AI는 노드 위치만 최적화

기존 STM_for_MCP.py의 KDS 검증 로직 활용
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import sys
import os

# 기존 STM_for_MCP 모듈 import 시도
try:
    from STM_for_MCP import STMDesignChecker, Node, Member, Material
    # Gradient 최적화에는 Simplified checker가 더 적합!
    USE_EXISTING_CHECKER = False  # ← 강제로 False
except ImportError:
    USE_EXISTING_CHECKER = False

if USE_EXISTING_CHECKER:
    print("✓ Using full KDS checker from STM_for_MCP.py")
else:
    print("✓ Using simplified KDS checker (optimized for gradient descent)")


# ═══════════════════════════════════════════════════════════
# 사용자 입력 정의
# ═══════════════════════════════════════════════════════════

@dataclass
class UserSTMDesign:
    """
    사용자가 제공하는 STM 설계
    - 노드 위치 (초안)
    - 부재 연결 (어떤 노드끼리 연결)
    """
    # 보 정보
    beam_width: float = 500.0  # mm
    beam_height: float = 2000.0  # mm  
    beam_length: float = 6900.0  # mm
    
    # 재료
    fck: float = 27.0  # MPa
    fy: float = 400.0  # MPa
    
    # 지지점 (노드 ID)
    supports: Dict[str, str] = None  # {'A': 'pin', 'F': 'roller'}
    
    # 하중 (노드 ID, Fx, Fy)
    loads: List[Tuple[str, float, float]] = None  # [('C', 0, -2000), ...]
    
    # 노드 초기 위치
    initial_nodes: Dict[str, Tuple[float, float]] = None  # {'A': (225, 125), ...}
    
    # 부재 연결 (사용자가 지정!)
    connections: List[Tuple[str, str]] = None  # [('B', 'C'), ('C', 'D'), ...]
    
    def __post_init__(self):
        """기본값 설정 (KDS Example 10.2)"""
        if self.supports is None:
            self.supports = {'A': 'pin', 'F': 'roller'}
        
        if self.loads is None:
            self.loads = [
                ('C', 0, -2000),
                ('D', 0, -2000)
            ]
        
        if self.initial_nodes is None:
            self.initial_nodes = {
                'A': (225, 125),
                'B': (1225, 1850),
                'C': (2225, 1850),
                'D': (4675, 1850),
                'E': (5675, 1850),
                'F': (6675, 125),
                'G': (1225, 125),
                'H': (5675, 125)
            }
        
        if self.connections is None:
            # KDS Example 10.2의 부재 연결
            self.connections = [
                ('B', 'C'),
                ('C', 'D'),
                ('D', 'E'),
                ('A', 'G'),
                ('G', 'H'),
                ('H', 'F'),
                ('B', 'G'),
                ('E', 'H'),
                ('A', 'B'),
                ('C', 'G'),
                ('D', 'H'),
                ('E', 'F')
            ]


# ═══════════════════════════════════════════════════════════
# Simplified KDS Checker (fallback)
# ═══════════════════════════════════════════════════════════

class SimplifiedKDSChecker:
    """기존 STM_for_MCP.py 없을 때 사용하는 간소화 버전"""
    
    def __init__(self, fck, fy, beam_width, beam_height, beam_length):
        self.fck = fck
        self.fy = fy
        self.b = beam_width
        self.h = beam_height
        self.L = beam_length
        self.f_cd = 0.85 * fck / 1.5
        self.f_yd = fy / 1.15
    
    def calculate_forces(self, nodes, connections, loads, supports):
        """평형 방정식 해"""
        try:
            n_members = len(connections)
            n_nodes = len(nodes)
            node_list = list(nodes.keys())
            
            # 반력 개수
            n_reactions = 0
            reaction_map = {}
            for node_id, support_type in supports.items():
                if support_type == 'pin':
                    reaction_map[f'Rx_{node_id}'] = n_reactions
                    reaction_map[f'Ry_{node_id}'] = n_reactions + 1
                    n_reactions += 2
                elif support_type == 'roller':
                    reaction_map[f'Ry_{node_id}'] = n_reactions
                    n_reactions += 1
            
            # 평형 행렬
            A = np.zeros((n_nodes * 2, n_members + n_reactions))
            b = np.zeros(n_nodes * 2)
            
            # 부재 기여
            for j, (n1_id, n2_id) in enumerate(connections):
                if n1_id not in nodes or n2_id not in nodes:
                    continue
                
                x1, y1 = nodes[n1_id]
                x2, y2 = nodes[n2_id]
                
                dx = x2 - x1
                dy = y2 - y1
                L = np.sqrt(dx**2 + dy**2)
                
                if L < 1e-6:
                    continue
                
                cos_val = dx / L
                sin_val = dy / L
                
                # 노드 1
                i1 = node_list.index(n1_id)
                A[i1*2, j] = cos_val
                A[i1*2+1, j] = sin_val
                
                # 노드 2
                i2 = node_list.index(n2_id)
                A[i2*2, j] = -cos_val
                A[i2*2+1, j] = -sin_val
            
            # 반력 기여
            for node_id in supports:
                i = node_list.index(node_id)
                if f'Rx_{node_id}' in reaction_map:
                    col = n_members + reaction_map[f'Rx_{node_id}']
                    A[i*2, col] = 1.0
                if f'Ry_{node_id}' in reaction_map:
                    col = n_members + reaction_map[f'Ry_{node_id}']
                    A[i*2+1, col] = 1.0
            
            # 하중
            for load_node_id, Fx, Fy in loads:
                i = node_list.index(load_node_id)
                b[i*2] -= Fx
                b[i*2+1] -= Fy
            
            # 해 구하기
            solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            forces = solution[:n_members]
            error = np.linalg.norm(A @ solution - b)
            
            return forces, error
        
        except Exception as e:
            print(f"⚠️ Force calculation error: {e}")
            return None, 1e10
    
    def check_design(self, nodes, connections, loads, supports):
        """설계 검증"""
        forces, error = self.calculate_forces(nodes, connections, loads, supports)
        
        if forces is None or error > 100.0:
            return {
                'ok': False,
                'equilibrium_error': error if error else 1e10,
                'total_width': 1e10
            }
        
        # 부재 폭 계산
        total_width = 0.0
        max_violation = 0.0
        
        for (n1_id, n2_id), force in zip(connections, forces):
            x1, y1 = nodes[n1_id]
            x2, y2 = nodes[n2_id]
            L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if L < 1e-6:
                continue
            
            # 필요 폭 계산 (간소화)
            if force > 0:  # Tension (tie)
                A_s = abs(force * 1000) / self.f_yd  # kN → N
                w_req = A_s / (0.01 * L)  # 가정: 1% 철근비
            else:  # Compression (strut)
                nu = 0.6
                w_req = abs(force * 1000) / (0.85 * self.f_cd * L * nu)
            
            w_req = max(200, min(w_req, 1000))  # 200-1000mm
            total_width += w_req
            
            # 강도 체크
            if force > 0:
                capacity = w_req * 0.01 * L * self.f_yd / 1000
            else:
                capacity = 0.85 * self.f_cd * w_req * L * 0.6 / 1000
            
            violation = max(0, abs(force) - capacity)
            max_violation = max(max_violation, violation)
        
        return {
            'ok': max_violation < 1.0 and error < 100.0,
            'equilibrium_error': error,
            'total_width': total_width,
            'max_violation': max_violation
        }


# ═══════════════════════════════════════════════════════════
# Node Type Classifier (절점 타입 분류)
# ═══════════════════════════════════════════════════════════

class NodeTypeClassifier:
    """절점 타입 분류 (CCC, CCT, CTT, TTT)"""
    
    def __init__(self):
        # KDS 기준 절점 효율계수
        self.beta_n = {
            'CCC': 1.0,   # 3개 압축
            'CCT': 0.8,   # 2개 압축 + 1개 인장
            'CTT': 0.6,   # 1개 압축 + 2개 인장
            'TTT': 0.6,   # 3개 인장
        }
    
    def classify_node(self, node_id, forces, connections, supports=None):
        """절점 타입 분류
        
        Args:
            node_id: 노드 ID
            forces: 부재력 배열
            connections: 부재 연결 리스트
            supports: 지지점 정보 (dict)
        """
        # 이 노드에 연결된 부재 찾기
        connected_members = []
        for i, (n1, n2) in enumerate(connections):
            if n1 == node_id or n2 == node_id:
                force = forces[i]
                member_type = 'Strut' if force < 0 else 'Tie'
                connected_members.append({
                    'member': (n1, n2),
                    'force': abs(force),
                    'type': member_type,
                    'index': i
                })
        
        # 압축재/인장재 개수
        n_strut = sum(1 for m in connected_members if m['type'] == 'Strut')
        n_tie = sum(1 for m in connected_members if m['type'] == 'Tie')
        
        # 지지점인 경우: 반력을 압축재로 간주
        is_support = False
        if supports is not None:
            if node_id in supports:  # 딕셔너리 key 확인
                is_support = True
                n_strut += 1  # 반력 = 압축재
        
        # C1T3 → CTT로 간주 (1개 압축 + 2개 인장)
        if n_strut == 1 and n_tie == 3:
            n_tie = 2  # 3개 타이 → 2개로 간주
        
        # 타입 결정 (표준 타입만 사용)
        if n_strut == 3 and n_tie == 0:
            node_type = 'CCC'
        elif n_strut == 2 and n_tie == 1:
            node_type = 'CCT'
        elif n_strut == 1 and n_tie == 2:
            node_type = 'CTT'
        elif n_strut == 0 and n_tie >= 3:
            node_type = 'TTT'
        else:
            # 특수 케이스 (기본값 사용)
            node_type = f'C{n_strut}T{n_tie}'
        
        return {
            'type': node_type,
            'n_strut': n_strut,
            'n_tie': n_tie,
            'members': connected_members,
            'beta_n': self.beta_n.get(node_type, 0.50),
            'is_support': is_support
        }


# ═══════════════════════════════════════════════════════════
# Detailed Member Calculator (상세 부재 폭 계산)
# ═══════════════════════════════════════════════════════════

class DetailedMemberCalculator:
    """상세 부재 폭 계산"""
    
    def __init__(self, fck=27.0, fy=400.0, beam_width=500.0):
        self.fck = fck
        self.fy = fy
        self.beam_width = beam_width

        # 강도 계수
        self.phi = 0.75
        
        # 기본 파라미터
        self.cover = 40.0
        self.d_s = 10.0
        self.spacing_min = 30.0
    
    def determine_beta_s(self, node1_info, node2_info):
        """스트럿 효율계수 β_s 결정
        
        - 평행 스트럿: β_s = 1.0
        - 병목형 스트럿: β_s = 0.6
        """
        if node1_info['type'] == 'CCC' and node2_info['type'] == 'CCC':
            return 1.0
        else:
            return 0.6
    
    def calculate_strut_width(self, force, beta_s=0.6):
        """스트럿 필요 폭: w_s = C / (φ × 0.85 × β_s × f_ck × b)"""
        C = abs(force) * 1000  # N
        w_s = C / (self.phi * 0.85 * beta_s * self.fck * self.beam_width)
        return max(200, w_s)
    
    def calculate_tie_width(self, force, node1, node2, all_nodes, d_b=None):
        """타이 필요 폭 계산 (수평/수직 케이스)
        
        Args:
            force: 인장력 (kN)
            node1, node2: 타이 양단 노드 ID
            all_nodes: 전체 노드 딕셔너리
            d_b: 철근 직경 (None이면 자동)
        """
        import math
        
        T = abs(force) * 1000  # N
        A_s = T / self.fy  # ← fy 직접 사용!
        
        # 주철근 직경 결정
        if d_b is None:
            if force < 500:
                d_b = 19
            elif force < 1000:
                d_b = 22
            elif force < 1500:
                d_b = 25
            elif force < 2500:
                d_b = 29
            else:
                d_b = 32
        
        A_bar = np.pi * (d_b/2)**2
        n_bars = int(np.ceil(A_s / A_bar))
        spacing = max(self.spacing_min, d_b, 25)
        
        # 노드 좌표
        x1, y1 = all_nodes[node1]
        x2, y2 = all_nodes[node2]
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # 수평 타이 (y 좌표 거의 같음)
        if dy < 100:
            w_t_calc = 2 * (self.cover + self.d_s + d_b + spacing/2)
            
        # 수직 타이 (x 좌표 거의 같음)
        elif dx < 100:
            w_t_calc = self._calculate_vertical_tie_width(
                node1, node2, all_nodes, d_b, spacing
            )
            
        # 대각선 (기본 공식 사용)
        else:
            w_t_calc = 2 * (self.cover + self.d_s + d_b + spacing/2)
        
        # 50의 배수로 올림
        w_t = math.ceil(w_t_calc / 50) * 50
        
        # 최소 250mm
        w_t = max(250, w_t)
        
        return {
            'w_t': w_t,
            'd_b': d_b,
            'n_bars': n_bars,
            'A_s': A_s,
            'spacing': spacing
        }

    def _calculate_vertical_tie_width(self, node1, node2, all_nodes, d_b, spacing):
        """수직 타이 w_t 계산 (인접 노드 중점 방식)"""
        x1, y1 = all_nodes[node1]
        x2, y2 = all_nodes[node2]
        
        # 상단 노드 (y가 큰 쪽)
        if y1 > y2:
            upper_node = node1
            x_upper = x1
            y_upper = y1
        else:
            upper_node = node2
            x_upper = x2
            y_upper = y2
        
        # 상단 노드와 비슷한 y 높이의 인접 노드 찾기
        adjacent = []
        for nid, (x, y) in all_nodes.items():
            if nid != node1 and nid != node2:
                if abs(y - y_upper) < 100:  # 같은 높이
                    adjacent.append((nid, x))
        
        if len(adjacent) < 2:
            # 인접 노드 부족 시 기본 공식
            return 2 * (self.cover + self.d_s + d_b + spacing/2)
        
        # x 좌표 정렬
        adjacent.sort(key=lambda item: item[1])
        
        # x_upper 좌우 찾기
        left_x = None
        right_x = None
        
        for nid, x in adjacent:
            if x < x_upper:
                left_x = x
            elif x > x_upper and right_x is None:
                right_x = x
        
        # 중점까지 거리
        distances = []
        if left_x is not None:
            mid_x = (x_upper + left_x) / 2
            distances.append(abs(x_upper - mid_x))
        
        if right_x is not None:
            mid_x = (x_upper + right_x) / 2
            distances.append(abs(x_upper - mid_x))
        
        if not distances:
            return 2 * (self.cover + self.d_s + d_b + spacing/2)
        
        # 최소 거리 × 2
        min_dist = min(distances)
        return 2 * min_dist


# ═══════════════════════════════════════════════════════════
# Node Design Checker (절점 설계 검토)
# ═══════════════════════════════════════════════════════════

class NodeDesignChecker:
    """절점 설계 검토 (표 10.2.4)"""
    
    def __init__(self, fck=27.0, fy = 400.0, beam_width=500.0):
        self.fck = fck
        self.fy = fy
        self.beam_width = beam_width

        # 강도 계수
        self.phi = 0.75
    
    def calculate_required_area(self, node_info, max_force):
        """절점 필요 단면적: A_req = F_max / (β_n × f_cd)"""
        F = abs(max_force) * 1000  # N
        beta_n = node_info['beta_n']
        f_cu = beta_n * self.fck
        f_cd_node = 0.85 * f_cu / 1.5
        A_req = F / f_cd_node
        return A_req
    
    def calculate_actual_area(self, node_info, member_widths):
        """절점 실제 단면적"""
        widths = []
        for member in node_info['members']:
            idx = member['index']
            if idx in member_widths:
                widths.append(member_widths[idx])
        
        if not widths:
            return 0
        
        avg_width = sum(widths) / len(widths)
        A_actual = avg_width * self.beam_width
        return A_actual
    
    def check_node(self, node_info, max_force, member_widths):
        """절점 설계 검토"""
        A_req = self.calculate_required_area(node_info, max_force)
        A_actual = self.calculate_actual_area(node_info, member_widths)
        
        ratio = A_actual / A_req if A_req > 0 else 999
        status = 'OK' if ratio >= 1.0 else 'NG'
        
        return {
            'A_req': A_req,
            'A_actual': A_actual,
            'ratio': ratio,
            'status': status
        }


# ═══════════════════════════════════════════════════════════
# Gradient-based Optimizer
# ═══════════════════════════════════════════════════════════

class GradientSTMOptimizer:
    """
    Gradient 기반 노드 위치 최적화
    
    고정:
    - 부재 연결 (사용자 제공)
    
    최적화:
    - 노드 위치 (±15% 범위)
    """
    
    def __init__(self, user_design: UserSTMDesign, max_change_ratio=0.15):
        self.user_design = user_design
        self.max_change_ratio = max_change_ratio
        
        # 초기 노드
        self.initial_nodes = user_design.initial_nodes.copy()
        self.node_ids = list(self.initial_nodes.keys())
        
        # ═══════════════════════════════════════════════════════════
        # 대칭 제약 설정
        # ═══════════════════════════════════════════════════════════
        self.x_center = user_design.beam_length / 2  # 3450mm
        
        # 고정 노드 (지지점 - 최적화 안 함)
        self.fixed_nodes = ['A', 'F']
        
        # 대칭 쌍 정의 (A, F 제외)
        self.symmetric_pairs = [
            ('B', 'E'),
            ('C', 'D'),
            ('G', 'H')
        ]
        
        # 좌측 노드만 최적화 변수로 (A 제외!)
        self.left_nodes = ['B', 'C', 'G']
        self.right_nodes = ['E', 'D', 'H']
        
        # 고정된 부재 연결
        self.connections = user_design.connections
        
        # KDS Checker
        if USE_EXISTING_CHECKER:
            self.material = Material(
                fck=user_design.fck,
                fy=user_design.fy
            )
            self.use_full_checker = True
            print("✓ Using full KDS checker from STM_for_MCP.py")
        else:
            self.checker = SimplifiedKDSChecker(
                fck=user_design.fck,
                fy=user_design.fy,
                beam_width=user_design.beam_width,
                beam_height=user_design.beam_height,
                beam_length=user_design.beam_length
            )
            self.use_full_checker = False
            print("✓ Using simplified KDS checker")
    
    def optimize(self, method='L-BFGS-B', verbose=True):
        """
        Gradient 최적화 실행 (대칭 제약 포함)
        
        Args:
            method: 'L-BFGS-B' or 'evolutionary'
        """
        if verbose:
            print(f"\n{'='*60}")
            print("Symmetric STM Optimization (X-axis only)")
            print(f"{'='*60}")
            print(f"Total nodes: {len(self.node_ids)}")
            print(f"Fixed nodes: {len(self.fixed_nodes)} (A, F - supports)")
            print(f"Optimization variables: {len(self.left_nodes)} X-coordinates (B, C, G)")
            print(f"Y-coordinates: FIXED (w_t, w_s determined)")
            print(f"Symmetric pairs: {len(self.symmetric_pairs)}")
            print(f"Symmetry axis: x = {self.x_center:.1f}mm")
            print(f"Connections: {len(self.connections)} (fixed by user)")
            print(f"Max X-position change: ±{self.max_change_ratio*100:.0f}%")
            print(f"Method: {method}")
        
        # 초기값 (좌측 노드만)
        x0 = self._nodes_to_vector(self.initial_nodes)
        
        # 경계
        bounds = self._get_bounds()
        
        # 초기 목적함수
        f0 = self.objective(x0)
        if verbose:
            print(f"Initial objective: {f0:.2f}")
        
        # 최적화
        if method == 'L-BFGS-B':
            result = minimize(
                self.objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'disp': verbose}
            )
        else:  # evolutionary
            result = differential_evolution(
                self.objective,
                bounds,
                maxiter=50,
                disp=verbose
            )
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final objective (continuous): {result.fun:.2f}")
            print(f"Improvement: {f0 - result.fun:.2f}")
            print(f"Success: {result.success}")
        
        # 최적화된 노드 (연속값)
        optimized_nodes_continuous = self._vector_to_nodes(result.x)
        
        # 5mm 그리드로 반올림
        optimized_nodes = self._round_to_grid(optimized_nodes_continuous, grid_size=5)
        
        # 반올림 후 목적함수 재계산
        x_rounded = self._nodes_to_vector(optimized_nodes)
        f_rounded = self.objective(x_rounded)
        
        if verbose:
            print(f"\n{'='*60}")
            print("Rounding to 5mm grid:")
            print(f"{'='*60}")
            print(f"Objective after rounding: {f_rounded:.2f}")
            print(f"Difference from continuous: {f_rounded - result.fun:.2f}")
            print(f"Total improvement: {f0 - f_rounded:.2f} ({(f0-f_rounded)/f0*100:.1f}%)")
        
        # 대칭 검증
        if verbose:
            self._verify_symmetry(optimized_nodes)
        
        return {
            'optimized_nodes': optimized_nodes,
            'initial_nodes': self.initial_nodes,
            'objective_value': f_rounded,
            'objective_value_continuous': result.fun,
            'initial_objective': f0,
            'success': result.success,
            'message': result.message,
            'connections': self.connections
        }
    
    def _nodes_to_vector(self, nodes):
        """좌측 노드의 X 좌표만 벡터로 변환 (Y는 고정)"""
        x = []
        for node_id in self.left_nodes:
            x_coord, y_coord = nodes[node_id]
            x.append(x_coord)  # X 좌표만!
        return np.array(x)
    
    def _vector_to_nodes(self, x):
        """벡터 → 노드 (X만 변경, Y는 초기값 고정, A/F는 완전 고정)
        
        w_t, w_s가 결정되면 Y 위치는 고정됨
        따라서 X 좌표만 최적화 가능
        A, F는 지지점이므로 완전 고정
        """
        nodes = {}
        
        # 고정 노드 (A, F) - 지지점은 항상 초기값
        for node_id in self.fixed_nodes:
            nodes[node_id] = self.initial_nodes[node_id]
        
        # 좌측 노드 (B, C, G만 최적화)
        for i, node_id in enumerate(self.left_nodes):
            x_new = x[i]
            y_fixed = self.initial_nodes[node_id][1]  # 초기 Y 값 유지!
            nodes[node_id] = (x_new, y_fixed)
        
        # 우측 노드 (E, D, H 대칭)
        for left_id, right_id in self.symmetric_pairs:
            x_left, y_left = nodes[left_id]
            x_right = 2 * self.x_center - x_left  # X 대칭
            y_right = y_left  # Y 동일 (고정)
            nodes[right_id] = (x_right, y_right)
        
        return nodes
    
    def _round_to_grid(self, nodes, grid_size=5):
        """노드 위치를 grid_size의 배수로 반올림
        
        Args:
            nodes: 노드 딕셔너리
            grid_size: 그리드 크기 (mm), 기본 5mm
            
        Returns:
            반올림된 노드 딕셔너리
        """
        rounded_nodes = {}
        
        # 고정 노드는 그대로
        for node_id in self.fixed_nodes:
            rounded_nodes[node_id] = self.initial_nodes[node_id]
        
        # 좌측 노드 반올림
        for node_id in self.left_nodes:
            x, y = nodes[node_id]
            x_rounded = round(x / grid_size) * grid_size
            rounded_nodes[node_id] = (x_rounded, y)
        
        # 우측 노드 대칭 재계산
        for left_id, right_id in self.symmetric_pairs:
            x_left, y_left = rounded_nodes[left_id]
            x_right = 2 * self.x_center - x_left
            y_right = y_left
            rounded_nodes[right_id] = (x_right, y_right)
        
        return rounded_nodes
    
    def _get_bounds(self):
        """X 좌표 범위만 설정 (Y는 고정이므로 불필요)"""
        bounds = []
        
        max_dx = self.max_change_ratio * self.user_design.beam_length
        
        for node_id in self.left_nodes:
            x0, y0 = self.initial_nodes[node_id]
            
            # X 범위만 (중심선 넘지 않음)
            x_min = max(0, x0 - max_dx)
            x_max = min(self.x_center, x0 + max_dx)  # 중심선이 상한
            bounds.append((x_min, x_max))
            
            # Y는 경계 설정 불필요 (고정됨)
        
        return bounds
    
    def objective(self, x):
        """
        목적함수: 최소화
        
        구성:
        1. KDS 위반 페널티
        2. 부재 폭 합 (재료 사용량)
        3. 위치 변경 최소화
        """
        nodes = self._vector_to_nodes(x)
        
        # KDS 체크
        if self.use_full_checker:
            result = self._check_with_full_kds(nodes)
        else:
            result = self.checker.check_design(
                nodes,
                self.connections,
                self.user_design.loads,
                self.user_design.supports
            )
        
        if not result['ok']:
            return 1e10 + result.get('equilibrium_error', 1e10)
        
        cost = 0.0
        
        # 1. 부재 폭 (재료 사용량) - 주 목표
        cost += result['total_width'] * 0.1  # 0.01 → 0.1 (10배 증가)
        
        # 2. 위치 변경 최소화 - 부차적 목표
        position_change = 0.0
        for node_id in self.node_ids:
            x0, y0 = self.initial_nodes[node_id]
            x1, y1 = nodes[node_id]
            position_change += np.sqrt((x1-x0)**2 + (y1-y0)**2)
        
        cost += position_change * 0.001  # 0.1 → 0.001 (100배 감소)
        
        # 3. 평형 오차
        cost += result.get('equilibrium_error', 0) * 0.01
        
        return cost
    
    def _check_with_full_kds(self, nodes):
        """기존 STM_for_MCP.py 사용"""
        try:
            # Checker 생성
            checker = STMDesignChecker(
                material=self.material,
                beam_width=self.user_design.beam_width,
                beam_height=self.user_design.beam_height,
                beam_length=self.user_design.beam_length
            )
            
            # 노드 추가
            for node_id, (x, y) in nodes.items():
                checker.add_node(Node(node_id, x, y))
            
            # 부재 추가
            for n1, n2 in self.connections:
                mid = f"{n1}{n2}"
                checker.add_member(Member(mid, n1, n2))
            
            # 하중 추가
            for node_id, Fx, Fy in self.user_design.loads:
                checker.add_load(node_id, Fx, Fy)
            
            # 지지점 추가
            for node_id, support_type in self.user_design.supports.items():
                checker.add_support(node_id, support_type)
            
            # 검증
            results = checker.verify_stm(l_b=450)
            
            # 부재 폭 합계
            total_width = 0.0
            for res in results['struts'].values():
                total_width += res.get('w_s', 300)
            for res in results['ties'].values():
                total_width += res.get('w_t', 300)
            
            return {
                'ok': results['overall'],
                'total_width': total_width,
                'equilibrium_error': 0.0  # 기존 checker는 평형 가정
            }
        
        except Exception as e:
            return {
                'ok': False,
                'total_width': 1e10,
                'equilibrium_error': 1e10
            }
    
    def _verify_symmetry(self, nodes):
        """대칭 검증 출력"""
        print(f"\n{'='*60}")
        print("Symmetry Verification")
        print(f"{'='*60}")
        
        # 고정 노드 표시
        print(f"\n[FIXED NODES - Supports]")
        for node_id in self.fixed_nodes:
            x, y = nodes[node_id]
            x_init, y_init = self.initial_nodes[node_id]
            print(f"{node_id}: ({x:.1f}, {y:.1f}) - FIXED (no change)")
        
        # 대칭 쌍 검증
        print(f"\n[OPTIMIZED SYMMETRIC PAIRS]")
        all_symmetric = True
        
        for left_id, right_id in self.symmetric_pairs:
            x_left, y_left = nodes[left_id]
            x_right, y_right = nodes[right_id]
            
            dist_left = abs(self.x_center - x_left)
            dist_right = abs(x_right - self.x_center)
            y_diff = abs(y_right - y_left)
            x_error = abs(dist_left - dist_right)
            
            print(f"{left_id} ↔ {right_id}:")
            print(f"  Position: ({x_left:.1f}, {y_left:.1f}) ↔ ({x_right:.1f}, {y_right:.1f})")
            print(f"  Distance from center: {dist_left:.2f}mm ↔ {dist_right:.2f}mm")
            print(f"  Y-coordinate: {y_left:.2f}mm ↔ {y_right:.2f}mm")
            
            if x_error < 1e-6 and y_diff < 1e-6:
                print(f"  ✓ Perfect symmetry")
            else:
                print(f"  ⚠️ Symmetry error: ΔX={x_error:.6f}mm, ΔY={y_diff:.6f}mm")
                all_symmetric = False
        
        print(f"{'='*60}")
        if all_symmetric:
            print("✓ All optimized nodes are perfectly symmetric!")
        else:
            print("⚠️ Warning: Some symmetry errors detected")
        print(f"{'='*60}\n")
    
    def print_member_forces_comparison(self, result):
        """부재력 전후 비교 출력"""
        
        print(f"\n{'='*100}")
        print("Member Forces Comparison (Before → After)")
        print(f"{'='*100}")
        
        # 초기 부재력
        initial_forces, _ = self.checker.calculate_forces(
            result['initial_nodes'],
            self.connections,
            self.user_design.loads,
            self.user_design.supports
        )
        
        # 최적화 부재력
        optimized_forces, _ = self.checker.calculate_forces(
            result['optimized_nodes'],
            self.connections,
            self.user_design.loads,
            self.user_design.supports
        )
        
        # 헤더
        print(f"{'Member':<12} {'Type':<10} {'Initial':<12} {'Optimized':<12} {'Change':<12} {'Change %':<12}")
        print(f"{'-'*100}")
        
        # 각 부재별 출력
        for i, (n1, n2) in enumerate(self.connections):
            member_name = f"{n1}-{n2}"
            
            f_initial = initial_forces[i]
            f_optimized = optimized_forces[i]
            change = f_optimized - f_initial
            change_pct = (change / f_initial * 100) if abs(f_initial) > 1e-6 else 0
            
            member_type = "Tie" if f_optimized > 0 else "Strut"
            
            print(f"{member_name:<12} {member_type:<10} "
                  f"{abs(f_initial):>10.1f}kN {abs(f_optimized):>10.1f}kN "
                  f"{change:>+10.1f}kN {change_pct:>+10.1f}%")
        
        print(f"{'-'*100}")
        
        # 통계
        print(f"\nStatistics:")
        print(f"  Total members: {len(self.connections)}")
        
        tie_mask = optimized_forces > 0
        if np.any(tie_mask):
            tie_initial = np.abs(initial_forces[tie_mask])
            tie_optimized = np.abs(optimized_forces[tie_mask])
            tie_reduction = np.mean((tie_initial - tie_optimized) / tie_initial * 100)
            print(f"  Ties: {np.sum(tie_mask)} members")
            print(f"    Average reduction: {tie_reduction:.1f}%")
        
        strut_mask = optimized_forces < 0
        if np.any(strut_mask):
            strut_initial = np.abs(initial_forces[strut_mask])
            strut_optimized = np.abs(optimized_forces[strut_mask])
            strut_reduction = np.mean((strut_initial - strut_optimized) / strut_initial * 100)
            print(f"  Struts: {np.sum(strut_mask)} members")
            print(f"    Average reduction: {strut_reduction:.1f}%")
        
        all_initial = np.abs(initial_forces)
        all_optimized = np.abs(optimized_forces)
        overall_reduction = np.mean((all_initial - all_optimized) / all_initial * 100)
        print(f"  Overall average reduction: {overall_reduction:.1f}%")
        
        print(f"{'='*100}\n")
    
    def print_member_forces_by_category(self, result):
        """카테고리별 부재력 출력"""
        
        print(f"\n{'='*100}")
        print("Member Forces by Category")
        print(f"{'='*100}")
        
        # 부재력 계산
        initial_forces, _ = self.checker.calculate_forces(
            result['initial_nodes'],
            self.connections,
            self.user_design.loads,
            self.user_design.supports
        )
        
        optimized_forces, _ = self.checker.calculate_forces(
            result['optimized_nodes'],
            self.connections,
            self.user_design.loads,
            self.user_design.supports
        )
        
        # 분류
        ties = []
        struts = []
        
        for i, (n1, n2) in enumerate(self.connections):
            member_name = f"{n1}-{n2}"
            f_initial = initial_forces[i]
            f_optimized = optimized_forces[i]
            change = f_optimized - f_initial
            change_pct = (change / f_initial * 100) if abs(f_initial) > 1e-6 else 0
            
            data = {
                'name': member_name,
                'initial': abs(f_initial),
                'optimized': abs(f_optimized),
                'change': change,
                'change_pct': change_pct
            }
            
            if f_optimized > 0:
                ties.append(data)
            else:
                struts.append(data)
        
        # Ties 출력
        print(f"\n[TIES - Tension Members]")
        print(f"{'-'*100}")
        print(f"{'Member':<12} {'Initial':<12} {'Optimized':<12} {'Change':<12} {'Change %':<12}")
        print(f"{'-'*100}")
        for tie in ties:
            print(f"{tie['name']:<12} {tie['initial']:>10.1f}kN {tie['optimized']:>10.1f}kN "
                  f"{tie['change']:>+10.1f}kN {tie['change_pct']:>+10.1f}%")
        
        if ties:
            avg_reduction = np.mean([t['change_pct'] for t in ties])
            print(f"{'-'*100}")
            print(f"Average reduction: {avg_reduction:.1f}%")
        
        # Struts 출력
        print(f"\n[STRUTS - Compression Members]")
        print(f"{'-'*100}")
        print(f"{'Member':<12} {'Initial':<12} {'Optimized':<12} {'Change':<12} {'Change %':<12}")
        print(f"{'-'*100}")
        for strut in struts:
            print(f"{strut['name']:<12} {strut['initial']:>10.1f}kN {strut['optimized']:>10.1f}kN "
                  f"{strut['change']:>+10.1f}kN {strut['change_pct']:>+10.1f}%")
        
        if struts:
            avg_reduction = np.mean([s['change_pct'] for s in struts])
            print(f"{'-'*100}")
            print(f"Average reduction: {avg_reduction:.1f}%")
        
        print(f"\n{'='*100}\n")
    
    def plot_comparison(self, result, save_path=None):
        """최적화 전후 비교 - 기본 버전"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # 초기
        self._plot_stm(ax1, self.initial_nodes, "Initial Design")
        
        # 최적화
        self._plot_stm(ax2, result['optimized_nodes'], "Optimized Design")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        
        return fig
    
    def plot_enhanced_comparison(self, result, save_path='stm_enhanced_comparison.png'):
        """
        개선된 비교 플롯
        
        Features:
        - 전후 비교 (좌우)
        - 노드 이동 벡터
        - 이동 거리 표시
        - 부재력 표시
        - 통계 정보
        """
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.lines import Line2D
        
        fig = plt.figure(figsize=(20, 12))
        
        # 3개 subplot: 초기, 최적화, 변화량
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, (4, 6))  # 하단 전체
        
        # Subplot 1: 초기 설계
        self._plot_stm_with_forces(ax1, self.initial_nodes, "Initial Design")
        
        # Subplot 2: 최적화된 설계
        self._plot_stm_with_forces(ax2, result['optimized_nodes'], "Optimized Design")
        
        # Subplot 3: 노드 이동 벡터
        self._plot_movement_vectors(ax3, result)
        
        # Subplot 4: 통계
        self._plot_movement_statistics(ax4, result)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Enhanced plot saved: {save_path}")
        
        return fig
    
    def _plot_stm_with_forces(self, ax, nodes, title):
        """힘 표시가 포함된 STM 플롯"""
        # 보 윤곽
        ax.add_patch(plt.Rectangle(
            (0, 0), 
            self.user_design.beam_length,
            self.user_design.beam_height,
            fill=False, edgecolor='gray', linewidth=2, linestyle='--', alpha=0.5
        ))
        
        # 힘 계산
        forces = None
        if not self.use_full_checker:
            forces, error = self.checker.calculate_forces(
                nodes, self.connections, 
                self.user_design.loads, self.user_design.supports
            )
        
        # 부재 그리기
        for i, (n1_id, n2_id) in enumerate(self.connections):
            x1, y1 = nodes[n1_id]
            x2, y2 = nodes[n2_id]
            
            if forces is not None and i < len(forces):
                force = forces[i]
                
                if force > 0:  # Tension
                    color = 'red'
                    linestyle = '--'
                    linewidth = 2 + min(abs(force) / 500, 2)
                else:  # Compression
                    color = 'blue'
                    linestyle = '-'
                    linewidth = 2 + min(abs(force) / 500, 2)
                
                # 부재력 표시
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax.text(mid_x, mid_y, f"{abs(force):.0f}",
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle='round,pad=0.2',
                               facecolor='white',
                               edgecolor=color,
                               alpha=0.8))
            else:
                dy = abs(y2 - y1)
                if dy < 100:
                    avg_y = (y1 + y2) / 2
                    if avg_y < self.user_design.beam_height * 0.3:
                        color, linestyle, linewidth = 'red', '--', 2
                    else:
                        color, linestyle, linewidth = 'blue', '-', 2.5
                else:
                    color, linestyle, linewidth = 'blue', '-', 2.5
            
            ax.plot([x1, x2], [y1, y2],
                   color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
        
        # 노드
        for node_id, (x, y) in nodes.items():
            ax.plot(x, y, 'ko', markersize=10, zorder=5)
            ax.text(x, y+120, node_id,
                   ha='center', fontsize=12, fontweight='bold', zorder=6)
        
        # 지지점
        for node_id, support_type in self.user_design.supports.items():
            x, y = nodes[node_id]
            if support_type == 'pin':
                ax.plot(x, y-100, '^', markersize=15, color='black', zorder=5)
            else:
                ax.plot(x, y-100, 'o', markersize=12, color='black', zorder=5)
        
        # 하중
        for node_id, Fx, Fy in self.user_design.loads:
            x, y = nodes[node_id]
            ax.arrow(x, y+220, 0, -150,
                    head_width=80, head_length=50,
                    fc='darkgreen', ec='darkgreen', linewidth=2, zorder=5)
            ax.text(x, y+300, f"{abs(Fy):.0f}kN",
                   ha='center', fontsize=11, color='darkgreen', fontweight='bold')
        
        ax.set_xlabel('Length (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Height (mm)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-300, self.user_design.beam_length + 300)
        ax.set_ylim(-400, self.user_design.beam_height + 400)
    
    def _plot_movement_vectors(self, ax, result):
        """노드 이동 벡터 표시"""
        from matplotlib.patches import FancyArrowPatch
        
        # 보 윤곽
        ax.add_patch(plt.Rectangle(
            (0, 0),
            self.user_design.beam_length,
            self.user_design.beam_height,
            fill=False, edgecolor='gray', linewidth=2, linestyle='--', alpha=0.3
        ))
        
        # 초기 노드 (연한 색)
        for node_id, (x, y) in result['initial_nodes'].items():
            ax.plot(x, y, 'o', color='lightblue', markersize=12, alpha=0.5, zorder=3)
        
        # 최적화된 노드 (진한 색)
        for node_id, (x, y) in result['optimized_nodes'].items():
            ax.plot(x, y, 'o', color='darkgreen', markersize=12, zorder=4)
            ax.text(x, y+120, node_id, ha='center', fontsize=11, fontweight='bold')
        
        # 이동 벡터
        max_distance = 0
        for node_id in result['initial_nodes'].keys():
            x0, y0 = result['initial_nodes'][node_id]
            x1, y1 = result['optimized_nodes'][node_id]
            
            dx = x1 - x0
            dy = y1 - y0
            distance = np.sqrt(dx**2 + dy**2)
            max_distance = max(max_distance, distance)
            
            if distance > 0.1:
                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    arrowstyle='->', 
                    mutation_scale=25,
                    linewidth=2.5,
                    color='red',
                    zorder=5
                )
                ax.add_patch(arrow)
                
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                ax.text(mid_x, mid_y, f"{distance:.1f}mm",
                       fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='yellow',
                               edgecolor='red',
                               alpha=0.8))
        
        ax.set_xlabel('Length (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Height (mm)', fontsize=12, fontweight='bold')
        ax.set_title('Node Movement Vectors', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-300, self.user_design.beam_length + 300)
        ax.set_ylim(-400, self.user_design.beam_height + 400)
        
        ax.text(0.02, 0.98, f"Max movement: {max_distance:.1f}mm",
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _plot_movement_statistics(self, ax, result):
        """통계 막대 그래프"""
        node_ids = list(result['initial_nodes'].keys())
        distances = []
        
        for node_id in node_ids:
            x0, y0 = result['initial_nodes'][node_id]
            x1, y1 = result['optimized_nodes'][node_id]
            distance = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            distances.append(distance)
        
        colors = ['skyblue' if d < 30 else 'orange' if d < 50 else 'red' for d in distances]
        bars = ax.bar(node_ids, distances, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, dist in zip(bars, distances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{dist:.1f}mm',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Node', fontsize=14, fontweight='bold')
        ax.set_ylabel('Movement Distance (mm)', fontsize=14, fontweight='bold')
        ax.set_title('Node Movement Statistics', fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        avg_distance = np.mean(distances)
        ax.axhline(y=avg_distance, color='darkgreen', linestyle='--', linewidth=2.5, 
                   label=f'Average: {avg_distance:.1f}mm')
        
        stats_text = f"""
        Average: {avg_distance:.1f} mm
        Max: {max(distances):.1f} mm ({node_ids[distances.index(max(distances))]})
        Min: {min(distances):.1f} mm ({node_ids[distances.index(min(distances))]})
        Total: {sum(distances):.1f} mm
        Objective: {result['objective_value']:.2f}
        """
        
        ax.text(0.98, 0.97, stats_text,
               transform=ax.transAxes,
               fontsize=11,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        color_legend = [
            plt.Rectangle((0,0),1,1, fc='skyblue', alpha=0.7, edgecolor='black', label='< 30mm'),
            plt.Rectangle((0,0),1,1, fc='orange', alpha=0.7, edgecolor='black', label='30-50mm'),
            plt.Rectangle((0,0),1,1, fc='red', alpha=0.7, edgecolor='black', label='> 50mm')
        ]
        ax.legend(handles=color_legend, loc='upper left', fontsize=10, title='Movement Range')
    
    def print_member_forces_comparison(self, result):
        """
        부재력 전후 비교 출력
        
        터미널에 깔끔한 표 형식으로 출력
        """
        
        print(f"\n{'='*100}")
        print("Member Forces Comparison (Before → After)")
        print(f"{'='*100}")
        
        # 초기 부재력 계산
        initial_forces, _ = self.checker.calculate_forces(
            result['initial_nodes'],
            self.connections,
            self.user_design.loads,
            self.user_design.supports
        )
        
        # 최적화 부재력 계산
        optimized_forces, _ = self.checker.calculate_forces(
            result['optimized_nodes'],
            self.connections,
            self.user_design.loads,
            self.user_design.supports
        )
        
        # 헤더
        print(f"{'Member':<12} {'Type':<10} {'Initial':<12} {'Optimized':<12} {'Change':<12} {'Change %':<12}")
        print(f"{'-'*100}")
        
        # 각 부재별 출력
        for i, (n1, n2) in enumerate(self.connections):
            member_name = f"{n1}-{n2}"
            
            f_initial = initial_forces[i]
            f_optimized = optimized_forces[i]
            change = f_optimized - f_initial
            change_pct = (change / f_initial * 100) if abs(f_initial) > 1e-6 else 0
            
            # 부재 타입 판단
            if f_optimized > 0:
                member_type = "Tie"
            else:
                member_type = "Strut"
            
            print(f"{member_name:<12} {member_type:<10} "
                  f"{abs(f_initial):>10.1f}kN {abs(f_optimized):>10.1f}kN "
                  f"{change:>+10.1f}kN {change_pct:>+10.1f}%")
        
        print(f"{'-'*100}")
        
        # 통계
        print(f"\nStatistics:")
        print(f"  Total members: {len(self.connections)}")
        
        # Tie 통계
        tie_mask = optimized_forces > 0
        if np.any(tie_mask):
            tie_initial = np.abs(initial_forces[tie_mask])
            tie_optimized = np.abs(optimized_forces[tie_mask])
            tie_reduction = np.mean((tie_initial - tie_optimized) / tie_initial * 100)
            print(f"  Ties: {np.sum(tie_mask)} members")
            print(f"    Average reduction: {tie_reduction:.1f}%")
        
        # Strut 통계
        strut_mask = optimized_forces < 0
        if np.any(strut_mask):
            strut_initial = np.abs(initial_forces[strut_mask])
            strut_optimized = np.abs(optimized_forces[strut_mask])
            strut_reduction = np.mean((strut_initial - strut_optimized) / strut_initial * 100)
            print(f"  Struts: {np.sum(strut_mask)} members")
            print(f"    Average reduction: {strut_reduction:.1f}%")
        
        # 전체 평균
        all_initial = np.abs(initial_forces)
        all_optimized = np.abs(optimized_forces)
        overall_reduction = np.mean((all_initial - all_optimized) / all_initial * 100)
        print(f"  Overall average reduction: {overall_reduction:.1f}%")
        
        print(f"{'='*100}\n")
    
    def print_member_forces_by_category(self, result):
        """
        부재를 카테고리별로 분류하여 출력
        """
        
        print(f"\n{'='*100}")
        print("Member Forces by Category")
        print(f"{'='*100}")
        
        # 부재력 계산
        initial_forces, _ = self.checker.calculate_forces(
            result['initial_nodes'],
            self.connections,
            self.user_design.loads,
            self.user_design.supports
        )
        
        optimized_forces, _ = self.checker.calculate_forces(
            result['optimized_nodes'],
            self.connections,
            self.user_design.loads,
            self.user_design.supports
        )
        
        # 카테고리 분류
        ties = []
        struts = []
        
        for i, (n1, n2) in enumerate(self.connections):
            member_name = f"{n1}-{n2}"
            f_initial = initial_forces[i]
            f_optimized = optimized_forces[i]
            change = f_optimized - f_initial
            change_pct = (change / f_initial * 100) if abs(f_initial) > 1e-6 else 0
            
            data = {
                'name': member_name,
                'initial': abs(f_initial),
                'optimized': abs(f_optimized),
                'change': change,
                'change_pct': change_pct
            }
            
            if f_optimized > 0:
                ties.append(data)
            else:
                struts.append(data)
        
        # Ties 출력
        print(f"\n[TIES - Tension Members]")
        print(f"{'-'*100}")
        print(f"{'Member':<12} {'Initial':<12} {'Optimized':<12} {'Change':<12} {'Change %':<12}")
        print(f"{'-'*100}")
        for tie in ties:
            print(f"{tie['name']:<12} {tie['initial']:>10.1f}kN {tie['optimized']:>10.1f}kN "
                  f"{tie['change']:>+10.1f}kN {tie['change_pct']:>+10.1f}%")
        
        if ties:
            avg_reduction = np.mean([t['change_pct'] for t in ties])
            print(f"{'-'*100}")
            print(f"Average reduction: {avg_reduction:.1f}%")
        
        # Struts 출력
        print(f"\n[STRUTS - Compression Members]")
        print(f"{'-'*100}")
        print(f"{'Member':<12} {'Initial':<12} {'Optimized':<12} {'Change':<12} {'Change %':<12}")
        print(f"{'-'*100}")
        for strut in struts:
            print(f"{strut['name']:<12} {strut['initial']:>10.1f}kN {strut['optimized']:>10.1f}kN "
                  f"{strut['change']:>+10.1f}kN {strut['change_pct']:>+10.1f}%")
        
        if struts:
            avg_reduction = np.mean([s['change_pct'] for s in struts])
            print(f"{'-'*100}")
            print(f"Average reduction: {avg_reduction:.1f}%")
        
        print(f"\n{'='*100}\n")
    
    def _plot_stm(self, ax, nodes, title):
        """
        STM 플롯
        
        특징:
        - 실제 부재력 계산 후 Tie/Strut 판단
        - 힘의 크기도 표시
        """
        # 보 윤곽
        ax.add_patch(plt.Rectangle(
            (0, 0), 
            self.user_design.beam_length,
            self.user_design.beam_height,
            fill=False, edgecolor='gray', linewidth=2, linestyle='--'
        ))
        
        # ═══════════════════════════════════════════════
        # 개선: 실제 힘 계산 ⭐
        # ═══════════════════════════════════════════════
        if self.use_full_checker:
            # 기존 KDS checker 사용 시
            # (힘 계산 로직 추가 필요)
            forces = self._calculate_forces_for_plot(nodes)
        else:
            # Simplified checker 사용
            forces, error = self.checker.calculate_forces(
                nodes,
                self.connections,
                self.user_design.loads,
                self.user_design.supports
            )
        
        # 부재 그리기
        for i, (n1_id, n2_id) in enumerate(self.connections):
            x1, y1 = nodes[n1_id]
            x2, y2 = nodes[n2_id]
            
            # ═══════════════════════════════════════════════
            # 개선: 힘의 부호로 정확히 판단 ⭐⭐⭐
            # ═══════════════════════════════════════════════
            if forces is not None and i < len(forces):
                force = forces[i]
                
                if force > 0:  # Tension → TIE
                    color = 'red'
                    linestyle = '--'
                    member_type = 'Tie'
                else:  # Compression → STRUT
                    color = 'blue'
                    linestyle = '-'
                    member_type = 'Strut'
                
                # 선 두께도 힘의 크기에 비례
                linewidth = 1.5 + min(abs(force) / 500, 3)
            
            else:
                # 힘 계산 실패 시 기존 방식 (fallback)
                dy = abs(y2 - y1)
                if dy < 100:
                    avg_y = (y1 + y2) / 2
                    if avg_y < self.user_design.beam_height * 0.3:
                        color, linestyle = 'red', '--'
                        member_type = 'Tie'
                    else:
                        color, linestyle = 'blue', '-'
                        member_type = 'Strut'
                else:
                    color, linestyle = 'blue', '-'
                    member_type = 'Strut'
                
                linewidth = 2.5
                force = None
            
            # 부재 그리기
            ax.plot([x1, x2], [y1, y2], 
                color=color, linestyle=linestyle, linewidth=linewidth,
                alpha=0.8)
            
            # 부재력 표시 (선택)
            if force is not None:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # 부재력 크기
                force_text = f"{abs(force):.0f}"
                
                ax.text(mid_x, mid_y, force_text,
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            edgecolor=color,
                            alpha=0.8))
        
        # 노드
        for node_id, (x, y) in nodes.items():
            ax.plot(x, y, 'ko', markersize=8, zorder=5)
            ax.text(x, y+100, node_id, 
                ha='center', fontsize=11, fontweight='bold')
        
        # 지지점
        for node_id, support_type in self.user_design.supports.items():
            x, y = nodes[node_id]
            if support_type == 'pin':
                ax.plot(x, y-80, '^', markersize=12, color='black', zorder=5)
            else:
                ax.plot(x, y-80, 'o', markersize=10, color='black', zorder=5)
        
        # 하중
        for node_id, Fx, Fy in self.user_design.loads:
            x, y = nodes[node_id]
            ax.arrow(x, y+180, 0, -130, 
                    head_width=70, head_length=40, 
                    fc='darkgreen', ec='darkgreen', zorder=5)
            ax.text(x, y+220, f"{abs(Fy):.0f}kN", 
                ha='center', fontsize=10, color='darkgreen',
                fontweight='bold')
        
        ax.set_xlabel('Length (mm)', fontsize=12)
        ax.set_ylabel('Height (mm)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 범례 개선
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', 
                label='Tie (Tension)'),
            Line2D([0], [0], color='blue', linewidth=2, linestyle='-', 
                label='Strut (Compression)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)


    def _calculate_forces_for_plot(self, nodes):
        """
        플롯용 힘 계산 헬퍼
        
        기존 checker 사용 시 힘만 추출
        """
        try:
            if self.use_full_checker:
                # STM_for_MCP.py 사용 시
                # solve_member_forces() 결과 활용
                checker = STMDesignChecker(
                    material=self.material,
                    beam_width=self.user_design.beam_width,
                    beam_height=self.user_design.beam_height,
                    beam_length=self.user_design.beam_length
                )
                
                for node_id, (x, y) in nodes.items():
                    checker.add_node(Node(node_id, x, y))
                
                for n1, n2 in self.connections:
                    mid = f"{n1}{n2}"
                    checker.add_member(Member(mid, n1, n2))
                
                for node_id, Fx, Fy in self.user_design.loads:
                    checker.add_load(node_id, Fx, Fy)
                
                for node_id, support_type in self.user_design.supports.items():
                    checker.add_support(node_id, support_type)
                
                # 힘만 계산
                member_forces = checker.solve_member_forces()
                
                # 순서 맞춰서 추출
                forces = []
                for n1, n2 in self.connections:
                    mid = f"{n1}{n2}"
                    forces.append(member_forces.get(mid, 0))
                
                return np.array(forces)
            
            else:
                # Simplified checker
                forces, error = self.checker.calculate_forces(
                    nodes,
                    self.connections,
                    self.user_design.loads,
                    self.user_design.supports
                )
                return forces
        
        except:
            return None


# ═══════════════════════════════════════════════════════════
# Output Functions (상세 출력)
# ═══════════════════════════════════════════════════════════

def print_member_design_table(optimizer, result):
    """부재 설계 표 출력 (표 10.2.3)"""
    
    print(f"\n{'='*120}")
    print("Member Design Table (예제표 10.2.3 - KDS Detailed Check)")
    print(f"{'='*120}")
    
    # 부재력 계산
    forces, _ = optimizer.checker.calculate_forces(
        result['optimized_nodes'],
        optimizer.connections,
        optimizer.user_design.loads,
        optimizer.user_design.supports
    )
    
    # 절점 분류
    classifier = NodeTypeClassifier()
    node_types = {}
    for node_id in optimizer.node_ids:
        node_types[node_id] = classifier.classify_node(
            node_id, forces, optimizer.connections, optimizer.user_design.supports
        )
    
    # 상세 계산기
    calc = DetailedMemberCalculator(
        fck=optimizer.user_design.fck,
        fy=optimizer.user_design.fy,
        beam_width=optimizer.user_design.beam_width
    )
    
    # 헤더
    print(f"{'Member':<12} {'Type':<8} {'β_s':<6} {'Force(kN)':<12} {'w_s(mm)':<10} {'w_t(mm)':<10} {'Remarks':<20}")
    print(f"{'-'*120}")
    
    member_widths = {}
    
    for i, (n1, n2) in enumerate(optimizer.connections):
        member_name = f"{n1}-{n2}"
        force = forces[i]
        force_abs = abs(force)
        
        if force < 0:  # Strut
            # β_s 결정
            beta_s = calc.determine_beta_s(node_types[n1], node_types[n2])
            
            # 스트럿 폭
            w_s = calc.calculate_strut_width(force_abs, beta_s)
            
            member_widths[i] = w_s
            
            remarks = "Bottle-shaped" if beta_s == 0.6 else "Prismatic"
            
            print(f"{member_name:<12} {'Strut':<8} {beta_s:<6.1f} {force_abs:<12.1f} {w_s:<10.1f} {'-':<10} {remarks:<20}")
        
        else:  # Tie
            # 타이 폭
            tie_info = calc.calculate_tie_width(force_abs, n1, n2, result['optimized_nodes'])
            w_t = tie_info['w_t']
            d_b = tie_info['d_b']
            n_bars = tie_info['n_bars']
            
            member_widths[i] = w_t
            
            remarks = f"D{d_b}×{n_bars}ea"
            
            print(f"{member_name:<12} {'Tie':<8} {'-':<6} {force_abs:<12.1f} {'-':<10} {w_t:<10.1f} {remarks:<20}")
    
    print(f"{'='*120}\n")
    
    return member_widths, node_types


def print_node_check_table(optimizer, result, member_widths, node_types):
    """절점 검토 표 출력 (표 10.2.4)"""
    
    print(f"\n{'='*120}")
    print("Node Design Check (예제표 10.2.4 - KDS Node Verification)")
    print(f"{'='*120}")
    
    # 부재력 계산
    forces, _ = optimizer.checker.calculate_forces(
        result['optimized_nodes'],
        optimizer.connections,
        optimizer.user_design.loads,
        optimizer.user_design.supports
    )
    
    # 절점 검토기
    checker = NodeDesignChecker(
        fck=optimizer.user_design.fck,
        beam_width=optimizer.user_design.beam_width
    )
    
    # 헤더
    print(f"{'Node':<8} {'Type':<8} {'β_n':<8} {'f_cu(MPa)':<12} {'A_req(mm²)':<14} {'A_act(mm²)':<14} {'Ratio':<10} {'Status':<8}")
    print(f"{'-'*120}")
    
    all_ok = True
    
    for node_id in optimizer.node_ids:
        node_info = node_types[node_id]
        node_type = node_info['type']
        beta_n = node_info['beta_n']
        
        # 이 절점의 최대 부재력
        max_force = 0
        for member in node_info['members']:
            max_force = max(max_force, member['force'])
        
        # f_cu
        f_cu = beta_n * optimizer.user_design.fck
        
        # 검토
        check_result = checker.check_node(node_info, max_force, member_widths)
        
        A_req = check_result['A_req']
        A_actual = check_result['A_actual']
        ratio = check_result['ratio']
        status = check_result['status']
        
        if status != 'OK':
            all_ok = False
        
        status_display = f"✓ {status}" if status == 'OK' else f"✗ {status}"
        
        print(f"{node_id:<8} {node_type:<8} {beta_n:<8.2f} {f_cu:<12.1f} {A_req:<14.0f} {A_actual:<14.0f} {ratio:<10.2f} {status_display:<8}")
    
    print(f"{'='*120}")
    
    if all_ok:
        print("✓ All nodes passed the design check!")
    else:
        print("⚠️ Warning: Some nodes failed the design check - review required")
    
    print(f"{'='*120}\n")


def print_summary_statistics(optimizer, result):
    """요약 통계 출력"""
    
    print(f"\n{'='*80}")
    print("Design Summary")
    print(f"{'='*80}")
    
    # 부재력
    initial_forces, _ = optimizer.checker.calculate_forces(
        result['initial_nodes'],
        optimizer.connections,
        optimizer.user_design.loads,
        optimizer.user_design.supports
    )
    
    optimized_forces, _ = optimizer.checker.calculate_forces(
        result['optimized_nodes'],
        optimizer.connections,
        optimizer.user_design.loads,
        optimizer.user_design.supports
    )
    
    # 통계
    print(f"\n[Material Efficiency]")
    if 'initial_objective' in result:
        print(f"  Initial objective: {result['initial_objective']:.2f}")
    print(f"  Optimized objective: {result['objective_value']:.2f}")
    if 'initial_objective' in result:
        improvement = result['initial_objective'] - result['objective_value']
        improvement_pct = improvement / result['initial_objective'] * 100
        print(f"  Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
    
    print(f"\n[Member Forces]")
    print(f"  Total members: {len(optimizer.connections)}")
    
    tie_mask = optimized_forces > 0
    strut_mask = optimized_forces < 0
    
    if np.any(tie_mask):
        tie_avg_reduction = np.mean((np.abs(initial_forces[tie_mask]) - np.abs(optimized_forces[tie_mask])) / np.abs(initial_forces[tie_mask]) * 100)
        print(f"  Ties: {np.sum(tie_mask)} members, Avg reduction: {tie_avg_reduction:.1f}%")
    
    if np.any(strut_mask):
        strut_avg_change = np.mean((np.abs(initial_forces[strut_mask]) - np.abs(optimized_forces[strut_mask])) / np.abs(initial_forces[strut_mask]) * 100)
        print(f"  Struts: {np.sum(strut_mask)} members, Avg change: {strut_avg_change:.1f}%")
    
    print(f"\n[Node Movement]")
    total_movement = 0
    max_movement = 0
    max_node = None
    
    for node_id in optimizer.node_ids:
        if node_id in optimizer.fixed_nodes:
            continue
        x0, y0 = result['initial_nodes'][node_id]
        x1, y1 = result['optimized_nodes'][node_id]
        movement = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        total_movement += movement
        if movement > max_movement:
            max_movement = movement
            max_node = node_id
    
    n_movable = len(optimizer.node_ids) - len(optimizer.fixed_nodes)
    avg_movement = total_movement / n_movable if n_movable > 0 else 0
    
    print(f"  Average movement: {avg_movement:.1f}mm")
    print(f"  Maximum movement: {max_movement:.1f}mm (Node {max_node})")
    
    print(f"\n[Constraints]")
    print(f"  Fixed nodes: {len(optimizer.fixed_nodes)} (A, F)")
    print(f"  Y-coordinates: FIXED (w_t, w_s determined)")
    print(f"  Optimized variables: {len(optimizer.left_nodes)} X-coordinates")
    print(f"  Grid size: 5mm")
    print(f"  Symmetry: ✓ Perfect")
    
    print(f"{'='*80}\n")


# ═══════════════════════════════════════════════════════════
# 사용 예시
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("Method 1: Gradient-based Position Optimization")
    print("="*60)
    
    # 사용자 설계
    user_design = UserSTMDesign()
    
    print("\nUser Design:")
    print(f"  Nodes: {len(user_design.initial_nodes)}")
    print(f"  Connections: {len(user_design.connections)} (user-provided)")
    print(f"  Loads: {len(user_design.loads)}")
    print(f"  Supports: {len(user_design.supports)}")
    
    # 최적화
    optimizer = GradientSTMOptimizer(user_design, max_change_ratio=0.15)
    result = optimizer.optimize(method='L-BFGS-B', verbose=True)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print("Position Changes:")
    print(f"{'='*60}")
    
    for node_id in user_design.initial_nodes.keys():
        x0, y0 = result['initial_nodes'][node_id]
        x1, y1 = result['optimized_nodes'][node_id]
        dx = x1 - x0
        dy = y1 - y0
        
        print(f"Node {node_id}:")
        print(f"  Initial:   ({x0:7.1f}, {y0:7.1f})")
        print(f"  Optimized: ({x1:7.1f}, {y1:7.1f})")
        print(f"  Change:    ({dx:+7.1f}, {dy:+7.1f})")
    
    # ═══════════════════════════════════════════════════════════
    # 부재력 비교 출력
    # ═══════════════════════════════════════════════════════════
    
    optimizer.print_member_forces_comparison(result)
    optimizer.print_member_forces_by_category(result)
    
    # ═══════════════════════════════════════════════════════════
    # 상세 설계 검증 (KDS 기준)
    # ═══════════════════════════════════════════════════════════
    
    # 부재 설계 표 (표 10.2.3)
    member_widths, node_types = print_member_design_table(optimizer, result)
    
    # 절점 검토 표 (표 10.2.4)
    print_node_check_table(optimizer, result, member_widths, node_types)
    
    # 요약 통계
    print_summary_statistics(optimizer, result)
    
    # ═══════════════════════════════════════════════════════════
    
    # 시각화
    fig1 = optimizer.plot_comparison(result, save_path='stm_gradient_comparison.png')
    fig2 = optimizer.plot_enhanced_comparison(result, save_path='stm_enhanced_comparison.png')
    
    print(f"\n{'='*60}")
    print("✓ Optimization complete!")
    print("✓ Basic plot saved: stm_gradient_comparison.png")
    print("✓ Enhanced plot saved: stm_enhanced_comparison.png")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════
# Output Functions (상세 출력)
# ═══════════════════════════════════════════════════════════

def print_node_check_table(optimizer, result, member_widths, node_types):
    """절점 검토 표 출력 (표 10.2.4)"""
    
    print(f"\n{'='*120}")
    print("Node Design Check (예제표 10.2.4 - KDS Node Verification)")
    print(f"{'='*120}")
    
    # 부재력 계산
    forces, _ = optimizer.checker.calculate_forces(
        result['optimized_nodes'],
        optimizer.connections,
        optimizer.user_design.loads,
        optimizer.user_design.supports
    )
    
    # 절점 검토기
    checker = NodeDesignChecker(
        fck=optimizer.user_design.fck,
        beam_width=optimizer.user_design.beam_width
    )
    
    # 헤더
    print(f"{'Node':<8} {'Type':<8} {'β_n':<8} {'f_cu(MPa)':<12} {'A_req(mm²)':<14} {'A_act(mm²)':<14} {'Ratio':<10} {'Status':<8}")
    print(f"{'-'*120}")
    
    all_ok = True
    
    for node_id in optimizer.node_ids:
        node_info = node_types[node_id]
        node_type = node_info['type']
        beta_n = node_info['beta_n']
        
        # 이 절점의 최대 부재력
        max_force = 0
        for member in node_info['members']:
            max_force = max(max_force, member['force'])
        
        # f_cu
        f_cu = beta_n * optimizer.user_design.fck
        
        # 검토
        check_result = checker.check_node(node_info, max_force, member_widths)
        
        A_req = check_result['A_req']
        A_actual = check_result['A_actual']
        ratio = check_result['ratio']
        status = check_result['status']
        
        if status != 'OK':
            all_ok = False
        
        status_display = f"✓ {status}" if status == 'OK' else f"✗ {status}"
        
        print(f"{node_id:<8} {node_type:<8} {beta_n:<8.2f} {f_cu:<12.1f} {A_req:<14.0f} {A_actual:<14.0f} {ratio:<10.2f} {status_display:<8}")
    
    print(f"{'='*120}")
    
    if all_ok:
        print("✓ All nodes passed the design check!")
    else:
        print("⚠️ Warning: Some nodes failed the design check - review required")
    
    print(f"{'='*120}\n")


def print_summary_statistics(optimizer, result):
    """요약 통계 출력"""
    
    print(f"\n{'='*80}")
    print("Design Summary")
    print(f"{'='*80}")
    
    # 부재력
    initial_forces, _ = optimizer.checker.calculate_forces(
        result['initial_nodes'],
        optimizer.connections,
        optimizer.user_design.loads,
        optimizer.user_design.supports
    )
    
    optimized_forces, _ = optimizer.checker.calculate_forces(
        result['optimized_nodes'],
        optimizer.connections,
        optimizer.user_design.loads,
        optimizer.user_design.supports
    )
    
    # 통계
    print(f"\n[Material Efficiency]")
    if 'initial_objective' in result:
        print(f"  Initial objective: {result['initial_objective']:.2f}")
    print(f"  Optimized objective: {result['objective_value']:.2f}")
    if 'initial_objective' in result:
        improvement = result['initial_objective'] - result['objective_value']
        improvement_pct = improvement / result['initial_objective'] * 100
        print(f"  Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
    
    print(f"\n[Member Forces]")
    print(f"  Total members: {len(optimizer.connections)}")
    
    tie_mask = optimized_forces > 0
    strut_mask = optimized_forces < 0
    
    if np.any(tie_mask):
        tie_avg_reduction = np.mean((np.abs(initial_forces[tie_mask]) - np.abs(optimized_forces[tie_mask])) / np.abs(initial_forces[tie_mask]) * 100)
        print(f"  Ties: {np.sum(tie_mask)} members, Avg reduction: {tie_avg_reduction:.1f}%")
    
    if np.any(strut_mask):
        strut_avg_change = np.mean((np.abs(initial_forces[strut_mask]) - np.abs(optimized_forces[strut_mask])) / np.abs(initial_forces[strut_mask]) * 100)
        print(f"  Struts: {np.sum(strut_mask)} members, Avg change: {strut_avg_change:.1f}%")
    
    print(f"\n[Node Movement]")
    total_movement = 0
    max_movement = 0
    max_node = None
    
    for node_id in optimizer.node_ids:
        if node_id in optimizer.fixed_nodes:
            continue
        x0, y0 = result['initial_nodes'][node_id]
        x1, y1 = result['optimized_nodes'][node_id]
        movement = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        total_movement += movement
        if movement > max_movement:
            max_movement = movement
            max_node = node_id
    
    n_movable = len(optimizer.node_ids) - len(optimizer.fixed_nodes)
    avg_movement = total_movement / n_movable if n_movable > 0 else 0
    
    print(f"  Average movement: {avg_movement:.1f}mm")
    print(f"  Maximum movement: {max_movement:.1f}mm (Node {max_node})")
    
    print(f"\n[Constraints]")
    print(f"  Fixed nodes: {len(optimizer.fixed_nodes)} (A, F)")
    print(f"  Y-coordinates: FIXED (w_t, w_s determined)")
    print(f"  Optimized variables: {len(optimizer.left_nodes)} X-coordinates")
    print(f"  Grid size: 5mm")
    print(f"  Symmetry: ✓ Perfect")
    
    print(f"{'='*80}\n")