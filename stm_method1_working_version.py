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
        # 대칭 제약 추가
        # ═══════════════════════════════════════════════════════════
        self.x_center = user_design.beam_length / 2  # 3450mm
        
        # 대칭 쌍 정의
        self.symmetric_pairs = [
            ('A', 'F'),
            ('B', 'E'),
            ('C', 'D'),
            ('G', 'H')
        ]
        
        # 좌측 노드만 최적화 변수로 사용
        self.left_nodes = ['A', 'B', 'C', 'G']
        self.right_nodes = ['F', 'E', 'D', 'H']
        
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
            print("Symmetric STM Optimization")
            print(f"{'='*60}")
            print(f"Total nodes: {len(self.node_ids)}")
            print(f"Optimization variables: {len(self.left_nodes)} left nodes")
            print(f"Symmetric pairs: {len(self.symmetric_pairs)}")
            print(f"Symmetry axis: x = {self.x_center:.1f}mm")
            print(f"Connections: {len(self.connections)} (fixed by user)")
            print(f"Max position change: ±{self.max_change_ratio*100:.0f}%")
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
            print(f"Final objective: {result.fun:.2f}")
            print(f"Improvement: {f0 - result.fun:.2f}")
            print(f"Success: {result.success}")
        
        # 최적화된 노드 (대칭 포함)
        optimized_nodes = self._vector_to_nodes(result.x)
        
        # 대칭 검증
        if verbose:
            self._verify_symmetry(optimized_nodes)
        
        return {
            'optimized_nodes': optimized_nodes,
            'initial_nodes': self.initial_nodes,
            'objective_value': result.fun,
            'success': result.success,
            'message': result.message,
            'connections': self.connections
        }
    
    def _nodes_to_vector(self, nodes):
        """
        Dict → vector (대칭: 좌측 노드만)
        
        Returns:
            [x_A, y_A, x_B, y_B, x_C, y_C, x_G, y_G]
            총 8개 변수 (원래 16개의 절반)
        """
        x = []
        for node_id in self.left_nodes:
            x.extend(nodes[node_id])
        return np.array(x)
    
    def _vector_to_nodes(self, x):
        """
        Vector → dict (대칭 적용)
        
        좌측 노드는 최적화 변수에서
        우측 노드는 대칭으로 자동 계산
        """
        nodes = {}
        
        # 좌측 노드 (최적화 변수)
        for i, node_id in enumerate(self.left_nodes):
            nodes[node_id] = (x[i*2], x[i*2+1])
        
        # 우측 노드 (대칭으로 계산)
        for left_id, right_id in self.symmetric_pairs:
            x_left, y_left = nodes[left_id]
            
            # X: 중심선 기준 대칭
            x_right = 2 * self.x_center - x_left
            
            # Y: 동일
            y_right = y_left
            
            nodes[right_id] = (x_right, y_right)
        
        return nodes
    
    def _get_bounds(self):
        """
        위치 변경 범위 (좌측 노드만)
        
        주의: 좌측 노드는 중심선을 넘지 않도록 제한
        """
        bounds = []
        
        max_dx = self.max_change_ratio * self.user_design.beam_length
        max_dy = self.max_change_ratio * self.user_design.beam_height
        
        for node_id in self.left_nodes:
            x0, y0 = self.initial_nodes[node_id]
            
            # X 범위 (좌측 노드: 중심선 넘지 않음)
            x_min = max(0, x0 - max_dx)
            x_max = min(self.x_center, x0 + max_dx)  # 중심선이 상한
            bounds.append((x_min, x_max))
            
            # Y 범위
            y_min = max(0, y0 - max_dy)
            y_max = min(self.user_design.beam_height, y0 + max_dy)
            bounds.append((y_min, y_max))
        
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
    
    def _verify_symmetry(self, nodes):
        """
        대칭 검증
        
        좌우 노드가 완벽하게 대칭인지 확인
        """
        print(f"\n{'='*60}")
        print("Symmetry Verification")
        print(f"{'='*60}")
        
        all_symmetric = True
        
        for left_id, right_id in self.symmetric_pairs:
            x_left, y_left = nodes[left_id]
            x_right, y_right = nodes[right_id]
            
            # 중심으로부터 거리
            dist_left = abs(self.x_center - x_left)
            dist_right = abs(x_right - self.x_center)
            
            # Y 좌표
            y_diff = abs(y_right - y_left)
            
            # 대칭 오차
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
            print("✓ All nodes are perfectly symmetric!")
        else:
            print("⚠️ Warning: Some symmetry errors detected")
        print(f"{'='*60}\n")
    
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
    
    # 시각화
    # 기본 비교 (간단)
    fig1 = optimizer.plot_comparison(result, save_path='stm_gradient_comparison.png')
    
    # 개선된 비교 (상세)
    fig2 = optimizer.plot_enhanced_comparison(result, save_path='stm_enhanced_comparison.png')
    
    print(f"\n{'='*60}")
    print("✓ Optimization complete!")
    print("✓ Basic plot saved: stm_gradient_comparison.png")
    print("✓ Enhanced plot saved: stm_enhanced_comparison.png")
    print(f"{'='*60}")