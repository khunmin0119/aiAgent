"""
사용자 주도형 STM 최적화(gradient-based optimization)
================================
사용자가 초기 노드 위치 제공 → Agent가 위치와 부재 폭 최적화

Approach: Gradient-based optimization (scipy.optimize)
- 연속 변수 최적화에 최적
- 제약 조건 처리 가능
- 미분 가능한 목적함수
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════
# 사용자 입력 정의
# ═══════════════════════════════════════════════════════════

@dataclass
class UserDesign:
    """사용자가 제공하는 초기 설계"""
    
    # 기본 정보
    span: float = 6.0  # m
    height: float = 2.0  # m
    
    # 하중 (x, y, magnitude)
    loads: List[Tuple[float, float, float]] = None
    
    # 지지점 (x, y)
    supports: List[Tuple[float, float]] = None
    
    # 후보 절점 (사용자가 배치)
    candidate_nodes: List[Tuple[float, float]] = None
    
    # 재료 특성
    fck: float = 27.0  # MPa
    fy: float = 400.0  # MPa
    cover: float = 0.05  # m
    
    def __post_init__(self):
        if self.loads is None:
            self.loads = [
                (2.0, 2.0, 50.0),
                (4.0, 2.0, 50.0)
            ]
        
        if self.supports is None:
            self.supports = [
                (0.0, 0.0),
                (self.span, 0.0)
            ]
        
        if self.candidate_nodes is None:
            self.candidate_nodes = [
                (1.5, 1.0),
                (3.0, 0.8),
                (4.5, 1.0)
            ]
    
    def get_all_nodes(self):
        """모든 노드 리스트"""
        return (
            self.supports +
            [(x, y) for x, y, _ in self.loads] +
            self.candidate_nodes
        )


# ═══════════════════════════════════════════════════════════
# STM 최적화기
# ═══════════════════════════════════════════════════════════

class STMOptimizer:
    """
    사용자 초기 설계를 최적화
    
    최적화 변수:
    - candidate_nodes 위치 (x, y)
    - 부재 폭 (optional)
    
    고정:
    - 하중 위치
    - 지지점 위치
    """
    
    def __init__(self, user_design: UserDesign, 
                 max_position_change_ratio: float = 0.15):
        """
        Args:
            user_design: 사용자 초기 설계
            max_position_change_ratio: 최대 위치 변경 (span/height의 비율)
        """
        self.design = user_design
        self.max_change_ratio = max_position_change_ratio
        
        # 초기 노드
        self.initial_candidates = np.array(user_design.candidate_nodes)
        self.n_candidates = len(self.initial_candidates)
        
        # 고정 노드
        self.fixed_nodes = (
            user_design.supports + 
            [(x, y) for x, y, _ in user_design.loads]
        )
        self.n_fixed = len(self.fixed_nodes)
        
        # 엣지 (사용자가 지정 또는 자동 생성)
        self.edges = self._generate_initial_edges()
        
        # 재료 강도
        self.f_cd = 0.85 * user_design.fck / 1.5
        self.f_yd = user_design.fy / 1.15
    
    def _generate_initial_edges(self):
        """
        초기 엣지 생성
        
        전략: 간단한 트러스 구조
        - 각 하중을 양쪽 지지점에 연결
        - 후보 노드를 인접 노드에 연결
        """
        edges = []
        n_supports = len(self.design.supports)
        n_loads = len(self.design.loads)
        
        # 하중 → 지지점
        for i in range(n_loads):
            load_idx = n_supports + i
            edges.append((0, load_idx))  # Left support
            edges.append((1, load_idx))  # Right support
        
        # 후보 노드 → 지지점 & 하중
        for i in range(self.n_candidates):
            cand_idx = n_supports + n_loads + i
            
            # 지지점에 연결
            edges.append((0, cand_idx))
            edges.append((1, cand_idx))
            
            # 가장 가까운 하중에 연결
            if n_loads > 0:
                cand_x = self.initial_candidates[i][0]
                load_x = [self.design.loads[j][0] for j in range(n_loads)]
                closest = np.argmin([abs(cand_x - lx) for lx in load_x])
                edges.append((n_supports + closest, cand_idx))
        
        return edges
    
    def optimize(self, method='gradient'):
        """
        최적화 실행
        
        Args:
            method: 'gradient' (빠름) 또는 'evolutionary' (전역 최적)
        
        Returns:
            result: 최적화 결과
        """
        print(f"\n{'='*60}")
        print(f"STM 최적화 시작")
        print(f"{'='*60}")
        print(f"초기 candidate 노드 수: {self.n_candidates}")
        print(f"초기 엣지 수: {len(self.edges)}")
        print(f"최적화 방법: {method}")
        print(f"최대 위치 변경: ±{self.max_change_ratio*100:.0f}%")
        
        # 초기값
        x0 = self.initial_candidates.flatten()
        
        # 경계 조건
        bounds = self._get_bounds()
        
        # 초기 목적함수 값
        f0 = self.objective(x0)
        print(f"\n초기 목적함수 값: {f0:.2f}")
        
        # 최적화
        if method == 'gradient':
            result = minimize(
                self.objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'disp': True}
            )
        else:  # evolutionary
            result = differential_evolution(
                self.objective,
                bounds,
                maxiter=50,
                disp=True
            )
        
        print(f"\n{'='*60}")
        print(f"최적화 완료")
        print(f"{'='*60}")
        print(f"최종 목적함수 값: {result.fun:.2f}")
        print(f"개선: {f0 - result.fun:.2f}")
        
        # 결과 파싱
        optimized_candidates = result.x.reshape(-1, 2)
        
        return {
            'optimized_candidates': optimized_candidates,
            'initial_candidates': self.initial_candidates,
            'objective_value': result.fun,
            'success': result.success,
            'message': result.message,
            'edges': self.edges
        }
    
    def _get_bounds(self):
        """위치 변경 범위 계산"""
        bounds = []
        
        max_dx = self.max_change_ratio * self.design.span
        max_dy = self.max_change_ratio * self.design.height
        
        for i in range(self.n_candidates):
            x0, y0 = self.initial_candidates[i]
            
            # X 범위
            x_min = max(0.0, x0 - max_dx)
            x_max = min(self.design.span, x0 + max_dx)
            bounds.append((x_min, x_max))
            
            # Y 범위
            y_min = max(0.0, y0 - max_dy)
            y_max = min(self.design.height, y0 + max_dy)
            bounds.append((y_min, y_max))
        
        return bounds
    
    def objective(self, x):
        """
        목적함수: 최소화할 값
        
        Args:
            x: [x1, y1, x2, y2, ...] 후보 노드 좌표
        
        Returns:
            cost: 낮을수록 좋음
        """
        # 노드 위치 복원
        candidate_positions = x.reshape(-1, 2)
        all_nodes = self.fixed_nodes + [tuple(pos) for pos in candidate_positions]
        
        # 힘 계산
        forces = self._calculate_forces(all_nodes)
        
        if forces is None:
            return 1e10  # 평형 실패 → 큰 페널티
        
        # 목적함수 구성
        cost = 0.0
        
        # ───────────────────────────────────────────────────
        # Term 1: 부재 폭 최소화 (재료 사용량)
        # ───────────────────────────────────────────────────
        total_width = 0.0
        max_violation = 0.0
        
        for (n1, n2), force in zip(self.edges, forces):
            x1, y1 = all_nodes[n1]
            x2, y2 = all_nodes[n2]
            L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if L < 1e-6:
                cost += 1e6
                continue
            
            # 필요 폭 계산
            if force > 0:  # Tension (tie)
                # A_s * f_yd >= F
                A_s = abs(force) / self.f_yd
                w_required = A_s / (0.01 * L)  # 가정: 철근비 1%
            else:  # Compression (strut)
                # 0.85 * f_cd * w * L * ν >= F
                # ν = 0.6 (strut 효율계수)
                nu = 0.6
                w_required = abs(force) / (0.85 * self.f_cd * L * nu)
            
            # 폭 제약: 0.2m ~ 1.0m
            w_required = np.clip(w_required, 0.2, 1.0)
            
            total_width += w_required
            
            # 강도 위반 체크
            if force > 0:
                capacity = w_required * 0.01 * L * self.f_yd
            else:
                capacity = 0.85 * self.f_cd * w_required * L * 0.6
            
            violation = max(0, abs(force) - capacity)
            max_violation = max(max_violation, violation)
        
        cost += total_width * 10.0  # 재료 사용량
        cost += max_violation * 100.0  # 강도 위반 큰 페널티
        
        # ───────────────────────────────────────────────────
        # Term 2: 위치 변경 최소화 (사용자 의도 존중)
        # ───────────────────────────────────────────────────
        position_change = np.linalg.norm(
            candidate_positions - self.initial_candidates
        )
        cost += position_change * 1.0
        
        # ───────────────────────────────────────────────────
        # Term 3: 기하학적 제약 (a/d ≤ 2 등)
        # ───────────────────────────────────────────────────
        d = self.design.height - self.design.cover
        
        for load_x, load_y, _ in self.design.loads:
            a = min(load_x, self.design.span - load_x)
            if a / d > 2.0:
                cost += (a/d - 2.0) * 50.0
        
        return cost
    
    def _calculate_forces(self, nodes):
        """평형 방정식 해"""
        try:
            n_edges = len(self.edges)
            n_nodes = len(nodes)
            
            # 평형 행렬
            A = np.zeros((2 * n_nodes, n_edges + 2))
            
            for idx, (n1, n2) in enumerate(self.edges):
                x1, y1 = nodes[n1]
                x2, y2 = nodes[n2]
                
                dx = x2 - x1
                dy = y2 - y1
                L = np.sqrt(dx**2 + dy**2)
                
                if L < 1e-6:
                    return None
                
                cos = dx / L
                sin = dy / L
                
                A[2*n1, idx] = -cos
                A[2*n1+1, idx] = -sin
                A[2*n2, idx] = cos
                A[2*n2+1, idx] = sin
            
            # 지지 조건
            A[0, -2] = 1.0   # Left support X
            A[1, -1] = 1.0   # Left support Y
            A[2, -2] = 0.0   # Right support X (roller)
            A[3, -1] = 1.0   # Right support Y
            
            # 하중 벡터
            b = np.zeros(2 * n_nodes)
            for i, (x, y, P) in enumerate(self.design.loads):
                node_idx = len(self.design.supports) + i
                b[2*node_idx + 1] = -P
            
            # 해
            solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # 평형 체크
            error = np.linalg.norm(A @ solution - b)
            if error > 10.0:  # 허용 오차
                return None
            
            return solution[:n_edges]
        
        except:
            return None
    
    def visualize_result(self, result):
        """최적화 결과 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 초기 설계
        self._plot_design(ax1, self.initial_candidates, "초기 설계")
        
        # 최적화된 설계
        self._plot_design(ax2, result['optimized_candidates'], "최적화된 설계")
        
        plt.tight_layout()
        return fig
    
    def _plot_design(self, ax, candidate_positions, title):
        """STM 플롯"""
        all_nodes = self.fixed_nodes + [tuple(pos) for pos in candidate_positions]
        
        # 노드
        supports = np.array(self.design.supports)
        loads = np.array([(x, y) for x, y, _ in self.design.loads])
        candidates = np.array(candidate_positions)
        
        ax.scatter(supports[:, 0], supports[:, 1], 
                  s=200, c='blue', marker='^', label='지지점', zorder=5)
        ax.scatter(loads[:, 0], loads[:, 1], 
                  s=200, c='red', marker='v', label='하중', zorder=5)
        ax.scatter(candidates[:, 0], candidates[:, 1], 
                  s=100, c='green', marker='o', label='후보 절점', zorder=5)
        
        # 엣지
        forces = self._calculate_forces(all_nodes)
        if forces is not None:
            for (n1, n2), force in zip(self.edges, forces):
                x1, y1 = all_nodes[n1]
                x2, y2 = all_nodes[n2]
                
                color = 'blue' if force > 0 else 'red'
                width = min(3, abs(force) / 20.0)
                
                ax.plot([x1, x2], [y1, y2], 
                       c=color, linewidth=width, alpha=0.7)
        
        ax.set_xlim(-0.5, self.design.span + 0.5)
        ax.set_ylim(-0.5, self.design.height + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')


# ═══════════════════════════════════════════════════════════
# 사용 예시
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("사용자 주도형 STM 최적화")
    print("="*60)
    
    # ───────────────────────────────────────────────────
    # 1. 사용자가 초기 설계 제공
    # ───────────────────────────────────────────────────
    user_design = UserDesign(
        span=6.0,
        height=2.0,
        loads=[
            (2.0, 2.0, 50.0),  # 하중 1
            (4.0, 2.0, 50.0)   # 하중 2
        ],
        candidate_nodes=[
            (1.5, 1.0),  # 사용자가 배치한 후보 절점
            (3.0, 0.8),
            (4.5, 1.0)
        ]
    )
    
    print("\n사용자 초기 설계:")
    print(f"  Span: {user_design.span}m")
    print(f"  Height: {user_design.height}m")
    print(f"  하중: {len(user_design.loads)}개")
    print(f"  후보 절점: {len(user_design.candidate_nodes)}개")
    
    # ───────────────────────────────────────────────────
    # 2. 최적화 실행
    # ───────────────────────────────────────────────────
    optimizer = STMOptimizer(user_design, max_position_change_ratio=0.15)
    result = optimizer.optimize(method='gradient')
    
    # ───────────────────────────────────────────────────
    # 3. 결과 출력
    # ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("최적화 결과")
    print(f"{'='*60}")
    
    print("\n위치 변화:")
    for i in range(len(result['optimized_candidates'])):
        initial = result['initial_candidates'][i]
        optimized = result['optimized_candidates'][i]
        
        dx = optimized[0] - initial[0]
        dy = optimized[1] - initial[1]
        
        print(f"  노드 {i+1}:")
        print(f"    초기: ({initial[0]:.3f}, {initial[1]:.3f})")
        print(f"    최적: ({optimized[0]:.3f}, {optimized[1]:.3f})")
        print(f"    변화: (Δx={dx:+.3f}, Δy={dy:+.3f})")
    
    # ───────────────────────────────────────────────────
    # 4. 시각화
    # ───────────────────────────────────────────────────
    fig = optimizer.visualize_result(result)
    plt.savefig('/home/claude/stm_optimization_result.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 결과 저장: stm_optimization_result.png")