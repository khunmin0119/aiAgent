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
    교재 예제 10.2 (STM-2) - 4절점 모델
    """
    
    def __init__(self, material: Material, beam_width: float, beam_height: float):
        self.material = material
        self.b = beam_width
        self.h = beam_height
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
        self.nodes[node.id] = node
        
    def add_member(self, member: Member):
        self.members[member.id] = member
        
    def add_load(self, node_id: str, Fx: float, Fy: float):
        self.loads.append((node_id, Fx, Fy))
        
    def add_support(self, node_id: str, support_type: str):
        self.supports[node_id] = support_type
    
    def get_member_length(self, member: Member) -> float:
        n1 = self.nodes[member.node_start]
        n2 = self.nodes[member.node_end]
        return np.sqrt((n2.x - n1.x)**2 + (n2.y - n1.y)**2)
    
    def get_member_angle(self, member: Member, in_degrees: bool = False) -> float:
        n1 = self.nodes[member.node_start]
        n2 = self.nodes[member.node_end]
        
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        
        angle_rad = np.arctan2(abs(dy), abs(dx))
        
        if in_degrees:
            return np.degrees(angle_rad)
        return angle_rad
    
    def solve_member_forces(self) -> Dict[str, float]:
        """대칭 구조를 이용한 부재력 계산"""
        print("\n[Step 1] Calculating member forces (symmetric structure)")
        
        # 대칭 구조이므로 반력 계산
        total_load = sum(abs(Fy) for _, _, Fy in self.loads)
        R_A = total_load / 2
        R_D = total_load / 2
        
        print(f"       Total load: {total_load} kN")
        print(f"       Reaction A: {R_A} kN")
        print(f"       Reaction D: {R_D} kN")
        
        # 부재 각도 계산
        member_AB = self.members['AB']
        angle_AB = self.get_member_angle(member_AB, in_degrees=True)
        angle_AB_rad = self.get_member_angle(member_AB)
        
        print(f"       Angle AB: {angle_AB:.1f}°")
        
        # 부재력 계산 (교재 방법)
        # AB 사재: F_AB = R_A / sin(θ)
        F_AB = -R_A / np.sin(angle_AB_rad)  # 압축 (음수)
        
        # BC 상단 수평재: F_BC = F_AB * cos(θ)  
        F_BC = F_AB * np.cos(angle_AB_rad)  # 압축 (음수)
        
        # AD 하단 수평재: F_AD = F_BC (대칭)
        F_AD = -F_BC  # 인장 (양수)
        
        # CD 사재: 대칭이므로 F_CD = F_AB
        F_CD = F_AB
        
        # 부재력 저장
        self.members['AB'].force = F_AB
        self.members['BC'].force = F_BC
        self.members['CD'].force = F_CD
        self.members['AD'].force = F_AD
        
        # 부재 타입 및 각도 저장
        self.members['AB'].member_type = 'strut'
        self.members['BC'].member_type = 'strut'
        self.members['CD'].member_type = 'strut'
        self.members['AD'].member_type = 'tie'
        
        for mid in ['AB', 'BC', 'CD', 'AD']:
            self.members[mid].angle = self.get_member_angle(self.members[mid], in_degrees=True)
        
        # 반력 저장
        self.reactions['A'] = {'Rx': 0, 'Ry': R_A}
        self.reactions['D'] = {'Rx': 0, 'Ry': R_D}
        
        print(f"\n       Member forces:")
        print(f"       AB = {F_AB:.1f} kN (압축)")
        print(f"       BC = {F_BC:.1f} kN (압축)")
        print(f"       CD = {F_CD:.1f} kN (압축)")
        print(f"       AD = {F_AD:.1f} kN (인장)")
        
        return {
            'AB': F_AB,
            'BC': F_BC,
            'CD': F_CD,
            'AD': F_AD
        }
    
    def infer_node_types(self):
        """절점 타입 결정"""
        # 교재 표 10.2.6 기준
        # A, D: CCT (2 struts + 1 tie)
        # B, C: CCC (3 struts)
        
        self.nodes['A'].node_type = 'CCT'
        self.nodes['B'].node_type = 'CCC'
        self.nodes['C'].node_type = 'CCC'
        self.nodes['D'].node_type = 'CCT'
    
    def get_beta_s(self, member_id: str) -> float:
        """교재 표 10.2.7 기준"""
        if member_id in ['AB', 'CD']:
            return 0.75  # Bottle-shaped diagonal struts
        elif member_id == 'BC':
            return 1.0   # Prismatic horizontal strut
        else:
            return 1.0
    
    def get_beta_n(self, node_type: str) -> float:
        """KDS Table 10.2"""
        beta_n_table = {
            'CCC': 1.0,
            'CCT': 0.80,
            'CTT': 0.60,
            'TTT': 0.60
        }
        return beta_n_table.get(node_type, 0.60)
    
    def calculate_strut_width(self, member: Member, w_t: float = 250, l_b: float = 450) -> float:
        """
        스트럿 폭 계산: w_s = l_b * sin(θ) + w_t * cos(θ)
        교재: w_s,AB = 450 * sin(40.8°) + 250 * cos(40.8°) = 483.3 mm
              w_s,BC = 300 mm (수평재)
        """
        angle_rad = self.get_member_angle(member)
        angle_deg = np.degrees(angle_rad)
        
        # BC는 수평재
        if member.id == 'BC':
            return 300  # mm
        
        # AB, CD는 사재
        w_s = l_b * np.sin(angle_rad) + w_t * np.cos(angle_rad)
        return w_s
    
    def check_strut_strength(self, member: Member) -> Tuple[bool, Dict]:
        """스트럿 강도 검토"""
        phi = 0.75
        beta_s = self.get_beta_s(member.id)
        f_ce = 0.85 * beta_s * self.material.fck
        
        # 스트럿 폭
        if member.id == 'BC':
            w_s = 300  # mm
        else:
            w_s = self.calculate_strut_width(member)
        
        # 요구 폭
        w_req = (abs(member.force) * 1000) / (phi * f_ce * self.b)
        
        # 강도
        F_ns = f_ce * w_s * self.b / 1000  # kN
        phi_Fns = phi * F_ns
        F_u = abs(member.force)
        
        is_ok = phi_Fns >= F_u
        
        result = {
            'member_id': member.id,
            'beta_s': beta_s,
            'f_ce': f_ce,
            'w_req': w_req,
            'w_s': w_s,
            'F_ns': F_ns,
            'phi_Fns': phi_Fns,
            'F_u': F_u,
            'ratio': F_u / phi_Fns if phi_Fns > 0 else float('inf'),
            'is_ok': is_ok
        }
        
        return is_ok, result
    
    def check_tie_strength(self, member: Member) -> Tuple[bool, Dict]:
        """타이 강도 검토"""
        phi = 0.85
        F_u = abs(member.force)
        
        # 요구 철근량
        A_s_req = (F_u * 1000) / (phi * self.material.fy)
        
        # 제공 철근량 (10% 여유)
        A_s = A_s_req * 1.1
        
        # 강도
        F_nt = A_s * self.material.fy / 1000
        phi_Fnt = phi * F_nt
        
        is_ok = phi_Fnt >= F_u
        
        result = {
            'member_id': member.id,
            'A_s_req': A_s_req,
            'A_s': A_s,
            'phi_Fnt': phi_Fnt,
            'F_u': F_u,
            'ratio': F_u / phi_Fnt if phi_Fnt > 0 else float('inf'),
            'is_ok': is_ok
        }
        
        return is_ok, result
    
    def check_node_strength_detailed(self, node: Node, l_b: float = 450) -> Dict:
        """
        절점 상세 검토 (교재 표 10.2.8 형식)
        
        각 절점에서 연결된 모든 요소(반력, 하중, 부재)에 대해
        요구 폭과 실제 폭을 비교
        """
        phi = 0.75
        beta_n = self.get_beta_n(node.node_type)
        f_cn = 0.85 * beta_n * self.material.fck
        
        node_checks = []
        
        # 1. 반력 검토 (지지점인 경우)
        if node.id in self.reactions:
            reaction = self.reactions[node.id]
            R_total = 0
            if 'Ry' in reaction:
                R_total = abs(reaction['Ry'])
            
            if R_total > 0:
                # 요구 폭
                w_req = (R_total * 1000) / (phi * f_cn * self.b)
                # 실제 폭 (지압판 크기)
                w_actual = l_b
                
                is_ok = w_actual >= w_req
                
                node_checks.append({
                    'element_type': 'R',
                    'element_id': 'R',
                    'force': R_total,
                    'w_req': w_req,
                    'w_actual': w_actual,
                    'is_ok': is_ok,
                    'note': '만족' if is_ok else '불만족'
                })
        
        # 2. 연결된 부재 검토
        for member in self.members.values():
            if member.node_start == node.id or member.node_end == node.id:
                if member.force is None:
                    continue
                
                force = abs(member.force)
                
                if member.member_type == 'strut':
                    # 스트럿의 경우
                    # 요구 폭 (절점에서)
                    w_req = (force * 1000) / (phi * f_cn * self.b)
                    
                    # 실제 폭 (스트럿 폭 계산)
                    if member.id == 'BC':
                        w_actual = 300  # mm
                    else:
                        w_actual = self.calculate_strut_width(member, w_t=250, l_b=l_b)
                    
                    is_ok = w_actual >= w_req
                    
                    node_checks.append({
                        'element_type': 'strut',
                        'element_id': member.id,
                        'force': force,
                        'w_req': w_req,
                        'w_actual': w_actual,
                        'is_ok': is_ok,
                        'note': '만족' if is_ok else '불만족'
                    })
                    
                elif member.member_type == 'tie':
                    # 타이의 경우
                    # 요구 폭
                    w_req = (force * 1000) / (phi * f_cn * self.b)
                    
                    # 타이 실제 폭 (교재: 수평 타이 = 250 mm)
                    w_actual = 250  # mm
                    
                    is_ok = w_actual >= w_req
                    
                    node_checks.append({
                        'element_type': 'tie',
                        'element_id': member.id,
                        'force': force,
                        'w_req': w_req,
                        'w_actual': w_actual,
                        'is_ok': is_ok,
                        'note': '불만족 (철근보강)' if not is_ok else '만족'
                    })
        
        # 3. 하중 검토
        for load_node_id, Fx, Fy in self.loads:
            if load_node_id == node.id:
                P_total = abs(Fy)
                
                if P_total > 0:
                    # 요구 폭
                    w_req = (P_total * 1000) / (phi * f_cn * self.b)
                    # 실제 폭 (지압판 크기, 교재에서는 250mm로 가정)
                    w_actual = 250  # mm (교재 표 10.2.8 참고)
                    
                    is_ok = w_actual >= w_req
                    
                    node_checks.append({
                        'element_type': 'P',
                        'element_id': '하중',
                        'force': P_total,
                        'w_req': w_req,
                        'w_actual': w_actual,
                        'is_ok': is_ok,
                        'note': '만족' if is_ok else '불만족'
                    })
        
        # 전체 판정
        all_ok = all(check['is_ok'] for check in node_checks)
        
        # 보완 사항 확인
        supplement = []
        for check in node_checks:
            if not check['is_ok']:
                if check['element_type'] == 'tie':
                    supplement.append("철근보강")
                else:
                    supplement.append(f"{check['element_id']} 폭 증대 필요")
        
        result = {
            'node_id': node.id,
            'node_type': node.node_type,
            'beta_n': beta_n,
            'f_cn': f_cn,
            'checks': node_checks,
            'all_ok': all_ok,
            'notes': supplement if supplement else []
        }
        
        return result
    
    def check_node_strength(self, node: Node) -> Tuple[bool, Dict]:
        """절점 강도 검토"""
        phi = 0.75
        beta_n = self.get_beta_n(node.node_type)
        f_cn = 0.85 * beta_n * self.material.fck
        
        # 절점 면적
        A_n = self.b * self.b
        
        # 강도
        F_nn = f_cn * A_n / 1000
        phi_Fnn = phi * F_nn
        
        # 최대 압축력
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
            'phi_Fnn': phi_Fnn,
            'F_u': F_u,
            'ratio': F_u / phi_Fnn if phi_Fnn > 0 else float('inf'),
            'is_ok': is_ok
        }
        
        return is_ok, result
    
    def verify_stm(self) -> Dict:
        """STM 검증"""
        print("\n" + "="*80)
        print("교재 예제 10.2 (STM-2) - 4절점 모델 검증")
        print("="*80)
        
        # 부재력 계산
        member_forces = self.solve_member_forces()
        
        # 절점 타입 결정
        print("\n[Step 2] Determining node types")
        self.infer_node_types()
        
        # 강도 검증
        print("\n[Step 3] Verifying strengths")
        
        # 스트럿 검증
        for mid in ['AB', 'BC', 'CD']:
            is_ok, result = self.check_strut_strength(self.members[mid])
            self.results['struts'][mid] = result
            if not is_ok:
                self.results['overall'] = False
        
        # 타이 검증
        is_ok, result = self.check_tie_strength(self.members['AD'])
        self.results['ties']['AD'] = result
        if not is_ok:
            self.results['overall'] = False
        
        # 절점 검증
        for nid in ['A', 'B', 'C', 'D']:
            is_ok, result = self.check_node_strength(self.nodes[nid])
            self.results['nodes'][nid] = result
            if not is_ok:
                self.results['overall'] = False
        
        # 상세 절점 검증 (표 10.2.8 형식)
        print("\n[Step 4] Detailed node verification (Table 10.2.8)")
        self.results['nodes_detailed'] = {}
        for nid in ['A', 'B', 'C', 'D']:
            detailed_result = self.check_node_strength_detailed(self.nodes[nid], l_b=450)
            self.results['nodes_detailed'][nid] = detailed_result
        
        if self.results['overall']:
            self.results['messages'].append("[PASS] All checks satisfied")
        else:
            self.results['messages'].append("[FAIL] Some checks failed")
        
        return self.results
    
    def export_to_excel(self, filename: str = None):
        """검증 결과를 엑셀 파일로 출력"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"STM_Verification_4Node_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: 프로젝트 정보
            info_data = {
                '항목': ['프로젝트명', '날짜', '보 폭 (mm)', '보 높이 (mm)', 
                        'fck (MPa)', 'fy (MPa)', '전체 판정'],
                '값': ['교재 예제 10.2 (STM-2)', 
                       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       self.b, self.h, self.material.fck, self.material.fy,
                       '만족' if self.results['overall'] else '불만족']
            }
            df_info = pd.DataFrame(info_data)
            df_info.to_excel(writer, sheet_name='프로젝트 정보', index=False)
            
            # Sheet 2: 부재력 (표 10.2.7)
            member_data = []
            for mid in ['AB', 'BC', 'CD', 'AD']:
                m = self.members[mid]
                member_data.append({
                    '부재': mid,
                    '시작절점': m.node_start,
                    '끝절점': m.node_end,
                    '부재 종류': '스트럿' if m.member_type == 'strut' else '타이',
                    '부재력 (kN)': round(abs(m.force), 1) if m.force else 0,
                    '각도 (deg)': round(m.angle, 1) if m.angle else 0,
                    'beta_s': self.get_beta_s(mid) if m.member_type == 'strut' else '-'
                })
            df_members = pd.DataFrame(member_data)
            df_members.to_excel(writer, sheet_name='부재력', index=False)
            
            # Sheet 3: 절점 분류 (표 10.2.6)
            node_data = []
            for nid in ['A', 'B', 'C', 'D']:
                n = self.nodes[nid]
                node_data.append({
                    '절점': nid,
                    'X 좌표 (mm)': n.x,
                    'Y 좌표 (mm)': n.y,
                    '절점 종류': n.node_type if n.node_type else 'N/A',
                    'beta_n': self.get_beta_n(n.node_type) if n.node_type else 0,
                    'f_cn (MPa)': round(0.85 * self.get_beta_n(n.node_type) * self.material.fck, 1) if n.node_type else 0
                })
            df_nodes = pd.DataFrame(node_data)
            df_nodes.to_excel(writer, sheet_name='절점 분류', index=False)
            
            # Sheet 4: 스트럿 검증 (표 10.2.7)
            strut_data = []
            for mid in ['AB', 'BC', 'CD']:
                res = self.results['struts'][mid]
                strut_data.append({
                    '부재': res['member_id'],
                    'beta_s': res['beta_s'],
                    'f_ce (MPa)': round(res['f_ce'], 1),
                    '부재력 (kN)': round(res['F_u'], 1),
                    '요구 폭 (mm)': round(res['w_req'], 1),
                    '실제 폭 (mm)': round(res['w_s'], 1),
                    'phi*F_ns (kN)': round(res['phi_Fns'], 1),
                    'D/C 비': round(res['ratio'], 3),
                    '판정': '만족' if res['is_ok'] else '불만족'
                })
            df_struts = pd.DataFrame(strut_data)
            df_struts.to_excel(writer, sheet_name='스트럿 검증', index=False)
            
            # Sheet 5: 타이 검증
            tie_data = []
            res = self.results['ties']['AD']
            tie_data.append({
                '부재': res['member_id'],
                '부재력 (kN)': round(res['F_u'], 1),
                '요구 철근량 (mm²)': round(res['A_s_req'], 1),
                '제공 철근량 (mm²)': round(res['A_s'], 1),
                'phi*F_nt (kN)': round(res['phi_Fnt'], 1),
                'D/C 비': round(res['ratio'], 3),
                '판정': '만족' if res['is_ok'] else '불만족'
            })
            df_ties = pd.DataFrame(tie_data)
            df_ties.to_excel(writer, sheet_name='타이 검증', index=False)
            
            # Sheet 6: 절점 검증
            node_verify_data = []
            for nid in ['A', 'B', 'C', 'D']:
                res = self.results['nodes'][nid]
                node_verify_data.append({
                    '절점': res['node_id'],
                    '절점 종류': res['node_type'],
                    'beta_n': res['beta_n'],
                    'f_cn (MPa)': round(res['f_cn'], 1),
                    '최대 압축력 (kN)': round(res['F_u'], 1),
                    'phi*F_nn (kN)': round(res['phi_Fnn'], 1),
                    'D/C 비': round(res['ratio'], 3),
                    '판정': '만족' if res['is_ok'] else '불만족'
                })
            df_node_verify = pd.DataFrame(node_verify_data)
            df_node_verify.to_excel(writer, sheet_name='절점 검증', index=False)
            
            # Sheet 7: 상세 절점 검증 (표 10.2.8 형식)
            if 'nodes_detailed' in self.results and self.results['nodes_detailed']:
                detailed_data = []
                for nid in ['A', 'B', 'C', 'D']:
                    if nid not in self.results['nodes_detailed']:
                        continue
                    res = self.results['nodes_detailed'][nid]
                    node_type = res['node_type']
                    beta_n = res['beta_n']
                    
                    for check in res['checks']:
                        notes_str = ', '.join(res['notes']) if res['notes'] else ''
                        
                        detailed_data.append({
                            '절점': nid,
                            '절점종류': node_type,
                            'beta_n': beta_n,
                            '절점종류(요소)': check['element_id'],
                            '부재력(kN)': round(check['force'], 1),
                            '요구된 폭(mm)': round(check['w_req'], 1),
                            '실제 폭(mm)': round(check['w_actual'], 1),
                            '판정': check['note'],
                            '보완': notes_str
                        })
                
                df_detailed = pd.DataFrame(detailed_data)
                df_detailed.to_excel(writer, sheet_name='상세 절점 검증(표10.2.8)', index=False)
            
            # Sheet 8: 반력
            reaction_data = []
            for node_id in ['A', 'D']:
                r = self.reactions[node_id]
                reaction_data.append({
                    '절점': node_id,
                    '지점 종류': self.supports[node_id],
                    'Rx (kN)': round(r.get('Rx', 0), 1),
                    'Ry (kN)': round(r.get('Ry', 0), 1)
                })
            df_reactions = pd.DataFrame(reaction_data)
            df_reactions.to_excel(writer, sheet_name='반력', index=False)
        
        print(f"\n[INFO] 결과가 엑셀 파일로 저장되었습니다: {filename}")
        return filename
    
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*80)
        print("검증 결과")
        print("="*80)
        
        # 부재력
        print("\n[표 10.2.7] 부재력 및 스트럿 검증")
        print("-"*80)
        print(f"{'부재':<8} {'beta_s':<8} {'부재력':<12} {'요구폭':<12} {'실제폭':<12} {'판정'}")
        print(f"{'':8} {'':8} {'(kN)':<12} {'(mm)':<12} {'(mm)':<12} {''}")
        print("-"*80)
        
        for mid in ['AB', 'BC', 'CD']:
            res = self.results['struts'][mid]
            status = '만족' if res['is_ok'] else '불만족'
            print(f"{mid:<8} {res['beta_s']:<8.2f} {res['F_u']:<12.1f} {res['w_req']:<12.1f} {res['w_s']:<12.1f} {status}")
        
        # 타이
        print("\n[타이 검증]")
        print("-"*80)
        res = self.results['ties']['AD']
        print(f"AD: 부재력 = {res['F_u']:.1f} kN, 요구 A_s = {res['A_s_req']:.1f} mm²")
        
        # 절점
        print("\n[표 10.2.6] 절점 분류 및 검증")
        print("-"*80)
        print(f"{'절점':<8} {'절점종류':<10} {'beta_n':<10} {'f_cn':<12} {'판정'}")
        print("-"*80)
        for nid in ['A', 'B', 'C', 'D']:
            res = self.results['nodes'][nid]
            status = '만족' if res['is_ok'] else '불만족'
            print(f"{nid:<8} {res['node_type']:<10} {res['beta_n']:<10.2f} {res['f_cn']:<12.1f} {status}")
        
        # 상세 절점 검증 (표 10.2.8)
        if 'nodes_detailed' in self.results and self.results['nodes_detailed']:
            print("\n[표 10.2.8] 절점의 설계 검토 (상세)")
            print("-"*100)
            print(f"{'절점':<6} {'절점종류':<8} {'β_n':<6} {'절점종류':<10} {'부재력':<12} {'요구된 폭':<12} {'실제 폭':<12} {'판정':<15} {'보완'}")
            print(f"{'':6} {'':8} {'':6} {'':10} {'(kN)':<12} {'(mm)':<12} {'(mm)':<12} {'':15} {''}")
            print("-"*100)
            
            for nid in ['A', 'B', 'C', 'D']:
                if nid not in self.results['nodes_detailed']:
                    continue
                res = self.results['nodes_detailed'][nid]
                node_type = res['node_type']
                beta_n = res['beta_n']
                
                for i, check in enumerate(res['checks']):
                    if i == 0:
                        # 첫 행에만 절점 정보 표시
                        notes_str = ', '.join(res['notes']) if res['notes'] else ''
                        print(f"{nid:<6} {node_type:<8} {beta_n:<6.1f} {check['element_id']:<10} {check['force']:<12.1f} {check['w_req']:<12.1f} {check['w_actual']:<12.1f} {check['note']:<15} {notes_str}")
                    else:
                        # 이후 행은 빈칸
                        print(f"{'':6} {'':8} {'':6} {check['element_id']:<10} {check['force']:<12.1f} {check['w_req']:<12.1f} {check['w_actual']:<12.1f} {check['note']:<15}")
        
        print("\n" + "="*80)
        for msg in self.results['messages']:
            print(msg)
        print("="*80 + "\n")
    
    def plot_stm(self, save_path: str = None):
        """STM 모델 그리기"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 부재 그리기
        for mid, m in self.members.items():
            n1 = self.nodes[m.node_start]
            n2 = self.nodes[m.node_end]
            
            if m.member_type == 'strut':
                color = 'blue'
                linestyle = '-'
                linewidth = 4
            else:
                color = 'red'
                linestyle = '--'
                linewidth = 3
            
            ax.plot([n1.x, n2.x], [n1.y, n2.y], 
                   color=color, linestyle=linestyle, linewidth=linewidth)
            
            # 부재력 표시
            mid_x = (n1.x + n2.x) / 2
            mid_y = (n1.y + n2.y) / 2
            if m.force:
                ax.text(mid_x, mid_y, f"{abs(m.force):.0f}kN", 
                       fontsize=12, ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 절점 그리기
        for nid, n in self.nodes.items():
            ax.plot(n.x, n.y, 'ko', markersize=12)
            ax.text(n.x, n.y + 120, nid, 
                   ha='center', fontsize=14, fontweight='bold')
            if n.node_type:
                ax.text(n.x, n.y - 120, n.node_type, 
                       ha='center', fontsize=11, color='green')
        
        # 지점 표시
        for nid, support_type in self.supports.items():
            n = self.nodes[nid]
            if support_type == 'pin':
                ax.plot(n.x, n.y - 150, '^', markersize=18, color='black')
        
        # 하중 표시
        for load_nid, Fx, Fy in self.loads:
            n = self.nodes[load_nid]
            ax.arrow(n.x, n.y + 250, 0, -200, 
                    head_width=80, head_length=40, fc='darkgreen', ec='darkgreen', linewidth=2)
            ax.text(n.x, n.y + 300, f"{abs(Fy)}kN", 
                   ha='center', fontsize=13, color='darkgreen', fontweight='bold')
        
        ax.set_xlabel('Length (mm)', fontsize=13)
        ax.set_ylabel('Height (mm)', fontsize=13)
        ax.set_title('STM-2 Model (4 Nodes)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 범례
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=4, label='Strut (Compression)'),
            Line2D([0], [0], color='red', linewidth=3, linestyle='--', label='Tie (Tension)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        
        # 저장 경로 처리 (Windows 호환)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[INFO] 그림이 저장되었습니다: {save_path}")
        
        return fig, ax


# 교재 예제 10.2 실행
if __name__ == "__main__":
    # 재료 물성
    material = Material(fck=27, fy=400)
    
    # 보 치수
    checker = STMDesignChecker(
        material=material,
        beam_width=500,
        beam_height=1725
    )
    
    # 절점 추가 (교재 그림 10.2.6 기준)
    # 좌표: 왼쪽 하단을 원점으로
    # 높이 = 1,725 mm (바닥에서부터) → y좌표 = 125 + 1,725 = 1,850 mm
    checker.add_node(Node('A', 450, 125, None))        # 왼쪽 하단 지점
    checker.add_node(Node('B', 2450, 1850, None))      # 왼쪽 상단 하중점 (125+1725)
    checker.add_node(Node('C', 4450, 1850, None))      # 오른쪽 상단 하중점 (125+1725)
    checker.add_node(Node('D', 6450, 125, None))       # 오른쪽 하단 지점
    
    # 부재 추가
    checker.add_member(Member('AB', 'A', 'B', None, None))
    checker.add_member(Member('BC', 'B', 'C', None, None))
    checker.add_member(Member('CD', 'C', 'D', None, None))
    checker.add_member(Member('AD', 'A', 'D', None, None))
    
    # 하중 추가 (B와 C에 각각 2,000 kN)
    checker.add_load('B', 0, -2000)
    checker.add_load('C', 0, -2000)
    
    # 지점 추가
    checker.add_support('A', 'pin')
    checker.add_support('D', 'pin')
    
    # 검증 수행
    results = checker.verify_stm()
    
    # 결과 출력
    checker.print_results()
    
    # 엑셀 파일 저장 (현재 디렉토리에 저장)
    excel_filename = checker.export_to_excel('STM_교재예제10.2_검증결과.xlsx')
    
    # 그림 저장 (현재 디렉토리에 저장)
    checker.plot_stm(save_path='STM_교재예제10.2_모델도.png')
    
    print(f"\n{'='*80}")
    print("실행 완료!")
    print(f"{'='*80}")
    print(f"생성된 파일:")
    print(f"  1. {excel_filename}")
    print(f"  2. STM_교재예제10.2_모델도.png")
    print(f"{'='*80}\n")