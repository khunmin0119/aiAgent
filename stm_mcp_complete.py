#!/usr/bin/env python3
"""
STM 검증 MCP 서버 - 완전 구현 버전
실제로 이미지 처리, 부재력 계산, STM 검증을 수행합니다.
"""

from mcp.server.models import InitializationOptions
import mcp.types as types
import mcp.server.stdio
from mcp.server import NotificationOptions, Server
import json
import traceback
import sys
import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

# 현재 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 자체 모듈 import
from stm_image_processor import STMImageProcessor
from stm_auto_analyzer import STMStaticAnalyzer, Node, Member, Load
from stm_table_output import STMDesignChecker, Material

# 서버 초기화
server = Server("stm-verifier")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """도구 목록"""
    return [
        types.Tool(
            name="verify_stm",
            description="""
            STM 다이어그램 이미지를 자동으로 검증합니다.
            
            **완전 자동화 프로세스:**
            1. 이미지에서 절점 자동 추출 (OpenCV)
            2. 부재 연결 자동 인식 (OpenCV)
            3. 부재력 자동 계산 (정역학 평형방정식)
            4. Strut/Tie 자동 판단
            5. 절점 형식 자동 결정
            6. KDS 강도 검증
            7. 표 형식 결과 출력
            
            **필수 입력:**
            - image_path: STM 다이어그램 이미지 경로 (절대 경로)
            
            **선택 입력:**
            - beam_width: 보 폭 (mm, 기본값 300)
            - beam_height: 보 높이 (mm, 기본값 1000)
            - beam_length: 보 길이 (mm, 기본값 2000)
            - fck: 콘크리트 압축강도 (MPa, 기본값 27)
            - fy: 철근 항복강도 (MPa, 기본값 400)
            - load_magnitude: 하중 크기 (kN, 기본값 3000)
            
            **출력:**
            - 자동 추출된 절점 좌표
            - 자동 계산된 부재력
            - Strut/Tie 판단 결과
            - 절점 형식
            - 강도 검증 결과 표
            - 합격/불합격 판정
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "STM 다이어그램 이미지 파일 경로 (절대 경로)"
                    },
                    "beam_width": {
                        "type": "number",
                        "description": "보 폭 (mm)",
                        "default": 300
                    },
                    "beam_height": {
                        "type": "number",
                        "description": "보 높이 (mm)",
                        "default": 1000
                    },
                    "beam_length": {
                        "type": "number",
                        "description": "보 길이 (mm)",
                        "default": 2000
                    },
                    "fck": {
                        "type": "number",
                        "description": "콘크리트 압축강도 (MPa)",
                        "default": 27
                    },
                    "fy": {
                        "type": "number",
                        "description": "철근 항복강도 (MPa)",
                        "default": 400
                    },
                    "load_magnitude": {
                        "type": "number",
                        "description": "하중 크기 (kN)",
                        "default": 3000
                    }
                },
                "required": ["image_path"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    """도구 실행 - 실제 STM 검증 수행"""
    
    if name != "verify_stm":
        raise ValueError(f"Unknown tool: {name}")
    
    try:
        # 파라미터 추출
        image_path = arguments.get("image_path", "")
        beam_width = arguments.get("beam_width", 300)
        beam_height = arguments.get("beam_height", 1000)
        beam_length = arguments.get("beam_length", 2000)
        fck = arguments.get("fck", 27)
        fy = arguments.get("fy", 400)
        load_magnitude = arguments.get("load_magnitude", 3000)
        
        result = f"""# STM 설계 자동 검증 결과

## 입력 정보
- 이미지: {image_path}
- 보 폭: {beam_width}mm
- 보 높이: {beam_height}mm
- 보 길이: {beam_length}mm
- fck: {fck}MPa
- fy: {fy}MPa
- 하중: {load_magnitude}kN

---

"""
        
        # 1단계: 이미지 검증
        if not os.path.exists(image_path):
            return [types.TextContent(
                type="text",
                text=result + f""" **오류: 이미지 파일을 찾을 수 없습니다**

경로: {image_path}

**해결 방법:**
1. 파일 경로가 정확한지 확인
2. 절대 경로를 사용 (예: C:/Users/.../image.png)
3. 파일이 실제로 존재하는지 확인
"""
            )]
        
        # 2단계: 이미지 처리
        result += "## 1단계: 이미지 분석\n\n"
        
        try:
            processor = STMImageProcessor()
            processor.load_image(image_path)
            result += f"[OK] 이미지 로드 완료 ({processor.image.shape})\n\n"
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=result + f"[ERROR] 이미지 로드 실패: {str(e)}\n"
            )]
        
        # 3단계: 절점 검출
        try:
            circles = processor.detect_circles(min_radius=5, max_radius=30, param2=20)
            
            if circles is None or len(circles) == 0:
                return [types.TextContent(
                    type="text",
                    text=result + """[ERROR] **절점을 검출할 수 없습니다**

**가능한 원인:**
1. 이미지가 너무 흐릿함
2. 절점이 명확하게 표시되지 않음
3. 이미지 품질이 낮음

**해결 방법:**
1. 고해상도 이미지 사용
2. 절점을 원으로 명확하게 표시
3. CAD 도면 사용 권장
4. JSON 형식으로 절점 직접 입력
"""
                )]
            
            # 절점 생성
            nodes = []
            node_labels = [chr(65 + i) for i in range(len(circles))]
            
            result += f" {len(circles)}개 절점 자동 검출\n\n"
            result += "| 절점 | X (픽셀) | Y (픽셀) |\n"
            result += "|------|----------|----------|\n"
            
            for i, (x, y, r) in enumerate(circles):
                node_id = node_labels[i]
                nodes.append(Node(node_id, float(x), float(y), None))
                result += f"| {node_id} | {int(x)} | {int(y)} |\n"
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=result + f"절점 검출 오류: {str(e)}\n"
            )]
        
        # 4단계: 부재 연결 검출
        result += "\n## 2단계: 부재 연결 추출\n\n"
        
        try:
            lines = processor.detect_lines(threshold=50, min_line_length=30)
            
            if lines is None or len(lines) == 0:
                return [types.TextContent(
                    type="text",
                    text=result + "부재 연결을 검출할 수 없습니다.\n"
                )]
            
            # 선분을 절점에 매칭
            members = []
            
            def find_nearest_node(x, y, threshold=50):
                min_dist = float('inf')
                nearest_idx = None
                for i, node in enumerate(nodes):
                    dist = np.sqrt((node.x - x)**2 + (node.y - y)**2)
                    if dist < min_dist and dist < threshold:
                        min_dist = dist
                        nearest_idx = i
                return nearest_idx
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                start_idx = find_nearest_node(x1, y1)
                end_idx = find_nearest_node(x2, y2)
                
                if start_idx is not None and end_idx is not None and start_idx != end_idx:
                    start_node = nodes[start_idx]
                    end_node = nodes[end_idx]
                    member_id = f"{start_node.id}{end_node.id}"
                    
                    # 중복 체크
                    if not any(m.id == member_id or m.id == f"{end_node.id}{start_node.id}" 
                              for m in members):
                        members.append(Member(member_id, start_node.id, end_node.id, None))
            
            result += f"{len(members)}개 부재 연결 추출\n\n"
            result += "| 부재 | 시작 | 끝 |\n"
            result += "|------|------|----|\n"
            for member in members:
                result += f"| {member.id} | {member.node_start} | {member.node_end} |\n"
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=result + f"부재 검출 오류: {str(e)}\n"
            )]
        
        # 5단계: 정역학 해석
        result += "\n## 3단계: 부재력 자동 계산\n\n"
        
        try:
            analyzer = STMStaticAnalyzer()
            
            # 지점 조건 설정 (양단 핀/롤러 가정)
            for i, node in enumerate(nodes):
                if i == 0:
                    node.is_support = True
                    node.support_type = 'pin'
                elif i == len(nodes) - 1:
                    node.is_support = True
                    node.support_type = 'roller'
                analyzer.add_node(node)
            
            # 부재 추가
            for member in members:
                analyzer.add_member(member)
            
            # 하중 추가 (중앙 절점에 하향 하중)
            mid_node = nodes[len(nodes) // 2]
            analyzer.add_load(Load(mid_node.id, Fx=0, Fy=-load_magnitude))
            
            # 부재력 계산
            member_forces = analyzer.solve_member_forces()
            analyzer.infer_member_types()
            analyzer.infer_node_types()
            
            result += "정역학 평형방정식 해결 완료\n\n"
            result += "| 부재 | 부재력 (kN) | 종류 |\n"
            result += "|------|-------------|------|\n"
            
            for member_id in sorted(member_forces.keys()):
                force = member_forces[member_id]
                member = analyzer.members[member_id]
                mtype = "Strut (압축)" if member.member_type == 'strut' else "Tie (인장)"
                result += f"| {member_id} | {force:.1f} | {mtype} |\n"
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=result + f"부재력 계산 오류: {str(e)}\n\n```\n{traceback.format_exc()}\n```\n"
            )]
        
        # 6단계: 절점 형식
        result += "\n## 4단계: 절점 형식 자동 결정\n\n"
        result += "| 절점 | 형식 | βn | 의미 |\n"
        result += "|------|------|-----|------|\n"
        
        for node_id in sorted(analyzer.nodes.keys()):
            node = analyzer.nodes[node_id]
            beta_n = {
                'CCC': 1.0, 'CCT': 0.8, 'CTT': 0.6, 'TTT': 0.6
            }.get(node.node_type, 0.6)
            meaning = {
                'CCC': '3개 압축',
                'CCT': '2압축+1인장',
                'CTT': '1압축+2인장',
                'TTT': '3개 인장'
            }.get(node.node_type, node.node_type)
            result += f"| {node_id} | {node.node_type} | {beta_n} | {meaning} |\n"
        
        # 7단계: STM 검증
        result += "\n## 5단계: 강도 검증\n\n"
        
        try:
            material = Material(fck=fck, fy=fy)
            checker = STMDesignChecker(
                material=material,
                beam_width=beam_width,
                beam_height=beam_height,
                beam_length=beam_length
            )
            
            # 데이터 전달
            for node in analyzer.nodes.values():
                checker.add_node(node)
            
            for member in analyzer.members.values():
                checker.add_member(member)
            
            # 검증 수행
            results = checker.verify_stm()
            df = checker.generate_node_table()
            
            result += "KDS 기준 강도 검증 완료\n\n"
            
            # 표 형식 결과
            if not df.empty:
                result += df.to_markdown(index=False)
            else:
                result += "(검증 결과 없음)\n"
            
            # 최종 판정
            result += "\n\n## 최종 판정\n\n"
            if results['overall']:
                result += "**합격** - 모든 부재가 설계 기준을 만족합니다.\n"
            else:
                result += "**불합격** - 일부 부재가 설계 기준을 만족하지 못합니다.\n\n"
                result += "### 불만족 부재\n\n"
                failed = df[df['판정'] == '불만족']
                if not failed.empty:
                    result += failed[['부재종류', '요구된 폭(mm)', '실제 폭(mm)', '보완']].to_markdown(index=False)
                    result += "\n"
            
        except Exception as e:
            result += f"검증 오류: {str(e)}\n\n```\n{traceback.format_exc()}\n```\n"
        
        result += "\n---\n\n"
        result += f"**처리 완료** | 검증 기준: KDS 14 20 52 (2021)\n"
        
        return [types.TextContent(type="text", text=result)]
        
    except Exception as e:
        error_msg = f"""# 오류 발생

{str(e)}

## 스택 트레이스

```
{traceback.format_exc()}
```

## 디버그 정보
- Python: {sys.version}
- 현재 디렉토리: {os.getcwd()}
- 파일 위치: {__file__}
"""
        return [types.TextContent(type="text", text=error_msg)]


async def main():
    """서버 실행"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        init_options = InitializationOptions(
            server_name="stm-verifier",
            server_version="2.0.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        )
        await server.run(
            read_stream,
            write_stream,
            init_options,
            raise_exceptions=True
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
