#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STM Verification MCP Server
Automatic image processing, force calculation, and STM verification
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

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import custom modules
from stm_image_processor import STMImageProcessor
from stm_auto_analyzer import STMStaticAnalyzer, Node, Member, Load
from stm_table_output import STMDesignChecker, Material

# Initialize server
server = Server("stm-verifier")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Tool list"""
    return [
        types.Tool(
            name="verify_stm",
            description="Automatic STM diagram verification from image",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "STM image file path"},
                    "beam_width": {"type": "number", "default": 300},
                    "beam_height": {"type": "number", "default": 1000},
                    "beam_length": {"type": "number", "default": 2000},
                    "fck": {"type": "number", "default": 27},
                    "fy": {"type": "number", "default": 400},
                    "load_magnitude": {"type": "number", "default": 3000}
                },
                "required": ["image_path"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    """Tool execution"""
    
    if name != "verify_stm":
        raise ValueError(f"Unknown tool: {name}")
    
    try:
        # Extract parameters
        image_path = arguments.get("image_path", "")
        beam_width = arguments.get("beam_width", 300)
        beam_height = arguments.get("beam_height", 1000)
        beam_length = arguments.get("beam_length", 2000)
        fck = arguments.get("fck", 27)
        fy = arguments.get("fy", 400)
        load_magnitude = arguments.get("load_magnitude", 3000)
        
        result = f"""# STM Design Verification Result

## Input Information
- Image: {image_path}
- Beam width: {beam_width}mm
- Beam height: {beam_height}mm
- Beam length: {beam_length}mm
- fck: {fck}MPa
- fy: {fy}MPa
- Load: {load_magnitude}kN

---

"""
        
        # Step 1: Image validation
        if not os.path.exists(image_path):
            return [types.TextContent(
                type="text",
                text=result + f"""[ERROR] Image file not found

Path: {image_path}

Solution:
1. Check file path
2. Use absolute path
3. Verify file exists
"""
            )]
        
        # Step 2: Image processing
        result += "## Step 1: Image Analysis\n\n"
        
        try:
            processor = STMImageProcessor()
            processor.load_image(image_path)
            result += f"[OK] Image loaded: {processor.image.shape}\n\n"
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=result + f"[ERROR] Image load failed: {str(e)}\n"
            )]
        
        # Step 3: Node detection
        try:
            circles = processor.detect_circles(min_radius=5, max_radius=50, param2=20)
            
            if circles is None or len(circles) == 0:
                return [types.TextContent(
                    type="text",
                    text=result + "[ERROR] No nodes detected\n"
                )]
            
            # Create nodes
            nodes = []
            node_labels = [chr(65 + i) for i in range(len(circles))]
            
            # Sort nodes by Y then X
            circles_sorted = sorted(circles, key=lambda c: (c[1], c[0]))
            y_coords = [c[1] for c in circles_sorted]
            y_mid = (max(y_coords) + min(y_coords)) / 2
            
            upper_nodes = sorted([c for c in circles_sorted if c[1] < y_mid], key=lambda c: c[0])
            lower_nodes = sorted([c for c in circles_sorted if c[1] >= y_mid], key=lambda c: c[0])
            
            result += f"[OK] {len(circles)} nodes detected\n\n"
            result += "Upper nodes:\n"
            for i, (x, y, r) in enumerate(upper_nodes):
                node_id = chr(65 + i)
                result += f"  {node_id}: X={int(x)}, Y={int(y)}\n"
            
            result += "\nLower nodes:\n"
            for i, (x, y, r) in enumerate(lower_nodes):
                node_id = chr(69 + i)
                result += f"  {node_id}: X={int(x)}, Y={int(y)}\n"
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=result + f"[ERROR] Node detection failed: {str(e)}\n"
            )]
        
        # Step 4: Create analyzer
        result += "\n## Step 2: Structural Analysis\n\n"
        
        try:
            analyzer = STMStaticAnalyzer()
            
            # Add upper nodes
            all_nodes = []
            for i, (x, y, r) in enumerate(upper_nodes):
                node_id = chr(65 + i)
                node = Node(node_id, float(x), float(y), None, False, None)
                analyzer.add_node(node)
                all_nodes.append((node_id, x, y))
            
            # Add lower nodes
            for i, (x, y, r) in enumerate(lower_nodes):
                node_id = chr(69 + i)
                is_support = (i == 0) or (i == len(lower_nodes)-1)
                support_type = 'pin' if i == 0 else ('roller' if i == len(lower_nodes)-1 else None)
                node = Node(node_id, float(x), float(y), None, is_support, support_type)
                analyzer.add_node(node)
                all_nodes.append((node_id, x, y))
            
            # Add members (geometric connections)
            members_data = [
                ("AB", "A", "B"), ("BC", "B", "C"), ("CD", "C", "D"),
                ("EF", "E", "F"), ("FG", "F", "G"), ("GH", "G", "H"),
                ("AE", "A", "E"), ("AF", "A", "F"), ("BF", "B", "F"),
                ("CG", "C", "G"), ("DG", "D", "G"), ("DH", "D", "H"),
            ]
            
            max_distance = 600
            added_members = []
            for mid, s, e in members_data:
                node_s = next((x, y) for id1, x, y in all_nodes if id1 == s)
                node_e = next((x, y) for id1, x, y in all_nodes if id1 == e)
                dist = np.sqrt((node_e[0]-node_s[0])**2 + (node_e[1]-node_s[1])**2)
                
                if dist < max_distance:
                    analyzer.add_member(Member(mid, s, e, None))
                    added_members.append(mid)
            
            result += f"[OK] {len(added_members)} members added\n"
            
            # Add loads (B and C nodes)
            analyzer.add_load(Load("B", 0, -load_magnitude))
            analyzer.add_load(Load("C", 0, -load_magnitude))
            
            # Calculate member forces
            member_forces = analyzer.solve_member_forces()
            analyzer.infer_member_types()
            analyzer.infer_node_types()
            
            result += "\n## Step 3: Member Forces\n\n"
            result += "| Member | Force (kN) | Type |\n"
            result += "|--------|------------|------|\n"
            
            for member_id in sorted(member_forces.keys()):
                force = member_forces[member_id]
                member = analyzer.members[member_id]
                mtype = "Strut" if member.member_type == 'strut' else "Tie"
                result += f"| {member_id:<6} | {force:>10.1f} | {mtype} |\n"
            
            # Reactions
            result += "\n## Step 4: Reactions\n\n"
            for node_id, reactions in sorted(analyzer.reactions.items()):
                rx = reactions.get('Rx', 0)
                ry = reactions.get('Ry', 0)
                result += f"  {node_id}: Rx={rx:.1f}kN, Ry={ry:.1f}kN\n"
            
            # Verification
            result += "\n## Step 5: KDS Strength Verification\n\n"
            
            material = Material(fck=fck, fy=fy)
            checker = STMDesignChecker(material, beam_width, beam_height, beam_length)
            
            for node in analyzer.nodes.values():
                checker.add_node(node)
            
            for member in analyzer.members.values():
                checker.add_member(member)
            
            results = checker.verify_stm()
            df = checker.generate_node_table()
            
            if not df.empty:
                result += df.to_string(index=False)
            else:
                result += "(No verification results)\n"
            
            result += "\n\n## Final Result\n\n"
            if results['overall']:
                result += "[PASS] All members satisfy design criteria\n"
            else:
                result += "[FAIL] Some members do not satisfy criteria\n"
            
        except Exception as e:
            result += f"[ERROR] Analysis failed: {str(e)}\n\n```\n{traceback.format_exc()}\n```\n"
        
        return [types.TextContent(type="text", text=result)]
        
    except Exception as e:
        error_msg = f"[ERROR] {str(e)}\n\n```\n{traceback.format_exc()}\n```\n"
        return [types.TextContent(type="text", text=error_msg)]

async def main():
    """Run server"""
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
