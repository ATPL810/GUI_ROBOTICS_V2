"""
ANALYZE GRAB POINTS - IMPROVED VERSION
Shows ALL tool locations including duplicates
"""

import json
import os
import re
from datetime import datetime
import math
from pathlib import Path

class GrabPointAnalyzer:
    def __init__(self, snapshot_folder=None):
        """Initialize analyzer with snapshot data"""
        print("ANALYZE GRAB POINTS - TOOL MAPPING SYSTEM (ALL LOCATIONS)")
        print("=" * 70)
        
        self.snapshot_folder = self.find_latest_snapshot(snapshot_folder)
        print(f"Analyzing snapshots from: {self.snapshot_folder}")
        
        self.grab_points = self.load_grab_points()
        print(f"Loaded {sum(len(v) for v in self.grab_points.values())} grab points")
        
        self.detections_data = self.parse_all_snapshots()
        print(f"Parsed detection data from {len(self.detections_data)} snapshots")
        
        self.output_dir = self.create_mapping_directory()
        
        # NEW: Store ALL assignments including duplicates
        self.all_tool_assignments = {}  # point_id -> tool_data
        self.tool_locations = {}  # tool_name -> list of locations
        
    def parse_all_snapshots(self):
        """Parse detection data from all 3 snapshots - IMPROVED"""
        detections_data = {}
        
        positions = ["initial_position", "second_position", "third_position"]
        
        for position in positions:
            detections = self.parse_snapshot_report(position)
            detections_data[position] = {
                "detections": detections,
                "count": len(detections)
            }
            
            print(f"  {position}: {len(detections)} tools detected")
        
        return detections_data
    
    def assign_tools_to_grab_points(self, max_distance=200):
        """Assign ALL tools to grab points, including duplicates"""
        print("\n" + "=" * 70)
        print("ASSIGNING ALL TOOLS TO GRAB POINTS (INCLUDING DUPLICATES)")
        print("=" * 70)
        
        assignments = {}
        tool_locations = {}  # NEW: Store ALL locations per tool
        
        # Process each position's grab points
        for position_name, grab_points in self.grab_points.items():
            print(f"\nProcessing {position_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            if position_name not in self.detections_data:
                continue
            
            position_detections = self.detections_data[position_name]["detections"]
            
            # Process each grab point
            for point_id, point_coords in grab_points.items():
                grab_point = (point_coords["x"], point_coords["y"])
                
                closest_tool = None
                min_distance = float('inf')
                closest_tool_data = None
                
                # Find closest tool to this grab point
                for detection in position_detections:
                    if "center" not in detection:
                        continue
                    
                    tool_center = detection["center"]
                    distance = self.calculate_distance(grab_point, tool_center)
                    
                    if distance < max_distance and distance < min_distance:
                        min_distance = distance
                        closest_tool = detection["class_name"]
                        closest_tool_data = {
                            "class_name": detection["class_name"],
                            "tool_name": detection["class_name"].lower(),
                            "confidence": detection["confidence"],
                            "distance": distance,
                            "tool_center": tool_center,
                            "grab_point": grab_point,
                            "point_id": point_id,
                            "position": position_name
                        }
                
                # Store assignment
                if closest_tool:
                    assignments[point_id] = closest_tool_data
                    
                    # ADD to tool_locations list (not replace)
                    tool_name = closest_tool.lower()
                    if tool_name not in tool_locations:
                        tool_locations[tool_name] = []
                    
                    # Check if this point already has this tool (avoid duplicates)
                    existing = False
                    for loc in tool_locations[tool_name]:
                        if loc["point_id"] == point_id:
                            existing = True
                            break
                    
                    if not existing:
                        tool_locations[tool_name].append({
                            "point_id": point_id,
                            "confidence": closest_tool_data["confidence"],
                            "distance": closest_tool_data["distance"],
                            "position": position_name
                        })
                    
                    print(f"  Point {point_id}: {closest_tool} ({min_distance:.0f}px, {closest_tool_data['confidence']*100:.1f}%)")
                else:
                    assignments[point_id] = {
                        "class_name": "none",
                        "tool_name": "none",
                        "confidence": 0.0,
                        "distance": float('inf'),
                        "tool_center": None,
                        "grab_point": grab_point,
                        "point_id": point_id,
                        "position": position_name
                    }
                    print(f"  Point {point_id}: No tool assigned")
        
        self.all_tool_assignments = assignments
        self.tool_locations = tool_locations
        
        # Sort each tool's locations by confidence (highest first)
        for tool_name in tool_locations:
            tool_locations[tool_name].sort(key=lambda x: (-x["confidence"], x["distance"]))
        
        return assignments, tool_locations
    
    def generate_tool_mapping(self):
        """Generate tool mapping showing ALL locations for each tool"""
        tool_mapping = {}
        
        for tool_name, locations in self.tool_locations.items():
            if tool_name != "none":
                tool_mapping[tool_name] = {
                    "total_count": len(locations),
                    "locations": locations,
                    "best_location": locations[0] if locations else None
                }
        
        self.tool_mapping = tool_mapping
        return tool_mapping
    
    def generate_master_report(self):
        """Generate report showing ALL tool locations"""
        print("\n" + "=" * 70)
        print("GENERATING MASTER REPORT (ALL TOOL LOCATIONS)")
        print("=" * 70)
        
        # Get summary
        total_points = len(self.all_tool_assignments)
        assigned_points = sum(1 for a in self.all_tool_assignments.values() if a["tool_name"] != "none")
        
        # Count total tools (including duplicates)
        total_tools = 0
        for tool_name, data in self.tool_mapping.items():
            total_tools += data["total_count"]
        
        # Create report
        report_lines = []
        report_lines.append("=" * 75)
        report_lines.append(" " * 15 + "GARAGE ASSISTANT - COMPLETE TOOL LOCATION REPORT")
        report_lines.append("=" * 75)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Snapshot Session: {os.path.basename(self.snapshot_folder).replace('robot_snapshots_', '')}")
        report_lines.append("")
        
        # Summary section
        report_lines.append("SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Grab Points: {total_points}")
        report_lines.append(f"Points with Tools: {assigned_points}")
        report_lines.append(f"Empty Points: {total_points - assigned_points}")
        report_lines.append(f"Total Tools (including duplicates): {total_tools}")
        report_lines.append("")
        
        # Show ALL tool locations
        report_lines.append("COMPLETE TOOL INVENTORY:")
        report_lines.append("=" * 75)
        
        for tool_name, data in sorted(self.tool_mapping.items()):
            report_lines.append(f"\n{tool_name.upper()} (Total: {data['total_count']}):")
            report_lines.append("-" * 40)
            
            for i, location in enumerate(data["locations"], 1):
                status = "⭐ BEST" if i == 1 else f"  Alt #{i}"
                report_lines.append(f"  {status}: Point {location['point_id']} ({location['position'].replace('_', ' ')})")
                report_lines.append(f"      Confidence: {location['confidence']*100:.1f}%, Distance: {location['distance']:.0f}px")
        
        # Grab point assignments
        report_lines.append("\n" + "=" * 75)
        report_lines.append("GRAB POINT ASSIGNMENTS:")
        report_lines.append("-" * 40)
        
        for position_name in ["initial_position", "second_position", "third_position"]:
            report_lines.append(f"\n{position_name.upper().replace('_', ' ')}:")
            for point_id, assignment in sorted(self.all_tool_assignments.items()):
                if assignment["position"] == position_name:
                    if assignment["tool_name"] != "none":
                        report_lines.append(f"  Point {point_id}: {assignment['class_name']} "
                                          f"({assignment['confidence']*100:.1f}%, {assignment['distance']:.0f}px)")
                    else:
                        report_lines.append(f"  Point {point_id}: EMPTY")
        
        # Robot action plan - shows ALL locations
        report_lines.append("\n" + "=" * 75)
        report_lines.append("ROBOT ACTION PLAN (WITH ALL LOCATIONS):")
        report_lines.append("-" * 40)
        
        for tool_name, data in sorted(self.tool_mapping.items()):
            report_lines.append(f"\nWhen user requests '{tool_name}':")
            if data["total_count"] > 1:
                report_lines.append(f"  Available at {data['total_count']} locations:")
                for i, location in enumerate(data["locations"], 1):
                    report_lines.append(f"    {i}. Point {location['point_id']} ({location['position'].replace('_', ' ')})")
                report_lines.append(f"  Will fetch from: Point {data['best_location']['point_id']} (highest confidence)")
            else:
                report_lines.append(f"  Will fetch from: Point {data['best_location']['point_id']}")
        
        # Tool status tracking
        report_lines.append("\n" + "=" * 75)
        report_lines.append("TOOL STATUS TRACKING SYSTEM:")
        report_lines.append("-" * 40)
        report_lines.append("\nAfter a tool is fetched, the system will:")
        report_lines.append("  1. Mark that location as 'fetched'")
        report_lines.append("  2. Update the 'best_location' to next available")
        report_lines.append("  3. Show remaining locations for that tool")
        
        # Save report
        report_file = os.path.join(self.output_dir, "complete_tool_report.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved complete report to: {report_file}")
        return report_file

def main():
    """Run the improved analyzer"""
    try:
        analyzer = GrabPointAnalyzer()
        
        # Get ALL assignments including duplicates
        assignments, tool_locations = analyzer.assign_tools_to_grab_points(max_distance=200)
        
        # Generate mapping with ALL locations
        tool_mapping = analyzer.generate_tool_mapping()
        
        # Save JSON
        json_file = os.path.join(analyzer.output_dir, "complete_tool_mapping.json")
        with open(json_file, 'w') as f:
            json.dump({
                "assignments": assignments,
                "tool_locations": tool_locations,
                "tool_mapping": tool_mapping
            }, f, indent=2, default=str)
        
        # Generate report
        analyzer.generate_master_report()
        
        print(f"\n✅ Found {len(tool_mapping)} unique tool types")
        total_tools = sum(data["total_count"] for data in tool_mapping.values())
        print(f"✅ Total tools (including duplicates): {total_tools}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()