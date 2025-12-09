"""
ANALYZE GRAB POINTS - TOOL MAPPING SYSTEM
Analyzes snapshots from take_snapshots.py and maps tools to grab points
Finds closest tool for each grab point and creates master report
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
        print("ANALYZE GRAB POINTS - TOOL MAPPING SYSTEM")
        print("=" * 70)
        
        # Find latest snapshot folder if not specified
        self.snapshot_folder = self.find_latest_snapshot(snapshot_folder)
        print(f"Analyzing snapshots from: {self.snapshot_folder}")
        
        # Load grab point coordinates
        self.grab_points = self.load_grab_points()
        print(f"Loaded {sum(len(v) for v in self.grab_points.values())} grab points")
        
        # Parse detection data from all snapshots
        self.detections_data = self.parse_all_snapshots()
        print(f"Parsed detection data from {len(self.detections_data)} snapshots")
        
        # Create output directory for mappings
        self.output_dir = self.create_mapping_directory()
        
        # Results storage
        self.grab_point_assignments = {}
        self.tool_mapping = {}
    
    def find_latest_snapshot(self, folder=None):
        """Find the latest snapshot folder in data/snapshots"""
        if folder and os.path.exists(folder):
            return folder
        
        # Look for snapshot folders
        snapshots_dir = "data/snapshots"
                
        if not os.path.exists(snapshots_dir):
            raise FileNotFoundError(f"Snapshot directory not found: {snapshots_dir}")
        
        # Get all folders that start with "robot_snapshots_"
        snapshot_folders = []
        for item in os.listdir(snapshots_dir):
            item_path = os.path.join(snapshots_dir, item)
            if os.path.isdir(item_path) and item.startswith("robot_snapshots_"):
                snapshot_folders.append(item_path)
        
        if not snapshot_folders:
            raise FileNotFoundError("No snapshot folders found. Run take_snapshots.py first.")
        
        # Sort by creation time (newest first)
        snapshot_folders.sort(key=os.path.getmtime, reverse=True)
        return snapshot_folders[0]
    
    def load_grab_points(self):
        """Load grab point coordinates from config or define inline"""
        # Grab points as specified in your description
        grab_points = {
            "initial_position": {
                "A": {"x": 75, "y": 260},
                "B": {"x": 243, "y": 280},
                "C": {"x": 410, "y": 155},
                "D": {"x": 500, "y": 300}
            },
            "second_position": {
                "E": {"x": 260, "y": 240},
                "F": {"x": 300, "y": 155},
                "G": {"x": 425, "y": 420}
            },
            "third_position": {
                "H": {"x": 160, "y": 300},
                "I": {"x": 470, "y": 140}
            }
        }
        
        # Save to config for reference
        config_dir = "config"
        os.makedirs(config_dir, exist_ok=True)
        with open(os.path.join(config_dir, "grab_points.json"), "w") as f:
            json.dump(grab_points, f, indent=2)
        
        return grab_points
    
    def parse_snapshot_report(self, position_name):
        """Parse detection data from a snapshot's text report"""
        report_file = os.path.join(self.snapshot_folder, f"{position_name}_detections.txt")
        
        if not os.path.exists(report_file):
            print(f"Warning: Report file not found: {report_file}")
            return []
        
        detections = []
        
        with open(report_file, 'r') as f:
            content = f.read()
            
            # Look for the DETECTED OBJECTS section
            if "DETECTED OBJECTS:" in content:
                # Extract the objects section
                objects_section = content.split("DETECTED OBJECTS:")[1]
                
                # Split into individual object blocks
                object_blocks = objects_section.split("Object #")
                
                for block in object_blocks[1:]:  # Skip first empty element
                    detection = {}
                    
                    # Extract class name
                    class_match = re.search(r"Class:\s*([^\n]+)", block)
                    if class_match:
                        detection["class_name"] = class_match.group(1).strip()
                    
                    # Extract confidence
                    conf_match = re.search(r"Confidence:\s*([^\n]+)", block)
                    if conf_match:
                        conf_str = conf_match.group(1).strip()
                        # Remove % sign and convert to float
                        detection["confidence"] = float(conf_str.replace('%', '')) / 100
                    
                    # Extract bounding box
                    bbox_match = re.search(r"Bounding Box:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", block)
                    if bbox_match:
                        x1, y1, x2, y2 = map(int, bbox_match.groups())
                        detection["bbox"] = (x1, y1, x2, y2)
                        
                        # Calculate center point
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        detection["center"] = (center_x, center_y)
                    
                    # Extract center coordinates
                    center_match = re.search(r"Center Coordinates:\s*\(([\d.]+),\s*([\d.]+)\)", block)
                    if center_match:
                        center_x, center_y = map(float, center_match.groups())
                        detection["center"] = (center_x, center_y)
                    
                    if detection:
                        detections.append(detection)
        
        return detections
    
    def parse_all_snapshots(self):
        """Parse detection data from all 3 snapshots"""
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
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points (x, y)"""
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def assign_tools_to_grab_points(self, max_distance=200):
        """Assign the closest tool to each grab point with one-to-one mapping"""
        print("\n" + "=" * 70)
        print("ASSIGNING TOOLS TO GRAB POINTS")
        print("=" * 70)
        
        assignments = {}
        
        # Process each position's grab points
        for position_name, grab_points in self.grab_points.items():
            print(f"\nProcessing {position_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            # Reset assigned_tools for each position
            assigned_tools = set()  # This ensures tools are only assigned within their position
            
            # Get detections for this position
            if position_name not in self.detections_data:
                print(f"  No detection data for {position_name}")
                continue
            
            position_detections = self.detections_data[position_name]["detections"]
            print(f"  Found {len(position_detections)} tools in this position")
            
            if not position_detections:
                print(f"  No tools detected in {position_name}")
                # Mark all grab points in this position as unassigned
                for point_id, point_coords in grab_points.items():
                    assignments[point_id] = {
                        "class_name": "none",
                        "confidence": 0.0,
                        "distance": float('inf'),
                        "tool_center": None,
                        "grab_point": (point_coords["x"], point_coords["y"])
                    }
                    print(f"  Point {point_id}: No tool assigned (no tools detected)")
                continue
            
            # Create a list of all possible grab point-tool pairs with distances
            all_pairs = []
            
            for point_id, point_coords in grab_points.items():
                grab_point = (point_coords["x"], point_coords["y"])
                
                for i, detection in enumerate(position_detections):
                    if "center" not in detection:
                        continue
                    
                    tool_center = detection["center"]
                    distance = self.calculate_distance(grab_point, tool_center)
                    
                    # Only consider tools within maximum distance
                    if distance <= max_distance:
                        all_pairs.append({
                            "point_id": point_id,
                            "grab_point": grab_point,
                            "tool_index": i,  # Track the tool by index
                            "distance": distance,
                            "detection": detection
                        })
            
            print(f"  Found {len(all_pairs)} valid grab point-tool pairs")
            
            # Sort all pairs by distance (closest first)
            all_pairs.sort(key=lambda x: x["distance"])
            
            # Assign tools to grab points in order of increasing distance
            for pair in all_pairs:
                point_id = pair["point_id"]
                tool_index = pair["tool_index"]
                
                # Skip if this grab point already has an assignment
                if point_id in assignments:
                    continue
                    
                # Skip if this tool is already assigned in this position
                if tool_index in assigned_tools:
                    continue
                    
                # Assign this tool to this grab point
                detection = pair["detection"]
                assignments[point_id] = {
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "distance": pair["distance"],
                    "tool_center": detection["center"],
                    "grab_point": pair["grab_point"]
                }
                
                # Mark this tool as assigned in this position
                assigned_tools.add(tool_index)
                
                print(f"  Point {point_id}: Assigned to {detection['class_name']}")
                print(f"      Distance: {pair['distance']:.1f}px, Confidence: {detection['confidence']*100:.1f}%")
            
            # Mark unassigned grab points in this position
            for point_id, point_coords in grab_points.items():
                if point_id not in assignments:
                    assignments[point_id] = {
                        "class_name": "none",
                        "confidence": 0.0,
                        "distance": float('inf'),
                        "tool_center": None,
                        "grab_point": (point_coords["x"], point_coords["y"])
                    }
                    print(f"  Point {point_id}: No tool assigned (no unassigned tools within {max_distance}px)")
        
        self.grab_point_assignments = assignments
        return assignments
    
    def create_mapping_directory(self):
        """Create directory for saving mappings"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mapping_dir = os.path.join("data", "mappings", f"mapping_{timestamp}")
        os.makedirs(mapping_dir, exist_ok=True)
        return mapping_dir
    
    def generate_tool_mapping(self):
        """Generate reverse mapping from tool name to grab point"""
        tool_mapping = {}
        
        for point_id, assignment in self.grab_point_assignments.items():
            if assignment["class_name"] != "none":
                tool_name = assignment["class_name"]
                
                if tool_name not in tool_mapping:
                    tool_mapping[tool_name] = []
                
                tool_mapping[tool_name].append({
                    "grab_point": point_id,
                    "distance": assignment["distance"],
                    "confidence": assignment["confidence"],
                    "position": self.get_position_from_point_id(point_id)
                })
        
        # For each tool, select the grab point with highest confidence (or smallest distance)
        best_mapping = {}
        for tool_name, points in tool_mapping.items():
            # Sort by confidence (highest first), then distance (smallest first)
            points.sort(key=lambda x: (-x["confidence"], x["distance"]))
            best_mapping[tool_name] = points[0]
        
        self.tool_mapping = best_mapping
        return best_mapping
    
    def get_position_from_point_id(self, point_id):
        """Determine which position a grab point belongs to"""
        if point_id in ["A", "B", "C", "D"]:
            return "initial_position"
        elif point_id in ["E", "F", "G"]:
            return "second_position"
        elif point_id in ["H", "I"]:
            return "third_position"
        return "unknown"
    
    def save_json_mapping(self):
        """Save grab point assignments and tool mapping to JSON file"""
        mapping_data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "snapshot_folder": self.snapshot_folder,
                "grab_points_count": len(self.grab_point_assignments)
            },
            "grab_point_assignments": self.grab_point_assignments,
            "tool_mapping": self.tool_mapping
        }
        
        json_file = os.path.join(self.output_dir, "tool_mapping.json")
        with open(json_file, 'w') as f:
            json.dump(mapping_data, f, indent=2, default=str)
        
        print(f"\nSaved JSON mapping to: {json_file}")
        return json_file
    
    def generate_master_report(self):
        """Generate comprehensive master report of all assignments"""
        print("\n" + "=" * 70)
        print("GENERATING MASTER REPORT")
        print("=" * 70)
        
        # Get summary information
        total_points = len(self.grab_point_assignments)
        assigned_points = sum(1 for a in self.grab_point_assignments.values() if a["class_name"] != "none")
        unassigned_points = total_points - assigned_points
        
        # Count unique tools
        unique_tools = set()
        for assignment in self.grab_point_assignments.values():
            if assignment["class_name"] != "none":
                unique_tools.add(assignment["class_name"])
        
        # Get snapshot timestamp from folder name
        snapshot_time = os.path.basename(self.snapshot_folder).replace("robot_snapshots_", "")
        
        # Create report lines
        report_lines = []
        report_lines.append("=" * 75)
        report_lines.append(" " * 20 + "GARAGE ASSISTANT - TOOL MAPPING REPORT")
        report_lines.append("=" * 75)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Snapshot Session: {snapshot_time}")
        report_lines.append("")
        
        # Summary section
        report_lines.append("SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Grab Points Analyzed: {total_points}")
        report_lines.append(f"Points with Assigned Tools: {assigned_points}")
        report_lines.append(f"Points without Tools: {unassigned_points}")
        report_lines.append(f"Unique Tools in Workshop: {len(unique_tools)}")
        report_lines.append("")
        
        # Initial Position assignments
        report_lines.append("GRAB POINT TOOL ASSIGNMENTS:")
        report_lines.append("=" * 75)
        
        report_lines.append("\nINITIAL POSITION (Base: 90 degrees):")
        report_lines.append("-" * 40)
        for point_id in ["A", "B", "C", "D"]:
            if point_id in self.grab_point_assignments:
                assignment = self.grab_point_assignments[point_id]
                if assignment["class_name"] != "none":
                    report_lines.append(f"  Point {point_id} ({assignment['grab_point'][0]},{assignment['grab_point'][1]}): "
                                      f"{assignment['class_name'].upper()} - "
                                      f"{assignment['confidence']*100:.1f}% confidence, "
                                      f"{assignment['distance']:.0f}px from tool center")
                else:
                    report_lines.append(f"  Point {point_id} ({self.grab_points['initial_position'][point_id]['x']},"
                                      f"{self.grab_points['initial_position'][point_id]['y']}): "
                                      f"No tool assigned")
        
        # Second Position assignments
        report_lines.append("\nSECOND POSITION (Base: 40 degrees):")
        report_lines.append("-" * 40)
        for point_id in ["E", "F", "G"]:
            if point_id in self.grab_point_assignments:
                assignment = self.grab_point_assignments[point_id]
                if assignment["class_name"] != "none":
                    report_lines.append(f"  Point {point_id} ({assignment['grab_point'][0]},{assignment['grab_point'][1]}): "
                                      f"{assignment['class_name'].upper()} - "
                                      f"{assignment['confidence']*100:.1f}% confidence, "
                                      f"{assignment['distance']:.0f}px from tool center")
                else:
                    report_lines.append(f"  Point {point_id} ({self.grab_points['second_position'][point_id]['x']},"
                                      f"{self.grab_points['second_position'][point_id]['y']}): "
                                      f"No tool assigned")
        
        # Third Position assignments
        report_lines.append("\nTHIRD POSITION (Base: 1 degree):")
        report_lines.append("-" * 40)
        for point_id in ["H", "I"]:
            if point_id in self.grab_point_assignments:
                assignment = self.grab_point_assignments[point_id]
                if assignment["class_name"] != "none":
                    report_lines.append(f"  Point {point_id} ({assignment['grab_point'][0]},{assignment['grab_point'][1]}): "
                                      f"{assignment['class_name'].upper()} - "
                                      f"{assignment['confidence']*100:.1f}% confidence, "
                                      f"{assignment['distance']:.0f}px from tool center")
                else:
                    report_lines.append(f"  Point {point_id} ({self.grab_points['third_position'][point_id]['x']},"
                                      f"{self.grab_points['third_position'][point_id]['y']}): "
                                      f"No tool assigned")
        
        # Tool mapping summary
        report_lines.append("\n" + "=" * 75)
        report_lines.append("TOOL TO GRAB POINT MAPPING:")
        report_lines.append("-" * 40)
        
        if self.tool_mapping:
            for tool_name, mapping in sorted(self.tool_mapping.items()):
                report_lines.append(f"  {tool_name.upper():<15} → Point {mapping['grab_point']} "
                                  f"({mapping['position'].replace('_', ' ')})")
        else:
            report_lines.append("  No tools mapped")
        
        # Robot action plan
        report_lines.append("\n" + "=" * 75)
        report_lines.append("ROBOT ACTION PLAN:")
        report_lines.append("-" * 40)
        
        if self.tool_mapping:
            for tool_name, mapping in sorted(self.tool_mapping.items()):
                position_desc = mapping["position"].replace("_", " ").title()
                report_lines.append(f"  When user requests '{tool_name.lower()}', robot will go to: "
                                  f"Point {mapping['grab_point']} ({position_desc})")
        else:
            report_lines.append("  No action plan available (no tools mapped)")
        
        # Notes section
        report_lines.append("\n" + "=" * 75)
        report_lines.append("NOTES:")
        report_lines.append("-" * 40)
        
        # Find empty points
        empty_points = []
        for point_id, assignment in self.grab_point_assignments.items():
            if assignment["class_name"] == "none":
                empty_points.append(point_id)
        
        if empty_points:
            report_lines.append(f"- Points {', '.join(empty_points)} are empty and available for new tools")
        else:
            report_lines.append("- All grab points have assigned tools")
        
        # Check distances
        far_assignments = []
        for point_id, assignment in self.grab_point_assignments.items():
            if assignment["class_name"] != "none" and assignment["distance"] > 150:
                far_assignments.append(f"{point_id} ({assignment['distance']:.0f}px)")
        
        if far_assignments:
            report_lines.append(f"- Some tools are far from grab points: {', '.join(far_assignments)}")
        else:
            report_lines.append("- All tools are within reasonable distance (<150px) from grab points")
        
        # Check confidence
        low_conf_assignments = []
        for point_id, assignment in self.grab_point_assignments.items():
            if assignment["class_name"] != "none" and assignment["confidence"] < 0.7:
                low_conf_assignments.append(f"{point_id} ({assignment['confidence']*100:.0f}%)")
        
        if low_conf_assignments:
            report_lines.append(f"- Low confidence assignments: {', '.join(low_conf_assignments)}")
        else:
            report_lines.append("- All assignments have good confidence (>70%)")
        
        report_lines.append("\n" + "=" * 75)
        
        # Save report to file
        report_file = os.path.join(self.output_dir, "master_report.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved master report to: {report_file}")
        
        # Also save a simplified version in the snapshot folder
        simple_report_file = os.path.join(self.snapshot_folder, "grab_point_assignments.txt")
        with open(simple_report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return report_file
    
    def print_summary(self):
        """Print a summary of the analysis"""
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("=" * 70)
        
        # Count statistics
        total_points = len(self.grab_point_assignments)
        assigned_points = sum(1 for a in self.grab_point_assignments.values() if a["class_name"] != "none")
        
        print(f"\nGRAB POINT ASSIGNMENTS:")
        print(f"  Total grab points: {total_points}")
        print(f"  Points with tools: {assigned_points}")
        print(f"  Empty points: {total_points - assigned_points}")
        
        print(f"\nTOOL MAPPING:")
        if self.tool_mapping:
            for tool_name, mapping in self.tool_mapping.items():
                print(f"  {tool_name}: Point {mapping['grab_point']} "
                      f"(Confidence: {mapping['confidence']*100:.1f}%, "
                      f"Distance: {mapping['distance']:.0f}px)")
        else:
            print("  No tools mapped")
        
        print(f"\nOUTPUT FILES:")
        print(f"  Mapping directory: {self.output_dir}")
        print(f"  Master report: {self.output_dir}/master_report.txt")
        print(f"  JSON mapping: {self.output_dir}/tool_mapping.json")
        print(f"  Grab points config: config/grab_points.json")
        
        print("\n" + "=" * 70)

def main():
    """Main execution function"""
    try:
        # Create data directories if they don't exist
        os.makedirs("data/mappings", exist_ok=True)
        os.makedirs("config", exist_ok=True)
        
        # Check if snapshots exist
        if not os.path.exists("data/snapshots"):
            print("ERROR: No snapshots found. Please run take_snapshots.py first.")
            print("Expected directory: data/snapshots/")
            return
        
        # Run the analysis
        print("Starting grab point analysis...")
        analyzer = GrabPointAnalyzer()
        
        # Step 1: Assign tools to grab points
        assignments = analyzer.assign_tools_to_grab_points(max_distance=200)
        
        # Step 2: Generate tool mapping
        tool_mapping = analyzer.generate_tool_mapping()
        
        # Step 3: Save results
        analyzer.save_json_mapping()
        analyzer.generate_master_report()
        
        # Step 4: Print summary
        analyzer.print_summary()
        
        print("\nAnalysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease make sure you have:")
        print("1. Run take_snapshots.py first to create snapshots")
        print("2. Check that the snapshot folder exists in data/snapshots/")
        
    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()