import os
import json
import re
import math
from datetime import datetime

class AnalysisSystem:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.grab_points = None
        self.detections_data = {}
        self.grab_point_assignments = {}
        self.tool_mapping = {}
        self.output_dir = None
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def analyze_grab_points(self, snapshot_folder):
        """Run complete grab point analysis with duplicate handling"""
        self.log("Starting grab point analysis...", "info")
        self.snapshot_folder = snapshot_folder
        
        self.load_grab_points()
        self.parse_all_snapshots()
        self.assign_tools_to_grab_points()
        self.generate_tool_mapping_with_duplicates()
        self.save_json_mapping()
        report_path = self.generate_master_report_with_duplicates()
        
        self.log("Grab point analysis COMPLETE!", "success")
        return report_path
    
    def load_grab_points(self):
        """Load grab point coordinates"""
        self.grab_points = {
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
        
        os.makedirs("config", exist_ok=True)
        with open("config/grab_points.json", "w") as f:
            json.dump(self.grab_points, f, indent=2)
        
        self.log(f"Loaded {sum(len(v) for v in self.grab_points.values())} grab points", "info")
    
    def parse_all_snapshots(self):
        """Parse detection data from all 3 snapshots"""
        positions = ["initial_position", "second_position", "third_position"]
        
        for position in positions:
            detections = self.parse_snapshot_report(position)
            self.detections_data[position] = {
                "detections": detections,
                "count": len(detections)
            }
            
            self.log(f"{position}: {len(detections)} tools detected", "info")
    
    def parse_snapshot_report(self, position_name):
        """Parse detection data from a snapshot's text report"""
        report_file = os.path.join(self.snapshot_folder, f"{position_name}_detections.txt")
        
        if not os.path.exists(report_file):
            self.log(f"Report file not found: {report_file}", "warning")
            return []
        
        detections = []
        
        with open(report_file, 'r') as f:
            content = f.read()
            
            if "DETECTED OBJECTS:" in content:
                objects_section = content.split("DETECTED OBJECTS:")[1]
                object_blocks = objects_section.split("Object #")
                
                for block in object_blocks[1:]:
                    detection = {}
                    
                    class_match = re.search(r"Class:\s*([^\n]+)", block)
                    if class_match:
                        detection["class_name"] = class_match.group(1).strip()
                    
                    conf_match = re.search(r"Confidence:\s*([^\n]+)", block)
                    if conf_match:
                        conf_str = conf_match.group(1).strip()
                        detection["confidence"] = float(conf_str.replace('%', '')) / 100
                    
                    bbox_match = re.search(r"Bounding Box:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", block)
                    if bbox_match:
                        x1, y1, x2, y2 = map(int, bbox_match.groups())
                        detection["bbox"] = (x1, y1, x2, y2)
                        detection["center"] = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    if detection:
                        detections.append(detection)
        
        return detections
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def assign_tools_to_grab_points(self, max_distance=200):
        """Assign the closest tool to each grab point"""
        self.log("Assigning tools to grab points...", "info")
        
        assignments = {}
        
        for position_name, grab_points in self.grab_points.items():
            self.log(f"Processing {position_name}...", "info")
            
            if position_name not in self.detections_data:
                continue
            
            position_detections = self.detections_data[position_name]["detections"]
            
            if not position_detections:
                for point_id, point_coords in grab_points.items():
                    assignments[point_id] = {
                        "class_name": "none",
                        "confidence": 0.0,
                        "distance": float('inf'),
                        "grab_point": (point_coords["x"], point_coords["y"])
                    }
                continue
            
            assigned_tools = set()
            all_pairs = []
            
            for point_id, point_coords in grab_points.items():
                grab_point = (point_coords["x"], point_coords["y"])
                
                for i, detection in enumerate(position_detections):
                    if "center" not in detection:
                        continue
                    
                    tool_center = detection["center"]
                    distance = self.calculate_distance(grab_point, tool_center)
                    
                    if distance <= max_distance:
                        all_pairs.append({
                            "point_id": point_id,
                            "grab_point": grab_point,
                            "tool_index": i,
                            "distance": distance,
                            "detection": detection
                        })
            
            all_pairs.sort(key=lambda x: x["distance"])
            
            for pair in all_pairs:
                point_id = pair["point_id"]
                tool_index = pair["tool_index"]
                
                if point_id in assignments or tool_index in assigned_tools:
                    continue
                    
                detection = pair["detection"]
                assignments[point_id] = {
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "distance": pair["distance"],
                    "tool_center": detection["center"],
                    "grab_point": pair["grab_point"]
                }
                
                assigned_tools.add(tool_index)
                self.log(f"  Point {point_id}: {detection['class_name']} ({pair['distance']:.1f}px)", "info")
            
            for point_id, point_coords in grab_points.items():
                if point_id not in assignments:
                    assignments[point_id] = {
                        "class_name": "none",
                        "confidence": 0.0,
                        "distance": float('inf'),
                        "tool_center": None,
                        "grab_point": (point_coords["x"], point_coords["y"])
                    }
        
        self.grab_point_assignments = assignments
        return assignments
    
    def generate_tool_mapping_with_duplicates(self):
        """Generate mapping from tool name to ALL grab points (including duplicates)"""
        tool_mapping = {}
        
        for point_id, assignment in self.grab_point_assignments.items():
            if assignment["class_name"] != "none":
                tool_name = assignment["class_name"].lower()
                
                if tool_name not in tool_mapping:
                    tool_mapping[tool_name] = []
                
                tool_mapping[tool_name].append({
                    "grab_point": point_id,
                    "distance": assignment["distance"],
                    "confidence": assignment["confidence"],
                    "position": self.get_position_from_point_id(point_id),
                    "position_desc": self.get_position_desc(self.get_position_from_point_id(point_id))
                })
        
        # Sort each tool's locations by confidence (highest first)
        for tool_name in tool_mapping:
            tool_mapping[tool_name].sort(key=lambda x: (-x["confidence"], x["distance"]))
        
        self.tool_mapping = tool_mapping
        
        if tool_mapping:
            self.log("Tool mapping generated (with duplicates):", "success")
            for tool_name, locations in tool_mapping.items():
                if len(locations) > 1:
                    self.log(f"  {tool_name.upper():<15} → {len(locations)} locations", "info")
                else:
                    self.log(f"  {tool_name.upper():<15} → Point {locations[0]['grab_point']} ({locations[0]['confidence']*100:.1f}%)", "info")
        
        return tool_mapping
    
    def get_position_from_point_id(self, point_id):
        """Determine which position a grab point belongs to"""
        if point_id in ["A", "B", "C", "D"]:
            return "initial_position"
        elif point_id in ["E", "F", "G"]:
            return "second_position"
        elif point_id in ["H", "I"]:
            return "third_position"
        return "unknown"
    
    def get_position_desc(self, position):
        """Get human-readable position description"""
        if position == "initial_position":
            return "initial position"
        elif position == "second_position":
            return "second position"
        elif position == "third_position":
            return "third position"
        return position
    
    def save_json_mapping(self):
        """Save grab point assignments and tool mapping to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("data", "mappings", f"mapping_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        
        self.log(f"Saved JSON mapping to: {json_file}", "info")
        return json_file
    
    def generate_master_report_with_duplicates(self):
        """Generate comprehensive master report with duplicate handling"""
        total_points = len(self.grab_point_assignments)
        assigned_points = sum(1 for a in self.grab_point_assignments.values() if a["class_name"] != "none")
        
        report_lines = []
        report_lines.append("=" * 75)
        report_lines.append(" " * 20 + "GARAGE ASSISTANT - TOOL MAPPING REPORT")
        report_lines.append("=" * 75)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Grab Points Analyzed: {total_points}")
        report_lines.append(f"Points with Assigned Tools: {assigned_points}")
        report_lines.append(f"Points without Tools: {total_points - assigned_points}")
        report_lines.append(f"Unique Tools: {len(self.tool_mapping)}")
        report_lines.append("")
        
        report_lines.append("GRAB POINT ASSIGNMENTS:")
        report_lines.append("=" * 75)
        
        for position_name, points in [("INITIAL POSITION", ["A", "B", "C", "D"]),
                                      ("SECOND POSITION", ["E", "F", "G"]),
                                      ("THIRD POSITION", ["H", "I"])]:
            report_lines.append(f"\n{position_name}:")
            report_lines.append("-" * 40)
            for point_id in points:
                if point_id in self.grab_point_assignments:
                    assignment = self.grab_point_assignments[point_id]
                    if assignment["class_name"] != "none":
                        report_lines.append(f"  Point {point_id}: {assignment['class_name'].upper()} "
                                          f"({assignment['confidence']*100:.1f}%)")
                    else:
                        report_lines.append(f"  Point {point_id}: No tool assigned")
        
        report_lines.append("\nALL TOOL LOCATIONS (INCLUDING DUPLICATES):")
        report_lines.append("-" * 40)
        
        if self.tool_mapping:
            for tool_name, locations in sorted(self.tool_mapping.items()):
                if len(locations) > 1:
                    report_lines.append(f"\n{tool_name.upper()} ({len(locations)} locations):")
                    for i, loc in enumerate(locations, 1):
                        star = "success " if i == 1 else "  "
                        position_desc = self.get_position_desc(loc["position"])
                        report_lines.append(f"  {star}Point {loc['grab_point']}: {loc['confidence']*100:.1f}% ({position_desc})")
                else:
                    report_lines.append(f"\n{tool_name.upper()} (1 location):")
                    position_desc = self.get_position_desc(locations[0]["position"])
                    report_lines.append(f"  Point {locations[0]['grab_point']}: {locations[0]['confidence']*100:.1f}% ({position_desc})")
        else:
            report_lines.append("  No tools found")
        
        report_lines.append("\nROBOT ACTION PLAN (WITH DUPLICATE HANDLING):")
        report_lines.append("-" * 40)
        
        if self.tool_mapping:
            for tool_name, locations in sorted(self.tool_mapping.items()):
                if len(locations) > 1:
                    report_lines.append(f"\nWhen user requests '{tool_name.lower()}':")
                    report_lines.append(f"  Available at {len(locations)} locations:")
                    for i, loc in enumerate(locations, 1):
                        star = " success " if i == 1 else "  "
                        position_desc = self.get_position_desc(loc["position"])
                        report_lines.append(f"  {star}Point {loc['grab_point']} ({position_desc})")
                    report_lines.append(f"  Will fetch from: Point {locations[0]['grab_point']} (highest confidence)")
                else:
                    report_lines.append(f"\nWhen user requests '{tool_name.lower()}':")
                    position_desc = self.get_position_desc(locations[0]["position"])
                    report_lines.append(f"  Will fetch from: Point {locations[0]['grab_point']} ({position_desc})")
        else:
            report_lines.append("  No action plan available (no tools mapped)")
        
        report_file = os.path.join(self.output_dir, "master_report.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.log(f"Saved master report to: {report_file}", "info")
        return report_file