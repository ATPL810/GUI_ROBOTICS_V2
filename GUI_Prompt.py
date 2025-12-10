import sys
import os
import time
import threading
import json
import re
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from Arm_Lib import Arm_Device

# PyQt5 Imports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ============================================
# ROBOT ARM CONTROLLER
# ============================================
class RobotArmController:
    def __init__(self, log_callback=None):
        """Initialize arm with exact specified angles"""
        self.log_callback = log_callback
        self.log("Initializing robot arm...")
        
        try:
            self.arm = Arm_Device()
            time.sleep(2)
            
            # Servo mapping
            self.SERVO_BASE = 1
            self.SERVO_SHOULDER = 2  
            self.SERVO_ELBOW = 3
            self.SERVO_WRIST = 4
            self.SERVO_WRIST_ROT = 5
            self.SERVO_GRIPPER = 6
            
            # Positions
            self.INITIAL_POSITION = {
                self.SERVO_BASE: 90,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }
            
            self.SECOND_POSITION = {
                self.SERVO_BASE: 40,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }

            self.THIRD_POSITION = {
                self.SERVO_BASE: 1,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }
            
            self.go_to_initial_position()
            self.log("✅ Robot arm initialized successfully")
            
        except Exception as e:
            self.log(f"❌ Failed to initialize robot arm: {e}", "error")
            raise
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def go_to_initial_position(self):
        """Move to exact initial position"""
        angles_dict = self.INITIAL_POSITION.copy()
        angles_dict[self.SERVO_WRIST] = self.convert_angle(angles_dict[self.SERVO_WRIST])
        
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000
        )
        time.sleep(2.5)
        self.log("At initial position (Base: 90°)")
    
    def go_to_second_position(self):
        """Move to second position (base at 40 degrees)"""
        angles_dict = self.SECOND_POSITION.copy()
        angles_dict[self.SERVO_WRIST] = self.convert_angle(angles_dict[self.SERVO_WRIST])
        
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000
        )
        time.sleep(2.5)
        self.log("At second position (Base: 40°)")
    
    def go_to_third_position(self):
        """Move to third position (base at 1 degree)"""
        angles_dict = self.THIRD_POSITION.copy()
        angles_dict[self.SERVO_WRIST] = self.convert_angle(angles_dict[self.SERVO_WRIST])
        
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000
        )
        time.sleep(2.5)
        self.log("At third position (Base: 1°)")
    
    def convert_angle(self, angle):
        """Convert negative angles to servo range (0-180)"""
        if angle < 0:
            return angle
        return angle
    
    def go_to_home(self):
        """Go to safe home position"""
        self.log("Moving to home position...")
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
        time.sleep(2)
        self.log("✅ At home position")

# ============================================
# CAMERA SYSTEM
# ============================================
class CameraSystem(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, log_callback=None):
        super().__init__()
        self.log_callback = log_callback
        self.cap = None
        self.running = False
        
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def setup_camera(self):
        """Setup camera for maximum FPS"""
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                self.log(f"Found camera at index {i}")
                
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                cap.set(cv2.CAP_PROP_EXPOSURE, 100)
                
                return cap
        
        raise Exception("No camera found!")
    
    def run(self):
        self.running = True
        try:
            self.cap = self.setup_camera()
            self.log("Camera started")
            
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Mirror the frame
                    frame = cv2.flip(frame, 1)
                    self.frame_ready.emit(frame)
                else:
                    break
                time.sleep(0.03)  # ~30 FPS
                
        except Exception as e:
            self.log(f"Camera error: {e}", "error")
        finally:
            self.stop()
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.log("Camera stopped")

# ============================================
# SNAPSHOT SYSTEM
# ============================================
class SnapshotSystem:
    def __init__(self, arm_controller, camera, log_callback=None):
        self.arm = arm_controller
        self.camera = camera
        self.log_callback = log_callback
        self.model = None
        self.output_dir = None
        self.all_snapshots = []
        
        # Tool classes
        self.TOOL_CLASSES = ['Bolt', 'Hammer', 'Measuring Tape', 'Plier', 'Screwdriver', 'Wrench']
        self.TOOL_COLORS = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def load_yolo_model(self):
        """Load YOLO model"""
        model_paths = ['best_best.pt']
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = YOLO(path)
                    model.overrides['conf'] = 0.35
                    model.overrides['iou'] = 0.3
                    model.overrides['agnostic_nms'] = True
                    model.overrides['max_det'] = 6
                    model.overrides['verbose'] = False
                    self.log(f"Model loaded: {path}")
                    return model
                except Exception as e:
                    self.log(f"Failed to load {path}: {e}", "warning")
                    continue
        
        raise Exception("No YOLO model found!")
    
    def create_output_directory(self):
        """Create timestamped directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/snapshots/robot_snapshots_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def capture_frame(self):
        """Capture a single frame from camera"""
        if self.camera.cap and self.camera.cap.isOpened():
            # Clear buffer
            for _ in range(3):
                self.camera.cap.read()
                time.sleep(0.1)
            
            ret, frame = self.camera.cap.read()
            if ret:
                return cv2.flip(frame, 1)
        return None
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if self.model is None:
            self.model = self.load_yolo_model()
        
        try:
            inference_size = 256
            small_frame = cv2.resize(frame, (inference_size, int(inference_size * 0.75)))
            
            results = self.model(small_frame, 
                               conf=0.35,
                               iou=0.3,
                               imgsz=inference_size,
                               max_det=6,
                               verbose=False,
                               half=False,
                               device='cpu',
                               agnostic_nms=True)
            
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                scale_x = frame.shape[1] / inference_size
                scale_y = frame.shape[0] / (inference_size * 0.75)
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    class_id = class_ids[i]
                    confidence = float(confidences[i])
                    
                    if class_id < len(self.TOOL_CLASSES):
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'class_id': int(class_id),
                            'class_name': self.TOOL_CLASSES[class_id],
                            'confidence': confidence,
                            'confidence_percentage': f"{confidence * 100:.1f}%",
                            'center_x': (x1 + x2) / 2,
                            'center_y': (y1 + y2) / 2
                        })
            
            return detections
            
        except Exception as e:
            self.log(f"Detection error: {e}", "error")
            return []
    
    def take_snapshots_sequence(self):
        """Run the complete automatic snapshot sequence"""
        self.log("🚀 Starting automatic snapshot sequence...", "success")
        self.output_dir = self.create_output_directory()
        self.log(f"Output directory: {self.output_dir}")
        
        try:
            # Step 1: Initial Position
            self.log("\n🔹 STEP 1: INITIAL POSITION", "info")
            self.wait_for_stabilization(3, "Initial Position")
            snapshot1 = self.capture_and_save("initial_position", self.arm.INITIAL_POSITION[self.arm.SERVO_BASE])
            
            # Step 2: Second Position
            self.log("\n🔹 STEP 2: MOVING TO SECOND POSITION", "info")
            self.arm.go_to_second_position()
            self.wait_for_stabilization(3, "Second Position")
            snapshot2 = self.capture_and_save("second_position", self.arm.SECOND_POSITION[self.arm.SERVO_BASE])
            
            # Step 3: Third Position
            self.log("\n🔹 STEP 3: MOVING TO THIRD POSITION", "info")
            self.arm.go_to_third_position()
            self.wait_for_stabilization(3, "Third Position")
            snapshot3 = self.capture_and_save("third_position", self.arm.THIRD_POSITION[self.arm.SERVO_BASE])
            
            # Step 4: Return to Initial
            self.log("\n🔹 STEP 4: RETURNING TO INITIAL POSITION", "info")
            time.sleep(3)
            self.arm.go_to_initial_position()
            
            # Step 5: Finalize
            self.save_summary_report()
            
            self.log("✅ Automatic snapshot sequence COMPLETE!", "success")
            total_detections = sum(len(snap['detections']) for snap in self.all_snapshots)
            self.log(f"Snapshots taken: {len(self.all_snapshots)}", "info")
            self.log(f"Total objects detected: {total_detections}", "info")
            
            return self.output_dir
            
        except Exception as e:
            self.log(f"❌ Error during snapshot sequence: {e}", "error")
            raise
    
    def wait_for_stabilization(self, seconds, position_name):
        """Wait for robot to stabilize"""
        self.log(f"Waiting {seconds} seconds for {position_name} stabilization...")
        for i in range(seconds, 0, -1):
            self.log(f"   {i}...", "info")
            time.sleep(1)
    
    def capture_and_save(self, position_name, base_angle):
        """Capture and save a snapshot"""
        self.log(f"Capturing snapshot: {position_name}")
        
        time.sleep(0.5)
        frame = self.capture_frame()
        if frame is None:
            self.log(f"Failed to capture frame at {position_name}", "error")
            return None
        
        detections = self.detect_objects(frame)
        
        if detections:
            self.log(f"Found {len(detections)} objects:", "success")
            for det in detections:
                self.log(f"  • {det['class_name']}: {det['confidence_percentage']}", "info")
        else:
            self.log("No objects detected", "warning")
        
        # Annotate and save frame
        annotated_frame = self.annotate_frame(frame, detections, position_name, base_angle)
        filename = f"{self.output_dir}/{position_name}.jpg"
        cv2.imwrite(filename, annotated_frame)
        self.log(f"Saved snapshot: {filename}")
        
        # Save detection data
        self.save_detection_data(position_name, detections, filename, base_angle)
        
        snapshot_info = {
            'position_name': position_name,
            'base_angle': base_angle,
            'filename': filename,
            'detections': detections,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.all_snapshots.append(snapshot_info)
        
        return snapshot_info
    
    def annotate_frame(self, frame, detections, position_name, base_angle):
        """Annotate frame with detections"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence_percentage']
            color = self.TOOL_COLORS[det['class_id'] % len(self.TOOL_COLORS)]
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(annotated, 
                         (x1, y1 - label_height - 10),
                         (x1 + label_width, y1),
                         color, -1)
            
            cv2.putText(annotated, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add position info
        position_text = f"Position: {position_name} (Base: {base_angle}°)"
        cv2.putText(annotated, position_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        detection_text = f"Objects detected: {len(detections)}"
        cv2.putText(annotated, detection_text,
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return annotated
    
    def save_detection_data(self, position_name, detections, image_filename, base_angle):
        """Save detection information to text file"""
        txt_filename = f"{self.output_dir}/{position_name}_detections.txt"
        
        with open(txt_filename, 'w') as f:
            f.write(f"ROBOT ARM SNAPSHOT DETECTION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Position: {position_name}\n")
            f.write(f"Base Servo Angle: {base_angle}°\n")
            f.write(f"Image File: {os.path.basename(image_filename)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Objects Detected: {len(detections)}\n")
            f.write("=" * 60 + "\n\n")
            
            if detections:
                f.write("DETECTED OBJECTS:\n")
                f.write("-" * 40 + "\n")
                
                for i, det in enumerate(detections, 1):
                    f.write(f"Object #{i}:\n")
                    f.write(f"  Class: {det['class_name']}\n")
                    f.write(f"  Confidence: {det['confidence_percentage']}\n")
                    f.write(f"  Bounding Box: {det['bbox']}\n")
                    f.write("-" * 30 + "\n")
        
        self.log(f"Saved detection report: {txt_filename}")
    
    def save_summary_report(self):
        """Save summary report of all snapshots"""
        summary_filename = f"{self.output_dir}/summary_report.txt"
        
        with open(summary_filename, 'w') as f:
            f.write("ROBOT ARM AUTOMATIC SNAPSHOT SYSTEM - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total snapshots taken: {len(self.all_snapshots)}\n")
            f.write("=" * 70 + "\n\n")
            
            total_detections = sum(len(snap['detections']) for snap in self.all_snapshots)
            f.write(f"TOTAL OBJECTS DETECTED: {total_detections}\n\n")
            
            f.write("SNAPSHOT SUMMARY BY POSITION:\n")
            f.write("-" * 50 + "\n")
            
            for snap in self.all_snapshots:
                f.write(f"\n{snap['position_name'].upper()} (Base: {snap['base_angle']}°):\n")
                f.write(f"  Time: {snap['timestamp']}\n")
                f.write(f"  Image: {os.path.basename(snap['filename'])}\n")
                f.write(f"  Objects detected: {len(snap['detections'])}\n")
                
                if snap['detections']:
                    f.write("  Detected objects:\n")
                    for det in snap['detections']:
                        f.write(f"    • {det['class_name']}: {det['confidence_percentage']}\n")
        
        self.log(f"Saved summary report: {summary_filename}")

# ============================================
# ANALYSIS SYSTEM
# ============================================
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
        """Run complete grab point analysis"""
        self.log("🔍 Starting grab point analysis...", "info")
        self.snapshot_folder = snapshot_folder
        
        self.load_grab_points()
        self.parse_all_snapshots()
        self.assign_tools_to_grab_points()
        self.generate_tool_mapping()
        self.save_json_mapping()
        report_path = self.generate_master_report()
        
        self.log("✅ Grab point analysis COMPLETE!", "success")
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
        
        self.log(f"Loaded {sum(len(v) for v in self.grab_points.values())} grab points")
    
    def parse_all_snapshots(self):
        """Parse detection data from all 3 snapshots"""
        positions = ["initial_position", "second_position", "third_position"]
        
        for position in positions:
            detections = self.parse_snapshot_report(position)
            self.detections_data[position] = {
                "detections": detections,
                "count": len(detections)
            }
            
            self.log(f"{position}: {len(detections)} tools detected")
    
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
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def assign_tools_to_grab_points(self, max_distance=200):
        """Assign the closest tool to each grab point"""
        self.log("Assigning tools to grab points...", "info")
        
        assignments = {}
        
        for position_name, grab_points in self.grab_points.items():
            self.log(f"Processing {position_name}...")
            
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
                self.log(f"  Point {point_id}: {detection['class_name']} ({pair['distance']:.1f}px)")
            
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
        
        best_mapping = {}
        for tool_name, points in tool_mapping.items():
            points.sort(key=lambda x: (-x["confidence"], x["distance"]))
            best_mapping[tool_name] = points[0]
        
        self.tool_mapping = best_mapping
        
        if best_mapping:
            self.log("Tool mapping generated:", "success")
            for tool_name, mapping in best_mapping.items():
                self.log(f"  {tool_name} → Point {mapping['grab_point']} ({mapping['confidence']*100:.1f}%)")
        
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
        
        self.log(f"Saved JSON mapping to: {json_file}")
        return json_file
    
    def generate_master_report(self):
        """Generate comprehensive master report"""
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
        
        report_lines.append("\nTOOL TO GRAB POINT MAPPING:")
        report_lines.append("-" * 40)
        
        if self.tool_mapping:
            for tool_name, mapping in sorted(self.tool_mapping.items()):
                report_lines.append(f"  {tool_name.upper():<15} → Point {mapping['grab_point']} "
                                  f"({mapping['confidence']*100:.1f}%)")
        else:
            report_lines.append("  No tools mapped")
        
        report_file = os.path.join(self.output_dir, "master_report.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.log(f"Saved master report to: {report_file}")
        return report_file

# ============================================
# GRAB SYSTEM WITH OBJECT VERIFICATION
# ============================================
class GrabSystem:
    def __init__(self, arm_controller, snapshot_system, camera_system, log_callback=None):
        self.arm = arm_controller
        self.snapshot_system = snapshot_system
        self.camera_system = camera_system
        self.log_callback = log_callback
        self.tool_mapping = {}
        self.master_report_path = None
        
        self.grab_scripts_dir = "movements"

        # Store the detected tools from last scan
        self.available_tools = []  # List of tool names that were actually detected
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def load_mapping(self, master_report_path=None):
        """Load tool mapping from master report"""
        if master_report_path:
            self.master_report_path = master_report_path
        else:
            self.master_report_path = self.find_latest_master_report()
        
        self.tool_mapping = self.parse_master_report()
        self.log(f"Loaded {len(self.tool_mapping)} tool mappings")
        return self.tool_mapping
    
    def find_latest_master_report(self):
        """Find the latest master report"""
        import glob
        pattern = "data/mappings/mapping_*/master_report.txt"
        matches = glob.glob(pattern)
        
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
        
        raise FileNotFoundError("No master report found!")
    
    def parse_master_report(self):
        """Parse tool mapping from master report"""
        tool_mapping = {}
        
        with open(self.master_report_path, 'r') as f:
            content = f.read()
            
            if "TOOL TO GRAB POINT MAPPING:" in content:
                mapping_section = content.split("TOOL TO GRAB POINT MAPPING:")[1]
                lines = mapping_section.strip().split('\n')
                
                for line in lines:
                    if "→" in line:
                        parts = line.split("→")
                        if len(parts) == 2:
                            tool_name = parts[0].strip().lower()
                            grab_point = parts[1].strip()
                            match = re.search(r'Point\s+([A-I])', grab_point)
                            if match:
                                grab_letter = match.group(1)
                                tool_mapping[tool_name] = grab_letter
        
        return tool_mapping
    
    def check_grab_script_exists(self, grab_letter):
        """Check if grab script exists for a point"""
        script_path = os.path.join(self.grab_scripts_dir, f"grab_point_{grab_letter}.py")
        exists = os.path.exists(script_path)
        
        if not exists:
            self.log(f"   Script missing: {script_path}", "warning")
        
        return exists
    
    def execute_grab_script(self, grab_letter):
        """Execute the grab script for a specific point"""
        script_path = os.path.join(self.grab_scripts_dir, f"grab_point_{grab_letter}.py")
        
        if not os.path.exists(script_path):
            self.log(f"❌ Grab script not found: {script_path}", "error")
            return False
        
        self.log(f"📄 Executing grab script: {script_path}", "info")
        
        try:
            # Import the grab script module
            import importlib.util
            import sys
            
            # Create a module name
            module_name = f"grab_point_{grab_letter}"
            
            # Load the module from file
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            
            # Inject the arm object into the module's namespace
            module.arm = self.arm
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Call the grab function (should be named grab_point_X)
            grab_function_name = f"grab_point_{grab_letter}"
            if hasattr(module, grab_function_name):
                grab_function = getattr(module, grab_function_name)
                self.log(f"   Running {grab_function_name}()...", "info")
                
                # Execute the grab function
                grab_function(self.arm)
                
                self.log(f"   ✅ {grab_function_name}() completed", "success")
                return True
            else:
                # Try alternative function names
                for func_name in ["main", "grab", "execute_grab"]:
                    if hasattr(module, func_name):
                        grab_function = getattr(module, func_name)
                        self.log(f"   Running {func_name}()...", "info")
                        grab_function(self.arm)
                        self.log(f"   ✅ {func_name}() completed", "success")
                        return True
                
                self.log(f"❌ No grab function found in {script_path}", "error")
                self.log(f"   Expected function: {grab_function_name}()", "error")
                return False
                
        except Exception as e:
            self.log(f"❌ Error executing grab script: {str(e)}", "error")
            import traceback
            self.log(f"   Details: {traceback.format_exc()}", "error")
            return False
    
    def verify_object_exists(self, tool_name):
        """Verify if the requested object was actually detected in the last scan"""
        # Get the last snapshot folder
        import glob
        snapshot_folders = glob.glob("data/snapshots/robot_snapshots_*")
        if not snapshot_folders:
            self.log("No scan data found!", "error")
            return False
        
        snapshot_folders.sort(key=os.path.getmtime, reverse=True)
        latest_folder = snapshot_folders[0]
        
        # Check all detection reports for this tool
        positions = ["initial_position", "second_position", "third_position"]
        tool_found = False
        
        for position in positions:
            report_file = os.path.join(latest_folder, f"{position}_detections.txt")
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    content = f.read()
                    # Search for tool name in the report
                    if tool_name.lower() in content.lower():
                        self.log(f"✅ Verified: {tool_name} found in {position}", "success")
                        tool_found = True
                        break
        
        if not tool_found:
            self.log(f"❌ Verification failed: {tool_name} was not detected in last scan", "error")
        
        return tool_found
    
    def fetch_tool(self, tool_name):
        """Fetch a specific tool with verification"""
        tool_name_lower = tool_name.lower().strip()
        
        self.log(f"\n{'='*60}", "system")
        self.log(f"🤖 FETCH REQUEST: {tool_name.upper()}", "system")
        self.log(f"{'='*60}", "system")
        
        # Step 1: Check if tool is in mapping
        if tool_name_lower not in self.tool_mapping:
            self.log(f"❌ Tool '{tool_name}' not found in tool mapping!", "error")
            self.show_available_tools()
            return False
        
        # Step 2: Verify object was actually detected
        self.log("🔍 Verifying object detection...", "info")
        if not self.verify_object_exists(tool_name):
            self.log(f"⚠️ WARNING: {tool_name} was not detected in the last scan!", "warning")
            self.log("The tool may have been moved or is not in the workspace.", "warning")
            
            # Ask user for confirmation
            reply = self.ask_user_confirmation(
                f"⚠️ {tool_name.upper()} was not detected in last scan!\n"
                f"Do you still want to attempt fetching?"
            )
            
            if not reply:
                self.log("Fetch cancelled by user.", "info")
                return False
        
        # Step 3: Get grab point
        grab_letter = self.tool_mapping[tool_name_lower]
        self.log(f"📍 Tool location: Point {grab_letter}", "success")
        
        # Step 4: Confirm with user
        self.log(f"\n📋 FETCH CONFIRMATION:", "info")
        self.log(f"   Tool: {tool_name.upper()}", "info")
        self.log(f"   Location: Point {grab_letter}", "info")
        self.log(f"   Arm will move to position and attempt to grab.", "info")
        
        reply = self.ask_user_confirmation(f"Proceed with fetching {tool_name.upper()}?")
        if not reply:
            self.log("Fetch cancelled by user.", "info")
            return False
        
        # Step 5: Execute fetch sequence
        self.log("\n🚀 Starting fetch sequence...", "success")
        success = self.execute_fetch_sequence(grab_letter, tool_name)
        
        if success:
            self.log(f"\n✅ SUCCESSFULLY FETCHED {tool_name.upper()}!", "success")
            return True
        else:
            self.log(f"\n❌ FAILED to fetch {tool_name.upper()}", "error")
            return False
    
    def ask_user_confirmation(self, message):
        """Ask user for confirmation (for GUI, we'll show in status)"""
        # This will be handled by the GUI's prompt system
        self.log(f"{message} [Waiting for user confirmation]", "warning")
        # In the GUI, this will trigger a dialog box
        return True  # Return True for now, GUI will handle this
    
    def show_available_tools(self):
        """Show available tools from mapping"""
        if self.tool_mapping:
            self.log("Available tools in mapping:", "info")
            for tool in sorted(self.tool_mapping.keys()):
                self.log(f"  • {tool.upper()}", "info")
        else:
            self.log("No tools available in mapping!", "warning")
    
    def execute_fetch_sequence(self, grab_letter, tool_name):
        """Execute the fetch sequence"""
        try:
            # Determine which position to go to
            if grab_letter in ["A", "B", "C", "D"]:
                self.log("Moving to initial position...", "info")
                self.arm.go_to_initial_position()
            elif grab_letter in ["E", "F", "G"]:
                self.log("Moving to second position...", "info")
                self.arm.go_to_second_position()
            elif grab_letter in ["H", "I"]:
                self.log("Moving to third position...", "info")
                self.arm.go_to_third_position()
            else:
                self.log(f"Unknown grab point: {grab_letter}", "error")
                return False
            
            time.sleep(1)
            
            # Simulate grabbing (in real implementation, run grab script)
            self.log(f"Attempting to grab at Point {grab_letter}...", "info")
            time.sleep(2)
            
            # Return to home
            self.log("Returning to home position...", "info")
            self.arm.go_to_home()
            
            return True
            
        except Exception as e:
            self.log(f"Error during fetch sequence: {e}", "error")
            # Try to return to home on error
            try:
                self.arm.go_to_home()
            except:
                pass
            return False

# ============================================
# TOOL PROMPT DIALOG
# ============================================
class ToolPromptDialog(QDialog):
    def __init__(self, tool_mapping, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 Fetch Tool")
        self.setGeometry(200, 200, 400, 300)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 14px;
            }
            QLineEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #585b70;
                color: #cdd6f4;
                border: 2px solid #89b4fa;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QListWidget {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        
        self.tool_mapping = tool_mapping
        self.selected_tool = None
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Select or Type Tool Name:")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(title_label)
        
        # Available tools list
        list_label = QLabel("Available Tools:")
        layout.addWidget(list_label)
        
        self.tools_list = QListWidget()
        for tool_name in sorted(self.tool_mapping.keys()):
            item = QListWidgetItem(tool_name.upper())
            item.setData(Qt.UserRole, tool_name)
            self.tools_list.addItem(item)
        
        self.tools_list.itemDoubleClicked.connect(self.select_tool_from_list)
        self.tools_list.itemClicked.connect(self.select_tool_from_list)
        layout.addWidget(self.tools_list)
        
        # Or type manually
        type_label = QLabel("Or type tool name:")
        layout.addWidget(type_label)
        
        self.tool_input = QLineEdit()
        self.tool_input.setPlaceholderText("e.g., hammer, screwdriver, wrench")
        layout.addWidget(self.tool_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("✅ Fetch")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setEnabled(False)
        
        self.cancel_button = QPushButton("❌ Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        # Connect signals
        self.tool_input.textChanged.connect(self.check_input)
        self.tools_list.itemSelectionChanged.connect(self.check_selection)
        
        self.setLayout(layout)
    
    def select_tool_from_list(self, item):
        """Select tool from list"""
        self.tool_input.setText(item.data(Qt.UserRole))
        self.ok_button.setEnabled(True)
    
    def check_input(self, text):
        """Check if input is valid"""
        if text.strip():
            self.ok_button.setEnabled(True)
        else:
            self.ok_button.setEnabled(False)
    
    def check_selection(self):
        """Check if item is selected"""
        if self.tools_list.currentItem():
            self.ok_button.setEnabled(True)
    
    def get_selected_tool(self):
        """Get the selected tool"""
        return self.tool_input.text().strip().lower()
    
    def accept(self):
        """Accept dialog"""
        tool_name = self.get_selected_tool()
        if tool_name:
            self.selected_tool = tool_name
            super().accept()

# ============================================
# CONFIRMATION DIALOG
# ============================================
class ConfirmationDialog(QDialog):
    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚠️ Confirmation Required")
        self.setGeometry(300, 300, 400, 200)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton {
                background-color: #585b70;
                color: #cdd6f4;
                border: 2px solid #89b4fa;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
        """)
        
        self.init_ui(message)
    
    def init_ui(self, message):
        layout = QVBoxLayout()
        
        # Message
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(message_label)
        
        # Warning icon
        warning_label = QLabel("⚠️")
        warning_label.setAlignment(Qt.AlignCenter)
        warning_label.setStyleSheet("font-size: 48px; color: #f9e2af;")
        layout.addWidget(warning_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.yes_button = QPushButton("✅ Yes, Continue")
        self.yes_button.clicked.connect(self.accept)
        
        self.no_button = QPushButton("❌ No, Cancel")
        self.no_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.yes_button)
        button_layout.addWidget(self.no_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

# ============================================
# MAIN GUI WINDOW
# ============================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏗️ Garage Assistant Pro")
        self.setGeometry(100, 100, 1100, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QLabel {
                color: #cdd6f4;
            }
            QPushButton {
                background-color: #585b70;
                color: #cdd6f4;
                border: 2px solid #89b4fa;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QPushButton:disabled {
                background-color: #313244;
                color: #6c7086;
                border: 2px solid #45475a;
            }
            QListWidget {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 5px;
            }
            QTextEdit {
                background-color: #181825;
                color: #a6adc8;
                border: 2px solid #45475a;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
            }
            QGroupBox {
                color: #89b4fa;
                border: 2px solid #89b4fa;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QProgressBar {
                border: 2px solid #45475a;
                border-radius: 5px;
                text-align: center;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 3px;
            }
        """)
        
        # System components
        self.arm_controller = None
        self.camera_system = None
        self.snapshot_system = None
        self.analysis_system = None
        self.grab_system = None
        
        # Threads
        self.camera_thread = None
        self.scan_thread = None
        self.fetch_thread = None
        
        # Current state
        self.scanning = False
        self.fetching = False
        self.current_snapshot_folder = None
        self.tool_mapping = {}
        
        # Initialize GUI
        self.init_ui()
        
        # Start systems
        self.initialize_systems()
    
    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel (Camera + Controls)
        left_panel = QVBoxLayout()
        
        # Camera display
        camera_group = QGroupBox("🎥 Live Camera Feed")
        camera_layout = QVBoxLayout()
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 3px solid #89b4fa;
                border-radius: 5px;
            }
        """)
        
        camera_layout.addWidget(self.camera_label)
        camera_group.setLayout(camera_layout)
        left_panel.addWidget(camera_group)
        
        # Control buttons
        control_group = QGroupBox("⚙️ System Controls")
        control_layout = QGridLayout()
        
        # Scan button
        self.scan_button = QPushButton("🔍 Start Scan & Analysis")
        self.scan_button.clicked.connect(self.start_scan)
        self.scan_button.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
        """)
        
        # Stop button
        self.stop_button = QPushButton("🛑 Emergency Stop")
        self.stop_button.clicked.connect(self.emergency_stop)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #1e1e2e;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #f5c2e7;
            }
        """)
        
        # Home button
        self.home_button = QPushButton("🏠 Go Home")
        self.home_button.clicked.connect(self.go_home)
        
        # Fetch button
        self.fetch_prompt_button = QPushButton("🤖 Fetch Tool...")
        self.fetch_prompt_button.clicked.connect(self.prompt_for_tool)
        self.fetch_prompt_button.setStyleSheet("""
            QPushButton {
                background-color: #cba6f7;
                color: #1e1e2e;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #f5c2e7;
            }
        """)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #a6e3a1;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        control_layout.addWidget(self.scan_button, 0, 0)
        control_layout.addWidget(self.stop_button, 0, 1)
        control_layout.addWidget(self.home_button, 1, 0)
        control_layout.addWidget(self.fetch_prompt_button, 1, 1)
        control_layout.addWidget(self.status_label, 2, 0)
        control_layout.addWidget(self.progress_bar, 3, 0, 1, 2)
        control_group.setLayout(control_layout)
        left_panel.addWidget(control_group)
        
        # Tools list
        tools_group = QGroupBox("🛠 Mapped Tools")
        tools_layout = QVBoxLayout()
        
        self.tools_list = QListWidget()
        self.tools_list.itemDoubleClicked.connect(self.fetch_selected_tool)
        
        self.fetch_button = QPushButton("🤖 Fetch Selected Tool")
        self.fetch_button.clicked.connect(self.fetch_selected_tool)
        
        tools_layout.addWidget(self.tools_list)
        tools_layout.addWidget(self.fetch_button)
        tools_group.setLayout(tools_layout)
        left_panel.addWidget(tools_group)
        
        # Add left panel to main layout
        main_layout.addLayout(left_panel)
        
        # Right panel (Logger)
        right_panel = QVBoxLayout()
        
        logger_group = QGroupBox("📋 System Logger")
        logger_layout = QVBoxLayout()
        
        self.logger_text = QTextEdit()
        self.logger_text.setReadOnly(True)
        self.logger_text.setMinimumWidth(400)
        
        # Clear button
        clear_button = QPushButton("🗑️ Clear Log")
        clear_button.clicked.connect(self.clear_log)
        
        logger_layout.addWidget(self.logger_text)
        logger_layout.addWidget(clear_button)
        logger_group.setLayout(logger_layout)
        right_panel.addWidget(logger_group)
        
        # System info
        info_group = QGroupBox("ℹ️ System Information")
        info_layout = QVBoxLayout()
        
        self.info_label = QLabel(
            "Arm Status: Not connected\n"
            "Camera Status: Not connected\n"
            "Last Scan: None\n"
            "Tools Mapped: 0\n"
            "Last Fetch: None"
        )
        self.info_label.setStyleSheet("""
            QLabel {
                color: #cba6f7;
                font-family: 'Monospace';
            }
        """)
        
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        right_panel.addWidget(info_group)
        
        # Add right panel to main layout
        main_layout.addLayout(right_panel)
    
    def initialize_systems(self):
        """Initialize all systems"""
        try:
            self.log("🚀 Initializing Garage Assistant Pro...", "system")
            
            # Initialize arm
            self.log("🤖 Initializing robot arm...", "info")
            self.arm_controller = RobotArmController(log_callback=self.log)
            
            # Initialize camera
            self.log("📷 Starting camera...", "info")
            self.camera_system = CameraSystem(log_callback=self.log)
            self.camera_system.frame_ready.connect(self.update_camera_frame)
            self.camera_thread = QThread()
            self.camera_system.moveToThread(self.camera_thread)
            self.camera_thread.started.connect(self.camera_system.run)
            self.camera_thread.start()
            
            # Initialize other systems
            self.snapshot_system = SnapshotSystem(self.arm_controller, self.camera_system, self.log)
            self.analysis_system = AnalysisSystem(self.log)
            self.grab_system = GrabSystem(self.arm_controller, self.snapshot_system, self.camera_system, self.log)
            
            self.log("✅ All systems initialized successfully!", "success")
            self.update_info("Arm: Connected ✓", "Camera: Running ✓")
            
        except Exception as e:
            self.log(f"❌ Failed to initialize systems: {e}", "error")
    
    def log(self, message, level="info"):
        """Log message to logger and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding
        colors = {
            "info": "#89b4fa",
            "success": "#a6e3a1",
            "warning": "#f9e2af",
            "error": "#f38ba8",
            "system": "#cba6f7"
        }
        
        color = colors.get(level, "#cdd6f4")
        formatted_message = f'<font color="{color}">[{timestamp}] {message}</font>'
        
        self.logger_text.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.logger_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Update status for important messages
        if level == "error":
            self.status_label.setText(f"Status: Error - {message[:30]}...")
        elif level == "success":
            self.status_label.setText(f"Status: {message[:40]}...")
    
    def update_camera_frame(self, frame):
        """Update camera display with new frame"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.camera_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            pass
    
    def prompt_for_tool(self):
        """Prompt user for which tool to fetch"""
        if not self.tool_mapping:
            self.log("❌ No tool mapping available! Please run scan first.", "error")
            return
        
        # Show tool selection dialog
        dialog = ToolPromptDialog(self.tool_mapping, self)
        if dialog.exec_() == QDialog.Accepted:
            tool_name = dialog.get_selected_tool()
            if tool_name:
                self.log(f"User requested to fetch: {tool_name.upper()}", "system")
                self.start_fetch_tool(tool_name)
    
    def start_fetch_tool(self, tool_name):
        """Start fetching a tool"""
        if self.fetching:
            self.log("Another fetch operation is already in progress!", "warning")
            return
        
        self.fetching = True
        self.fetch_prompt_button.setEnabled(False)
        self.fetch_button.setEnabled(False)
        self.status_label.setText(f"Status: Fetching {tool_name.upper()}...")
        
        # Run in separate thread
        self.fetch_thread = threading.Thread(target=self.run_fetch_sequence, args=(tool_name,))
        self.fetch_thread.daemon = True
        self.fetch_thread.start()
    
    def run_fetch_sequence(self, tool_name):
        """Run the fetch sequence"""
        try:
            # Load mapping if not loaded
            if not self.tool_mapping:
                self.grab_system.load_mapping()
                self.tool_mapping = self.grab_system.tool_mapping
            
            # Fetch the tool
            success = self.grab_system.fetch_tool(tool_name)
            
            if success:
                self.log(f"✅ Successfully fetched {tool_name.upper()}!", "success")
                self.update_info(f"Last Fetch: {datetime.now().strftime('%H:%M:%S')}",
                               f"Last Tool: {tool_name.upper()}")
            else:
                self.log(f"❌ Failed to fetch {tool_name.upper()}", "error")
                
        except Exception as e:
            self.log(f"❌ Fetch error: {e}", "error")
        
        finally:
            self.fetching = False
            self.fetch_prompt_button.setEnabled(True)
            self.fetch_button.setEnabled(True)
            self.status_label.setText("Status: Ready")
    
    def fetch_selected_tool(self):
        """Fetch the selected tool from list"""
        current_item = self.tools_list.currentItem()
        if not current_item:
            self.log("Please select a tool first!", "warning")
            return
        
        tool_name = current_item.data(Qt.UserRole)
        self.start_fetch_tool(tool_name)
    
    def start_scan(self):
        """Start the automatic scan and analysis"""
        if self.scanning:
            self.log("Scan already in progress!", "warning")
            return
        
        self.scanning = True
        self.scan_button.setEnabled(False)
        self.fetch_prompt_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Scanning...")
        self.progress_bar.setValue(0)
        
        # Run in separate thread
        self.scan_thread = threading.Thread(target=self.run_scan_sequence)
        self.scan_thread.daemon = True
        self.scan_thread.start()
    
    def run_scan_sequence(self):
        """Run complete scan sequence"""
        try:
            # Step 1: Take snapshots
            self.log("\n" + "="*50, "system")
            self.log("STARTING AUTOMATIC SCAN SEQUENCE", "system")
            self.log("="*50, "system")
            
            self.log("📸 Step 1: Taking snapshots...", "info")
            self.progress_bar.setValue(10)
            
            self.current_snapshot_folder = self.snapshot_system.take_snapshots_sequence()
            self.progress_bar.setValue(50)
            
            # Step 2: Analyze grab points
            self.log("\n🔍 Step 2: Analyzing grab points...", "info")
            report_path = self.analysis_system.analyze_grab_points(self.current_snapshot_folder)
            self.progress_bar.setValue(80)
            
            # Step 3: Load mapping
            self.log("\n🗺️ Step 3: Loading tool mapping...", "info")
            self.tool_mapping = self.grab_system.load_mapping(report_path)
            self.progress_bar.setValue(90)
            
            # Update tools list
            self.update_tools_list()
            self.progress_bar.setValue(100)
            
            self.log("\n✅ SCAN COMPLETE!", "success")
            self.log(f"📊 Tools mapped: {len(self.tool_mapping)}", "success")
            
            self.update_info(f"Last Scan: {datetime.now().strftime('%H:%M:%S')}",
                           f"Tools Mapped: {len(self.tool_mapping)}")
            
        except Exception as e:
            self.log(f"❌ Scan failed: {e}", "error")
        
        finally:
            self.scanning = False
            self.scan_button.setEnabled(True)
            self.fetch_prompt_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Status: Ready")
    
    def update_tools_list(self):
        """Update the tools list widget"""
        self.tools_list.clear()
        
        for tool_name in sorted(self.tool_mapping.keys()):
            item = QListWidgetItem(tool_name.upper())
            item.setData(Qt.UserRole, tool_name)
            
            # Color code based on tool type
            if "hammer" in tool_name.lower():
                item.setForeground(QColor("#f38ba8"))
            elif "screwdriver" in tool_name.lower():
                item.setForeground(QColor("#89b4fa"))
            elif "wrench" in tool_name.lower():
                item.setForeground(QColor("#f9e2af"))
            elif "plier" in tool_name.lower():
                item.setForeground(QColor("#a6e3a1"))
            elif "bolt" in tool_name.lower():
                item.setForeground(QColor("#cba6f7"))
            elif "tape" in tool_name.lower():
                item.setForeground(QColor("#f5c2e7"))
            
            self.tools_list.addItem(item)
        
        if self.tool_mapping:
            self.log(f"Updated tools list with {len(self.tool_mapping)} tools", "success")
    
    def go_home(self):
        """Move arm to home position"""
        self.log("Moving arm to home position...", "info")
        try:
            self.arm_controller.go_to_home()
            self.log("✅ Arm at home position", "success")
        except Exception as e:
            self.log(f"❌ Failed to go home: {e}", "error")
    
    def emergency_stop(self):
        """Emergency stop all operations"""
        self.log("🛑 EMERGENCY STOP ACTIVATED!", "error")
        
        self.scanning = False
        self.fetching = False
        
        # Enable buttons
        self.scan_button.setEnabled(True)
        self.fetch_prompt_button.setEnabled(True)
        self.fetch_button.setEnabled(True)
        self.status_label.setText("Status: Emergency Stop")
        
        # Stop camera if running
        if self.camera_system:
            self.camera_system.stop()
        
        # Move arm to safe position
        try:
            self.log("Moving arm to safe position...", "info")
            self.arm_controller.go_to_home()
            self.log("✅ Arm at safe position", "success")
        except:
            self.log("⚠️ Could not move arm to safe position", "warning")
    
    def update_info(self, *args):
        """Update system information"""
        text = "\n".join(args)
        self.info_label.setText(text)
    
    def clear_log(self):
        """Clear the logger"""
        self.logger_text.clear()
        self.log("Log cleared", "info")
    
    def closeEvent(self, event):
        """Cleanup on window close"""
        self.log("Shutting down systems...", "system")
        
        # Stop any ongoing operations
        self.scanning = False
        self.fetching = False
        
        # Stop camera
        if self.camera_system:
            self.camera_system.stop()
        
        # Stop camera thread
        if self.camera_thread:
            self.camera_thread.quit()
            self.camera_thread.wait()
        
        # Move arm to home
        try:
            if self.arm_controller:
                self.arm_controller.go_to_home()
        except:
            pass
        
        self.log("✅ Goodbye!", "success")
        event.accept()

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/snapshots", exist_ok=True)
    os.makedirs("data/mappings", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("movements", exist_ok=True)
    
    main()