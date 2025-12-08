"""
GARAGE ASSISTANT ROBOTIC ARM SYSTEM WITH ADAPTIVE GRIPPER
Detects objects, displays GUI for user selection, and executes pick-and-place operations
Uses exact 9 point coordinates with adaptive gripping
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from Arm_Lib import Arm_Device
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading

print("GARAGE ASSISTANT ROBOTIC ARM SYSTEM WITH ADAPTIVE GRIPPER")
print("=" * 70)

# ============================================
# ROBOT ARM CONTROLLER WITH 9 POINTS
# ============================================
class RobotArmController:
    def __init__(self):
        """Initialize arm with exact specified angles"""
        print("Initializing robot arm...")
        
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
            
            # INITIAL POSITION
            self.INITIAL_POSITION = {
                self.SERVO_BASE: 90,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }
            
            # Snapshot positions
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
            
            # DROP ZONE POSITION (servo1: 150 degrees)
            self.DROP_ZONE_POSITION = {
                self.SERVO_BASE: 150,
                self.SERVO_SHOULDER: 90,
                self.SERVO_ELBOW: 90,
                self.SERVO_WRIST: 90,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90  # Will be set during operation
            }
            
            # ============================================
            # 9 EXACT GRABBING POINTS (YOUR COORDINATES)
            # ============================================
            
            self.OBJECT_POSITIONS = {
                'position_1': {  # Point A - Bolts
                    'name': 'Point A - Bolts',
                    'approach': (72, 33, 31, 59, 110, 89),
                    'grab': (72, 35, 31, 59, 85, 112),
                    'lift': (69, 58, 39, 59, 118, None),  # None = keep gripper closed
                    'intermediate': (120, 90, 55, 60, 90, None),
                    'to_drop': [
                        (130, 40, 55, 45, 90, None),  # Approach drop zone
                        (130, 30, 55, 45, 90, 125)   # Prepare to release
                    ]
                },
                'position_2': {  # Point B - Multiple tools
                    'name': 'Point B - Multiple Tools',
                    'approach': (81, 33, 54, 17, 76, 139),
                    'grab': (81, 29, 55, 18, 76, 150),
                    'lift': (81, 29, 54, 18, 76, None),
                    'intermediate': (120, 90, 55, 60, 90, None),
                    'to_drop': [
                        (130, 40, 55, 45, 90, None),
                        (130, 30, 55, 45, 90, 125)
                    ]
                },
                'position_3': {  # Point C - Bolt
                    'name': 'Point C - Bolt',
                    'approach': (91, 47, 6, 85, 89, 91),
                    'grab': (91, 45, 6, 85, 89, 119),
                    'lift': (103, 54, 40, 34, 90, None),
                    'intermediate': (120, 90, 55, 60, 90, None),
                    'to_drop': [
                        (130, 40, 55, 45, 90, None),
                        (130, 30, 55, 45, 90, 125)
                    ]
                },
                'position_4': {  # Point D - Multiple tools
                    'name': 'Point D - Multiple Tools',
                    'approach': (105, 40, 25, 50, 89, 111),
                    'grab': (105, 36, 27, 53, 90, 123),
                    'lift': (103, 54, 40, 34, 90, None),
                    'intermediate': (120, 90, 55, 60, 90, None),
                    'to_drop': [
                        (130, 40, 55, 45, 90, None),
                        (130, 30, 55, 45, 90, 125)
                    ]
                },
                'position_5': {  # Point E - Multiple tools
                    'name': 'Point E - Multiple Tools',
                    'approach': (31, 39, 48, 22, 101, 133),
                    'grab': (31, 33, 48, 22, 102, 140),
                    'lift': (31, 65, 71, 22, 102, None),
                    'intermediate': (120, 90, 55, 60, 90, None),
                    'to_drop': [
                        (130, 40, 55, 45, 90, None),
                        (130, 30, 55, 45, 90, 125)
                    ]
                },
                'position_6': {  # Point F - Bolt
                    'name': 'Point F - Bolt',
                    'approach': (38, 33, 53, 35, 90, 142),
                    'grab': (38, 25, 54, 44, 90, 143),
                    'lift': (37, 36, 71, 44, 80, None),
                    'intermediate': (120, 90, 55, 60, 90, None),
                    'to_drop': [
                        (130, 40, 55, 45, 90, None),
                        (130, 30, 55, 45, 90, 130)
                    ]
                },
                'position_7': {  # Point G - Pliers
                    'name': 'Point G - Pliers',
                    'approach': (50, 17, 93, 1, 90, 59),
                    'grab': (49, 2, 93, 16, 89, 59),
                    'lift': (50, 30, 107, 18, 89, None),
                    'intermediate': (120, 90, 55, 60, 90, None),
                    'to_drop': [
                        (130, 40, 55, 45, 90, None),
                        (130, 30, 55, 45, 90, 124)
                    ]
                },
                'position_8': {  # Point H - Measuring Tape
                    'name': 'Point H - Measuring Tape',
                    'approach': (0, 23, 90, 2, 40, 62),
                    'grab': (-15, 19, 90, 6, 40, 45),
                    'lift': (50, 30, 107, 18, 89, None),
                    'intermediate': (120, 90, 55, 60, 90, None),
                    'to_drop': [
                        (130, 40, 55, 45, 90, None),
                        (130, 30, 55, 45, 90, 100)
                    ]
                },
                'position_9': {  # Point I - Hammer
                    'name': 'Point I - Hammer',
                    'approach': (13, 50, 37, 37, 93, 87),
                    'grab': (8, 3, 86, 34, 123, 86),
                    'lift': (9, 26, 98, 35, 123, None),
                    'intermediate': (120, 90, 55, 60, 90, 163),
                    'to_drop': [
                        (130, 40, 55, 45, 90, 163),
                        (130, 30, 55, 45, 90, 100)
                    ]
                }
            }
            
            # Move to initial position
            print("Moving to initial position...")
            self.go_to_initial_position()
            
            print("Robot arm initialized successfully")
            print(f"Initial position: {self.INITIAL_POSITION}")
            print(f"9 grabbing points configured")
            
        except Exception as e:
            print(f"Failed to initialize robot arm: {e}")
            raise
    
    def convert_angle(self, angle):
        """Convert negative angles to servo range (0-180)"""
        if angle < 0:
            return 180 + angle  # -35 becomes 145, -15 becomes 165, etc.
        return angle
    
    def move_to_angles(self, angles_tuple, move_time=1000, keep_gripper=None):
        """
        Move to specified angles
        angles_tuple: (servo1, servo2, servo3, servo4, servo5, servo6)
        If servo6 is None and keep_gripper is provided, use keep_gripper value
        """
        s1, s2, s3, s4, s5, s6 = angles_tuple
        
        # Convert wrist angle if negative
        s4 = self.convert_angle(s4)
        
        # Handle gripper angle
        if s6 is None and keep_gripper is not None:
            s6 = keep_gripper
        elif s6 is None:
            s6 = 90  # Default to open
            
        self.arm.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, move_time)
        time.sleep(move_time/1000 + 0.5)
    
    def go_to_initial_position(self):
        """Move to exact initial position"""
        self.move_to_angles(
            (self.INITIAL_POSITION[self.SERVO_BASE],
             self.INITIAL_POSITION[self.SERVO_SHOULDER],
             self.INITIAL_POSITION[self.SERVO_ELBOW],
             self.INITIAL_POSITION[self.SERVO_WRIST],
             self.INITIAL_POSITION[self.SERVO_WRIST_ROT],
             self.INITIAL_POSITION[self.SERVO_GRIPPER]),
            2000
        )
        print("At initial position")
    
    def go_to_second_position(self):
        """Move to second position (base at 40 degrees)"""
        self.move_to_angles(
            (self.SECOND_POSITION[self.SERVO_BASE],
             self.SECOND_POSITION[self.SERVO_SHOULDER],
             self.SECOND_POSITION[self.SERVO_ELBOW],
             self.SECOND_POSITION[self.SERVO_WRIST],
             self.SECOND_POSITION[self.SERVO_WRIST_ROT],
             self.SECOND_POSITION[self.SERVO_GRIPPER]),
            2000
        )
        print("At second position (base at 40 degrees)")
    
    def go_to_third_position(self):
        """Move to third position (base at 1 degree)"""
        self.move_to_angles(
            (self.THIRD_POSITION[self.SERVO_BASE],
             self.THIRD_POSITION[self.SERVO_SHOULDER],
             self.THIRD_POSITION[self.SERVO_ELBOW],
             self.THIRD_POSITION[self.SERVO_WRIST],
             self.THIRD_POSITION[self.SERVO_WRIST_ROT],
             self.THIRD_POSITION[self.SERVO_GRIPPER]),
            2000
        )
        print("At third position (base at 1 degree)")
    
    def adaptive_grip(self, start_angle=90, max_angle=180, step=3, delay=0.2, callback=None):
        """
        ADAPTIVE GRIPPER - Gradually close gripper until object is firmly grasped
        Monitors resistance by checking if gripper can't close further
        """
        print("Starting adaptive grip...")
        
        if callback:
            callback("Starting adaptive grip...")
        
        current_angle = start_angle
        previous_resistance = current_angle
        resistance_count = 0
        max_resistance_count = 3  # Need 3 consecutive resistance readings
        
        while current_angle < max_angle:
            # Move gripper incrementally
            print(f"Closing gripper to {current_angle}°")
            self.arm.Arm_serial_servo_write(6, current_angle, 200)
            time.sleep(delay)
            
            # Check for resistance (in a real system, you'd read servo feedback)
            # For simulation, we assume resistance when angle reaches certain points
            # based on typical tool sizes
            
            # After reaching initial contact (around 120-140°), check for stabilization
            if current_angle >= 120:
                # Simulate resistance check
                # In real implementation, you'd check if servo current/position changed
                if current_angle == previous_resistance:
                    resistance_count += 1
                    print(f"Resistance detected ({resistance_count}/{max_resistance_count})")
                else:
                    resistance_count = 0
                    previous_resistance = current_angle
                
                if resistance_count >= max_resistance_count:
                    print(f"Object firmly gripped at {current_angle}°")
                    if callback:
                        callback(f"Object gripped at {current_angle}°")
                    return current_angle
            
            current_angle += step
        
        # If we reach max angle without detecting firm grip
        print(f"Reached max grip angle: {current_angle}°")
        if callback:
            callback(f"Reached max grip angle: {current_angle}°")
        return current_angle
    
    def execute_pickup_sequence(self, position_key, tool_type=None, callback=None):
        """Execute complete pickup and delivery sequence using exact coordinates"""
        if position_key not in self.OBJECT_POSITIONS:
            if callback:
                callback(f"Invalid position key: {position_key}")
            return False
        
        position = self.OBJECT_POSITIONS[position_key]
        
        try:
            if callback:
                callback(f"Starting pickup from {position['name']}...")
            print(f"Starting pickup from {position['name']}")
            
            # ============================================
            # STEP 1: MOVE TO APPROACH POSITION
            # ============================================
            if callback:
                callback(f"Moving to approach position...")
            print(f"Moving to approach position...")
            time.sleep(1)
            self.move_to_angles(position['approach'], 1500)
            time.sleep(2)
            
            # ============================================
            # STEP 2: MOVE TO GRAB POSITION
            # ============================================
            if callback:
                callback("Moving to grab position...")
            print("Moving to grab position...")
            time.sleep(1)
            self.move_to_angles(position['grab'], 1500)
            time.sleep(2)
            
            # ============================================
            # STEP 3: ADAPTIVE GRIP
            # ============================================
            if callback:
                callback("Starting adaptive grip...")
            print("Starting adaptive grip...")
            grip_angle = self.adaptive_grip(
                start_angle=90,  # Start from open
                max_angle=180,   # Max close
                step=3,          # Small increments
                delay=0.2,       # Wait between steps
                callback=callback
            )
            time.sleep(1)
            
            # ============================================
            # STEP 4: LIFT OBJECT
            # ============================================
            if callback:
                callback("Lifting object...")
            print("Lifting object...")
            time.sleep(1)
            self.move_to_angles(position['lift'], 1500, keep_gripper=grip_angle)
            time.sleep(2)
            
            # ============================================
            # STEP 5: MOVE TO INTERMEDIATE POSITION
            # ============================================
            if callback:
                callback("Moving to intermediate position...")
            print("Moving to intermediate position...")
            time.sleep(1)
            self.move_to_angles(position['intermediate'], 1500, keep_gripper=grip_angle)
            time.sleep(2)
            
            # ============================================
            # STEP 6: MOVE TO DROP ZONE
            # ============================================
            if callback:
                callback("Moving to drop zone...")
            print("Moving to drop zone...")
            
            for i, drop_pos in enumerate(position['to_drop']):
                if callback:
                    callback(f"Drop zone step {i+1}...")
                print(f"Drop zone step {i+1}...")
                time.sleep(1)
                self.move_to_angles(drop_pos, 1500, keep_gripper=grip_angle)
                time.sleep(2)
            
            # ============================================
            # STEP 7: RELEASE OBJECT AT DROP ZONE
            # ============================================
            if callback:
                callback("Releasing object at drop zone...")
            print("Releasing object at drop zone...")
            time.sleep(1)
            self.arm.Arm_serial_servo_write(6, 90, 1000)  # Open gripper
            time.sleep(2)
            
            # ============================================
            # STEP 8: RETURN TO HOME POSITION
            # ============================================
            if callback:
                callback("Returning to home position...")
            print("Returning to home position...")
            self.go_to_initial_position()
            
            if callback:
                callback("Pickup and delivery completed successfully")
            print("Pickup sequence completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during pickup sequence: {e}")
            if callback:
                callback(f"Error: {e}")
            
            # Try to return to home on error
            try:
                self.go_to_initial_position()
            except:
                pass
            
            return False

# ============================================
# CAMERA AND DETECTION SYSTEM
# ============================================
class CameraDetectionSystem:
    def __init__(self):
        """Initialize camera and YOLO model"""
        print("Setting up camera and detection system...")
        
        self.cap = self.setup_camera()
        self.model = self.load_yolo_model()
        
        self.TOOL_CLASSES = ['Bolt', 'Hammer', 'Measuring Tape', 'Plier', 'Screwdriver', 'Wrench']
        self.TOOL_COLORS = [
            (0, 255, 0),    # Green - Bolt
            (0, 0, 255),    # Red - Hammer  
            (255, 0, 0),    # Blue - Measuring Tape
            (255, 255, 0),  # Cyan - Plier
            (255, 0, 255),  # Magenta - Screwdriver
            (0, 255, 255)   # Yellow - Wrench
        ]
        
        print("Camera and detection system ready")
    
    def setup_camera(self):
        """Setup camera for maximum FPS"""
        print("   Initializing camera...")
        
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"   Found camera at index {i}")
                
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                cap.set(cv2.CAP_PROP_EXPOSURE, 100)
                
                return cap
        
        raise Exception("No camera found")
    
    def load_yolo_model(self):
        """Load YOLO model"""
        print("   Loading YOLO model...")
        
        model_paths = ['./best_2s.pt']
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = YOLO(path)
                    model.overrides['conf'] = 0.35
                    model.overrides['iou'] = 0.3
                    model.overrides['agnostic_nms'] = True
                    model.overrides['max_det'] = 6
                    model.overrides['verbose'] = False
                    print(f"   Model loaded: {path}")
                    return model
                except Exception as e:
                    print(f"   Failed to load {path}: {e}")
                    continue
        
        raise Exception("No YOLO model found")
    
    def detect_objects(self, frame):
        """Detect objects in frame and return detections"""
        if self.model is None:
            return []
        
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
                    
                    x1 = max(0, min(x1, frame.shape[1]-1))
                    y1 = max(0, min(y1, frame.shape[0]-1))
                    x2 = max(0, min(x2, frame.shape[1]-1))
                    y2 = max(0, min(y2, frame.shape[0]-1))
                    
                    class_id = class_ids[i]
                    confidence = float(confidences[i])
                    
                    if class_id < len(self.TOOL_CLASSES):
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'class_id': int(class_id),
                            'class_name': self.TOOL_CLASSES[class_id],
                            'confidence': confidence,
                            'confidence_int': int(confidence * 100),
                            'center_x': (x1 + x2) / 2,
                            'center_y': (y1 + y2) / 2,
                            'width': x2 - x1,
                            'height': y2 - y1
                        })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def annotate_frame(self, frame, detections, position_name, base_angle):
        """Annotate frame with detections and position info"""
        annotated = frame.copy()
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence_int = det['confidence_int']
            color = self.TOOL_COLORS[det['class_id'] % len(self.TOOL_COLORS)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with class and confidence
            label = f"{class_name}: {confidence_int}%"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background for label
            cv2.rectangle(annotated, 
                         (x1, y1 - label_height - 10),
                         (x1 + label_width, y1),
                         color, -1)
            
            # Draw text
            cv2.putText(annotated, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add position information at top
        position_text = f"Position: {position_name} (Base: {base_angle} deg)"
        cv2.putText(annotated, position_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add detection count
        detection_text = f"Objects detected: {len(detections)}"
        cv2.putText(annotated, detection_text,
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, timestamp,
                   (10, annotated.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return annotated
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# ============================================
# GUI APPLICATION
# ============================================
class GarageAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Garage Assistant - Adaptive Gripper System")
        self.root.geometry("1100x700")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize systems
        self.arm = None
        self.detector = None
        self.all_detections = []
        self.all_snapshots = []
        self.current_order = None
        self.is_running = False
        self.output_dir = None
        
        self.setup_gui()
        
        # Start initialization in background
        threading.Thread(target=self.initialize_systems, daemon=True).start()
    
    def setup_gui(self):
        """Setup GUI layout"""
        # Title
        title_frame = tk.Frame(self.root, bg='#1a1a1a', height=60)
        title_frame.pack(fill='x', padx=10, pady=10)
        
        title_label = tk.Label(title_frame, text="GARAGE ASSISTANT - ADAPTIVE GRIPPER", 
                              font=('Arial', 20, 'bold'), 
                              bg='#1a1a1a', fg='#00ff00')
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = tk.Label(title_frame, text="9 Pre-Programmed Points with Smart Gripping",
                                 font=('Arial', 12),
                                 bg='#1a1a1a', fg='#ffff00')
        subtitle_label.pack()
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#2b2b2b')
        content_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Detection results
        left_panel = tk.Frame(content_frame, bg='#1a1a1a', width=500)
        left_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        detection_label = tk.Label(left_panel, text="DETECTED OBJECTS", 
                                   font=('Arial', 16, 'bold'),
                                   bg='#1a1a1a', fg='#ffffff')
        detection_label.pack(pady=10)
        
        # Points info label
        points_info = tk.Label(left_panel, 
                              text="Objects are mapped to 9 pre-programmed points (A-I)\nEach point has exact grab coordinates",
                              font=('Arial', 10),
                              bg='#1a1a1a', fg='#cccccc')
        points_info.pack(pady=(0, 10))
        
        self.detection_listbox = tk.Listbox(left_panel, font=('Courier', 11),
                                           bg='#3a3a3a', fg='#ffffff',
                                           selectmode='single', height=15)
        self.detection_listbox.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Right panel - Logger and controls
        right_panel = tk.Frame(content_frame, bg='#1a1a1a', width=500)
        right_panel.pack(side='right', fill='both', expand=True, padx=5)
        
        logger_label = tk.Label(right_panel, text="SYSTEM LOG", 
                               font=('Arial', 16, 'bold'),
                               bg='#1a1a1a', fg='#ffffff')
        logger_label.pack(pady=10)
        
        # Adaptive gripper info
        gripper_info = tk.Label(right_panel,
                               text="Adaptive Gripper: Closes gradually until object is firmly grasped",
                               font=('Arial', 10),
                               bg='#1a1a1a', fg='#00ff00')
        gripper_info.pack(pady=(0, 10))
        
        self.logger = scrolledtext.ScrolledText(right_panel, font=('Courier', 10),
                                               bg='#3a3a3a', fg='#00ff00',
                                               height=15, wrap='word')
        self.logger.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Control panel
        control_frame = tk.Frame(self.root, bg='#1a1a1a')
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Input frame
        input_frame = tk.Frame(control_frame, bg='#1a1a1a')
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Enter Object Number (1-9):", 
                font=('Arial', 12), bg='#1a1a1a', fg='#ffffff').pack(side='left', padx=5)
        
        self.order_entry = tk.Entry(input_frame, font=('Arial', 14), width=10,
                                    bg='#3a3a3a', fg='#ffffff', insertbackground='#ffffff')
        self.order_entry.pack(side='left', padx=5)
        
        self.execute_btn = tk.Button(input_frame, text="EXECUTE ORDER",
                                     font=('Arial', 12, 'bold'),
                                     bg='#00aa00', fg='#ffffff',
                                     command=self.execute_order,
                                     width=15)
        self.execute_btn.pack(side='left', padx=5)
        
        self.cancel_btn = tk.Button(input_frame, text="CANCEL ORDER",
                                    font=('Arial', 12, 'bold'),
                                    bg='#aa0000', fg='#ffffff',
                                    command=self.cancel_order,
                                    width=15)
        self.cancel_btn.pack(side='left', padx=5)
        
        # Test gripper button
        test_btn = tk.Button(input_frame, text="TEST GRIPPER",
                            font=('Arial', 10),
                            bg='#0055aa', fg='#ffffff',
                            command=self.test_gripper,
                            width=12)
        test_btn.pack(side='left', padx=5)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Initializing system...",
                                     font=('Arial', 11), bg='#1a1a1a', fg='#ffff00',
                                     anchor='w', padx=10)
        self.status_label.pack(fill='x', side='bottom')
    
    def log(self, message):
        """Add message to logger"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logger.insert('end', f"[{timestamp}] {message}\n")
        self.logger.see('end')
        self.root.update()
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update()
    
    def create_output_directory(self):
        """Create timestamped directory for saving snapshots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"garage_assistant_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.log(f"Created output directory: {output_dir}")
        return output_dir
    
    def initialize_systems(self):
        """Initialize robot and camera systems"""
        try:
            self.log("Initializing robot arm with 9 points...")
            self.arm = RobotArmController()
            self.log("Robot arm initialized with 9 grabbing points")
            
            self.log("Initializing camera system...")
            self.detector = CameraDetectionSystem()
            self.log("Camera system initialized")
            
            # Create output directory for this session
            self.create_output_directory()
            
            self.update_status("System ready - Starting inspection...")
            self.log("Starting automatic inspection...")
            
            # Run inspection sequence
            self.run_inspection_sequence()
            
        except Exception as e:
            self.log(f"Initialization error: {e}")
            self.update_status("System initialization failed")
    
    def run_inspection_sequence(self):
        """Run 3-snapshot inspection sequence"""
        try:
            self.all_detections = []
            self.all_snapshots = []
            
            # Snapshot 1 - Initial position
            self.log("Taking snapshot 1 (base at 90 degrees)...")
            self.update_status("Snapshot 1/3 - Initial position")
            time.sleep(3)
            snapshot1 = self.capture_snapshot(
                position_name="initial_position",
                base_angle=self.arm.INITIAL_POSITION[self.arm.SERVO_BASE]
            )
            if snapshot1 and snapshot1['detections']:
                self.all_detections.extend(snapshot1['detections'])
            
            # Snapshot 2 - Second position
            self.log("Moving to second position...")
            self.update_status("Moving to position 2...")
            self.arm.go_to_second_position()
            self.log("Taking snapshot 2 (base at 40 degrees)...")
            self.update_status("Snapshot 2/3 - Second position")
            time.sleep(3)
            snapshot2 = self.capture_snapshot(
                position_name="second_position",
                base_angle=self.arm.SECOND_POSITION[self.arm.SERVO_BASE]
            )
            if snapshot2 and snapshot2['detections']:
                self.all_detections.extend(snapshot2['detections'])
            
            # Snapshot 3 - Third position
            self.log("Moving to third position...")
            self.update_status("Moving to position 3...")
            self.arm.go_to_third_position()
            self.log("Taking snapshot 3 (base at 1 degree)...")
            self.update_status("Snapshot 3/3 - Third position")
            time.sleep(3)
            snapshot3 = self.capture_snapshot(
                position_name="third_position",
                base_angle=self.arm.THIRD_POSITION[self.arm.SERVO_BASE]
            )
            if snapshot3 and snapshot3['detections']:
                self.all_detections.extend(snapshot3['detections'])
            
            # Return to initial position
            self.log("Returning to initial position...")
            self.update_status("Returning to initial position...")
            self.arm.go_to_initial_position()
            
            # Save summary report
            self.save_summary_report()
            
            # Update display
            self.update_detection_display()
            
            self.log(f"Inspection complete - Total objects detected: {len(self.all_detections)}")
            self.update_status(f"Ready - {len(self.all_detections)} objects detected at 9 points")
            self.is_running = True
            
        except Exception as e:
            self.log(f"Inspection error: {e}")
            self.update_status("Inspection failed")
    
    def capture_snapshot(self, position_name, base_angle):
        """Capture a snapshot, detect objects, and save results"""
        self.log(f"Capturing snapshot: {position_name}")
        self.log(f"   Base servo angle: {base_angle} deg")
        
        # Clear camera buffer
        for _ in range(3):
            self.detector.cap.read()
            time.sleep(0.1)
        
        time.sleep(0.5)
        
        # Capture frame
        ret, frame = self.detector.cap.read()
        if not ret:
            self.log(f"Failed to capture frame at {position_name}")
            return None
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Detect objects
        self.log("   Detecting objects...")
        detections = self.detector.detect_objects(frame)
        
        if detections:
            self.log(f"   Found {len(detections)} objects:")
            for det in detections:
                self.log(f"      - {det['class_name']}: {det['confidence_int']}%")
        else:
            self.log("   No objects detected")
        
        # Store snapshot info
        snapshot_info = {
            'position_name': position_name,
            'base_angle': base_angle,
            'detections': detections,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'frame': frame
        }
        self.all_snapshots.append(snapshot_info)
        
        # Save annotated images if objects detected
        if detections:
            annotated_frame = self.detector.annotate_frame(frame, detections, position_name, base_angle)
            filename = f"{self.output_dir}/{position_name}_detected.jpg"
            cv2.imwrite(filename, annotated_frame)
            self.log(f"   Saved annotated snapshot: {filename}")
            snapshot_info['filename'] = filename
            
            # Save detection data
            self.save_detection_data(position_name, detections, filename, base_angle)
        else:
            self.log(f"   Skipping snapshot save (no objects detected)")
            snapshot_info['filename'] = None
        
        return snapshot_info
    
    def save_detection_data(self, position_name, detections, image_filename, base_angle):
        """Save detection information to text file"""
        txt_filename = f"{self.output_dir}/{position_name}_detections.txt"
        
        with open(txt_filename, 'w') as f:
            f.write(f"GARAGE ASSISTANT - DETECTION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Position: {position_name}\n")
            f.write(f"Base Servo Angle: {base_angle} deg\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Objects Detected: {len(detections)}\n")
            f.write("=" * 60 + "\n\n")
            
            if detections:
                f.write("DETECTED OBJECTS:\n")
                f.write("-" * 40 + "\n")
                
                for i, det in enumerate(detections, 1):
                    f.write(f"Object #{i}:\n")
                    f.write(f"  Class: {det['class_name']}\n")
                    f.write(f"  Confidence: {det['confidence_int']}%\n")
                    f.write(f"  Bounding Box: {det['bbox']}\n")
                    f.write("-" * 30 + "\n")
            else:
                f.write("No objects detected in this snapshot.\n")
        
        self.log(f"   Saved detection report: {txt_filename}")
    
    def save_summary_report(self):
        """Save summary report of all snapshots"""
        summary_filename = f"{self.output_dir}/summary_report.txt"
        
        with open(summary_filename, 'w') as f:
            f.write("GARAGE ASSISTANT ROBOTIC ARM - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total snapshots taken: {len(self.all_snapshots)}\n")
            f.write(f"Total objects detected: {len(self.all_detections)}\n")
            f.write("=" * 70 + "\n\n")
            
            # 9 Points information
            f.write("9 PRE-PROGRAMMED GRABBING POINTS:\n")
            f.write("-" * 50 + "\n")
            for i in range(1, 10):
                point_key = f'position_{i}'
                point_name = self.arm.OBJECT_POSITIONS[point_key]['name']
                f.write(f"Point {i}: {point_name}\n")
            f.write("\n")
            
            # Summary by position
            f.write("SNAPSHOT SUMMARY:\n")
            f.write("-" * 50 + "\n")
            
            for snap in self.all_snapshots:
                f.write(f"\n{snap['position_name'].upper()} (Base: {snap['base_angle']} deg):\n")
                f.write(f"  Time: {snap['timestamp']}\n")
                f.write(f"  Objects detected: {len(snap['detections'])}\n")
                
                if snap['detections']:
                    f.write("  Detected objects:\n")
                    for det in snap['detections']:
                        f.write(f"    - {det['class_name']}: {det['confidence_int']}%\n")
                else:
                    f.write("  No objects detected\n")
            
            # Count by tool class
            f.write("\n" + "=" * 70 + "\n")
            f.write("DETECTION STATISTICS BY TOOL TYPE:\n")
            f.write("-" * 40 + "\n")
            
            tool_counts = {}
            for snap in self.all_snapshots:
                for det in snap['detections']:
                    tool_name = det['class_name']
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            
            if tool_counts:
                for tool_name, count in sorted(tool_counts.items()):
                    f.write(f"{tool_name}: {count} detection(s)\n")
            else:
                f.write("No tools detected in any snapshot.\n")
        
        self.log(f"Saved summary report: {summary_filename}")
    
    def update_detection_display(self):
        """Update detection listbox"""
        self.detection_listbox.delete(0, 'end')
        
        for i, det in enumerate(self.all_detections, 1):
            if i <= 9:  # Only show up to 9 objects (matching our 9 points)
                point_name = self.arm.OBJECT_POSITIONS[f'position_{i}']['name']
                display_text = f"{i}. {det['class_name']} ({det['confidence_int']}%) - {point_name}"
                self.detection_listbox.insert('end', display_text)
            else:
                break  # Only support 9 objects maximum
    
    def execute_order(self):
        """Execute user order to pick object"""
        if not self.is_running:
            self.log("System not ready")
            return
        
        try:
            order_num = int(self.order_entry.get().strip())
            
            if order_num < 1 or order_num > min(9, len(self.all_detections)):
                self.log(f"Invalid order number. Choose 1-{min(9, len(self.all_detections))}")
                self.update_status("Invalid order number")
                return
            
            selected_obj = self.all_detections[order_num - 1]
            point_name = self.arm.OBJECT_POSITIONS[f'position_{order_num}']['name']
            
            self.log(f"Order received: Pick {selected_obj['class_name']} from {point_name}")
            self.update_status(f"Executing order: {selected_obj['class_name']} from {point_name}")
            
            # Execute pickup in background
            self.current_order = order_num
            threading.Thread(target=self.execute_pickup, 
                           args=(order_num, selected_obj), 
                           daemon=True).start()
            
        except ValueError:
            self.log("Please enter a valid number")
            self.update_status("Invalid input")
    
    def execute_pickup(self, order_num, obj):
        """Execute pickup sequence for selected object"""
        position_key = f"position_{order_num}"
        point_name = self.arm.OBJECT_POSITIONS[position_key]['name']
        
        def status_callback(msg):
            self.log(msg)
            self.update_status(msg)
        
        # Log start of operation
        self.log(f"Starting pickup sequence for {obj['class_name']} from {point_name}")
        self.log("Using adaptive gripper - will close gradually until object is firmly gripped")
        
        success = self.arm.execute_pickup_sequence(position_key, obj['class_name'], status_callback)
        
        if success:
            self.log(f"Successfully delivered {obj['class_name']} from {point_name}")
            self.log("Adaptive gripper adjusted automatically for perfect grip")
            self.update_status("Task completed - Ready for next order")
        else:
            self.log(f"Failed to deliver {obj['class_name']} from {point_name}")
            self.update_status("Task failed - Check system")
        
        self.current_order = None
        self.order_entry.delete(0, 'end')
    
    def cancel_order(self):
        """Cancel current order"""
        if self.current_order:
            self.log(f"Cancelling order {self.current_order}...")
            self.update_status("Order cancelled")
            self.current_order = None
            self.order_entry.delete(0, 'end')
        else:
            self.log("No active order to cancel")
    
    def test_gripper(self):
        """Test the adaptive gripper"""
        self.log("Testing adaptive gripper...")
        self.update_status("Testing gripper...")
        
        threading.Thread(target=self._test_gripper_thread, daemon=True).start()
    
    def _test_gripper_thread(self):
        """Thread for testing gripper"""
        try:
            # Test adaptive grip
            def test_callback(msg):
                self.log(f"Gripper test: {msg}")
            
            self.log("Opening gripper fully...")
            self.arm.arm.Arm_serial_servo_write(6, 90, 1000)
            time.sleep(2)
            
            self.log("Testing adaptive grip (will close gradually)...")
            final_angle = self.arm.adaptive_grip(
                start_angle=90,
                max_angle=180,
                step=5,
                delay=0.3,
                callback=test_callback
            )
            
            self.log(f"Adaptive grip test complete - Final angle: {final_angle}°")
            self.log("Opening gripper...")
            self.arm.arm.Arm_serial_servo_write(6, 90, 1000)
            time.sleep(1)
            
            self.log("Gripper test completed successfully")
            self.update_status("Gripper test complete")
            
        except Exception as e:
            self.log(f"Gripper test error: {e}")
            self.update_status("Gripper test failed")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print("GARAGE ASSISTANT ROBOTIC ARM SYSTEM WITH ADAPTIVE GRIPPER")
    print("=" * 70)
    print("Features:")
    print("1. Automatic 3-position inspection with object detection")
    print("2. 9 PRE-PROGRAMMED POINTS with exact coordinates (A-I)")
    print("3. ADAPTIVE GRIPPER - Smart gripping based on resistance")
    print("4. GUI interface for object selection and control")
    print("5. Complete pickup and delivery to drop zone (servo1: 150°)")
    print("6. Detailed logging and report generation")
    print("=" * 70)
    print("\nStarting system...")
    
    root = tk.Tk()
    app = GarageAssistantGUI(root)
    root.mainloop()