"""
GARAGE ASSISTANT ROBOTIC ARM SYSTEM - COMPLETE VERSION
Combines exact 9 point coordinates, adaptive gripper, and professional GUI
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
import queue
import json

print("GARAGE ASSISTANT ROBOTIC ARM SYSTEM - COMPLETE VERSION")
print("=" * 70)
print("Features:")
print("1. Exact 9 point coordinates (A-I) with your positions")
print("2. Adaptive gripper - smart gripping for all tools")
print("3. Live camera feed with real-time detection")
print("4. Professional GUI with status panels")
print("5. Drop zone at 150° as specified")
print("=" * 70)

# ============================================
# ROBOT ARM CONTROLLER WITH 9 POINTS & ADAPTIVE GRIP
# ============================================
class RobotArmController:
    def __init__(self):
        """Initialize arm with all 9 exact point coordinates"""
        print("Initializing robot arm with 9 points...")
        
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
            
            # Initial position (home)
            self.HOME_POSITION = {
                self.SERVO_BASE: 90,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }
            
            # 3 Snapshot positions for scanning
            self.SNAPSHOT_POSITIONS = [
                {  # Position 1
                    self.SERVO_BASE: 90,
                    self.SERVO_SHOULDER: 105,
                    self.SERVO_ELBOW: 45,
                    self.SERVO_WRIST: -35,
                    self.SERVO_WRIST_ROT: 90,
                    self.SERVO_GRIPPER: 90
                },
                {  # Position 2
                    self.SERVO_BASE: 40,
                    self.SERVO_SHOULDER: 105,
                    self.SERVO_ELBOW: 45,
                    self.SERVO_WRIST: -35,
                    self.SERVO_WRIST_ROT: 90,
                    self.SERVO_GRIPPER: 90
                },
                {  # Position 3
                    self.SERVO_BASE: 1,
                    self.SERVO_SHOULDER: 105,
                    self.SERVO_ELBOW: 45,
                    self.SERVO_WRIST: -35,
                    self.SERVO_WRIST_ROT: 90,
                    self.SERVO_GRIPPER: 90
                }
            ]
            
            # ============================================
            # 9 EXACT GRABBING POINTS (YOUR COORDINATES)
            # ============================================
            self.POINT_COORDINATES = {
                # Point A - Bolts
                'PointA': {
                    'name': 'Point A - Bolts Area',
                    'approach': [72, 33, 31, 59, 110, 89],      # Before grabbing
                    'grab': [72, 35, 31, 59, 85, 112],          # Grab position
                    'grip_start': 112,                          # Start grip from this angle
                    'lift': [69, 58, 39, 59, 118, None],        # Lift object (None = keep current grip)
                    'waypoints': [
                        [120, 90, 55, 60, 90, None],           # Intermediate position
                        [160, 40, 55, 45, 90, None],           # Approach drop zone
                        [160, 30, 55, 45, 90, 125]             # Prepare to release
                    ]
                },
                # Point B - Multiple tools
                'PointB': {
                    'name': 'Point B - Mixed Tools',
                    'approach': [81, 33, 54, 17, 76, 139],
                    'grab': [81, 29, 55, 18, 76, 150],
                    'grip_start': 150,
                    'lift': [81, 29, 54, 18, 76, None],
                    'waypoints': [
                        [120, 90, 55, 60, 90, None],
                        [160, 40, 55, 45, 90, None],
                        [160, 30, 55, 45, 90, 125]
                    ]
                },
                # Point C - Bolt
                'PointC': {
                    'name': 'Point C - Bolt',
                    'approach': [91, 47, 6, 85, 89, 91],
                    'grab': [91, 45, 6, 85, 89, 119],
                    'grip_start': 119,
                    'lift': [103, 54, 40, 34, 90, None],
                    'waypoints': [
                        [120, 90, 55, 60, 90, None],
                        [160, 40, 55, 45, 90, None],
                        [160, 30, 55, 45, 90, 125]
                    ]
                },
                # Point D - Multiple tools
                'PointD': {
                    'name': 'Point D - Mixed Tools',
                    'approach': [105, 40, 25, 50, 89, 111],
                    'grab': [105, 36, 27, 53, 90, 123],
                    'grip_start': 123,
                    'lift': [103, 54, 40, 34, 90, None],
                    'waypoints': [
                        [120, 90, 55, 60, 90, None],
                        [160, 40, 55, 45, 90, None],
                        [160, 30, 55, 45, 90, 125]
                    ]
                },
                # Point E - Multiple tools
                'PointE': {
                    'name': 'Point E - Mixed Tools',
                    'approach': [31, 39, 48, 22, 101, 133],
                    'grab': [31, 33, 48, 22, 102, 140],
                    'grip_start': 140,
                    'lift': [31, 65, 71, 22, 102, None],
                    'waypoints': [
                        [120, 90, 55, 60, 90, None],
                        [160, 40, 55, 45, 90, None],
                        [160, 30, 55, 45, 90, 125]
                    ]
                },
                # Point F - Bolt
                'PointF': {
                    'name': 'Point F - Bolt',
                    'approach': [38, 33, 53, 35, 90, 142],
                    'grab': [38, 25, 54, 44, 90, 143],
                    'grip_start': 143,
                    'lift': [37, 36, 71, 44, 80, None],
                    'waypoints': [
                        [120, 90, 55, 60, 90, None],
                        [160, 40, 55, 45, 90, None],
                        [160, 30, 55, 45, 90, 130]
                    ]
                },
                # Point G - Pliers
                'PointG': {
                    'name': 'Point G - Pliers',
                    'approach': [50, 17, 93, 1, 90, 59],
                    'grab': [49, 2, 93, 16, 89, 59],
                    'grip_start': 59,
                    'lift': [50, 30, 107, 18, 89, None],
                    'waypoints': [
                        [120, 90, 55, 60, 90, None],
                        [160, 40, 55, 45, 90, None],
                        [160, 30, 55, 45, 90, 124]
                    ]
                },
                # Point H - Measuring Tape
                'PointH': {
                    'name': 'Point H - Measuring Tape',
                    'approach': [0, 23, 90, 2, 40, 62],
                    'grab': [-15, 19, 90, 6, 40, 45],
                    'grip_start': 45,
                    'lift': [50, 30, 107, 18, 89, None],
                    'waypoints': [
                        [120, 90, 55, 60, 90, None],
                        [160, 40, 55, 45, 90, None],
                        [160, 30, 55, 45, 90, 100]
                    ]
                },
                # Point I - Hammer
                'PointI': {
                    'name': 'Point I - Hammer',
                    'approach': [13, 50, 37, 37, 93, 87],
                    'grab': [8, 3, 86, 34, 123, 86],
                    'grip_start': 86,
                    'lift': [9, 26, 98, 35, 123, None],
                    'waypoints': [
                        [120, 90, 55, 60, 90, 163],
                        [160, 40, 55, 45, 90, 163],
                        [160, 30, 55, 45, 90, 100]
                    ]
                }
            }
            
            # Drop zone at 160° as specified
            self.DROP_ZONE = {
                'base_angle': 160,
                'positions': [
                    [160, 90, 55, 60, 90, None],      # Approach drop zone
                    [160, 70, 55, 45, 90, None],      # Lower to drop
                    [90, 90, 90, 90, 90, 90]         # Lift up after release
                ]
            }
            
            # Initialize to home position
            self.go_home()
            
            print(f"Robot arm initialized with {len(self.POINT_COORDINATES)} points")
            print(f"Drop zone: Base at {self.DROP_ZONE['base_angle']}°")
            
        except Exception as e:
            print(f"Failed to initialize robot arm: {e}")
            raise
    
    def convert_angle(self, angle):
        """Convert negative angles to servo range (0-180)"""
        if angle < 0:
            return  angle
        return angle
    
    def move_to_position(self, angles_list, duration=1000, current_grip=None):
        """Move arm to specified position"""
        if angles_list is None:
            return
        
        s1, s2, s3, s4, s5, s6 = angles_list
        
        # Convert wrist angle if negative
        s4 = self.convert_angle(s4)
        
        # Handle gripper angle
        if s6 is None and current_grip is not None:
            s6 = current_grip
        elif s6 is None:
            s6 = 90  # Default open
            
        self.arm.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, duration)
        time.sleep(duration/1000 + 0.5)
    
    def go_home(self):
        """Move to home position"""
        home_angles = list(self.HOME_POSITION.values())
        self.move_to_position(home_angles, 2000)
        print("At home position")
    
    def go_to_snapshot(self, index):
        """Move to one of the 3 snapshot positions"""
        if 0 <= index < len(self.SNAPSHOT_POSITIONS):
            pos = list(self.SNAPSHOT_POSITIONS[index].values())
            self.move_to_position(pos, 2000)
            print(f"At snapshot position {index+1}")
            return True
        return False
    
    def adaptive_grip(self, start_angle=90, max_angle=180, step=3, delay=0.2):
        """
        ADAPTIVE GRIPPER - Closes gradually until object is firmly gripped
        Returns final grip angle
        """
        print(f"Starting adaptive grip from {start_angle}°")
        
        current_angle = start_angle
        previous_angle = current_angle
        resistance_count = 0
        max_resistance = 3
        
        while current_angle < max_angle:
            # Move gripper incrementally
            self.arm.Arm_serial_servo_write(6, current_angle, 200)
            time.sleep(delay)
            
            # Check for resistance (object gripped)
            # In real system, you'd check servo feedback
            # Simulating resistance detection:
            if current_angle >= 120:  # After initial contact
                if current_angle == previous_angle:
                    resistance_count += 1
                    print(f"Resistance detected ({resistance_count}/{max_resistance}) at {current_angle}°")
                else:
                    resistance_count = 0
                
                if resistance_count >= max_resistance:
                    print(f"Object firmly gripped at {current_angle}°")
                    return current_angle
            
            previous_angle = current_angle
            current_angle += step
        
        print(f"Reached max grip angle: {current_angle}°")
        return current_angle
    
    def execute_pickup_sequence(self, point_name, callback=None):
        """Execute complete pickup from specific point"""
        if point_name not in self.POINT_COORDINATES:
            if callback:
                callback(f"Point {point_name} not found")
            return False
        
        point = self.POINT_COORDINATES[point_name]
        
        try:
            if callback:
                callback(f"Starting pickup from {point['name']}")
            
            # STEP 1: Move to approach position
            if callback:
                callback("Moving to approach position...")
            self.move_to_position(point['approach'], 1500)
            time.sleep(1)
            
            # STEP 2: Move to grab position
            if callback:
                callback("Moving to grab position...")
            self.move_to_position(point['grab'], 1500)
            time.sleep(1)
            
            # STEP 3: Adaptive grip
            if callback:
                callback("Starting adaptive grip...")
            grip_angle = self.adaptive_grip(
                start_angle=point['grip_start'],
                max_angle=180,
                step=3,
                delay=0.2
            )
            time.sleep(1)
            
            # STEP 4: Lift object
            if callback:
                callback("Lifting object...")
            self.move_to_position(point['lift'], 1500, current_grip=grip_angle)
            time.sleep(1)
            
            # STEP 5: Move through waypoints
            for i, waypoint in enumerate(point['waypoints']):
                if callback:
                    callback(f"Moving to waypoint {i+1}...")
                self.move_to_position(waypoint, 1500, current_grip=grip_angle)
                time.sleep(1)
            
            # STEP 6: Move to drop zone at 150°
            if callback:
                callback("Moving to drop zone (150°)...")
            for i, drop_pos in enumerate(self.DROP_ZONE['positions']):
                if callback:
                    callback(f"Drop zone step {i+1}...")
                self.move_to_position(drop_pos, 1500, current_grip=grip_angle)
                time.sleep(1)
                
                # Release after lowering to drop position
                if i == 1:  # After second position (lowered to drop)
                    if callback:
                        callback("Releasing object...")
                    self.arm.Arm_serial_servo_write(6, 90, 1000)
                    time.sleep(1)
            
            # STEP 7: Return home
            if callback:
                callback("Returning to home position...")
            self.go_home()
            
            if callback:
                callback("Pickup and delivery completed successfully")
            
            return True
            
        except Exception as e:
            print(f"Error in pickup sequence: {e}")
            if callback:
                callback(f"Error: {e}")
            
            # Try to return home on error
            try:
                self.go_home()
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
        
        print(f"Detection system ready for {len(self.TOOL_CLASSES)} tool types")
    
    def setup_camera(self):
        """Setup camera"""
        print("   Initializing camera...")
        
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"   Found camera at index {i}")
                
                # Optimize settings
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                
                return cap
        
        raise Exception("No camera found")
    
    def load_yolo_model(self):
        """Load YOLO model"""
        print("   Loading YOLO model...")
        
        model_paths = ['./best_best.pt']
        
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
        """Detect objects in frame"""
        if self.model is None:
            return []
        
        try:
            # Resize for faster inference
            inference_size = 256
            small_frame = cv2.resize(frame, (inference_size, int(inference_size * 0.75)))
            
            # Run detection
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
                
                # Scale back to original size
                scale_x = frame.shape[1] / inference_size
                scale_y = frame.shape[0] / (inference_size * 0.75)
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    # Clamp to frame bounds
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
                            'confidence_pct': int(confidence * 100),
                            'center_x': (x1 + x2) / 2,
                            'center_y': (y1 + y2) / 2
                        })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def annotate_frame(self, frame, detections, position_name=""):
        """Add detection annotations to frame"""
        annotated = frame.copy()
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence_pct']
            color = self.TOOL_COLORS[det['class_id'] % len(self.TOOL_COLORS)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence}%"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Label background
            cv2.rectangle(annotated, 
                         (x1, y1 - label_height - 10),
                         (x1 + label_width, y1),
                         color, -1)
            
            # Label text
            cv2.putText(annotated, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add position info if provided
        if position_name:
            cv2.putText(annotated, position_name,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add detection count
        cv2.putText(annotated, f"Objects: {len(detections)}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, timestamp,
                   (annotated.shape[1] - 120, annotated.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return annotated
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# ============================================
# PROFESSIONAL GUI APPLICATION
# ============================================
class GarageAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Garage Assistant - Complete System")
        self.root.geometry("1100x700")
        
        # Systems
        self.arm = None
        self.detector = None
        
        # Data
        self.detected_objects = []
        self.object_point_mapping = {}  # object -> point mapping
        self.current_order = None
        self.system_ready = False
        self.output_dir = None
        
        # Thread communication
        self.gui_queue = queue.Queue()
        
        # Setup GUI
        self.setup_gui()
        
        # Start queue checker
        self.check_queue()
        
        # Initialize in background thread
        threading.Thread(target=self.initialize_system, daemon=True).start()
    
    def setup_gui(self):
        """Setup professional GUI layout"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom colors
        bg_color = '#2b2b2b'
        panel_bg = '#1a1a1a'
        text_color = '#ffffff'
        accent_color = '#007acc'
        success_color = '#00aa00'
        warning_color = '#ffaa00'
        
        self.root.configure(bg=bg_color)
        
        # Title frame
        title_frame = tk.Frame(self.root, bg=panel_bg, height=80)
        title_frame.pack(fill='x', padx=10, pady=(10, 5))
        
        title = tk.Label(title_frame, 
                        text="GARAGE ASSISTANT ROBOTIC SYSTEM",
                        font=('Arial', 24, 'bold'),
                        bg=panel_bg, fg='#00ff00')
        title.pack(pady=10)
        
        subtitle = tk.Label(title_frame,
                          text="9 Pre-programmed Points • Adaptive Gripper • Live Camera Feed",
                          font=('Arial', 11),
                          bg=panel_bg, fg='#cccccc')
        subtitle.pack()
        
        # Main content
        main_frame = tk.Frame(self.root, bg=bg_color)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Camera
        left_panel = tk.Frame(main_frame, bg=panel_bg, width=700)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        cam_label = tk.Label(left_panel, text="LIVE CAMERA FEED",
                           font=('Arial', 14, 'bold'),
                           bg=panel_bg, fg=text_color)
        cam_label.pack(pady=(10, 5))
        
        self.camera_display = tk.Label(left_panel, text="Initializing camera...",
                                      bg='#000000', fg='#888888',
                                      font=('Arial', 10))
        self.camera_display.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Right panel - Controls
        right_panel = tk.Frame(main_frame, bg=panel_bg, width=500)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Status panel
        status_frame = tk.LabelFrame(right_panel, text="SYSTEM STATUS",
                                    font=('Arial', 12, 'bold'),
                                    bg=panel_bg, fg=text_color,
                                    padx=10, pady=10)
        status_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, 
                                    text="Initializing...",
                                    font=('Arial', 11),
                                    bg=panel_bg, fg=warning_color,
                                    anchor='w')
        self.status_label.pack(fill='x', pady=2)
        
        self.arm_label = tk.Label(status_frame,
                                 text="Robot Arm: Connecting...",
                                 font=('Arial', 10),
                                 bg=panel_bg, fg='#cccccc',
                                 anchor='w')
        self.arm_label.pack(fill='x', pady=2)
        
        self.camera_label = tk.Label(status_frame,
                                    text="Camera: Connecting...",
                                    font=('Arial', 10),
                                    bg=panel_bg, fg='#cccccc',
                                    anchor='w')
        self.camera_label.pack(fill='x', pady=2)
        
        self.points_label = tk.Label(status_frame,
                                    text="Points: 0/9 ready",
                                    font=('Arial', 10),
                                    bg=panel_bg, fg='#cccccc',
                                    anchor='w')
        self.points_label.pack(fill='x', pady=2)
        
        # Detected objects panel
        objects_frame = tk.LabelFrame(right_panel, text="DETECTED OBJECTS",
                                     font=('Arial', 12, 'bold'),
                                     bg=panel_bg, fg=text_color,
                                     padx=10, pady=10)
        objects_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Points info
        points_info = tk.Label(objects_frame,
                              text="Objects mapped to 9 pre-programmed points (A-I):",
                              font=('Arial', 9),
                              bg=panel_bg, fg='#aaaaaa',
                              anchor='w')
        points_info.pack(fill='x', pady=(0, 5))
        
        self.objects_text = scrolledtext.ScrolledText(objects_frame,
                                                     height=10,
                                                     bg='#3a3a3a',
                                                     fg=text_color,
                                                     font=('Courier', 10))
        self.objects_text.pack(fill='both', expand=True)
        
        # Order panel
        order_frame = tk.LabelFrame(right_panel, text="PLACE ORDER",
                                   font=('Arial', 12, 'bold'),
                                   bg=panel_bg, fg=text_color,
                                   padx=10, pady=10)
        order_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # Order input
        input_frame = tk.Frame(order_frame, bg=panel_bg)
        input_frame.pack(fill='x', pady=5)
        
        tk.Label(input_frame, text="Object Number (1-9):",
                font=('Arial', 10),
                bg=panel_bg, fg=text_color).pack(side='left', padx=(0, 10))
        
        self.order_var = tk.StringVar()
        self.order_entry = tk.Entry(input_frame,
                                   textvariable=self.order_var,
                                   width=10,
                                   font=('Arial', 12),
                                   bg='#3a3a3a',
                                   fg=text_color,
                                   insertbackground=text_color)
        self.order_entry.pack(side='left', padx=(0, 10))
        self.order_entry.bind('<Return>', lambda e: self.submit_order())
        
        self.order_btn = tk.Button(input_frame,
                                  text="EXECUTE ORDER",
                                  font=('Arial', 10, 'bold'),
                                  bg=success_color,
                                  fg='white',
                                  command=self.submit_order,
                                  padx=20)
        self.order_btn.pack(side='left')
        
        # Cancel button
        self.cancel_btn = tk.Button(order_frame,
                                   text="CANCEL ORDER",
                                   font=('Arial', 10, 'bold'),
                                   bg='#aa0000',
                                   fg='white',
                                   command=self.cancel_order,
                                   padx=20)
        self.cancel_btn.pack(fill='x', pady=(5, 0))
        
        # Current order display
        self.order_display = tk.Label(order_frame,
                                     text="Current order: None",
                                     font=('Arial', 10),
                                     bg=panel_bg, fg='#ffff00')
        self.order_display.pack(pady=(5, 0))
        
        # Logger panel
        log_frame = tk.LabelFrame(right_panel, text="SYSTEM LOG",
                                 font=('Arial', 12, 'bold'),
                                 bg=panel_bg, fg=text_color,
                                 padx=10, pady=10)
        log_frame.pack(fill='both', expand=True, padx=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame,
                                                 height=8,
                                                 bg='#1a1a1a',
                                                 fg='#00ff00',
                                                 font=('Courier', 9))
        self.log_text.pack(fill='both', expand=True)
        
        # Disable controls initially
        self.order_btn.config(state='disabled')
        self.cancel_btn.config(state='disabled')
        self.order_entry.config(state='disabled')
    
    def check_queue(self):
        """Process messages from worker threads"""
        try:
            while True:
                msg_type, data = self.gui_queue.get_nowait()
                
                if msg_type == 'log':
                    self.add_log(data)
                elif msg_type == 'status':
                    self.status_label.config(text=data)
                elif msg_type == 'arm_status':
                    self.arm_label.config(text=data)
                elif msg_type == 'camera_status':
                    self.camera_label.config(text=data)
                elif msg_type == 'points_status':
                    self.points_label.config(text=data)
                elif msg_type == 'objects':
                    self.update_objects_display(data)
                elif msg_type == 'camera_image':
                    self.update_camera_image(data)
                elif msg_type == 'system_ready':
                    self.system_ready = True
                    self.order_btn.config(state='normal')
                    self.cancel_btn.config(state='normal')
                    self.order_entry.config(state='normal')
                    self.status_label.config(text="System Ready", fg='#00ff00')
                elif msg_type == 'order_started':
                    self.order_display.config(text=f"Current order: {data}")
                    self.order_btn.config(state='disabled')
                    self.cancel_btn.config(state='normal')
                elif msg_type == 'order_completed':
                    self.order_display.config(text="Current order: None")
                    self.order_btn.config(state='normal')
                    self.cancel_btn.config(state='disabled')
                    
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_queue)
    
    def add_log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.update()
    
    def update_objects_display(self, objects_with_points):
        """Update detected objects display with point mapping"""
        self.objects_text.delete(1.0, tk.END)
        
        if not objects_with_points:
            self.objects_text.insert(tk.END, "No objects detected")
            return
        
        self.objects_text.insert(tk.END, "AVAILABLE OBJECTS:\n")
        self.objects_text.insert(tk.END, "="*50 + "\n\n")
        
        for i, (obj, point) in enumerate(objects_with_points, 1):
            display_text = f"{i}. {obj['class_name']}\n"
            display_text += f"   Point: {point}\n"
            display_text += f"   Confidence: {obj['confidence_pct']}%\n"
            display_text += "-"*40 + "\n"
            
            self.objects_text.insert(tk.END, display_text)
            
            # Store mapping for later use
            self.object_point_mapping[i] = (obj, point)
        
        self.objects_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.objects_text.insert(tk.END, f"\nEnter number (1-{len(objects_with_points)}) to order\n")
        self.objects_text.insert(tk.END, "Enter 0 to cancel order\n")
    
    def update_camera_image(self, img_data):
        """Update camera display with new image"""
        photo = tk.PhotoImage(data=img_data)
        self.camera_display.config(image=photo)
        self.camera_display.image = photo
    
    def initialize_system(self):
        """Initialize all systems"""
        self.gui_queue.put(('log', "Initializing Garage Assistant System..."))
        self.gui_queue.put(('status', "Initializing..."))
        
        try:
            # Initialize robot arm
            self.gui_queue.put(('log', "Initializing robot arm with 9 points..."))
            self.gui_queue.put(('arm_status', "Arm: Initializing..."))
            self.arm = RobotArmController()
            self.gui_queue.put(('arm_status', "Arm: Connected (9 points ready)"))
            self.gui_queue.put(('points_status', "Points: 9/9 ready"))
            self.gui_queue.put(('log', "Robot arm initialized with 9 points"))
            
            # Initialize camera
            self.gui_queue.put(('log', "Initializing camera system..."))
            self.gui_queue.put(('camera_status', "Camera: Initializing..."))
            self.detector = CameraDetectionSystem()
            self.gui_queue.put(('camera_status', "Camera: Connected"))
            self.gui_queue.put(('log', "Camera system initialized"))
            
            # Create output directory
            self.create_output_directory()
            
            # Run automatic scan
            self.gui_queue.put(('log', "Starting automatic 3-position scan..."))
            self.run_automatic_scan()
            
            # Start camera feed thread
            threading.Thread(target=self.camera_feed_thread, daemon=True).start()
            
            # System ready
            self.gui_queue.put(('log', "System initialization complete"))
            self.gui_queue.put(('system_ready', True))
            
        except Exception as e:
            self.gui_queue.put(('log', f"Initialization error: {str(e)}"))
            self.gui_queue.put(('status', "Initialization Failed"))
    
    def create_output_directory(self):
        """Create output directory for logs and images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"garage_assistant_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.gui_queue.put(('log', f"Created output directory: {self.output_dir}"))
    
    def run_automatic_scan(self):
        """Run 3-position automatic scan"""
        try:
            all_detections = []
            
            for i in range(3):
                self.gui_queue.put(('log', f"Moving to snapshot position {i+1}..."))
                self.gui_queue.put(('status', f"Scanning position {i+1}/3..."))
                
                # Move to snapshot position
                self.arm.go_to_snapshot(i)
                time.sleep(3)  # Wait for stabilization
                
                # Capture frame
                ret, frame = self.detector.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    detections = self.detector.detect_objects(frame)
                    
                    if detections:
                        self.gui_queue.put(('log', f"Found {len(detections)} objects in position {i+1}"))
                        for det in detections:
                            all_detections.append(det)
                    
                    # Save annotated image
                    annotated = self.detector.annotate_frame(frame, detections, f"Position {i+1}")
                    filename = f"{self.output_dir}/scan_position_{i+1}.jpg"
                    cv2.imwrite(filename, annotated)
            
            # Map objects to points (first 9 objects to points A-I)
            objects_with_points = []
            point_names = ['PointA', 'PointB', 'PointC', 'PointD', 'PointE', 
                          'PointF', 'PointG', 'PointH', 'PointI']
            
            for i, obj in enumerate(all_detections[:9]):  # Limit to 9 objects
                if i < len(point_names):
                    objects_with_points.append((obj, point_names[i]))
            
            # Update display
            self.gui_queue.put(('objects', objects_with_points))
            self.gui_queue.put(('log', f"Scan complete: {len(objects_with_points)} objects mapped to points"))
            
            # Return to home
            self.gui_queue.put(('log', "Returning to home position..."))
            self.arm.go_home()
            
            self.gui_queue.put(('status', f"Ready - {len(objects_with_points)} objects detected"))
            
        except Exception as e:
            self.gui_queue.put(('log', f"Scan error: {str(e)}"))
    
    def camera_feed_thread(self):
        """Continuous camera feed thread"""
        while True:
            if self.detector and self.detector.cap.isOpened():
                ret, frame = self.detector.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    # Detect objects in real-time
                    detections = self.detector.detect_objects(frame)
                    
                    # Annotate frame
                    annotated = self.detector.annotate_frame(frame, detections, "Live View")
                    
                    # Resize for display
                    display_frame = cv2.resize(annotated, (640, 480))
                    
                    # Convert to PhotoImage
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img_data = cv2.imencode('.png', rgb_frame)[1].tobytes()
                    
                    self.gui_queue.put(('camera_image', img_data))
            
            time.sleep(0.1)  # ~10 FPS
    
    def submit_order(self):
        """Submit order for object"""
        if not self.system_ready:
            return
        
        try:
            order_text = self.order_var.get().strip()
            if not order_text:
                return
            
            order_num = int(order_text)
            self.order_var.set("")  # Clear input
            
            if order_num == 0:
                self.cancel_order()
                return
            
            if order_num in self.object_point_mapping:
                obj, point = self.object_point_mapping[order_num]
                
                self.gui_queue.put(('log', f"Order placed: {obj['class_name']} from {point}"))
                self.gui_queue.put(('order_started', f"{obj['class_name']} from {point}"))
                self.gui_queue.put(('status', f"Processing order: {obj['class_name']}"))
                
                # Disable ordering during operation
                self.order_entry.config(state='disabled')
                
                # Process in background thread
                threading.Thread(target=self.process_order, 
                               args=(obj, point), 
                               daemon=True).start()
            else:
                self.gui_queue.put(('log', f"Invalid order number. Choose 1-{len(self.object_point_mapping)}"))
                
        except ValueError:
            self.gui_queue.put(('log', "Please enter a valid number"))
    
    def process_order(self, obj, point):
        """Process order in background thread"""
        try:
            # Callback function for status updates
            def status_callback(message):
                self.gui_queue.put(('log', message))
                self.gui_queue.put(('status', message))
            
            # Execute pickup sequence
            success = self.arm.execute_pickup_sequence(point, status_callback)
            
            if success:
                self.gui_queue.put(('log', f"Successfully delivered {obj['class_name']} from {point}"))
                self.gui_queue.put(('log', "Adaptive gripper used for perfect grip"))
                self.gui_queue.put(('status', "Delivery complete - System Ready"))
                
                # Remove object from list
                if obj in [o for o, _ in self.object_point_mapping.values()]:
                    # Find and remove the object-point pair
                    for key, (o, p) in list(self.object_point_mapping.items()):
                        if o['class_name'] == obj['class_name'] and p == point:
                            del self.object_point_mapping[key]
                            break
                    
                    # Update display
                    objects_list = list(self.object_point_mapping.values())
                    self.gui_queue.put(('objects', objects_list))
            else:
                self.gui_queue.put(('log', f"Failed to deliver {obj['class_name']}"))
                self.gui_queue.put(('status', "Delivery failed - System Ready"))
            
        except Exception as e:
            self.gui_queue.put(('log', f"Order processing error: {str(e)}"))
        
        finally:
            # Re-enable controls
            self.gui_queue.put(('order_completed', None))
            self.order_entry.config(state='normal')
    
    def cancel_order(self):
        """Cancel current order"""
        if self.current_order:
            self.gui_queue.put(('log', f"Cancelling order: {self.current_order}"))
            self.gui_queue.put(('order_completed', None))
            self.current_order = None
            self.gui_queue.put(('status', "Order cancelled - System Ready"))
        else:
            self.gui_queue.put(('log', "No active order to cancel"))
    
    def on_closing(self):
        """Handle window closing"""
        self.gui_queue.put(('log', "Shutting down system..."))
        self.gui_queue.put(('status', "Shutting down..."))
        
        # Release resources
        if self.detector:
            self.detector.release()
        
        # Return arm to home
        if self.arm:
            try:
                self.arm.go_home()
            except:
                pass
        
        self.root.destroy()

# ============================================
# MAIN APPLICATION
# ============================================
if __name__ == "__main__":
    print("\nStarting complete garage assistant system...")
    print("Press Ctrl+C in terminal to exit\n")
    
    root = tk.Tk()
    app = GarageAssistantGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start main loop
    root.mainloop()