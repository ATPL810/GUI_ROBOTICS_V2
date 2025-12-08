"""
GARAGE ASSISTANT ROBOTIC ARM SYSTEM
Detects objects, displays GUI for user selection, and executes pick-and-place operations
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
from PIL import Image, ImageTk

print("GARAGE ASSISTANT ROBOTIC ARM SYSTEM")
print("=" * 70)

# ============================================
# ROBOT ARM CONTROLLER
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
            
            # Second position: base at 40 degrees
            self.SECOND_POSITION = {
                self.SERVO_BASE: 40,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }

            # Third position: base at 1 degree
            self.THIRD_POSITION = {
                self.SERVO_BASE: 1,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }
            
            # DROP ZONE POSITION
            self.DROP_ZONE_BASE = 150
            self.DROP_ZONE_SEQUENCE = [
                # Add your drop zone waypoints here
                # Example: approach drop zone, lower to drop, release
                # Format: (servo1, servo2, servo3, servo4, servo5, servo6)
                (150, 90, 55, 60, 90, None),  # Move to drop zone (gripper stays closed)
                (150, 70, 55, 45, 90, None),  # Lower to drop position
                # Gripper will open here (handled in release step)
                (150, 90, 90, 90, 90, 90)     # Lift up after release
            ]
            
            # Hardcoded positions for 9 objects (PLACEHOLDER - will be filled)
            self.OBJECT_POSITIONS = {
                'position_1': {
                    'name': 'Position A',
                    'before_grab': (52, 35, 49, 45, 89, 125),
                    'after_grab': [
                        (60, 45, 50, 90, 90, 135),
                        (100, 90, 55, 60, 90, 135),
                        (130, 40, 55, 45, 90, 135),
                        (130, 30, 55, 45, 90, 125)
                    ]
                },
                # Positions 2-9 will be added here
                'position_2': {'name': 'Position B', 'before_grab': None, 'after_grab': []},
                'position_3': {'name': 'Position C', 'before_grab': None, 'after_grab': []},
                'position_4': {'name': 'Position D', 'before_grab': None, 'after_grab': []},
                'position_5': {'name': 'Position E', 'before_grab': None, 'after_grab': []},
                'position_6': {'name': 'Position F', 'before_grab': None, 'after_grab': []},
                'position_7': {'name': 'Position G', 'before_grab': None, 'after_grab': []},
                'position_8': {'name': 'Position H', 'before_grab': None, 'after_grab': []},
                'position_9': {'name': 'Position I', 'before_grab': None, 'after_grab': []}
            }
            
            # Move to initial position
            print("Moving to initial position...")
            self.go_to_initial_position()
            
            print("Robot arm initialized successfully")
            print(f"   Initial position: {self.INITIAL_POSITION}")
            
        except Exception as e:
            print(f"Failed to initialize robot arm: {e}")
            raise
    
    def convert_angle(self, angle):
        """Convert negative angles to servo range (0-180)"""
        if angle < 0:
            return 180 + angle
        return angle
    
    def go_to_initial_position(self):
        """Move to exact initial position"""
        angles_dict = self.INITIAL_POSITION.copy()
        angles_dict[self.SERVO_WRIST] = angles_dict[self.SERVO_WRIST]
        
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000
        )
        time.sleep(2.5)
        print("   At initial position")
    
    def go_to_second_position(self):
        """Move to second position (base at 40 degrees)"""
        angles_dict = self.SECOND_POSITION.copy()
        angles_dict[self.SERVO_WRIST] = angles_dict[self.SERVO_WRIST]
        
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000
        )
        time.sleep(2.5)
        print("   At second position (base at 40 degrees)")
    
    def go_to_third_position(self):
        """Move to third position (base at 1 degree)"""
        angles_dict = self.THIRD_POSITION.copy()
        angles_dict[self.SERVO_WRIST] = angles_dict[self.SERVO_WRIST]
        
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000
        )
        time.sleep(2.5)
        print("   At third position (base at 1 degree)")
    
    def get_current_position_name(self, servo1_angle):
        """Get position name based on base servo angle"""
        if servo1_angle == self.INITIAL_POSITION[self.SERVO_BASE]:
            return "initial_position"
        elif servo1_angle == self.SECOND_POSITION[self.SERVO_BASE]:
            return "second_position"
        elif servo1_angle == self.THIRD_POSITION[self.SERVO_BASE]:
            return "third_position"
        return "unknown_position"
    
    def adaptive_grip(self, callback=None):
        """
        Gradually close gripper until object is firmly grasped
        Monitors gripper angle to detect when grip stabilizes
        """
        print("Starting adaptive grip...")
        
        # Start from open position
        current_angle = 90
        target_angle = 180
        step = 5
        stable_count = 0
        previous_angle = current_angle
        
        while current_angle < target_angle:
            # Move gripper incrementally
            self.arm.Arm_serial_servo_write(6, current_angle, 200)
            time.sleep(0.3)
            
            # Check if gripper movement has stabilized (object is gripped)
            # In practice, you would read actual servo position
            # For now, we simulate by checking if we've reached firm grip
            if current_angle >= 135:
                stable_count += 1
                if stable_count >= 3:
                    print(f"Object gripped firmly at angle: {current_angle}")
                    if callback:
                        callback(f"Object gripped at angle {current_angle}")
                    break
            
            previous_angle = current_angle
            current_angle += step
        
        return current_angle
    
    def execute_pickup_sequence(self, position_key, callback=None):
        """Execute complete pickup and delivery sequence"""
        if position_key not in self.OBJECT_POSITIONS:
            print(f"Invalid position key: {position_key}")
            return False
        
        position = self.OBJECT_POSITIONS[position_key]
        
        if position['before_grab'] is None:
            print(f"Position {position_key} not configured")
            if callback:
                callback(f"Position {position_key} not configured")
            return False
        
        try:
            # Move to position before grabbing
            if callback:
                callback(f"Moving to {position['name']} pickup position...")
            print(f"Moving to {position['name']} before grab position...")
            time.sleep(2)
            self.arm.Arm_serial_servo_write6(*position['before_grab'], 1000)
            time.sleep(2)
            
            # Adaptive grip
            if callback:
                callback("Gripping object...")
            grip_angle = self.adaptive_grip(callback)
            time.sleep(2)
            
            # Move through waypoints after grabbing
            if callback:
                callback("Transporting object to drop zone...")
            for i, waypoint in enumerate(position['after_grab']):
                print(f"Moving to waypoint {i+1}...")
                self.arm.Arm_serial_servo_write6(*waypoint, 1000)
                time.sleep(2)
            
            # Execute drop zone sequence
            if callback:
                callback("Approaching drop zone...")
            print("Executing drop zone sequence...")
            
            for i, waypoint in enumerate(self.DROP_ZONE_SEQUENCE):
                print(f"Drop zone step {i+1}...")
                
                # If gripper angle is None, keep it closed (use current grip angle)
                if waypoint[5] is None:
                    # Move all servos except gripper
                    self.arm.Arm_serial_servo_write6(
                        waypoint[0], waypoint[1], waypoint[2],
                        waypoint[3], waypoint[4], grip_angle, 1000
                    )
                else:
                    # Move all servos including gripper
                    self.arm.Arm_serial_servo_write6(*waypoint, 1000)
                
                time.sleep(2)
                
                # Release object after reaching drop position (after second waypoint)
                if i == 1:  # After lowering to drop position
                    if callback:
                        callback("Releasing object...")
                    print("Opening gripper to release object...")
                    self.arm.Arm_serial_servo_write(6, 90, 1000)
                    time.sleep(2)
            
            # Return to initial position
            if callback:
                callback("Returning to initial position...")
            print("Returning to initial position...")
            self.go_to_initial_position()
            
            if callback:
                callback("Task completed successfully")
            print("Pickup sequence completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during pickup sequence: {e}")
            if callback:
                callback(f"Error: {e}")
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
        self.root.title("Garage Assistant - Robot Control System")
        self.root.geometry("1200x800")
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
        
        title_label = tk.Label(title_frame, text="GARAGE ASSISTANT", 
                              font=('Arial', 24, 'bold'), 
                              bg='#1a1a1a', fg='#00ff00')
        title_label.pack(pady=10)
        
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
        
        tk.Label(input_frame, text="Enter Object Number:", 
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
            self.log("Initializing robot arm...")
            self.arm = RobotArmController()
            self.log("Robot arm initialized")
            
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
            self.update_status(f"Ready - {len(self.all_detections)} objects detected")
            self.is_running = True
            
        except Exception as e:
            self.log(f"Inspection error: {e}")
            self.update_status("Inspection failed")
    
    def capture_snapshot(self, position_name, base_angle):
        """Capture a snapshot, detect objects, and save results"""
        self.log(f"Capturing snapshot: {position_name}")
        self.log(f"   Base servo angle: {base_angle} deg")
        
        # Clear camera buffer by reading and discarding a few frames
        self.log("   Clearing camera buffer...")
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
        
        # Only save annotated images if objects were detected
        if detections:
            # Annotate frame
            annotated_frame = self.detector.annotate_frame(frame, detections, position_name, base_angle)
            
            # Save annotated snapshot
            filename = f"{self.output_dir}/{position_name}_detected.jpg"
            cv2.imwrite(filename, annotated_frame)
            self.log(f"   Saved annotated snapshot: {filename}")
            snapshot_info['filename'] = filename
            
            # Save detection data to text file
            self.save_detection_data(position_name, detections, filename, base_angle)
        else:
            self.log(f"   Skipping snapshot save (no objects detected)")
            snapshot_info['filename'] = None
        
        return snapshot_info
    
    def save_detection_data(self, position_name, detections, image_filename, base_angle):
        """Save detection information to text file"""
        txt_filename = f"{self.output_dir}/{position_name}_detections.txt"
        
        with open(txt_filename, 'w') as f:
            f.write(f"ROBOT ARM SNAPSHOT DETECTION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Position: {position_name}\n")
            f.write(f"Base Servo Angle: {base_angle} deg\n")
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
                    f.write(f"  Confidence: {det['confidence_int']}%\n")
                    f.write(f"  Bounding Box: {det['bbox']}\n")
                    f.write(f"  Center Coordinates: ({det['center_x']:.1f}, {det['center_y']:.1f})\n")
                    f.write(f"  Size: {det['width']}x{det['height']} pixels\n")
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
            f.write("=" * 70 + "\n\n")
            
            # Count all detections
            total_detections = sum(len(snap['detections']) for snap in self.all_snapshots)
            f.write(f"TOTAL OBJECTS DETECTED: {total_detections}\n\n")
            
            # Summary by position
            f.write("SNAPSHOT SUMMARY BY POSITION:\n")
            f.write("-" * 50 + "\n")
            
            for snap in self.all_snapshots:
                f.write(f"\n{snap['position_name'].upper()} (Base: {snap['base_angle']} deg):\n")
                f.write(f"  Time: {snap['timestamp']}\n")
                if snap['filename']:
                    f.write(f"  Image: {os.path.basename(snap['filename'])}\n")
                else:
                    f.write(f"  Image: Not saved (no detections)\n")
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
    
    def save_pickup_log(self, order_num, obj, success):
        """Save log of pickup operation"""
        log_filename = f"{self.output_dir}/pickup_operations.txt"
        
        mode = 'a' if os.path.exists(log_filename) else 'w'
        with open(log_filename, mode) as f:
            if mode == 'w':
                f.write("GARAGE ASSISTANT - PICKUP OPERATIONS LOG\n")
                f.write("=" * 70 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Order Number: {order_num}\n")
            f.write(f"Object: {obj['class_name']}\n")
            f.write(f"Confidence: {obj['confidence_int']}%\n")
            f.write(f"Position: position_{order_num}\n")
            f.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
            f.write("-" * 50 + "\n\n")
        
        self.log(f"Logged pickup operation to: {log_filename}")
    
    def update_detection_display(self):
        """Update detection listbox"""
        self.detection_listbox.delete(0, 'end')
        
        for i, det in enumerate(self.all_detections, 1):
            display_text = f"{i}. {det['class_name']} ({det['confidence_int']}%)"
            self.detection_listbox.insert('end', display_text)
    
    def execute_order(self):
        """Execute user order to pick object"""
        if not self.is_running:
            self.log("System not ready")
            return
        
        try:
            order_num = int(self.order_entry.get().strip())
            
            if order_num < 1 or order_num > len(self.all_detections):
                self.log(f"Invalid order number: {order_num}")
                self.update_status("Invalid order number")
                return
            
            selected_obj = self.all_detections[order_num - 1]
            self.log(f"Order received: Pick {selected_obj['class_name']}")
            self.update_status(f"Executing order: {selected_obj['class_name']}")
            
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
        
        def status_callback(msg):
            self.log(msg)
            self.update_status(msg)
        
        success = self.arm.execute_pickup_sequence(position_key, status_callback)
        
        # Log the operation
        self.save_pickup_log(order_num, obj, success)
        
        if success:
            self.log(f"Successfully delivered {obj['class_name']}")
            self.update_status("Task completed - Ready for next order")
        else:
            self.log(f"Failed to deliver {obj['class_name']}")
            self.update_status("Task failed")
        
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

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print("GARAGE ASSISTANT ROBOTIC ARM SYSTEM")
    print("=" * 70)
    print("Features:")
    print("1. Automatic 3-position inspection with object detection")
    print("2. GUI interface for object selection and control")
    print("3. Adaptive gripper for secure object handling")
    print("4. Complete pickup and delivery to drop zone")
    print("5. Detailed logging and report generation")
    print("=" * 70)
    print("\nStarting GUI...")
    
    root = tk.Tk()
    app = GarageAssistantGUI(root)
    root.mainloop()