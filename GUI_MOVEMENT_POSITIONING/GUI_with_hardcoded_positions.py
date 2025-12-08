"""
GARAGE ASSISTANT ROBOTIC ARM SYSTEM
Complete system with 9 precise grabbing positions
"""

import cv2
import time
import os
import json
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from ultralytics import YOLO
from Arm_Lib import Arm_Device
import numpy as np

print("GARAGE ASSISTANT ROBOTIC ARM SYSTEM")
print("=" * 70)

# ============================================
# GRABBING FUNCTIONS FOR 9 POINTS
# ============================================
class GrabbingFunctions:
    @staticmethod
    def grab_point_A(arm, tool_type="bolt"):
        """Grabbing point A - BOLTS"""
        print("Starting Grab Point A - BOLTS")
        
        # Define grip force for each tool
        grip_forces = {
            "bolt": 177,
            "wrench": 170,
            "screwdriver": 169,
            "hammer": 169,
            "plier": 125,
            "measuring_tape": 163
        }
        
        grip_force = grip_forces.get(tool_type.lower(), 177)
        
        # move to pos Before grabbing
        arm.Arm_serial_servo_write6(72, 33, 31, 59, 110, 89, 1000)
        time.sleep(2)   
        arm.Arm_serial_servo_write6(72, 35, 31, 59, 85, 112, 1000)
        
        # grabbing
        time.sleep(2)
        arm.Arm_serial_servo_write(6, grip_force, 1000)

        # move to pos After grabbing object while holding it
        time.sleep(2)
        arm.Arm_serial_servo_write6(69, 58, 39, 59, 118, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        # tightening
        arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

        time.sleep(2)
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        print("Grab Point A completed")
        return True

    @staticmethod
    def grab_point_B(arm, tool_type="wrench"):
        """Grabbing point B - Multiple tools"""
        print(f"Starting Grab Point B - {tool_type}")
        
        # Define grip force for each tool
        grip_forces = {
            "wrench": 176,
            "hammer": 170,
            "screwdriver": 169,
            "bolt": 177,
            "plier": 125,
            "measuring_tape": 163
        }
        
        grip_force = grip_forces.get(tool_type.lower(), 176)
        
        # move to pos Before grabbing
        arm.Arm_serial_servo_write6(81, 33, 54, 17, 76, 139, 1000)
        time.sleep(2)   
        arm.Arm_serial_servo_write6(81, 29, 55, 18, 76, 150, 1000)
        
        # grabbing
        time.sleep(2)
        arm.Arm_serial_servo_write(6, grip_force, 1000)

        # move to pos After grabbing object while holding it
        time.sleep(2)
        arm.Arm_serial_servo_write6(81, 29, 54, 18, 76, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        # tightening
        arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

        time.sleep(2)
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        print(f"Grab Point B completed for {tool_type}")
        return True

    @staticmethod
    def grab_point_C(arm, tool_type="bolt"):
        """Grabbing point C - Bolt"""
        print(f"Starting Grab Point C - {tool_type}")
        
        # Define grip force for each tool
        grip_forces = {
            "bolt": 176,
            "wrench": 170,
            "screwdriver": 169,
            "hammer": 169,
            "plier": 125,
            "measuring_tape": 163
        }
        
        grip_force = grip_forces.get(tool_type.lower(), 176)
        
        # move to pos Before grabbing
        arm.Arm_serial_servo_write6(91, 47, 6, 85, 89, 91, 1000)
        time.sleep(2)   
        arm.Arm_serial_servo_write6(91, 45, 6, 85, 89, 119, 1000)
        
        # grabbing
        time.sleep(2)
        arm.Arm_serial_servo_write(6, grip_force, 1000)

        # move to pos After grabbing object while holding it
        time.sleep(2)
        arm.Arm_serial_servo_write6(103, 54, 40, 34, 90, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        # tightening
        arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

        time.sleep(2)
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        print(f"Grab Point C completed for {tool_type}")
        return True

    @staticmethod
    def grab_point_D(arm, tool_type="wrench"):
        """Grabbing point D - Multiple tools"""
        print(f"Starting Grab Point D - {tool_type}")
        
        # Define grip force for each tool
        grip_forces = {
            "wrench": 153,
            "screwdriver": 165,
            "hammer": 169,
            "bolt": 177,
            "plier": 125,
            "measuring_tape": 163
        }
        
        grip_force = grip_forces.get(tool_type.lower(), 153)
        
        # move to pos Before grabbing
        arm.Arm_serial_servo_write6(105, 40, 25, 50, 89, 111, 1000)
        time.sleep(2)   
        arm.Arm_serial_servo_write6(105, 36, 27, 53, 90, 123, 1000)
        
        # grabbing
        time.sleep(2)
        arm.Arm_serial_servo_write(6, grip_force, 1000)

        # move to pos After grabbing
        time.sleep(2)
        arm.Arm_serial_servo_write6(103, 54, 40, 34, 90, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        # tightening
        arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

        time.sleep(2)
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        print(f"Grab Point D completed for {tool_type}")
        return True

    @staticmethod
    def grab_point_E(arm, tool_type="hammer"):
        """Grabbing point E - Multiple tools"""
        print(f"Starting Grab Point E - {tool_type}")
        
        # Define grip force for each tool
        grip_forces = {
            "hammer": 169,
            "screwdriver": 169,
            "wrench": 154,
            "bolt": 177,
            "plier": 125,
            "measuring_tape": 163
        }
        
        grip_force = grip_forces.get(tool_type.lower(), 169)
        
        # move to pos Before grabbing
        arm.Arm_serial_servo_write6(31, 39, 48, 22, 101, 133, 1000)
        time.sleep(2)   
        arm.Arm_serial_servo_write6(31, 33, 48, 22, 102, 140, 1000)
        
        # grabbing
        time.sleep(2)
        arm.Arm_serial_servo_write(6, grip_force, 1000)

        # move to pos After grabbing object while holding it
        time.sleep(2)
        arm.Arm_serial_servo_write6(31, 65, 71, 22, 102, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        # tightening
        arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

        time.sleep(2)
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        print(f"Grab Point E completed for {tool_type}")
        return True

    @staticmethod
    def grab_point_F(arm, tool_type="bolt"):
        """Grabbing point F - Bolt"""
        print(f"Starting Grab Point F - {tool_type}")
        
        # Define grip force for each tool
        grip_forces = {
            "bolt": 177,
            "wrench": 170,
            "screwdriver": 169,
            "hammer": 169,
            "plier": 125,
            "measuring_tape": 163
        }
        
        grip_force = grip_forces.get(tool_type.lower(), 177)
        
        # move to pos Before grabbing
        arm.Arm_serial_servo_write6(38, 33, 53, 35, 90, 142, 1000)
        time.sleep(2)   
        arm.Arm_serial_servo_write6(38, 25, 54, 44, 90, 143, 1000)
        
        # grabbing
        time.sleep(2)
        arm.Arm_serial_servo_write(6, grip_force, 1000)

        # move to pos After grabbing object while holding it
        time.sleep(2)
        arm.Arm_serial_servo_write6(37, 36, 71, 44, 80, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        # tightening
        arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 130, 1000)

        time.sleep(2)
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        print(f"Grab Point F completed for {tool_type}")
        return True

    @staticmethod
    def grab_point_G(arm, tool_type="plier"):
        """Grabbing point G - Pliers"""
        print(f"Starting Grab Point G - {tool_type}")
        
        # Define grip force for each tool
        grip_forces = {
            "plier": 125,
            "bolt": 177,
            "wrench": 170,
            "screwdriver": 169,
            "hammer": 169,
            "measuring_tape": 163
        }
        
        grip_force = grip_forces.get(tool_type.lower(), 125)
        
        # move to pos Before grabbing
        arm.Arm_serial_servo_write6(50, 17, 93, 1, 90, 59, 1000)
        time.sleep(2)   
        arm.Arm_serial_servo_write6(49, 2, 93, 16, 89, 59, 1000)
        
        # grabbing
        time.sleep(2)
        arm.Arm_serial_servo_write(6, grip_force, 1000)

        # move to pos After grabbing object while holding it
        time.sleep(2)
        arm.Arm_serial_servo_write6(50, 30, 107, 18, 89, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        # tightening
        arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 124, 1000)

        time.sleep(2)
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 80, 1000)
        print(f"Grab Point G completed for {tool_type}")
        return True

    @staticmethod
    def grab_point_H(arm, tool_type="measuring_tape"):
        """Grabbing point H - Measuring Tape"""
        print(f"Starting Grab Point H - {tool_type}")
        
        # Define grip force for each tool
        grip_forces = {
            "measuring_tape": 163,
            "bolt": 177,
            "wrench": 170,
            "screwdriver": 169,
            "hammer": 169,
            "plier": 125
        }
        
        grip_force = grip_forces.get(tool_type.lower(), 163)
        
        # move to pos Before grabbing
        arm.Arm_serial_servo_write6(0, 23, 90, 2, 40, 62, 1000)
        time.sleep(2)   
        arm.Arm_serial_servo_write6(-15, 19, 90, 6, 40, 45, 1000)
        
        # grabbing
        time.sleep(2)
        arm.Arm_serial_servo_write(6, grip_force, 1000)

        # move to pos After grabbing object while holding it
        time.sleep(2)
        arm.Arm_serial_servo_write6(50, 30, 107, 18, 89, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        # tightening
        arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 100, 1000)

        time.sleep(2)
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 80, 1000)
        print(f"Grab Point H completed for {tool_type}")
        return True

    @staticmethod
    def grab_point_I(arm, tool_type="hammer"):
        """Grabbing point I - Hammer"""
        print(f"Starting Grab Point I - {tool_type}")
        
        # Define grip force for each tool
        grip_forces = {
            "hammer": 167,
            "bolt": 177,
            "wrench": 170,
            "screwdriver": 169,
            "plier": 125,
            "measuring_tape": 163
        }
        
        grip_force = grip_forces.get(tool_type.lower(), 167)
        
        # move to pos Before grabbing
        arm.Arm_serial_servo_write6(13, 50, 37, 37, 93, 87, 1000)
        time.sleep(2)   
        arm.Arm_serial_servo_write6(8, 3, 86, 34, 123, 86, 1000)
        
        # grabbing
        time.sleep(2)
        arm.Arm_serial_servo_write(6, grip_force, 1000)

        # move to pos After grabbing object while holding it
        time.sleep(2)
        arm.Arm_serial_servo_write6(9, 26, 98, 35, 123, grip_force, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, 163, 1000)
        time.sleep(2)
        
        # tightening
        arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, 163, 1000)
        time.sleep(2)
        arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 100, 1000)

        time.sleep(2)
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 80, 1000)
        print(f"Grab Point I completed for {tool_type}")
        return True

# ============================================
# ROBOT ARM CONTROLLER
# ============================================
class RobotArmController:
    def __init__(self):
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
            
            # Initial position
            self.INITIAL_POSITION = {
                self.SERVO_BASE: 90,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }
            
            # Snapshot positions
            self.SNAPSHOT_POSITIONS = [
                {"name": "initial", "base": 90},
                {"name": "second", "base": 40},
                {"name": "third", "base": 1}
            ]
            
            # Map points to grabbing functions
            self.GRAB_FUNCTIONS = {
                "PointA": GrabbingFunctions.grab_point_A,
                "PointB": GrabbingFunctions.grab_point_B,
                "PointC": GrabbingFunctions.grab_point_C,
                "PointD": GrabbingFunctions.grab_point_D,
                "PointE": GrabbingFunctions.grab_point_E,
                "PointF": GrabbingFunctions.grab_point_F,
                "PointG": GrabbingFunctions.grab_point_G,
                "PointH": GrabbingFunctions.grab_point_H,
                "PointI": GrabbingFunctions.grab_point_I
            }
            
            # Tool grip force mapping
            self.TOOL_GRIP_FORCES = {
                "bolt": 177,
                "hammer": 169,
                "measuring tape": 163,
                "plier": 125,
                "screwdriver": 169,
                "wrench": 176
            }
            
            self.go_to_initial_position()
            print("Robot arm initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize robot arm: {e}")
            raise
    
    def convert_angle(self, angle):
        if angle < 0:
            return 180 + angle
        return angle
    
    def go_to_initial_position(self):
        self.move_to_position(self.INITIAL_POSITION)
    
    def move_to_position(self, position_dict, move_time=1500):
        """Move arm to specific position"""
        servo1 = position_dict.get(self.SERVO_BASE, 90)
        servo2 = position_dict.get(self.SERVO_SHOULDER, 105)
        servo3 = position_dict.get(self.SERVO_ELBOW, 45)
        servo4 = self.convert_angle(position_dict.get(self.SERVO_WRIST, -35))
        servo5 = position_dict.get(self.SERVO_WRIST_ROT, 90)
        servo6 = position_dict.get(self.SERVO_GRIPPER, 90)
        
        self.arm.Arm_serial_servo_write6(servo1, servo2, servo3, servo4, servo5, servo6, move_time)
        time.sleep(move_time/1000 + 0.5)
    
    def go_to_snapshot_position(self, index):
        if 0 <= index < len(self.SNAPSHOT_POSITIONS):
            pos = self.SNAPSHOT_POSITIONS[index]
            position = {
                self.SERVO_BASE: pos["base"],
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }
            self.move_to_position(position)
            return True
        return False
    
    def execute_grab_sequence(self, point_name, tool_name):
        """Execute the specific grab sequence for a point"""
        if point_name in self.GRAB_FUNCTIONS:
            try:
                # Convert tool name to match grip force dictionary
                tool_key = tool_name.lower()
                if "measuring" in tool_key:
                    tool_key = "measuring_tape"
                elif "plier" in tool_key:
                    tool_key = "plier"
                
                # Execute the grab function
                return self.GRAB_FUNCTIONS[point_name](self.arm, tool_key)
            except Exception as e:
                print(f"Error executing grab sequence for {point_name}: {e}")
                return False
        return False

# ============================================
# CAMERA AND DETECTION SYSTEM
# ============================================
class CameraDetectionSystem:
    def __init__(self):
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
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"Found camera at index {i}")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
        raise Exception("No camera found!")
    
    def load_yolo_model(self):
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
                    print(f"Model loaded: {path}")
                    return model
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    continue
        raise Exception("No YOLO model found!")
    
    def detect_objects(self, frame):
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
                            'center_x': (x1 + x2) / 2,
                            'center_y': (y1 + y2) / 2
                        })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# ============================================
# GUI APPLICATION
# ============================================
class GarageAssistantGUI:
    def __init__(self, root, arm_controller, detector):
        self.root = root
        self.arm = arm_controller
        self.detector = detector
        self.detected_objects = []
        self.object_positions = {}  # Maps object to position
        self.current_order = None
        self.is_scanning = False
        self.scan_results = []
        
        # Configure window
        self.root.title("Garage Assistant - Robotic Arm Controller")
        self.root.geometry("1100x700")
        self.root.configure(bg='#2b2b2b')
        
        # Apply theme
        self.setup_styles()
        
        # Setup GUI
        self.setup_gui()
        
        # Start initial scan in separate thread
        threading.Thread(target=self.initial_scan, daemon=True).start()
    
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.bg_color = '#2b2b2b'
        self.fg_color = '#ffffff'
        self.accent_color = '#007acc'
        self.button_color = '#3c3c3c'
        
        self.style.configure('TLabel', background=self.bg_color, foreground=self.fg_color, font=('Arial', 10))
        self.style.configure('TButton', background=self.button_color, foreground=self.fg_color, 
                           font=('Arial', 10, 'bold'), borderwidth=1)
        self.style.map('TButton', background=[('active', self.accent_color)])
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'), foreground=self.accent_color)
        self.style.configure('Status.TLabel', font=('Arial', 11), foreground='#00ff00')
        self.style.configure('Tool.TLabel', font=('Arial', 10, 'bold'), foreground='#ffff00')
    
    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Header
        header_label = ttk.Label(left_frame, text="GARAGE ASSISTANT", style='Header.TLabel')
        header_label.pack(pady=(0, 20))
        
        # Status display
        self.status_label = ttk.Label(left_frame, text="Status: Initializing...", style='Status.TLabel')
        self.status_label.pack(pady=(0, 10))
        
        # Detected objects display
        objects_frame = ttk.LabelFrame(left_frame, text="Detected Objects", padding=10)
        objects_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.objects_text = scrolledtext.ScrolledText(objects_frame, height=15, width=45,
                                                     bg='#1e1e1e', fg='#ffffff',
                                                     font=('Courier', 10))
        self.objects_text.pack(fill=tk.BOTH, expand=True)
        
        # Points info
        points_frame = ttk.LabelFrame(left_frame, text="Available Points", padding=10)
        points_frame.pack(fill=tk.X, pady=(0, 10))
        
        points_info = "Points A-I are programmed\nEach point has specific grab sequence"
        points_label = ttk.Label(points_frame, text=points_info, font=('Arial', 9))
        points_label.pack()
        
        # Order input section
        order_frame = ttk.LabelFrame(left_frame, text="Order Object", padding=10)
        order_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(order_frame, text="Enter object number (1-9) or 0 to cancel:").pack(anchor=tk.W)
        
        input_frame = ttk.Frame(order_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.order_var = tk.StringVar()
        order_entry = ttk.Entry(input_frame, textvariable=self.order_var, width=10, font=('Arial', 12))
        order_entry.pack(side=tk.LEFT, padx=(0, 10))
        order_entry.bind('<Return>', lambda e: self.process_order())
        
        order_btn = ttk.Button(input_frame, text="Order", command=self.process_order)
        order_btn.pack(side=tk.LEFT)
        
        # Current order display
        self.order_display = ttk.Label(order_frame, text="Current order: None", font=('Arial', 10, 'bold'))
        self.order_display.pack(pady=5)
        
        # Control buttons
        control_frame = ttk.LabelFrame(left_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack()
        
        ttk.Button(btn_frame, text="Rescan Objects", command=self.rescan_objects).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Test Grip", command=self.test_grip).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Go Home", command=self.go_home).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Emergency Stop", command=self.emergency_stop).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Logger and Camera
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Camera feed
        camera_frame = ttk.LabelFrame(right_frame, text="Camera Feed", padding=10)
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.camera_label = ttk.Label(camera_frame, text="Camera feed will appear here", 
                                     background='#1e1e1e', foreground='#888888')
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Logger
        log_frame = ttk.LabelFrame(right_frame, text="System Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=50,
                                                 bg='#1e1e1e', fg='#ffffff',
                                                 font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Start camera feed
        self.update_camera_feed()
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.update()
        
        print(message)
    
    def update_status(self, message, is_error=False):
        color = "#ff4444" if is_error else "#00ff00"
        self.status_label.configure(text=f"Status: {message}", foreground=color)
        self.log_message(message)
    
    def update_detected_objects(self, scan_results):
        self.objects_text.delete(1.0, tk.END)
        
        if not scan_results:
            self.objects_text.insert(tk.END, "No objects detected")
            return
        
        self.detected_objects = []
        self.object_positions = {}
        self.scan_results = scan_results
        
        # Display objects
        self.objects_text.insert(tk.END, "AVAILABLE OBJECTS:\n")
        self.objects_text.insert(tk.END, "="*50 + "\n\n")
        
        object_counter = 1
        for scan in scan_results:
            for det in scan['detections']:
                # Assign position based on scan
                position_name = f"Point{chr(65+scan['position_index'])}"  # A, B, C, etc.
                
                # Add to display
                display_text = f"{object_counter}. {det['class_name']}\n"
                display_text += f"   Position: {position_name}\n"
                display_text += f"   Scan: {scan['position_name']}\n"
                display_text += f"   Confidence: {det['confidence']:.2f}\n"
                display_text += "-"*30 + "\n"
                
                self.objects_text.insert(tk.END, display_text)
                
                # Store mapping
                self.detected_objects.append({
                    'number': object_counter,
                    'name': det['class_name'],
                    'position': position_name,
                    'scan_position': scan['position_name']
                })
                
                object_counter += 1
        
        if object_counter == 1:
            self.objects_text.insert(tk.END, "No objects detected in any position\n")
        
        self.objects_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.objects_text.insert(tk.END, f"\n0. Cancel current order\n")
        
        self.update_status(f"Found {object_counter-1} objects")
    
    def process_order(self):
        try:
            order_num = int(self.order_var.get())
            self.order_var.set("")  # Clear input field
            
            if order_num == 0:
                self.cancel_order()
                return
            
            # Find the ordered object
            ordered_object = None
            for obj in self.detected_objects:
                if obj['number'] == order_num:
                    ordered_object = obj
                    break
            
            if ordered_object:
                self.current_order = ordered_object
                
                self.order_display.configure(
                    text=f"Current order: {order_num} - {ordered_object['name']} at {ordered_object['position']}"
                )
                
                self.log_message(f"Order placed: {ordered_object['name']} from {ordered_object['position']}")
                self.update_status(f"Processing order: {ordered_object['name']}")
                
                # Execute order in separate thread
                threading.Thread(target=self.execute_order, daemon=True).start()
            else:
                self.update_status(f"Invalid order number. Choose 1-{len(self.detected_objects)} or 0", True)
                
        except ValueError:
            self.update_status("Please enter a valid number", True)
    
    def cancel_order(self):
        if self.current_order:
            self.log_message(f"Cancelled order: {self.current_order['name']}")
            self.current_order = None
            self.order_display.configure(text="Current order: None")
            self.update_status("Order cancelled")
        else:
            self.update_status("No active order to cancel")
    
    def execute_order(self):
        if not self.current_order:
            return
        
        object_name = self.current_order['name']
        position_name = self.current_order['position']
        
        self.update_status(f"Executing order: {object_name} from {position_name}")
        
        try:
            # Execute the specific grab sequence for this position
            success = self.arm.execute_grab_sequence(position_name, object_name)
            
            if success:
                self.update_status(f"Successfully delivered {object_name}")
                self.log_message(f"Object {object_name} delivered to drop zone")
                
                # Remove object from detected list
                self.detected_objects = [obj for obj in self.detected_objects 
                                       if not (obj['number'] == self.current_order['number'])]
                
                # Update display
                self.root.after(0, self.update_detected_objects, self.scan_results)
            else:
                self.update_status(f"Failed to pick up {object_name}", True)
                self.log_message(f"Failed to execute grab sequence for {position_name}")
        
        except Exception as e:
            self.update_status(f"Error executing order: {str(e)}", True)
            self.log_message(f"Order execution error: {e}")
        
        self.current_order = None
        self.order_display.configure(text="Current order: None")
    
    def initial_scan(self):
        self.update_status("Starting initial scan...")
        self.is_scanning = True
        
        all_detections = []
        
        try:
            for i in range(3):
                self.log_message(f"Moving to snapshot position {i+1}")
                self.arm.go_to_snapshot_position(i)
                time.sleep(3)  # Wait for stabilization
                
                # Capture frame
                ret, frame = self.detector.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    detections = self.detector.detect_objects(frame)
                    
                    snapshot_info = {
                        'position_index': i,
                        'position_name': f"Position {i+1}",
                        'detections': detections,
                        'snapshot_time': datetime.now().strftime("%H:%M:%S")
                    }
                    all_detections.append(snapshot_info)
                    
                    if detections:
                        self.log_message(f"Found {len(detections)} objects in position {i+1}")
                        for det in detections:
                            self.log_message(f"  - {det['class_name']} (confidence: {det['confidence']:.2f})")
                    else:
                        self.log_message(f"No objects in position {i+1}")
            
            # Return to initial position
            self.arm.go_to_initial_position()
            
            # Update GUI with detections
            self.root.after(0, self.update_detected_objects, all_detections)
            self.update_status("Initial scan complete")
            
        except Exception as e:
            self.update_status(f"Scan error: {str(e)}", True)
            self.log_message(f"Scan error: {e}")
        
        self.is_scanning = False
    
    def rescan_objects(self):
        if not self.is_scanning:
            self.log_message("Manual rescan initiated...")
            threading.Thread(target=self.initial_scan, daemon=True).start()
    
    def test_grip(self):
        self.log_message("Testing gripper...")
        threading.Thread(target=self._test_grip_thread, daemon=True).start()
    
    def _test_grip_thread(self):
        try:
            self.arm.arm.Arm_serial_servo_write(6, 90, 500)  # Open
            time.sleep(1)
            self.arm.arm.Arm_serial_servo_write(6, 135, 500)  # Close
            time.sleep(1)
            self.arm.arm.Arm_serial_servo_write(6, 90, 500)  # Open
            self.log_message("Gripper test complete")
        except Exception as e:
            self.log_message(f"Gripper test failed: {e}")
    
    def go_home(self):
        self.log_message("Returning to home position...")
        threading.Thread(target=self.arm.go_to_initial_position, daemon=True).start()
    
    def emergency_stop(self):
        self.log_message("EMERGENCY STOP ACTIVATED")
        self.update_status("EMERGENCY STOP", True)
        self.current_order = None
        self.order_display.configure(text="Current order: None")
        
        # Return to home position safely
        threading.Thread(target=self.arm.go_to_initial_position, daemon=True).start()
    
    def update_camera_feed(self):
        if hasattr(self, 'detector') and self.detector.cap is not None:
            try:
                ret, frame = self.detector.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    # Detect objects in real-time
                    detections = self.detector.detect_objects(frame)
                    
                    # Draw detections
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        class_name = det['class_name']
                        confidence = det['confidence']
                        
                        # Choose color based on class
                        color_idx = self.detector.TOOL_CLASSES.index(class_name) if class_name in self.detector.TOOL_CLASSES else 0
                        color = self.detector.TOOL_COLORS[color_idx % len(self.detector.TOOL_COLORS)]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Resize for display
                    display_frame = cv2.resize(frame, (500, 375))
                    
                    # Convert to RGB for tkinter
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PhotoImage
                    img = tk.PhotoImage(data=cv2.imencode('.png', rgb_frame)[1].tobytes())
                    
                    self.camera_label.configure(image=img)
                    self.camera_label.image = img
            except Exception as e:
                pass
        
        # Update every 100ms
        self.root.after(100, self.update_camera_feed)
    
    def on_closing(self):
        self.log_message("Shutting down system...")
        self.detector.release()
        self.root.destroy()

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    print("Starting Garage Assistant System...")
    print("=" * 70)
    print("System features:")
    print("1. Automatic initial scan (3 positions)")
    print("2. 9 programmed grab points (A-I)")
    print("3. Tool-specific grip forces")
    print("4. Real-time camera feed with detection")
    print("5. GUI with order system (type 1-9 or 0 to cancel)")
    print("=" * 70)
    
    try:
        # Initialize hardware
        arm = RobotArmController()
        detector = CameraDetectionSystem()
        
        # Create GUI
        root = tk.Tk()
        app = GarageAssistantGUI(root, arm, detector)
        
        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Start GUI
        root.mainloop()
        
    except Exception as e:
        print(f"Failed to start system: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("System shutdown complete")

# ============================================
# EXECUTION
# ============================================
if __name__ == "__main__":
    main()