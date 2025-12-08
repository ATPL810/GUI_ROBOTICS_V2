"""
GARAGE ASSISTANT ROBOTIC ARM SYSTEM
Combines automatic snapshot detection with GUI control for object retrieval
"""

import cv2
import time
import os
import json
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext
from ultralytics import YOLO
from Arm_Lib import Arm_Device
import numpy as np

print("GARAGE ASSISTANT ROBOTIC ARM SYSTEM")
print("=" * 70)

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
            
            # 9 hardcoded positions for objects
            self.OBJECT_POSITIONS = {
                "PointA": {"servo1": 52, "servo2": 35, "servo3": 49, "servo4": 45, "servo5": 89, "servo6": 90},
                "PointB": {"servo1": 60, "servo2": 45, "servo3": 50, "servo4": 90, "servo5": 90, "servo6": 90},
                "PointC": {"servo1": 100, "servo2": 90, "servo3": 55, "servo4": 60, "servo5": 90, "servo6": 90},
                "PointD": {"servo1": 130, "servo2": 40, "servo3": 55, "servo4": 45, "servo5": 90, "servo6": 90},
                "PointE": {"servo1": 130, "servo2": 30, "servo3": 55, "servo4": 45, "servo5": 90, "servo6": 90},
                "PointF": {"servo1": 70, "servo2": 60, "servo3": 40, "servo4": 70, "servo5": 90, "servo6": 90},
                "PointG": {"servo1": 80, "servo2": 70, "servo3": 30, "servo4": 80, "servo5": 90, "servo6": 90},
                "PointH": {"servo1": 110, "servo2": 50, "servo3": 35, "servo4": 55, "servo5": 90, "servo6": 90},
                "PointI": {"servo1": 95, "servo2": 75, "servo3": 45, "servo4": 65, "servo5": 90, "servo6": 90}
            }
            
            # Drop zone position
            self.DROP_ZONE = {"servo1": 150, "servo2": 90, "servo3": 90, "servo4": 90, "servo5": 90, "servo6": 90}
            
            self.go_to_initial_position()
            print("Robot arm initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize robot arm: {e}")
            raise
    
    def convert_angle(self, angle):
        if angle < 0:
            return 180 + angle
        return angle
    
    def move_to_position(self, position_dict, move_time=1500):
        """Move arm to specific position"""
        servo1 = position_dict.get("servo1", 90)
        servo2 = position_dict.get("servo2", 105)
        servo3 = position_dict.get("servo3", 45)
        servo4 = self.convert_angle(position_dict.get("servo4", -35))
        servo5 = position_dict.get("servo5", 90)
        servo6 = position_dict.get("servo6", 90)
        
        self.arm.Arm_serial_servo_write6(servo1, servo2, servo3, servo4, servo5, servo6, move_time)
        time.sleep(move_time/1000 + 0.5)
    
    def go_to_initial_position(self):
        self.move_to_position(self.INITIAL_POSITION)
    
    def go_to_snapshot_position(self, index):
        if 0 <= index < len(self.SNAPSHOT_POSITIONS):
            pos = self.SNAPSHOT_POSITIONS[index]
            position = {
                "servo1": pos["base"],
                "servo2": 105,
                "servo3": 45,
                "servo4": -35,
                "servo5": 90,
                "servo6": 90
            }
            self.move_to_position(position)
            return True
        return False
    
    def grip_object(self, initial_grip=125, target_grip=145):
        """Gradually grip object until resistance is detected"""
        # Open gripper slightly more than initial
        self.arm.Arm_serial_servo_write(6, initial_grip - 10, 500)
        time.sleep(0.5)
        
        # Move to pre-grab position
        self.arm.Arm_serial_servo_write(6, initial_grip, 300)
        time.sleep(0.3)
        
        # Gradual gripping with small increments
        grip_angle = initial_grip
        grip_increment = 2
        max_grip_angle = target_grip
        
        for i in range(20):  # Max 20 iterations
            grip_angle += grip_increment
            self.arm.Arm_serial_servo_write(6, grip_angle, 100)
            time.sleep(0.1)
            
            # In a real system, you would check current feedback here
            # For simulation, we'll stop when reaching target or after certain angle
            if grip_angle >= max_grip_angle:
                break
        
        return grip_angle
    
    def move_to_object_position(self, position_name):
        """Move to one of the 9 hardcoded object positions"""
        if position_name in self.OBJECT_POSITIONS:
            pos = self.OBJECT_POSITIONS[position_name].copy()
            # Move to approach position (slightly above)
            approach_pos = pos.copy()
            approach_pos["servo2"] += 10  # Lift shoulder
            approach_pos["servo6"] = 90   # Open gripper
            
            self.move_to_position(approach_pos, 1500)
            time.sleep(1)
            
            # Move to actual grab position
            self.move_to_position(pos, 1500)
            return True
        return False
    
    def pick_and_drop_object(self, object_position_name):
        """Complete pick and drop sequence"""
        try:
            # 1. Move to object position
            if not self.move_to_object_position(object_position_name):
                return False
            
            time.sleep(1)
            
            # 2. Grip object
            final_grip = self.grip_object(125, 145)
            print(f"Gripped object at angle: {final_grip}")
            time.sleep(1)
            
            # 3. Lift object
            current_pos = self.OBJECT_POSITIONS[object_position_name].copy()
            lift_pos = current_pos.copy()
            lift_pos["servo2"] += 20  # Lift up
            
            self.move_to_position(lift_pos, 1500)
            time.sleep(1)
            
            # 4. Move to drop zone
            self.move_to_position(self.DROP_ZONE, 2000)
            time.sleep(1)
            
            # 5. Release object
            self.arm.Arm_serial_servo_write(6, 90, 500)
            time.sleep(1)
            
            # 6. Return to initial
            self.go_to_initial_position()
            
            return True
            
        except Exception as e:
            print(f"Error in pick and drop: {e}")
            self.go_to_initial_position()
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
        
        # Configure window
        self.root.title("Garage Assistant - Robotic Arm Controller")
        self.root.geometry("1000x700")
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
        
        self.objects_text = scrolledtext.ScrolledText(objects_frame, height=10, width=40,
                                                     bg='#1e1e1e', fg='#ffffff',
                                                     font=('Courier', 10))
        self.objects_text.pack(fill=tk.BOTH, expand=True)
        
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
        
        # Right panel - Logger
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        log_frame = ttk.LabelFrame(right_frame, text="System Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=30, width=50,
                                                 bg='#1e1e1e', fg='#ffffff',
                                                 font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Camera feed placeholder
        camera_frame = ttk.LabelFrame(right_frame, text="Camera Feed", padding=10)
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.camera_label = ttk.Label(camera_frame, text="Camera feed will appear here", 
                                     background='#1e1e1e', foreground='#888888')
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
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
    
    def update_detected_objects(self, detections_list):
        self.objects_text.delete(1.0, tk.END)
        
        if not detections_list:
            self.objects_text.insert(tk.END, "No objects detected")
            return
        
        self.detected_objects = []
        self.object_positions = {}
        
        # Combine detections from all snapshots
        all_detections = []
        for idx, snapshot in enumerate(detections_list):
            for det in snapshot['detections']:
                # Assign position based on snapshot
                position_name = f"Point{chr(65+idx)}"  # A, B, C, etc.
                all_detections.append({
                    'object': det['class_name'],
                    'position': position_name,
                    'snapshot': snapshot['position_name']
                })
        
        # Display objects
        self.objects_text.insert(tk.END, "Available objects:\n")
        self.objects_text.insert(tk.END, "="*40 + "\n")
        
        for i, det in enumerate(all_detections[:9], 1):  # Limit to 9 objects
            display_text = f"{i}. {det['object']} at {det['position']}\n"
            self.objects_text.insert(tk.END, display_text)
            
            # Store mapping
            self.detected_objects.append(det['object'])
            self.object_positions[det['object']] = det['position']
        
        self.objects_text.insert(tk.END, "\n0. Cancel current order\n")
        self.objects_text.insert(tk.END, "="*40)
        
        self.update_status(f"Found {len(all_detections)} objects")
    
    def process_order(self):
        try:
            order_num = int(self.order_var.get())
            
            if order_num == 0:
                self.cancel_order()
                return
            
            if 1 <= order_num <= len(self.detected_objects):
                object_name = self.detected_objects[order_num-1]
                position_name = self.object_positions.get(object_name, "Unknown")
                
                self.current_order = {
                    'number': order_num,
                    'object': object_name,
                    'position': position_name
                }
                
                self.order_display.configure(text=f"Current order: {order_num} - {object_name}")
                self.log_message(f"Order placed: {object_name} from {position_name}")
                
                # Execute order in separate thread
                threading.Thread(target=self.execute_order, daemon=True).start()
            else:
                self.update_status(f"Invalid order number. Choose 1-{len(self.detected_objects)} or 0", True)
                
        except ValueError:
            self.update_status("Please enter a valid number", True)
    
    def cancel_order(self):
        if self.current_order:
            self.log_message(f"Cancelled order: {self.current_order['object']}")
            self.current_order = None
            self.order_display.configure(text="Current order: None")
            self.update_status("Order cancelled")
        else:
            self.update_status("No active order to cancel")
    
    def execute_order(self):
        if not self.current_order:
            return
        
        object_name = self.current_order['object']
        position_name = self.current_order['position']
        
        self.update_status(f"Executing order: {object_name}")
        
        # Search for object in 3 positions
        found = False
        for pos_idx in range(3):
            self.log_message(f"Checking position {pos_idx+1}...")
            
            # Move to snapshot position
            self.arm.go_to_snapshot_position(pos_idx)
            time.sleep(2)
            
            # Capture and check for object
            ret, frame = self.detector.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                detections = self.detector.detect_objects(frame)
                
                for det in detections:
                    if det['class_name'] == object_name:
                        found = True
                        self.log_message(f"Found {object_name} in position {pos_idx+1}")
                        
                        # Pick and drop object
                        success = self.arm.pick_and_drop_object(position_name)
                        
                        if success:
                            self.update_status(f"Successfully delivered {object_name}")
                            self.log_message(f"Object delivered to drop zone")
                        else:
                            self.update_status(f"Failed to pick up {object_name}", True)
                        
                        break
                
                if found:
                    break
        
        if not found:
            self.update_status(f"Object '{object_name}' not found in any position", True)
            self.log_message("Object not found")
        
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
                        'position_name': f"Position {i+1}",
                        'detections': detections
                    }
                    all_detections.append(snapshot_info)
                    
                    if detections:
                        self.log_message(f"Found {len(detections)} objects in position {i+1}")
                    else:
                        self.log_message(f"No objects in position {i+1}")
            
            # Return to initial position
            self.arm.go_to_initial_position()
            
            # Update GUI with detections
            self.root.after(0, self.update_detected_objects, all_detections)
            self.update_status("Initial scan complete")
            
        except Exception as e:
            self.update_status(f"Scan error: {str(e)}", True)
        
        self.is_scanning = False
    
    def rescan_objects(self):
        if not self.is_scanning:
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
    
    def update_camera_feed(self):
        if hasattr(self, 'detector') and self.detector.cap is not None:
            try:
                ret, frame = self.detector.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    # Resize for display
                    display_frame = cv2.resize(frame, (400, 300))
                    
                    # Convert to RGB for tkinter
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PhotoImage
                    img = tk.PhotoImage(data=cv2.imencode('.png', rgb_frame)[1].tobytes())
                    
                    self.camera_label.configure(image=img)
                    self.camera_label.image = img
            except:
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
    finally:
        print("System shutdown complete")

# ============================================
# EXECUTION
# ============================================
if __name__ == "__main__":
    main()