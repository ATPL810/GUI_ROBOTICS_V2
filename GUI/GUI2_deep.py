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
from collections import defaultdict

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
            
            # Drop zone position
            self.DROP_ZONE = {
                self.SERVO_BASE: 150,
                self.SERVO_SHOULDER: 30,
                self.SERVO_ELBOW: 55,
                self.SERVO_WRIST: 45,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 125
            }
            
            # Predefined object positions (example - you'll replace these)
            self.OBJECT_POSITIONS = {
                'Bolt': {
                    'pre_grab': [52, 35, 49, 45, 89, 125],
                    'post_grab': [60, 45, 50, 90, 90, 135],
                    'mid_points': [
                        [100, 90, 55, 60, 90, 135],
                        [130, 40, 55, 45, 90, 135]
                    ]
                },
                'Hammer': {
                    'pre_grab': [45, 30, 40, 35, 85, 120],
                    'post_grab': [55, 40, 45, 85, 90, 130],
                    'mid_points': [
                        [95, 85, 50, 55, 90, 130],
                        [125, 35, 50, 40, 90, 130]
                    ]
                },
                'Measuring Tape': {
                    'pre_grab': [58, 38, 52, 48, 92, 128],
                    'post_grab': [65, 48, 55, 95, 90, 138],
                    'mid_points': [
                        [105, 92, 58, 65, 90, 138],
                        [135, 42, 58, 48, 90, 138]
                    ]
                }
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
        angles = self.INITIAL_POSITION.copy()
        angles[self.SERVO_WRIST] = (angles[self.SERVO_WRIST])
        
        self.arm.Arm_serial_servo_write6(
            angles[1], angles[2], angles[3],
            angles[4], angles[5], angles[6],
            2000
        )
        time.sleep(2.5)
        print("At initial position")
    
    def go_to_second_position(self):
        angles = self.SECOND_POSITION.copy()
        angles[self.SERVO_WRIST] = (angles[self.SERVO_WRIST])
        
        self.arm.Arm_serial_servo_write6(
            angles[1], angles[2], angles[3],
            angles[4], angles[5], angles[6],
            2000
        )
        time.sleep(2.5)
        print("At second position")
    
    def go_to_third_position(self):
        angles = self.THIRD_POSITION.copy()
        angles[self.SERVO_WRIST] = (angles[self.SERVO_WRIST])
        
        self.arm.Arm_serial_servo_write6(
            angles[1], angles[2], angles[3],
            angles[4], angles[5], angles[6],
            2000
        )
        time.sleep(2.5)
        print("At third position")
    
    def go_to_position(self, position_dict):
        angles = position_dict.copy()
        angles[self.SERVO_WRIST] = (angles[self.SERVO_WRIST])
        
        self.arm.Arm_serial_servo_write6(
            angles[1], angles[2], angles[3],
            angles[4], angles[5], angles[6],
            1000
        )
        time.sleep(1.5)
    
    def move_to_coordinates(self, angles, duration=1000):
        self.arm.Arm_serial_servo_write6(*angles, duration)
        time.sleep(duration/1000 + 0.5)
    
    def smart_grip_object(self, initial_grip=90):
        print("Starting smart grip sequence...")
        
        grip_angle = initial_grip
        prev_angle = grip_angle
        stalled_count = 0
        
        while grip_angle < 180 and stalled_count < 3:
            self.arm.Arm_serial_servo_write(6, grip_angle, 200)
            time.sleep(0.3)
            
            current_angle = grip_angle
            
            if current_angle == prev_angle:
                stalled_count += 1
                print(f"Grip stalled at {current_angle} degrees (count: {stalled_count})")
            else:
                stalled_count = 0
            
            prev_angle = current_angle
            
            if stalled_count >= 2:
                print("Object gripped successfully")
                break
            
            grip_angle += 5
        
        time.sleep(0.5)
        return grip_angle
    
    def release_gripper(self):
        print("Releasing gripper...")
        self.arm.Arm_serial_servo_write(6, 90, 500)
        time.sleep(0.5)
    
    def pick_and_deliver_object(self, object_name):
        if object_name not in self.OBJECT_POSITIONS:
            print(f"No position data for {object_name}")
            return False
        
        try:
            positions = self.OBJECT_POSITIONS[object_name]
            
            # Move to pre-grab position
            print(f"Moving to pre-grab position for {object_name}")
            self.move_to_coordinates(positions['pre_grab'])
            time.sleep(1)
            
            # Smart gripping
            print("Starting smart grip...")
            grip_angle = self.smart_grip_object(initial_grip=125)
            
            # Move to post-grab position
            print("Moving to post-grab position")
            post_grab = positions['post_grab'].copy()
            post_grab[5] = grip_angle
            self.move_to_coordinates(post_grab)
            
            # Move through mid points
            for i, point in enumerate(positions['mid_points']):
                print(f"Moving through point {i+1}")
                point_copy = point.copy()
                point_copy[5] = grip_angle
                self.move_to_coordinates(point_copy)
            
            # Move to drop zone
            print("Moving to drop zone")
            drop_pos = list(self.DROP_ZONE.values())
            drop_pos[5] = grip_angle
            self.move_to_coordinates(drop_pos, 1500)
            
            # Release object
            time.sleep(1)
            self.release_gripper()
            
            # Return to initial position
            time.sleep(1)
            self.go_to_initial_position()
            
            print(f"Successfully delivered {object_name}")
            return True
            
        except Exception as e:
            print(f"Error during pick and deliver: {e}")
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
        
        raise Exception("No camera found!")
    
    def load_yolo_model(self):
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
                            'center_y': (y1 + y2) / 2,
                            'width': x2 - x1,
                            'height': y2 - y1
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
# GARAGE ASSISTANT GUI
# ============================================
class GarageAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Garage Assistant Robotic System")
        self.root.geometry("1200x700")
        
        # Variables
        self.detected_objects = []
        self.object_list = []
        self.current_order = None
        self.arm_controller = None
        self.detector = None
        self.system_ready = False
        
        # Queue for thread-safe GUI updates
        self.gui_queue = queue.Queue()
        
        # Setup GUI
        self.setup_gui()
        
        # Start checking for GUI updates
        self.check_queue()
        
        # Initialize system in separate thread
        threading.Thread(target=self.initialize_system, daemon=True).start()
    
    def setup_gui(self):
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.root.configure(bg='#2c3e50')
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), foreground='white', background='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), foreground='#ecf0f1', background='#2c3e50')
        style.configure('Status.TLabel', font=('Arial', 10), foreground='#bdc3c7', background='#2c3e50')
        style.configure('Object.TLabel', font=('Arial', 10), foreground='white', background='#34495e')
        style.configure('Control.TButton', font=('Arial', 10, 'bold'), padding=10)
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera view
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera frame
        cam_frame = ttk.LabelFrame(left_panel, text="Live Camera View", padding=10)
        cam_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_label = ttk.Label(cam_frame, text="Initializing camera...")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Controls
        right_panel = ttk.Frame(main_container, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status frame
        status_frame = ttk.LabelFrame(right_panel, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Initializing...", style='Status.TLabel')
        self.status_label.pack(fill=tk.X)
        
        self.arm_status_label = ttk.Label(status_frame, text="Arm: Not Connected", style='Status.TLabel')
        self.arm_status_label.pack(fill=tk.X)
        
        self.camera_status_label = ttk.Label(status_frame, text="Camera: Not Connected", style='Status.TLabel')
        self.camera_status_label.pack(fill=tk.X)
        
        # Detected objects frame
        objects_frame = ttk.LabelFrame(right_panel, text="Detected Objects", padding=10)
        objects_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.objects_text = scrolledtext.ScrolledText(objects_frame, height=8, width=40, bg='#34495e', fg='white')
        self.objects_text.pack(fill=tk.BOTH, expand=True)
        
        # Order frame
        order_frame = ttk.LabelFrame(right_panel, text="Place Order", padding=10)
        order_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(order_frame, text="Enter object number to retrieve:", style='Subtitle.TLabel').pack(anchor=tk.W)
        
        order_input_frame = ttk.Frame(order_frame)
        order_input_frame.pack(fill=tk.X, pady=5)
        
        self.order_var = tk.StringVar()
        self.order_entry = ttk.Entry(order_input_frame, textvariable=self.order_var, width=10)
        self.order_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        self.order_button = ttk.Button(order_input_frame, text="Submit Order", 
                                      command=self.submit_order, style='Control.TButton')
        self.order_button.pack(side=tk.LEFT)
        
        # Cancel order button
        self.cancel_button = ttk.Button(order_frame, text="Cancel Current Order", 
                                       command=self.cancel_order, style='Control.TButton')
        self.cancel_button.pack(fill=tk.X, pady=(5, 0))
        
        # Logger frame
        logger_frame = ttk.LabelFrame(right_panel, text="System Log", padding=10)
        logger_frame.pack(fill=tk.BOTH, expand=True)
        
        self.logger_text = scrolledtext.ScrolledText(logger_frame, height=10, bg='#1c2833', fg='#ecf0f1')
        self.logger_text.pack(fill=tk.BOTH, expand=True)
        
        # Disable controls initially
        self.order_button.config(state='disabled')
        self.cancel_button.config(state='disabled')
        self.order_entry.config(state='disabled')
    
    def check_queue(self):
        """Check for messages from worker threads"""
        try:
            while True:
                msg_type, data = self.gui_queue.get_nowait()
                
                if msg_type == 'log':
                    self.add_log(data)
                elif msg_type == 'status':
                    self.status_label.config(text=data)
                elif msg_type == 'arm_status':
                    self.arm_status_label.config(text=data)
                elif msg_type == 'camera_status':
                    self.camera_status_label.config(text=data)
                elif msg_type == 'objects':
                    self.update_objects_display(data)
                elif msg_type == 'camera_image':
                    self.update_camera_image(data)
                elif msg_type == 'system_ready':
                    self.system_ready = True
                    self.order_button.config(state='normal')
                    self.cancel_button.config(state='normal')
                    self.order_entry.config(state='normal')
                    self.status_label.config(text="System Ready")
                    
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_queue)
    
    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logger_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.logger_text.see(tk.END)
    
    def update_objects_display(self, objects):
        self.detected_objects = objects
        self.objects_text.delete(1.0, tk.END)
        
        if objects:
            self.objects_text.insert(tk.END, "Detected Objects:\n")
            self.objects_text.insert(tk.END, "-" * 30 + "\n")
            
            self.object_list = []
            for i, obj in enumerate(objects, 1):
                class_name = obj['class_name']
                self.object_list.append(class_name)
                self.objects_text.insert(tk.END, f"{i}. {class_name}\n")
            
            self.objects_text.insert(tk.END, "\nEnter number (1-{}) to retrieve object.".format(len(objects)))
            self.objects_text.insert(tk.END, "\nEnter 0 to cancel order.")
        else:
            self.objects_text.insert(tk.END, "No objects detected.")
    
    def update_camera_image(self, img_data):
        photo = tk.PhotoImage(data=img_data)
        self.camera_label.config(image=photo)
        self.camera_label.image = photo
    
    def initialize_system(self):
        self.gui_queue.put(('log', "Initializing Garage Assistant System..."))
        self.gui_queue.put(('status', "Initializing..."))
        
        try:
            # Initialize robot arm
            self.gui_queue.put(('log', "Initializing robot arm..."))
            self.gui_queue.put(('arm_status', "Arm: Initializing..."))
            self.arm_controller = RobotArmController()
            self.gui_queue.put(('arm_status', "Arm: Connected"))
            self.gui_queue.put(('log', "Robot arm initialized successfully"))
            
            # Initialize camera
            self.gui_queue.put(('log', "Initializing camera..."))
            self.gui_queue.put(('camera_status', "Camera: Initializing..."))
            self.detector = CameraDetectionSystem()
            self.gui_queue.put(('camera_status', "Camera: Connected"))
            self.gui_queue.put(('log', "Camera initialized successfully"))
            
            # Run automatic snapshot sequence
            self.gui_queue.put(('log', "Starting automatic snapshot sequence..."))
            self.run_automatic_snapshots()
            
            # Start camera feed
            threading.Thread(target=self.camera_feed, daemon=True).start()
            
            self.gui_queue.put(('log', "System initialization complete"))
            self.gui_queue.put(('system_ready', True))
            
        except Exception as e:
            self.gui_queue.put(('log', f"Initialization error: {str(e)}"))
            self.gui_queue.put(('status', "Initialization Failed"))
    
    def run_automatic_snapshots(self):
        try:
            positions = [
                ("Initial Position", self.arm_controller.INITIAL_POSITION, self.arm_controller.go_to_initial_position),
                ("Second Position", self.arm_controller.SECOND_POSITION, self.arm_controller.go_to_second_position),
                ("Third Position", self.arm_controller.THIRD_POSITION, self.arm_controller.go_to_third_position)
            ]
            
            all_detections = []
            
            for pos_name, pos_dict, move_func in positions:
                self.gui_queue.put(('log', f"Moving to {pos_name}..."))
                move_func()
                time.sleep(3)
                
                ret, frame = self.detector.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    detections = self.detector.detect_objects(frame)
                    all_detections.extend(detections)
                    
                    if detections:
                        for det in detections:
                            self.gui_queue.put(('log', f"Detected: {det['class_name']}"))
            
            # Remove duplicates based on object type
            unique_objects = {}
            for det in all_detections:
                obj_name = det['class_name']
                if obj_name not in unique_objects:
                    unique_objects[obj_name] = det
            
            self.gui_queue.put(('objects', list(unique_objects.values())))
            
            # Return to initial position
            self.gui_queue.put(('log', "Returning to initial position..."))
            self.arm_controller.go_to_initial_position()
            
            self.gui_queue.put(('log', "Automatic scan complete"))
            
        except Exception as e:
            self.gui_queue.put(('log', f"Snapshot error: {str(e)}"))
    
    def camera_feed(self):
        while True:
            if self.detector and self.detector.cap.isOpened():
                ret, frame = self.detector.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    # Detect objects in real-time
                    detections = self.detector.detect_objects(frame)
                    
                    # Draw detections
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        color = self.detector.TOOL_COLORS[det['class_id'] % len(self.detector.TOOL_COLORS)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = det['class_name']
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Convert to PhotoImage
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (640, 480))
                    
                    img_data = cv2.imencode('.png', frame)[1].tobytes()
                    self.gui_queue.put(('camera_image', img_data))
            
            time.sleep(0.1)
    
    def submit_order(self):
        if not self.system_ready:
            return
        
        order_text = self.order_var.get().strip()
        if not order_text:
            return
        
        try:
            order_num = int(order_text)
            
            if order_num == 0:
                self.cancel_order()
                return
            
            if 1 <= order_num <= len(self.object_list):
                object_name = self.object_list[order_num - 1]
                self.current_order = object_name
                
                self.gui_queue.put(('log', f"Order placed for: {object_name}"))
                self.gui_queue.put(('status', f"Processing order: {object_name}"))
                
                # Disable controls during processing
                self.order_button.config(state='disabled')
                self.order_entry.config(state='disabled')
                self.cancel_button.config(state='disabled')
                
                # Process order in separate thread
                threading.Thread(target=self.process_order, args=(object_name,), daemon=True).start()
                
                self.order_var.set("")
            else:
                self.gui_queue.put(('log', f"Invalid order number. Please enter 1-{len(self.object_list)} or 0 to cancel."))
                
        except ValueError:
            self.gui_queue.put(('log', "Please enter a valid number"))
    
    def process_order(self, object_name):
        try:
            self.gui_queue.put(('log', f"Starting to retrieve: {object_name}"))
            
            success = self.arm_controller.pick_and_deliver_object(object_name)
            
            if success:
                self.gui_queue.put(('log', f"Successfully delivered: {object_name}"))
                
                # Remove object from list
                if object_name in self.object_list:
                    self.object_list.remove(object_name)
                    self.gui_queue.put(('objects', self.detected_objects))
            else:
                self.gui_queue.put(('log', f"Failed to retrieve: {object_name}"))
            
        except Exception as e:
            self.gui_queue.put(('log', f"Order processing error: {str(e)}"))
        
        finally:
            self.current_order = None
            self.gui_queue.put(('status', "System Ready"))
            self.order_button.config(state='normal')
            self.order_entry.config(state='normal')
            self.cancel_button.config(state='normal')
    
    def cancel_order(self):
        if self.current_order:
            self.gui_queue.put(('log', f"Cancelling order for: {self.current_order}"))
            self.current_order = None
            self.gui_queue.put(('status', "System Ready"))
        else:
            self.gui_queue.put(('log', "No active order to cancel"))
    
    def on_closing(self):
        self.gui_queue.put(('log', "Shutting down system..."))
        
        if self.detector:
            self.detector.release()
        
        if self.arm_controller:
            try:
                self.arm_controller.go_to_initial_position()
            except:
                pass
        
        self.root.destroy()

# ============================================
# MAIN APPLICATION
# ============================================
if __name__ == "__main__":
    print("GARAGE ASSISTANT ROBOTIC SYSTEM")
    print("=" * 70)
    print("Starting GUI application...")
    
    root = tk.Tk()
    app = GarageAssistantGUI(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("GUI initialized. Starting main loop...")
    print("=" * 70)
    
    root.mainloop()