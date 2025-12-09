"""
GARAGE ASSISTANT - SIMPLIFIED VERSION WITH ALL 9 POINTS
Core functionality with complete grabbing sequences
"""

import cv2
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from ultralytics import YOLO
from Arm_Lib import Arm_Device
import numpy as np

class GarageAssistant:
    def __init__(self):
        # Initialize components
        self.arm = Arm_Device()
        time.sleep(2)
        
        # Load YOLO model
        try:
            self.model = YOLO('./best_best.pt')
            self.model.overrides['conf'] = 0.35
            self.model.overrides['iou'] = 0.3
            self.model.overrides['max_det'] = 6
            self.model.overrides['verbose'] = False
        except:
            print("Warning: Could not load YOLO model")
            self.model = None
        
        # Camera setup
        self.cap = self.setup_camera()
        
        # Tool classes
        self.TOOL_CLASSES = ['Bolt', 'Hammer', 'Measuring Tape', 'Plier', 'Screwdriver', 'Wrench']
        
        # Store detected objects
        self.objects_detected = []
        self.object_positions = {}  # object_name -> point_name
        self.current_order = None
        
        # Initialize arm
        self.init_arm()
        
    def setup_camera(self):
        """Setup camera"""
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
        return None
    
    def init_arm(self):
        """Initialize arm to home position"""
        # Initial position (servo4: -35 becomes 145)
        self.arm.Arm_serial_servo_write6(90, 105, 45, -35, 90, 90, 2000)
        time.sleep(2.5)
    
    def take_snapshots(self):
        """Take 3 snapshots at different positions"""
        positions = [
            (90, "Position 1"),
            (40, "Position 2"), 
            (1, "Position 3")
        ]
        
        detected_objects = []
        
        for angle, pos_name in positions:
            print(f"Moving to {pos_name} (base: {angle}°)")
            self.arm.Arm_serial_servo_write6(angle, 105, 45, -35, 90, 90, 2000)
            time.sleep(3)
            
            # Detect objects
            objects_in_position = self.detect_objects_in_position(pos_name, angle)
            detected_objects.extend(objects_in_position)
        
        # Return to home
        self.arm.Arm_serial_servo_write6(90, 105, 45, -35, 90, 90, 2000)
        time.sleep(1)
        
        # Store detected objects
        self.objects_detected = detected_objects
        
        return detected_objects
    
    def detect_objects_in_position(self, pos_name, base_angle):
        """Detect objects in current position"""
        if self.cap is None or self.model is None:
            return []
        
        # Clear camera buffer
        for _ in range(3):
            self.cap.read()
        
        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            return []
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Run detection
        try:
            results = self.model(frame, conf=0.35, iou=0.3, max_det=6, verbose=False)
            detections = []
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for i, class_id in enumerate(class_ids):
                    if class_id < len(self.TOOL_CLASSES):
                        object_name = self.TOOL_CLASSES[class_id]
                        confidence = float(confidences[i])
                        
                        # Map position to point (Position 1 -> PointA, etc.)
                        point_name = f"Point{chr(65 + len(detections))}"  # A, B, C, etc.
                        
                        detections.append({
                            'name': object_name,
                            'point': point_name,
                            'position': pos_name,
                            'confidence': confidence,
                            'base_angle': base_angle
                        })
                        
                        # Store mapping
                        self.object_positions[object_name] = point_name
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    # ============================================
    # ALL 9 GRABBING FUNCTIONS
    # ============================================
    
    def grab_point_A(self, tool_type="bolt"):
        """Grabbing point A"""
        print(f"Grabbing from Point A - {tool_type}")
        
        # Get grip force based on tool
        grip_force = self.get_grip_force(tool_type)
        
        # Move to pre-grab position
        self.arm.Arm_serial_servo_write6(72, 33, 31, 59, 110, 89, 1000)
        time.sleep(2)   
        self.arm.Arm_serial_servo_write6(72, 35, 31, 59, 85, 112, 1000)
        
        # Grab
        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, grip_force, 1000)

        # Move while holding
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(69, 58, 39, 59, 118, grip_force, 1000)
        time.sleep(2)
        
        # Move to drop zone approach
        self.arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        # Final approach to drop zone
        self.arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        
        # Drop object
        self.arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, 90, 500)  # Release
        
        # Return home
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        print(f"Point A completed for {tool_type}")
        return True
    
    def grab_point_B(self, tool_type="wrench"):
        """Grabbing point B"""
        print(f"Grabbing from Point B - {tool_type}")
        
        grip_force = self.get_grip_force(tool_type)
        
        self.arm.Arm_serial_servo_write6(81, 33, 54, 17, 76, 139, 1000)
        time.sleep(2)   
        self.arm.Arm_serial_servo_write6(81, 29, 55, 18, 76, 150, 1000)
        
        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, grip_force, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write6(81, 29, 54, 18, 76, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        self.arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, 90, 500)
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        return True
    
    def grab_point_C(self, tool_type="bolt"):
        """Grabbing point C"""
        print(f"Grabbing from Point C - {tool_type}")
        
        grip_force = self.get_grip_force(tool_type)
        
        self.arm.Arm_serial_servo_write6(91, 47, 6, 85, 89, 91, 1000)
        time.sleep(2)   
        self.arm.Arm_serial_servo_write6(91, 45, 6, 85, 89, 119, 1000)
        
        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, grip_force, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write6(103, 54, 40, 34, 90, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        self.arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, 90, 500)
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        return True
    
    def grab_point_D(self, tool_type="wrench"):
        """Grabbing point D"""
        print(f"Grabbing from Point D - {tool_type}")
        
        grip_force = self.get_grip_force(tool_type)
        
        self.arm.Arm_serial_servo_write6(105, 40, 25, 50, 89, 111, 1000)
        time.sleep(2)   
        self.arm.Arm_serial_servo_write6(105, 36, 27, 53, 90, 123, 1000)
        
        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, grip_force, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write6(103, 54, 40, 34, 90, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        self.arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, 90, 500)
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        return True
    
    def grab_point_E(self, tool_type="hammer"):
        """Grabbing point E"""
        print(f"Grabbing from Point E - {tool_type}")
        
        grip_force = self.get_grip_force(tool_type)
        
        self.arm.Arm_serial_servo_write6(31, 39, 48, 22, 101, 133, 1000)
        time.sleep(2)   
        self.arm.Arm_serial_servo_write6(31, 33, 48, 22, 102, 140, 1000)
        
        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, grip_force, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write6(31, 65, 71, 22, 102, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        self.arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, 90, 500)
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        return True
    
    def grab_point_F(self, tool_type="bolt"):
        """Grabbing point F"""
        print(f"Grabbing from Point F - {tool_type}")
        
        grip_force = self.get_grip_force(tool_type)
        
        self.arm.Arm_serial_servo_write6(38, 33, 53, 35, 90, 142, 1000)
        time.sleep(2)   
        self.arm.Arm_serial_servo_write6(38, 25, 54, 44, 90, 143, 1000)
        
        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, grip_force, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write6(37, 36, 71, 44, 80, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        self.arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 130, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, 90, 500)
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        return True
    
    def grab_point_G(self, tool_type="plier"):
        """Grabbing point G"""
        print(f"Grabbing from Point G - {tool_type}")
        
        grip_force = self.get_grip_force(tool_type)
        
        self.arm.Arm_serial_servo_write6(50, 17, 93, 1, 90, 59, 1000)
        time.sleep(2)   
        self.arm.Arm_serial_servo_write6(49, 2, 93, 16, 89, 59, 1000)
        
        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, grip_force, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write6(50, 30, 107, 18, 89, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        self.arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 124, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, 80, 500)
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 80, 1000)
        return True
    
    def grab_point_H(self, tool_type="measuring_tape"):
        """Grabbing point H"""
        print(f"Grabbing from Point H - {tool_type}")
        
        grip_force = self.get_grip_force(tool_type)
        
        self.arm.Arm_serial_servo_write6(0, 23, 90, 2, 40, 62, 1000)
        time.sleep(2)   
        self.arm.Arm_serial_servo_write6(-15, 19, 90, 6, 40, 45, 1000)
        
        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, grip_force, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write6(50, 30, 107, 18, 89, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
        time.sleep(2)
        
        self.arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 100, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, 80, 500)
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 80, 1000)
        return True
    
    def grab_point_I(self, tool_type="hammer"):
        """Grabbing point I"""
        print(f"Grabbing from Point I - {tool_type}")
        
        grip_force = self.get_grip_force(tool_type)
        
        self.arm.Arm_serial_servo_write6(13, 50, 37, 37, 93, 87, 1000)
        time.sleep(2)   
        self.arm.Arm_serial_servo_write6(8, 3, 86, 34, 123, 86, 1000)
        
        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, grip_force, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write6(9, 26, 98, 35, 123, grip_force, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, 163, 1000)
        time.sleep(2)
        
        self.arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, 163, 1000)
        time.sleep(2)
        self.arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 100, 1000)

        time.sleep(2)
        self.arm.Arm_serial_servo_write(6, 80, 500)
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 80, 1000)
        return True
    
    def get_grip_force(self, tool_type):
        """Get appropriate grip force for tool type"""
        tool_lower = tool_type.lower()
        
        if 'bolt' in tool_lower:
            return 177
        elif 'hammer' in tool_lower:
            return 169
        elif 'measuring' in tool_lower:
            return 163
        elif 'plier' in tool_lower:
            return 125
        elif 'screwdriver' in tool_lower:
            return 169
        elif 'wrench' in tool_lower:
            return 176
        else:
            return 160  # Default
    
    def pick_object(self, object_name):
        """Pick and deliver an object"""
        # Get point for this object
        point_name = self.object_positions.get(object_name)
        
        if not point_name:
            print(f"No point found for {object_name}")
            return False
        
        # Convert tool name for grip function
        tool_key = object_name.lower()
        if 'measuring' in tool_key:
            tool_key = 'measuring_tape'
        elif 'plier' in tool_key:
            tool_key = 'plier'
        
        # Execute the appropriate grab function
        if point_name == 'PointA':
            return self.grab_point_A(tool_key)
        elif point_name == 'PointB':
            return self.grab_point_B(tool_key)
        elif point_name == 'PointC':
            return self.grab_point_C(tool_key)
        elif point_name == 'PointD':
            return self.grab_point_D(tool_key)
        elif point_name == 'PointE':
            return self.grab_point_E(tool_key)
        elif point_name == 'PointF':
            return self.grab_point_F(tool_key)
        elif point_name == 'PointG':
            return self.grab_point_G(tool_key)
        elif point_name == 'PointH':
            return self.grab_point_H(tool_key)
        elif point_name == 'PointI':
            return self.grab_point_I(tool_key)
        else:
            print(f"Unknown point: {point_name}")
            return False
    
    def smart_grip(self):
        """Smart gripping algorithm"""
        for angle in range(125, 180, 5):
            self.arm.Arm_serial_servo_write(6, angle, 200)
            time.sleep(0.3)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


class SimpleGUI:
    """Simplified GUI for garage assistant"""
    
    def __init__(self, assistant):
        self.assistant = assistant
        self.root = tk.Tk()
        self.root.title("Garage Assistant - Simplified")
        self.root.geometry("800x600")
        
        self.setup_gui()
        
        # Auto-start detection
        self.auto_scan()
    
    def setup_gui(self):
        # Main frames
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top section - Status and controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(top_frame, text="Status: Initializing...", font=('Arial', 12, 'bold'))
        self.status_label.pack(side=tk.LEFT)
        
        # Control buttons
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(side=tk.RIGHT)
        
        ttk.Button(control_frame, text="Rescan", 
                  command=self.auto_scan).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Home",
                  command=self.go_home).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Test Grip",
                  command=self.test_grip).pack(side=tk.LEFT, padx=2)
        
        # Middle section - Objects and order
        middle_frame = ttk.LabelFrame(main_frame, text="Available Objects", padding=10)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Objects display
        self.objects_text = scrolledtext.ScrolledText(middle_frame, height=8)
        self.objects_text.pack(fill=tk.BOTH, expand=True)
        
        # Order section
        order_frame = ttk.Frame(middle_frame)
        order_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(order_frame, text="Order #:").pack(side=tk.LEFT)
        
        self.order_entry = ttk.Entry(order_frame, width=10)
        self.order_entry.pack(side=tk.LEFT, padx=5)
        self.order_entry.bind('<Return>', lambda e: self.submit_order())
        
        ttk.Button(order_frame, text="Get Object", 
                  command=self.submit_order).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(order_frame, text="Cancel Order",
                  command=self.cancel_order).pack(side=tk.LEFT)
        
        # Current order display
        self.order_label = ttk.Label(order_frame, text="Current: None")
        self.order_label.pack(side=tk.RIGHT)
        
        # Bottom section - Logger
        log_frame = ttk.LabelFrame(main_frame, text="System Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.logger = scrolledtext.ScrolledText(log_frame, height=8)
        self.logger.pack(fill=tk.BOTH, expand=True)
    
    def log(self, message):
        self.logger.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.logger.see(tk.END)
        self.logger.update()
    
    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        self.log(message)
    
    def auto_scan(self):
        """Automatic scanning in background"""
        self.update_status("Starting scan...")
        self.order_entry.config(state='disabled')
        threading.Thread(target=self._scan_thread, daemon=True).start()
    
    def _scan_thread(self):
        objects = self.assistant.take_snapshots()
        self.root.after(0, self.update_objects, objects)
        self.root.after(0, lambda: self.order_entry.config(state='normal'))
        self.update_status(f"Scan complete: {len(objects)} objects")
    
    def update_objects(self, objects):
        self.objects_text.delete(1.0, tk.END)
        
        if not objects:
            self.objects_text.insert(tk.END, "No objects detected")
            return
        
        for i, obj in enumerate(objects, 1):
            display_text = f"{i}. {obj['name']}\n"
            display_text += f"   Point: {obj['point']}, Position: {obj['position']}\n"
            display_text += f"   Confidence: {obj['confidence']:.2f}\n"
            display_text += "-" * 40 + "\n"
            self.objects_text.insert(tk.END, display_text)
    
    def submit_order(self):
        try:
            order_num = int(self.order_entry.get())
            self.order_entry.delete(0, tk.END)
            
            if order_num == 0:
                self.cancel_order()
                return
            
            if 1 <= order_num <= len(self.assistant.objects_detected):
                obj = self.assistant.objects_detected[order_num-1]
                obj_name = obj['name']
                
                self.update_status(f"Order placed: {obj_name}")
                self.order_label.config(text=f"Current: {obj_name}")
                self.assistant.current_order = obj_name
                
                # Process in background
                threading.Thread(target=self._process_order, 
                               args=(obj_name,), daemon=True).start()
            else:
                self.log(f"Invalid order number. Choose 1-{len(self.assistant.objects_detected)}")
                
        except ValueError:
            self.log("Please enter a valid number")
    
    def _process_order(self, obj_name):
        self.update_status(f"Retrieving {obj_name}...")
        
        success = self.assistant.pick_object(obj_name)
        
        if success:
            self.update_status(f"Delivered {obj_name}")
            self.log(f"Successfully delivered {obj_name}")
        else:
            self.update_status(f"Failed to retrieve {obj_name}")
            self.log(f"Failed to retrieve {obj_name}")
        
        self.assistant.current_order = None
        self.root.after(0, lambda: self.order_label.config(text="Current: None"))
    
    def cancel_order(self):
        if self.assistant.current_order:
            self.log(f"Cancelled order for {self.assistant.current_order}")
            self.assistant.current_order = None
            self.order_label.config(text="Current: None")
            self.update_status("Order cancelled")
        else:
            self.log("No active order")
    
    def go_home(self):
        self.log("Returning to home position...")
        threading.Thread(target=self._go_home_thread, daemon=True).start()
    
    def _go_home_thread(self):
        self.assistant.init_arm()
        self.log("At home position")
    
    def test_grip(self):
        self.log("Testing gripper...")
        threading.Thread(target=self._test_grip_thread, daemon=True).start()
    
    def _test_grip_thread(self):
        try:
            self.assistant.arm.Arm_serial_servo_write(6, 90, 500)
            time.sleep(1)
            self.assistant.arm.Arm_serial_servo_write(6, 135, 500)
            time.sleep(1)
            self.assistant.arm.Arm_serial_servo_write(6, 90, 500)
            self.log("Gripper test complete")
        except Exception as e:
            self.log(f"Gripper test failed: {e}")
    
    def run(self):
        self.root.mainloop()
    
    def on_closing(self):
        self.log("Shutting down...")
        self.assistant.cleanup()
        self.root.destroy()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("Starting Garage Assistant - Simplified Version")
    print("=" * 60)
    print("Features:")
    print("1. Automatic 3-position scan")
    print("2. All 9 grab points integrated")
    print("3. Tool-specific grip forces")
    print("4. Simple GUI with order system")
    print("=" * 60)
    
    try:
        # Create assistant
        assistant = GarageAssistant()
        
        # Create and run GUI
        gui = SimpleGUI(assistant)
        
        # Handle window closing
        gui.root.protocol("WM_DELETE_WINDOW", gui.on_closing)
        
        # Run
        gui.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("Program ended")