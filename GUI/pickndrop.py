"""
GARAGE ASSISTANT - SIMPLIFIED VERSION
Core functionality only
"""

import cv2
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from ultralytics import YOLO
from Arm_Lib import Arm_Device

class GarageAssistant:
    def __init__(self):
        # Initialize components
        self.arm = Arm_Device()
        self.model = YOLO('./best_2s.pt')  # Your model
        self.objects_detected = []
        self.current_order = None
        
        # Initialize servos
        self.init_arm()
        
    def init_arm(self):
        time.sleep(2)
        # Move to initial position
        self.arm.Arm_serial_servo_write6(90, 105, 45, 145, 90, 90, 2000)
        time.sleep(2.5)
    
    def take_snapshots(self):
        """Take 3 snapshots at different positions"""
        positions = [
            (90, "Position 1"),
            (40, "Position 2"),
            (1, "Position 3")
        ]
        
        all_objects = []
        
        for angle, pos_name in positions:
            # Move to position
            self.arm.Arm_serial_servo_write6(angle, 105, 45, 145, 90, 90, 2000)
            time.sleep(3)
            
            # Detect objects (simplified)
            objects = self.detect_objects()
            all_objects.extend(objects)
            
        # Return to start
        self.arm.Arm_serial_servo_write6(90, 105, 45, 145, 90, 90, 2000)
        
        # Remove duplicates
        unique_objects = list(set(all_objects))
        self.objects_detected = unique_objects
        return unique_objects
    
    def detect_objects(self):
        """Simplified detection - returns object names"""
        # Your detection logic here
        return ['Bolt', 'Hammer', 'Wrench']  # Example
    
    def pick_object(self, object_name):
        """Pick and deliver an object"""
        # Your pre-programmed positions for this object
        positions = self.get_object_positions(object_name)
        
        # Move through positions
        for pos in positions['pre_grab']:
            self.arm.Arm_serial_servo_write6(*pos, 1000)
            time.sleep(1)
        
        # Smart grip
        self.smart_grip()
        
        # Move to drop zone
        for pos in positions['to_drop']:
            self.arm.Arm_serial_servo_write6(*pos, 1000)
            time.sleep(1)
        
        # Release and return
        self.arm.Arm_serial_servo_write(6, 90, 500)
        self.arm.Arm_serial_servo_write6(90, 105, 45, 145, 90, 90, 2000)
    
    def smart_grip(self):
        """Smart gripping algorithm"""
        for angle in range(125, 180, 5):
            self.arm.Arm_serial_servo_write(6, angle, 200)
            time.sleep(0.3)
            # Check if grip stalled (object gripped)
            # Add your stall detection logic here
    
    def get_object_positions(self, object_name):
        """Get pre-programmed positions for an object"""
        # Your 9 positions for each object
        positions = {
            'Bolt': {
                'pre_grab': [[52, 35, 49, 45, 89, 125]],
                'to_drop': [[60, 45, 50, 90, 90, 135],
                           [100, 90, 55, 60, 90, 135],
                           [150, 90, 90, 90, 90, 135]]  # Drop zone
            }
            # Add other objects...
        }
        return positions.get(object_name, positions['Bolt'])


class SimpleGUI:
    """Simplified GUI for garage assistant"""
    
    def __init__(self, assistant):
        self.assistant = assistant
        self.root = tk.Tk()
        self.root.title("Garage Assistant")
        self.root.geometry("800x600")
        
        self.setup_gui()
        
        # Auto-start detection
        self.auto_scan()
    
    def setup_gui(self):
        # Main frames
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Objects display
        self.objects_text = tk.Text(control_frame, height=8, width=30)
        self.objects_text.pack(side=tk.LEFT, padx=(0, 10))
        
        # Order section
        order_frame = ttk.Frame(control_frame)
        order_frame.pack(side=tk.LEFT)
        
        ttk.Label(order_frame, text="Enter object number:").pack()
        
        self.order_entry = ttk.Entry(order_frame, width=10)
        self.order_entry.pack(pady=5)
        
        ttk.Button(order_frame, text="Get Object", 
                  command=self.submit_order).pack()
        
        ttk.Button(order_frame, text="Cancel Order",
                  command=self.cancel_order).pack(pady=5)
        
        ttk.Button(order_frame, text="Rescan",
                  command=self.auto_scan).pack()
        
        # Logger
        log_frame = ttk.LabelFrame(self.root, text="System Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.logger = scrolledtext.ScrolledText(log_frame, height=10)
        self.logger.pack(fill=tk.BOTH, expand=True)
    
    def log(self, message):
        self.logger.insert(tk.END, f"{message}\n")
        self.logger.see(tk.END)
    
    def auto_scan(self):
        """Automatic scanning in background"""
        self.log("Starting automatic scan...")
        threading.Thread(target=self._scan_thread, daemon=True).start()
    
    def _scan_thread(self):
        objects = self.assistant.take_snapshots()
        self.root.after(0, self.update_objects, objects)
        self.log(f"Scan complete. Found: {len(objects)} objects")
    
    def update_objects(self, objects):
        self.objects_text.delete(1.0, tk.END)
        for i, obj in enumerate(objects, 1):
            self.objects_text.insert(tk.END, f"{i}. {obj}\n")
    
    def submit_order(self):
        try:
            order_num = int(self.order_entry.get())
            if 1 <= order_num <= len(self.assistant.objects_detected):
                obj_name = self.assistant.objects_detected[order_num-1]
                self.log(f"Order placed for: {obj_name}")
                self.current_order = obj_name
                
                # Process in background
                threading.Thread(target=self._process_order, 
                               args=(obj_name,), daemon=True).start()
            else:
                self.log("Invalid order number")
        except:
            self.log("Enter a valid number")
    
    def _process_order(self, obj_name):
        self.log(f"Retrieving {obj_name}...")
        self.assistant.pick_object(obj_name)
        self.log(f"Delivered {obj_name}")
        self.current_order = None
    
    def cancel_order(self):
        if self.current_order:
            self.log(f"Cancelled order for {self.current_order}")
            self.current_order = None
        else:
            self.log("No active order")
    
    def run(self):
        self.root.mainloop()


# ============================================
# MAIN EXECUTION - SUPER SIMPLE
# ============================================

if __name__ == "__main__":
    print("Starting Garage Assistant...")
    
    # Create assistant
    assistant = GarageAssistant()
    
    # Create and run GUI
    gui = SimpleGUI(assistant)
    gui.run()