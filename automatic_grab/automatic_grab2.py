"""
FULLY AUTOMATIC ROBOTIC ARM SYSTEM
Auto-detection and grabbing from fixed neutral position
NO USER INPUT REQUIRED - Runs automatically
"""

import cv2
import time
import numpy as np
import math
from ultralytics import YOLO
import threading
from collections import deque, defaultdict
import sys
import os

print("🤖 FULLY AUTOMATIC ROBOTIC ARM SYSTEM")
print("=" * 70)

# ============================================
# ARM_LIB SERVO CONTROL - SIMPLIFIED
# ============================================
class ArmController:
    def __init__(self):
        """Initialize arm with your specified neutral angles"""
        self.connected = False
        self.arm = None
        
        # Try to import Arm_Lib
        try:
            from Arm_Lib import Arm_Device
            self.Arm_Device = Arm_Device
            ARM_LIB_AVAILABLE = True
            print("✅ Arm_Lib found")
        except ImportError:
            print("⚠️ Arm_Lib not found, running in simulation")
            ARM_LIB_AVAILABLE = False
            return
        
        # Servo mapping (YOUR SPECIFIED ANGLES)
        self.SERVO_BASE = 1       # Servo 1: Base
        self.SERVO_SHOULDER = 2   # Servo 2: Shoulder  
        self.SERVO_ELBOW = 3      # Servo 3: Elbow
        self.SERVO_WRIST = 4      # Servo 4: Wrist
        self.SERVO_WRIST_ROT = 5  # Servo 5: Wrist rotation
        self.SERVO_GRIPPER = 6    # Servo 6: Gripper
        
        # YOUR SPECIFIED NEUTRAL ANGLES
        self.NEUTRAL_ANGLES = {
            self.SERVO_BASE: 90,      # Your specified: 90
            self.SERVO_SHOULDER: 115, # Your specified: 115
            self.SERVO_ELBOW: 4,      # Your specified: 4
            self.SERVO_WRIST: 15,     # Your specified: 15
            self.SERVO_WRIST_ROT: 90, # Your specified: 90
            self.SERVO_GRIPPER: 70    # Gripper open (adjust as needed)
        }
        
        # Gripper settings
        self.GRIPPER_OPEN = 70
        self.GRIPPER_CLOSED = 110
        
        # Current state
        self.current_angles = [0] + list(self.NEUTRAL_ANGLES.values())  # Index 0 unused
        self.gripper_open = True
        
        # Initialize
        self.initialize_arm()
    
    def initialize_arm(self):
        """Initialize arm connection"""
        try:
            self.arm = self.Arm_Device()
            time.sleep(2)
            
            # Move to neutral position immediately
            print("📸 Moving to SNAPSHOT position...")
            self.go_to_neutral()
            
            self.connected = True
            print("✅ Arm initialized at snapshot position")
            print(f"   Neutral angles: {self.NEUTRAL_ANGLES}")
            
        except Exception as e:
            print(f"⚠️ Arm init failed: {e}")
            self.connected = False
    
    def go_to_neutral(self):
        """Move to snapshot position (YOUR ANGLES)"""
        success = self.set_multiple_angles(self.NEUTRAL_ANGLES, 2000)
        self.gripper_open = True
        return success
    
    def set_servo_angle(self, servo_id, angle, move_time=1000):
        """Set single servo angle"""
        angle = max(0, min(180, angle))
        
        if self.connected:
            try:
                self.arm.Arm_serial_servo_write(servo_id, angle, move_time)
                self.current_angles[servo_id] = angle
                time.sleep(move_time / 1000)
                return True
            except Exception as e:
                print(f"    [ERROR] Servo {servo_id}: {e}")
                return False
        else:
            # Simulation
            print(f"    [SIM] Servo {servo_id} → {angle}°")
            self.current_angles[servo_id] = angle
            time.sleep(move_time / 1000)
            return True
    
    def set_multiple_angles(self, angles_dict, move_time=1000):
        """Set multiple servos"""
        try:
            if self.connected and len(angles_dict) == 6:
                # Set all 6 servos at once
                angles = [angles_dict.get(i, 90) for i in range(1, 7)]
                self.arm.Arm_serial_servo_write6(*angles, move_time)
                
                for servo_id, angle in angles_dict.items():
                    self.current_angles[servo_id] = angle
                
                time.sleep(move_time / 1000)
                return True
        except:
            pass
        
        # Fallback: Set individually
        for servo_id, angle in angles_dict.items():
            self.set_servo_angle(servo_id, angle, 0)
        
        time.sleep(move_time / 1000)
        return True
    
    def open_gripper(self):
        """Open gripper"""
        success = self.set_servo_angle(self.SERVO_GRIPPER, self.GRIPPER_OPEN, 500)
        if success:
            self.gripper_open = True
        return success
    
    def close_gripper(self):
        """Close gripper"""
        success = self.set_servo_angle(self.SERVO_GRIPPER, self.GRIPPER_CLOSED, 500)
        if success:
            self.gripper_open = False
        return success

# ============================================
# SIMPLE COORDINATE SYSTEM
# ============================================
class SimpleCoordinateSystem:
    def __init__(self):
        # Camera settings
        self.camera_width = 640
        self.camera_height = 480
        
        # Simple mapping: pixel to servo offsets
        # These need calibration for YOUR setup!
        self.pixel_to_servo_offset = {
            'base': 0.15,      # Degrees per pixel offset from center
            'shoulder': 0.08,  # Adjust these values!
            'elbow': 0.06,
            'wrist': 0.04
        }
        
        print("📍 Simple coordinate system initialized")
    
    def calculate_servo_angles(self, pixel_x, pixel_y, bbox_width, bbox_height):
        """
        Calculate servo angles based on object position
        Returns: Dictionary of servo angles for pickup
        """
        # Center of image (where arm is pointing)
        center_x = 320
        center_y = 240
        
        # Calculate offsets from center
        offset_x = pixel_x - center_x
        offset_y = pixel_y - center_y
        
        # Print coordinates to console (as requested)
        print(f"📐 Object at: Pixel({pixel_x}, {pixel_y}), "
              f"Offset({offset_x}, {offset_y}), "
              f"Size({bbox_width}x{bbox_height})")
        
        # Simple calculation: adjust base angle based on X position
        # More X offset = more base rotation
        base_adjust = offset_x * self.pixel_to_servo_offset['base']
        
        # Adjust shoulder based on Y position and object size
        # Higher objects (smaller Y) need shoulder down, larger objects need more adjustment
        size_factor = (bbox_width * bbox_height) / (640 * 480)
        shoulder_adjust = offset_y * self.pixel_to_servo_offset['shoulder'] + size_factor * 20
        
        # Calculate target angles (add to neutral)
        target_angles = {
            1: 90 + base_adjust,           # Base
            2: 115 + shoulder_adjust,      # Shoulder
            3: 4 + (size_factor * 10),     # Elbow (lower for larger objects)
            4: 15,                         # Wrist (keep level)
            5: 90,                         # Wrist rotation
            6: 70                          # Gripper (open)
        }
        
        # Constrain angles
        for servo_id in target_angles:
            target_angles[servo_id] = max(0, min(180, target_angles[servo_id]))
        
        return target_angles

# ============================================
# AUTOMATIC DETECTION AND GRABBING SYSTEM
# ============================================
class AutoGrabSystem:
    def __init__(self):
        # Initialize components
        self.arm = ArmController()
        self.coord_system = SimpleCoordinateSystem()
        
        # Tool classes
        self.class_names = {
            0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
            3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
        }
        
        # YOLO model
        self.model = self.load_yolo_model()
        
        # Camera
        self.cap = self.setup_camera()
        
        # Detection buffer
        self.detection_buffer = deque(maxlen=10)
        
        # State
        self.is_grabbing = False
        self.grabbed_count = 0
        self.max_grab_attempts = 3
        
        print("🔍 Auto-grab system ready")
        print("   Will run AUTOMATICALLY - no user input needed")
    
    def setup_camera(self):
        """Setup camera"""
        print("📷 Initializing camera...")
        
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Camera found at index {i}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
        
        print("⚠️ No camera found")
        return None
    
    def load_yolo_model(self):
        """Load YOLO model"""
        print("📦 Loading YOLO model...")
        
        model_paths = ['best.pt', 'yolo11n.pt', 'yolov8n.pt']
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = YOLO(path)
                    model.overrides['conf'] = 0.4
                    model.overrides['iou'] = 0.3
                    model.overrides['max_det'] = 10
                    print(f"✅ Model loaded: {path}")
                    return model
                except:
                    continue
        
        print("❌ No YOLO model found")
        return None
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, conf=0.4, iou=0.3, verbose=False)
            detections = []
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    
                    if class_id in self.class_names:
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'class_id': class_id,
                            'class_name': self.class_names[class_id],
                            'confidence': confidence,
                            'center_x': (x1 + x2) / 2,
                            'center_y': (y1 + y2) / 2,
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'timestamp': time.time()
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return []
    
    def choose_target(self, detections):
        """Choose which object to grab"""
        if not detections:
            return None
        
        # Pick the object closest to center
        closest = None
        min_distance = float('inf')
        
        for det in detections:
            # Distance from image center (320, 240)
            distance = math.sqrt((det['center_x'] - 320)**2 + (det['center_y'] - 240)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest = det
        
        return closest
    
    def execute_grab(self, detection):
        """Execute grab sequence for one object"""
        if self.is_grabbing:
            return False
        
        self.is_grabbing = True
        
        try:
            print(f"\n{'='*60}")
            print(f"🤖 GRABBING: {detection['class_name']}")
            print(f"{'='*60}")
            
            # Calculate servo angles for this object
            target_angles = self.coord_system.calculate_servo_angles(
                detection['center_x'], detection['center_y'],
                detection['width'], detection['height']
            )
            
            print(f"📐 Target angles: {target_angles}")
            
            # ----- GRAB SEQUENCE -----
            
            # 1. Ensure gripper is open
            self.arm.open_gripper()
            time.sleep(0.5)
            
            # 2. Move to approach position (slightly above object)
            approach_angles = target_angles.copy()
            approach_angles[2] += 10  # Lift elbow a bit
            approach_angles[3] -= 5   # Adjust wrist
            
            print("   [1] Moving to approach position...")
            self.arm.set_multiple_angles(approach_angles, 1500)
            time.sleep(0.5)
            
            # 3. Move down to object
            print("   [2] Descending to object...")
            self.arm.set_multiple_angles(target_angles, 1000)
            time.sleep(0.3)
            
            # 4. Close gripper
            print("   [3] Closing gripper...")
            self.arm.close_gripper()
            time.sleep(0.5)
            
            # 5. Lift object
            print("   [4] Lifting object...")
            lift_angles = target_angles.copy()
            lift_angles[2] += 15  # Lift elbow
            lift_angles[3] -= 5   # Adjust wrist
            self.arm.set_multiple_angles(lift_angles, 1000)
            time.sleep(0.5)
            
            # 6. Move to drop position (simple right turn)
            print("   [5] Moving to drop position...")
            drop_angles = {
                1: 135,   # Base turned right
                2: 100,   # Shoulder
                3: 20,    # Elbow
                4: 15,    # Wrist
                5: 90,    # Wrist rotation
                6: self.arm.GRIPPER_CLOSED  # Keep closed
            }
            self.arm.set_multiple_angles(drop_angles, 1500)
            time.sleep(0.5)
            
            # 7. Open gripper to drop
            print("   [6] Dropping object...")
            self.arm.open_gripper()
            time.sleep(0.5)
            
            # 8. Return to snapshot position
            print("   [7] Returning to snapshot position...")
            self.arm.go_to_neutral()
            time.sleep(1.0)
            
            self.grabbed_count += 1
            print(f"\n✅ Successfully grabbed {detection['class_name']}")
            print(f"📊 Total grabbed: {self.grabbed_count}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Grab failed: {e}")
            # Emergency recovery
            self.arm.open_gripper()
            self.arm.go_to_neutral()
            return False
        
        finally:
            self.is_grabbing = False
    
    def run(self):
        """Main automatic loop"""
        print("\n🚀 Starting FULLY AUTOMATIC system...")
        print("   System will run until stopped with Ctrl+C")
        print("=" * 70)
        
        # Ensure arm is at snapshot position
        self.arm.go_to_neutral()
        time.sleep(2.0)
        
        # Detection cooldown to prevent rapid consecutive grabs
        last_grab_time = 0
        grab_cooldown = 3.0  # Seconds between grabs
        
        frame_count = 0
        detection_interval = 5  # Process every 5th frame
        
        try:
            while True:
                # Read camera frame
                if self.cap is not None:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("⚠️ Camera error")
                        time.sleep(1)
                        continue
                else:
                    # Test frame if no camera
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Process detection every N frames
                if frame_count % detection_interval == 0 and not self.is_grabbing:
                    # Detect objects
                    detections = self.detect_objects(frame)
                    
                    if detections:
                        print(f"\n📸 Snapshot analysis: Found {len(detections)} objects")
                        
                        # Choose target
                        target = self.choose_target(detections)
                        
                        if target and time.time() - last_grab_time > grab_cooldown:
                            print(f"🎯 Selected: {target['class_name']} "
                                  f"(Confidence: {target['confidence']:.2f})")
                            
                            # Execute grab in background thread
                            threading.Thread(
                                target=self.execute_grab,
                                args=(target,),
                                daemon=True
                            ).start()
                            
                            last_grab_time = time.time()
                
                # Simple display (optional - can be removed)
                if frame_count % 30 == 0:  # Update display every 30 frames
                    display_frame = frame.copy()
                    
                    # Draw status
                    status = f"Grabbed: {self.grabbed_count} | State: {'Grabbing' if self.is_grabbing else 'Ready'}"
                    cv2.putText(display_frame, status, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Auto Grab System (Press Q to quit)', display_frame)
                
                # Check for quit (only Q key, no other controls)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 User requested quit")
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user (Ctrl+C)")
        
        finally:
            # Cleanup
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Return to neutral
            self.arm.go_to_neutral()
            
            print(f"\n📊 Final count: Grabbed {self.grabbed_count} objects")
            print("✅ System stopped cleanly")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    # Install required packages if missing
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️ Installing ultralytics...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO
    
    # Create and run system
    system = AutoGrabSystem()
    system.run()