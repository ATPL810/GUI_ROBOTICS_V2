"""
FULLY AUTOMATIC ROBOTIC ARM SYSTEM
Takes 3 snapshots (front, right, left) and saves images with detections
Grabs all detected objects sequentially
"""

import cv2
import time
import numpy as np
import math
from ultralytics import YOLO
import threading
from collections import deque
import os
from datetime import datetime

print("🤖 FULLY AUTOMATIC ROBOTIC ARM SYSTEM - 3 POSITION SCAN")
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
        
        # YOUR SPECIFIED NEUTRAL ANGLES (FRONT POSITION)
        self.NEUTRAL_ANGLES = {
            self.SERVO_BASE: 90,      # Your specified: 90
            self.SERVO_SHOULDER: 115, # Your specified: 115
            self.SERVO_ELBOW: 4,      # Your specified: 4
            self.SERVO_WRIST: 15,     # Your specified: 15
            self.SERVO_WRIST_ROT: 90, # Your specified: 90
            self.SERVO_GRIPPER: 70    # Gripper open (adjust as needed)
        }
        
        # Position angles for 3 snapshots
        self.POSITION_ANGLES = {
            'front': {
                self.SERVO_BASE: 90,      # Center
                self.SERVO_SHOULDER: 115,
                self.SERVO_ELBOW: 4,
                self.SERVO_WRIST: 15,
                self.SERVO_WRIST_ROT: 90,
            },
            'right': {
                self.SERVO_BASE: 135,     # Right side
                self.SERVO_SHOULDER: 115,
                self.SERVO_ELBOW: 4,
                self.SERVO_WRIST: 15,
                self.SERVO_WRIST_ROT: 90,
            },
            'left': {
                self.SERVO_BASE: 45,      # Left side
                self.SERVO_SHOULDER: 115,
                self.SERVO_ELBOW: 4,
                self.SERVO_WRIST: 15,
                self.SERVO_WRIST_ROT: 90,
            }
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
            print("📸 Moving to FRONT position...")
            self.go_to_position('front')
            
            self.connected = True
            print("✅ Arm initialized at front position")
            print(f"   Front angles: {self.POSITION_ANGLES['front']}")
            
        except Exception as e:
            print(f"⚠️ Arm init failed: {e}")
            self.connected = False
    
    def go_to_position(self, position_name):
        """Move to specified position (front, right, left)"""
        if position_name in self.POSITION_ANGLES:
            position_angles = self.POSITION_ANGLES[position_name].copy()
            position_angles[self.SERVO_GRIPPER] = self.GRIPPER_OPEN if self.gripper_open else self.GRIPPER_CLOSED
            success = self.set_multiple_angles(position_angles, 2000)
            print(f"   Moved to {position_name} position")
            return success
        return False
    
    def go_to_neutral(self):
        """Move to front position"""
        return self.go_to_position('front')
    
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
# COORDINATE SYSTEM WITH POSITION AWARENESS
# ============================================
class PositionAwareCoordinateSystem:
    def __init__(self):
        # Camera settings
        self.camera_width = 640
        self.camera_height = 480
        
        # Position-specific adjustments
        self.position_adjustments = {
            'front': {
                'base_offset': 0,
                'x_scale': 0.15,
                'y_scale': 0.08
            },
            'right': {
                'base_offset': 45,  # Already at 135 degrees
                'x_scale': 0.12,
                'y_scale': 0.08
            },
            'left': {
                'base_offset': -45,  # Already at 45 degrees
                'x_scale': 0.12,
                'y_scale': 0.08
            }
        }
        
        print("📍 Position-aware coordinate system initialized")
    
    def calculate_servo_angles(self, pixel_x, pixel_y, bbox_width, bbox_height, current_position):
        """
        Calculate servo angles based on object position and current arm position
        Returns: Dictionary of servo angles for pickup
        """
        # Center of image (where arm is pointing)
        center_x = 320
        center_y = 240
        
        # Calculate offsets from center
        offset_x = pixel_x - center_x
        offset_y = pixel_y - center_y
        
        # Get position-specific adjustments
        adj = self.position_adjustments[current_position]
        
        # Print coordinates to console
        print(f"📐 [{current_position.upper()}] Object at: Pixel({pixel_x}, {pixel_y}), "
              f"Offset({offset_x}, {offset_y}), Size({bbox_width}x{bbox_height})")
        
        # Calculate base angle considering current position
        base_adjust = offset_x * adj['x_scale']
        
        # Adjust shoulder based on Y position and object size
        size_factor = (bbox_width * bbox_height) / (640 * 480)
        shoulder_adjust = offset_y * adj['y_scale'] + size_factor * 20
        
        # Calculate target angles
        # Base needs to be calculated relative to current position
        if current_position == 'front':
            base_angle = 90 + base_adjust
        elif current_position == 'right':
            base_angle = 135 + base_adjust
        elif current_position == 'left':
            base_angle = 45 + base_adjust
        else:
            base_angle = 90 + base_adjust
        
        target_angles = {
            1: base_angle,               # Base (position-aware)
            2: 115 + shoulder_adjust,    # Shoulder
            3: 4 + (size_factor * 10),   # Elbow
            4: 15,                       # Wrist
            5: 90,                       # Wrist rotation
            6: 70                        # Gripper (open)
        }
        
        # Constrain angles
        for servo_id in target_angles:
            target_angles[servo_id] = max(0, min(180, target_angles[servo_id]))
        
        return target_angles

# ============================================
# 3-POSITION AUTO GRAB SYSTEM
# ============================================
class ThreePositionAutoGrabSystem:
    def __init__(self):
        # Initialize components
        self.arm = ArmController()
        self.coord_system = PositionAwareCoordinateSystem()
        
        # Tool classes
        self.class_names = {
            0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
            3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
        }
        
        # YOLO model
        self.model = self.load_yolo_model()
        
        # Camera
        self.cap = self.setup_camera()
        
        # Detection storage
        self.all_detections = []  # Stores all detected objects with position info
        
        # State
        self.is_grabbing = False
        self.grabbed_count = 0
        self.max_grab_attempts = 3
        
        # Create output directory
        self.output_dir = self.create_output_directory()
        
        print("🔍 3-Position Auto-grab system ready")
        print(f"📁 Output folder: {self.output_dir}")
    
    def create_output_directory(self):
        """Create timestamped directory for saving snapshots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"snapshots_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
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
        
        model_paths = ['./best_2s.pt']
        
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
    
    def capture_and_save_snapshot(self, position_name):
        """
        Capture a snapshot at current position, detect objects, and save image with detections
        Returns: List of detections with position info
        """
        print(f"\n📸 Capturing {position_name} snapshot...")
        
        # Ensure we're at the right position
        self.arm.go_to_position(position_name)
        time.sleep(1.5)  # Wait for arm to stabilize
        
        # Capture frame
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print(f"⚠️ Failed to capture {position_name} snapshot")
                return []
        else:
            # Create test frame if no camera
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"{position_name.upper()} POSITION", 
                       (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Annotate frame with detections
        annotated_frame = self.annotate_frame(frame, detections, position_name)
        
        # Save annotated frame
        filename = f"{self.output_dir}/{position_name}_snapshot.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"   💾 Saved {position_name} snapshot: {filename}")
        
        # Save detection info to text file
        self.save_detection_info(position_name, detections, filename)
        
        # Add position info to each detection
        for det in detections:
            det['position'] = position_name
            det['snapshot_file'] = filename
        
        return detections
    
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
    
    def annotate_frame(self, frame, detections, position_name):
        """Annotate frame with detection results"""
        annotated = frame.copy()
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x, center_y = int(det['center_x']), int(det['center_y'])
            cv2.circle(annotated, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add position label
        cv2.putText(annotated, f"Position: {position_name.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add detection count
        cv2.putText(annotated, f"Detections: {len(detections)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, timestamp, 
                   (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return annotated
    
    def save_detection_info(self, position_name, detections, image_filename):
        """Save detection information to text file"""
        info_filename = f"{self.output_dir}/{position_name}_detections.txt"
        
        with open(info_filename, 'w') as f:
            f.write(f"Position: {position_name}\n")
            f.write(f"Image: {os.path.basename(image_filename)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total detections: {len(detections)}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, det in enumerate(detections, 1):
                f.write(f"Detection #{i}:\n")
                f.write(f"  Class: {det['class_name']}\n")
                f.write(f"  Confidence: {det['confidence']:.3f}\n")
                f.write(f"  Bounding Box: {det['bbox']}\n")
                f.write(f"  Center: ({det['center_x']:.1f}, {det['center_y']:.1f})\n")
                f.write(f"  Size: {det['width']}x{det['height']} pixels\n")
                f.write("-" * 30 + "\n")
    
    def scan_all_positions(self):
        """
        Scan all 3 positions and collect all detections
        Returns: List of all detected objects
        """
        print("\n" + "="*70)
        print("🔄 STARTING 3-POSITION SCAN")
        print("="*70)
        
        self.all_detections = []
        positions = ['front', 'right', 'left']
        
        for position in positions:
            # Move to position and capture snapshot
            detections = self.capture_and_save_snapshot(position)
            
            if detections:
                print(f"   ✅ Found {len(detections)} objects in {position} position")
                self.all_detections.extend(detections)
            else:
                print(f"   ⚠️ No objects found in {position} position")
            
            time.sleep(1)  # Brief pause between positions
        
        # Return to front position
        self.arm.go_to_position('front')
        
        print(f"\n📊 TOTAL DETECTIONS: {len(self.all_detections)} objects")
        
        # Save summary file
        self.save_scan_summary()
        
        return self.all_detections
    
    def save_scan_summary(self):
        """Save summary of all detections"""
        summary_filename = f"{self.output_dir}/scan_summary.txt"
        
        with open(summary_filename, 'w') as f:
            f.write("3-POSITION SCAN SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Scan time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total objects detected: {len(self.all_detections)}\n\n")
            
            # Count by class
            class_counts = {}
            position_counts = {'front': 0, 'right': 0, 'left': 0}
            
            for det in self.all_detections:
                class_name = det['class_name']
                position = det['position']
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                position_counts[position] = position_counts.get(position, 0) + 1
            
            f.write("Objects by class:\n")
            for class_name, count in class_counts.items():
                f.write(f"  {class_name}: {count}\n")
            
            f.write("\nObjects by position:\n")
            for position, count in position_counts.items():
                f.write(f"  {position}: {count}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("DETAILED LIST:\n\n")
            
            for i, det in enumerate(self.all_detections, 1):
                f.write(f"{i}. {det['class_name']} (Confidence: {det['confidence']:.3f})\n")
                f.write(f"   Position: {det['position']}\n")
                f.write(f"   Center: ({det['center_x']:.1f}, {det['center_y']:.1f})\n")
                f.write(f"   Snapshot: {os.path.basename(det['snapshot_file'])}\n")
                f.write("-"*40 + "\n")
    
    def execute_grab_for_detection(self, detection):
        """
        Execute grab sequence for one detected object
        Modified to handle objects from different positions
        """
        if self.is_grabbing:
            return False
        
        self.is_grabbing = True
        
        try:
            print(f"\n{'='*60}")
            print(f"🤖 GRABBING: {detection['class_name']} from {detection['position']} position")
            print(f"{'='*60}")
            
            # First, move to the position where object was detected
            print(f"   [0] Moving to {detection['position']} position...")
            self.arm.go_to_position(detection['position'])
            time.sleep(1.5)
            
            # Calculate servo angles for this object
            target_angles = self.coord_system.calculate_servo_angles(
                detection['center_x'], detection['center_y'],
                detection['width'], detection['height'],
                detection['position']
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
            
            # 8. Return to front position
            print("   [7] Returning to front position...")
            self.arm.go_to_position('front')
            time.sleep(1.0)
            
            self.grabbed_count += 1
            print(f"\n✅ Successfully grabbed {detection['class_name']} from {detection['position']}")
            print(f"📊 Total grabbed: {self.grabbed_count}")
            
            # Mark this detection as grabbed
            detection['grabbed'] = True
            detection['grab_time'] = datetime.now().strftime("%H:%M:%S")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Grab failed for {detection['class_name']}: {e}")
            # Emergency recovery
            self.arm.open_gripper()
            self.arm.go_to_position('front')
            
            # Mark as failed
            detection['grab_failed'] = True
            detection['error'] = str(e)
            
            return False
        
        finally:
            self.is_grabbing = False
    
    def grab_all_objects(self):
        """Grab all detected objects one by one"""
        if not self.all_detections:
            print("\n⚠️ No objects to grab!")
            return
        
        print(f"\n{'='*70}")
        print(f"🤖 STARTING GRAB SEQUENCE FOR {len(self.all_detections)} OBJECTS")
        print(f"{'='*70}")
        
        # Sort objects by position for efficiency
        # Start with front, then left, then right (but you can customize this)
        position_order = ['front', 'left', 'right']
        sorted_detections = sorted(
            self.all_detections,
            key=lambda x: position_order.index(x['position']) if x['position'] in position_order else 3
        )
        
        grab_cooldown = 2.0  # Seconds between grabs
        
        for i, detection in enumerate(sorted_detections, 1):
            print(f"\n🎯 Processing object {i}/{len(sorted_detections)}")
            print(f"   {detection['class_name']} from {detection['position']} position")
            
            # Execute grab
            success = self.execute_grab_for_detection(detection)
            
            if success:
                print(f"   ✅ Grab successful")
            else:
                print(f"   ❌ Grab failed")
            
            # Cooldown between grabs (except after last one)
            if i < len(sorted_detections):
                print(f"   ⏳ Waiting {grab_cooldown} seconds before next grab...")
                time.sleep(grab_cooldown)
        
        print(f"\n{'='*70}")
        print(f"✅ GRAB SEQUENCE COMPLETE")
        print(f"📊 Successfully grabbed: {self.grabbed_count}/{len(self.all_detections)} objects")
        print(f"{'='*70}")
        
        # Save grab results
        self.save_grab_results()
    
    def save_grab_results(self):
        """Save results of grab operations"""
        results_filename = f"{self.output_dir}/grab_results.txt"
        
        with open(results_filename, 'w') as f:
            f.write("GRAB OPERATION RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total objects attempted: {len(self.all_detections)}\n")
            f.write(f"Successfully grabbed: {self.grabbed_count}\n")
            f.write(f"Success rate: {self.grabbed_count/len(self.all_detections)*100:.1f}%\n\n")
            
            grabbed_count = sum(1 for d in self.all_detections if d.get('grabbed', False))
            f.write(f"Grabbed: {grabbed_count}\n")
            f.write(f"Failed: {len(self.all_detections) - grabbed_count}\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-"*50 + "\n")
            
            for i, det in enumerate(self.all_detections, 1):
                status = "✅ GRABBED" if det.get('grabbed', False) else "❌ FAILED"
                f.write(f"{i}. {det['class_name']} ({det['position']}) - {status}\n")
                if det.get('grab_time'):
                    f.write(f"   Time: {det['grab_time']}\n")
                if det.get('error'):
                    f.write(f"   Error: {det['error']}\n")
                f.write("\n")
    
    def run(self):
        """Main automatic sequence: Scan -> Save -> Grab All"""
        print("\n🚀 Starting 3-POSITION AUTO GRAB SYSTEM...")
        print("   Sequence: 1. Scan 3 positions → 2. Save snapshots → 3. Grab all objects")
        print("=" * 70)
        
        # Ensure arm is at front position
        self.arm.go_to_position('front')
        time.sleep(2.0)
        
        try:
            # STEP 1: Scan all 3 positions
            print("\n🔍 STEP 1: Scanning all 3 positions...")
            self.scan_all_positions()
            
            if not self.all_detections:
                print("\n⚠️ No objects detected. Stopping.")
                return
            
            # Brief pause before grabbing
            print("\n⏳ Preparing to grab objects...")
            time.sleep(2)
            
            # STEP 2: Grab all detected objects
            print("\n🤖 STEP 2: Grabbing all detected objects...")
            self.grab_all_objects()
            
            print(f"\n🎉 MISSION COMPLETE!")
            print(f"📁 All snapshots and results saved in: {self.output_dir}")
            
            # Show final statistics
            print(f"\n📊 FINAL STATISTICS:")
            print(f"   Total objects detected: {len(self.all_detections)}")
            print(f"   Successfully grabbed: {self.grabbed_count}")
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\n❌ System error: {e}")
        finally:
            # Cleanup
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Return to front position
            self.arm.go_to_position('front')
            
            print(f"\n📁 All data saved in: {self.output_dir}")
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
    system = ThreePositionAutoGrabSystem()
    system.run()