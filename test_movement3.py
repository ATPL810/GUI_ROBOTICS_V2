"""
COMPLETE YAHBOOM ROBOTIC ARM CONTROL SYSTEM
Real hardware control with YOLO11 detection
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

# Import configuration
from GUI_ROBOTICS_V2.arm_config import *

# ============================================
# DYNAMIC ARM LIBRARY IMPORT
# ============================================
YAHBOOM_AVAILABLE = False
Arm_Device = None

try:
    # Try different import patterns based on your Yahboom kit
    if ARM_TYPE in ["DOFBOT", "ALPHABOT", "ROBOTIC_ARM_PI"]:
        import Arm_Lib
        from Arm_Lib import Arm_Device
        YAHBOOM_AVAILABLE = True
        print(f"✅ Successfully imported Arm_Lib for {ARM_TYPE}")
        
    elif ARM_TYPE == "AI_ROBOTIC_ARM":
        import yahboom
        Arm_Device = yahboom.Arm()
        YAHBOOM_AVAILABLE = True
        print("✅ Successfully imported yahboom library")
        
    else:
        print("⚠️ Unknown ARM_TYPE in config, trying default import...")
        import Arm_Lib
        from Arm_Lib import Arm_Device
        YAHBOOM_AVAILABLE = True
        
except ImportError as e:
    print(f"⚠️ Failed to import Yahboom libraries: {e}")
    print("Running in simulation mode")
    YAHBOOM_AVAILABLE = False

print("🤖 YAHBOOM ROBOTIC ARM CONTROL SYSTEM")
print("=" * 70)

# ============================================
# REAL YAHBOOM ARM CONTROL CLASS
# ============================================
class YahboomRoboticArm:
    def __init__(self):
        """Initialize real Yahboom robotic arm"""
        self.connected = False
        self.arm_device = None
        self.current_angles = [90, 90, 90, 90, GRIPPER_OPEN, 90]  # Initial angles
        self.gripper_open = True
        
        # Initialize arm
        self.initialize_arm()
    
    def initialize_arm(self):
        """Initialize connection to Yahboom arm"""
        try:
            if YAHBOOM_AVAILABLE:
                # Initialize Yahboom arm
                self.arm_device = Arm_Device()
                time.sleep(2)  # Wait for initialization
                
                # Test communication
                self.arm_device.Arm_serial_servo_write(1, 90, 500)
                time.sleep(0.5)
                
                # Set to neutral position
                self.go_to_neutral()
                self.connected = True
                print("✅ Yahboom Arm Connected and Initialized")
                print(f"   Servo Count: {SERVO_COUNT}")
                print(f"   Arm Type: {ARM_TYPE}")
            else:
                print("⚠️ Running in simulation mode")
                self.connected = False
                
        except Exception as e:
            print(f"❌ Failed to initialize Yahboom arm: {e}")
            print("Check: 1) Arm is powered ON")
            print("       2) USB cable is connected")
            print("       3) Correct library is installed")
            self.connected = False
    
    def go_to_neutral(self):
        """Move arm to neutral position (all 90 degrees)"""
        print("    [INIT] Moving to neutral position...")
        
        if self.connected and self.arm_device:
            try:
                # Move all servos to 90 degrees
                for servo in range(1, SERVO_COUNT + 1):
                    min_angle, max_angle = SERVO_LIMITS.get(servo, (0, 180))
                    angle = 90
                    # Ensure within limits
                    angle = max(min_angle, min(max_angle, angle))
                    self.arm_device.Arm_serial_servo_write(servo, angle, 1000)
                
                time.sleep(1.5)
                self.current_angles = [90, 90, 90, 90, GRIPPER_OPEN, 90]
                self.gripper_open = True
                print("    [INIT] Neutral position reached")
                
            except Exception as e:
                print(f"    [ERROR] Failed to go to neutral: {e}")
        else:
            print("    [SIM] Moving to neutral position")
            time.sleep(0.5)
            self.current_angles = [90, 90, 90, 90, GRIPPER_OPEN, 90]
            self.gripper_open = True
    
    def set_servo_angle(self, servo_id, angle, move_time=MOVE_SPEED):
        """Set single servo angle with safety limits"""
        # Get limits for this servo
        min_angle, max_angle = SERVO_LIMITS.get(servo_id, (0, 180))
        
        # Clamp angle to limits
        angle = max(min_angle, min(max_angle, angle))
        
        # Update current angles
        self.current_angles[servo_id - 1] = angle
        
        if self.connected and self.arm_device:
            try:
                print(f"    [SERVO {servo_id}] Moving to {angle}° ({move_time}ms)")
                self.arm_device.Arm_serial_servo_write(servo_id, angle, move_time)
                
                # Wait for movement to complete
                wait_time = move_time / 1000.0 + 0.1
                time.sleep(wait_time)
                return True
                
            except Exception as e:
                print(f"    [ERROR] Servo {servo_id} move failed: {e}")
                return False
        else:
            # Simulation
            print(f"    [SIM] Servo {servo_id} → {angle}° ({move_time}ms)")
            time.sleep(move_time / 1000.0)
            return True
    
    def set_multiple_angles(self, angles_dict, move_time=MOVE_SPEED):
        """Set multiple servo angles simultaneously"""
        if self.connected and self.arm_device:
            try:
                # Prepare servo angles array
                servo_angles = [0] * 6  # For 6 servos
                
                for servo_id, angle in angles_dict.items():
                    # Apply limits
                    min_angle, max_angle = SERVO_LIMITS.get(servo_id, (0, 180))
                    angle = max(min_angle, min(max_angle, angle))
                    servo_angles[servo_id - 1] = angle
                    self.current_angles[servo_id - 1] = angle
                
                # Use appropriate method based on library
                print(f"    [MOVE] Multi-servo move ({move_time}ms)")
                
                # Method 1: Write all servos at once
                if hasattr(self.arm_device, 'Arm_serial_servo_write6'):
                    self.arm_device.Arm_serial_servo_write6(
                        servo_angles[0], servo_angles[1], servo_angles[2],
                        servo_angles[3], servo_angles[4], servo_angles[5],
                        move_time
                    )
                # Method 2: Write individually
                else:
                    for servo_id, angle in angles_dict.items():
                        self.arm_device.Arm_serial_servo_write(servo_id, angle, move_time)
                
                time.sleep(move_time / 1000.0 + 0.1)
                return True
                
            except Exception as e:
                print(f"    [ERROR] Multi-servo move failed: {e}")
                return False
        else:
            # Simulation
            for servo_id, angle in angles_dict.items():
                print(f"    [SIM] Servo {servo_id} → {angle}°")
                self.current_angles[servo_id - 1] = angle
            time.sleep(move_time / 1000.0)
            return True
    
    def open_gripper(self):
        """Open the gripper"""
        print("    [GRIPPER] Opening...")
        success = self.set_servo_angle(SERVO_GRIPPER, GRIPPER_OPEN, 800)
        if success:
            self.gripper_open = True
            print("    [GRIPPER] Opened")
        return success
    
    def close_gripper(self):
        """Close the gripper"""
        print("    [GRIPPER] Closing...")
        success = self.set_servo_angle(SERVO_GRIPPER, GRIPPER_CLOSED, 800)
        if success:
            self.gripper_open = False
            print("    [GRIPPER] Closed")
        return success
    
    def move_to_position(self, base_angle, shoulder_angle, elbow_angle, wrist_angle=None):
        """Move arm to specific joint angles"""
        angles = {
            SERVO_BASE: base_angle,
            SERVO_SHOULDER: shoulder_angle,
            SERVO_ELBOW: elbow_angle
        }
        
        if wrist_angle is not None:
            angles[SERVO_WRIST] = wrist_angle
        
        return self.set_multiple_angles(angles, MOVE_SPEED)
    
    def move_to_pick_position(self, x_norm, y_norm):
        """
        Move to pick position based on normalized coordinates
        x_norm: 0.0 (left) to 1.0 (right)
        y_norm: 0.0 (far) to 1.0 (near)
        """
        # Convert normalized coordinates to servo angles
        # These formulas need calibration for your specific arm!
        
        # Base rotation (left-right)
        base_angle = 30 + (x_norm * 120)  # 30° to 150°
        
        # Shoulder (forward-back)
        shoulder_angle = 140 - (y_norm * 60)  # 140° to 80°
        
        # Elbow (height)
        elbow_angle = 80 + (y_norm * 40)  # 80° to 120°
        
        # Wrist (keep level)
        wrist_angle = 90
        
        print(f"    [PICK] Moving to ({x_norm:.2f}, {y_norm:.2f})")
        print(f"    [PICK] Angles: Base={base_angle:.0f}°, Shoulder={shoulder_angle:.0f}°, Elbow={elbow_angle:.0f}°")
        
        return self.move_to_position(base_angle, shoulder_angle, elbow_angle, wrist_angle)
    
    def move_to_drop_position(self, tool_index):
        """Move to drop position for specific tool"""
        # Different drop positions for each tool
        drop_positions = [
            (90, 100, 100, 90),   # Bolt - Position 0
            (60, 100, 100, 90),   # Hammer - Position 1
            (120, 100, 100, 90),  # Measuring Tape - Position 2
            (90, 120, 80, 90),    # Plier - Position 3
            (60, 120, 80, 90),    # Screwdriver - Position 4
            (120, 120, 80, 90)    # Wrench - Position 5
        ]
        
        if 0 <= tool_index < len(drop_positions):
            base, shoulder, elbow, wrist = drop_positions[tool_index]
            print(f"    [DROP] Moving to drop position for tool {tool_index}")
            return self.move_to_position(base, shoulder, elbow, wrist)
        
        return False
    
    def execute_pick_and_place(self, pick_x, pick_y, tool_index):
        """Execute complete pick and place sequence"""
        try:
            # 1. Open gripper
            self.open_gripper()
            time.sleep(0.5)
            
            # 2. Move above pick position
            self.move_to_pick_position(pick_x, pick_y + 0.1)  # Slightly above
            time.sleep(0.5)
            
            # 3. Move to pick position
            self.move_to_pick_position(pick_x, pick_y)
            time.sleep(0.5)
            
            # 4. Close gripper
            self.close_gripper()
            time.sleep(0.8)  # Let gripper settle
            
            # 5. Lift up
            self.move_to_pick_position(pick_x, pick_y + 0.1)
            time.sleep(0.5)
            
            # 6. Move to drop position
            self.move_to_drop_position(tool_index)
            time.sleep(0.5)
            
            # 7. Open gripper (drop)
            self.open_gripper()
            time.sleep(0.5)
            
            # 8. Return to neutral
            self.go_to_neutral()
            
            return True
            
        except Exception as e:
            print(f"    [ERROR] Pick and place failed: {e}")
            self.open_gripper()
            self.go_to_neutral()
            return False
    
    def emergency_stop(self):
        """Emergency stop - move to safe position"""
        print("    [EMERGENCY] Stopping arm!")
        if self.connected and self.arm_device:
            try:
                self.arm_device.Arm_serial_servo_write6(90, 90, 90, 90, GRIPPER_OPEN, 90, 500)
                time.sleep(0.5)
            except:
                pass
        self.current_angles = [90, 90, 90, 90, GRIPPER_OPEN, 90]
        self.gripper_open = True

# ============================================
# MAIN VISION AND CONTROL SYSTEM
# ============================================
class YahboomVisionSystem:
    def __init__(self):
        print("Initializing Yahboom Vision System...")
        
        # Initialize camera
        self.cap = self.initialize_camera()
        if self.cap is None:
            print("❌ ERROR: Camera not found!")
            sys.exit(1)
        
        # Initialize YOLO model
        self.model = self.initialize_yolo()
        
        # Initialize Yahboom arm
        self.arm = YahboomRoboticArm()
        
        # Tool classes
        self.class_names = {
            0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
            3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
        }
        
        # Tool colors
        self.tool_colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Yellow
        ]
        
        # System state
        self.detections = []
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.is_processing = False
        self.paused = False
        self.emergency_stop = False
        
        # Displacement tracking
        self.displaced_counts = defaultdict(int)
        self.target_count = 1
        
        print("\n✅ System Initialization Complete!")
        print("=" * 70)
        print("Controls:")
        print("  SPACE - Toggle pause")
        print("  e     - Emergency stop")
        print("  n     - Move to neutral")
        print("  o     - Open gripper")
        print("  c     - Close gripper")
        print("  t     - Test movement")
        print("  q     - Quit")
        print("=" * 70)
    
    def initialize_camera(self):
        """Initialize camera with configuration"""
        print("📷 Initializing camera...")
        
        for i in range(4):  # Try different indices
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"✅ Camera found at index {i}")
                
                # Configure camera
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test frame capture
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Camera working: {frame.shape[1]}x{frame.shape[0]}")
                    return cap
                else:
                    cap.release()
        
        print("❌ No working camera found")
        return None
    
    def initialize_yolo(self):
        """Initialize YOLO model"""
        print("🤖 Loading YOLO model...")
        
        model_path = "best.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Please place your 'best.pt' file in the same directory")
            return None
        
        try:
            model = YOLO(model_path)
            
            # Configure model
            model.overrides['conf'] = 0.4
            model.overrides['iou'] = 0.3
            model.overrides['agnostic_nms'] = True
            model.overrides['max_det'] = 6
            model.overrides['verbose'] = False
            
            print("✅ YOLO model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            return None
    
    def process_frame(self, frame):
        """Process frame for object detection"""
        if self.model is None or self.is_processing:
            return []
        
        self.is_processing = True
        
        try:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (320, 240))
            
            # Run detection
            results = self.model(small_frame, 
                               conf=0.4,
                               iou=0.3,
                               imgsz=320,
                               max_det=6,
                               verbose=False,
                               device='cpu')
            
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Scale back to original frame size
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    
                    # Scale coordinates
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    confidence_percent = confidence * 100
                    
                    # Calculate center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Normalize coordinates (0 to 1)
                    norm_x = center_x / frame.shape[1]
                    norm_y = center_y / frame.shape[0]
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'norm_pos': (norm_x, norm_y),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'confidence': confidence,
                        'confidence_percent': confidence_percent
                    }
                    
                    detections.append(detection)
            
            self.detections = detections
            return detections
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return []
        
        finally:
            self.is_processing = False
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence_percent']
            center_x, center_y = det['center']
            
            # Get color
            color = self.tool_colors[class_id % len(self.tool_colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class_name']}: {confidence:.1f}%"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Background for text
            cv2.rectangle(frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            
            # Text
            cv2.putText(frame, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
            cv2.circle(frame, (center_x, center_y), 7, color, 2)
        
        return frame
    
    def draw_status(self, frame):
        """Draw system status on frame"""
        # Status background
        cv2.rectangle(frame, (10, 10), (400, 160), (0, 0, 0, 180), -1)
        cv2.rectangle(frame, (10, 10), (400, 160), (255, 255, 255), 1)
        
        y_pos = 35
        line_height = 25
        
        # System status
        status_color = (0, 255, 0) if not self.emergency_stop else (0, 0, 255)
        status = "RUNNING" if not self.paused else "PAUSED"
        if self.emergency_stop:
            status = "EMERGENCY STOP"
        
        cv2.putText(frame, f"Status: {status}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y_pos += line_height
        
        # Arm status
        arm_status = "CONNECTED" if self.arm.connected else "SIMULATION"
        arm_color = (0, 255, 0) if self.arm.connected else (0, 255, 255)
        cv2.putText(frame, f"Arm: {arm_status}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, arm_color, 1)
        y_pos += line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_height
        
        # Detections count
        cv2.putText(frame, f"Detections: {len(self.detections)}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += line_height
        
        # Displacement status
        cv2.putText(frame, "Displaced:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        # Show counts for each tool
        y_pos += 20
        for i in range(3):  # First row
            tool_name = self.class_names[i][:3]
            count = self.displaced_counts[i]
            color = (0, 255, 0) if count >= self.target_count else (255, 255, 255)
            cv2.putText(frame, f"{tool_name}:{count}", 
                       (20 + i * 60, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        y_pos += 20
        for i in range(3, 6):  # Second row
            tool_name = self.class_names[i][:3]
            count = self.displaced_counts[i]
            color = (0, 255, 0) if count >= self.target_count else (255, 255, 255)
            cv2.putText(frame, f"{tool_name}:{count}", 
                       (20 + (i-3) * 60, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def handle_displacement(self, detection):
        """Handle displacement for a detected object"""
        if self.emergency_stop or self.paused:
            return False
        
        class_id = detection['class_id']
        
        # Check if already displaced enough
        if self.displaced_counts[class_id] >= self.target_count:
            return False
        
        print(f"\n🚀 Starting displacement for {detection['class_name']}")
        print(f"   Confidence: {detection['confidence_percent']:.1f}%")
        print(f"   Position: ({detection['norm_pos'][0]:.2f}, {detection['norm_pos'][1]:.2f})")
        
        # Execute pick and place
        success = self.arm.execute_pick_and_place(
            detection['norm_pos'][0],
            detection['norm_pos'][1],
            class_id
        )
        
        if success:
            self.displaced_counts[class_id] += 1
            print(f"✅ Successfully displaced {detection['class_name']}")
            print(f"   Total: {self.displaced_counts[class_id]}/{self.target_count}")
            return True
        else:
            print(f"❌ Failed to displace {detection['class_name']}")
            return False
    
    def run(self):
        """Main loop"""
        print("\n🎬 Starting main loop...")
        print("Press SPACE to start/stop detection")
        
        last_detection_time = 0
        detection_interval = 2.0  # Seconds between detections
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Mirror for intuitive control
                frame = cv2.flip(frame, 1)
                
                # Update FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Process detections if not paused
                if not self.paused and not self.emergency_stop:
                    current_time = time.time()
                    if current_time - last_detection_time > detection_interval:
                        detections = self.process_frame(frame)
                        last_detection_time = current_time
                        
                        # Handle displacement if any detection
                        if detections and not self.arm.is_displacing:
                            # Sort by confidence
                            detections.sort(key=lambda x: x['confidence'], reverse=True)
                            best_detection = detections[0]
                            
                            # Start displacement in separate thread
                            if best_detection['confidence'] > 0.6:  # Minimum confidence
                                threading.Thread(
                                    target=self.handle_displacement,
                                    args=(best_detection,),
                                    daemon=True
                                ).start()
                    
                    # Draw detections
                    frame = self.draw_detections(frame, detections if 'detections' in locals() else [])
                
                # Draw status
                frame = self.draw_status(frame)
                
                # Show frame
                cv2.imshow('Yahboom Robotic Arm Control', frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord(' '):  # SPACE
                    self.paused = not self.paused
                    print(f"\n⏸️  System {'PAUSED' if self.paused else 'RESUMED'}")
                elif key == ord('e'):  # Emergency stop
                    self.emergency_stop = True
                    self.arm.emergency_stop()
                    print("\n🚨 EMERGENCY STOP!")
                elif key == ord('n'):  # Neutral
                    self.arm.go_to_neutral()
                    print("\n🔄 Moving to neutral")
                elif key == ord('o'):  # Open gripper
                    self.arm.open_gripper()
                elif key == ord('c'):  # Close gripper
                    self.arm.close_gripper()
                elif key == ord('t'):  # Test movement
                    print("\n🧪 Testing arm movement...")
                    self.arm.move_to_position(60, 120, 80, 90)
                    time.sleep(1)
                    self.arm.move_to_position(120, 120, 80, 90)
                    time.sleep(1)
                    self.arm.go_to_neutral()
                
                # Check if all done
                all_done = all(count >= self.target_count for count in self.displaced_counts.values())
                if all_done:
                    print("\n" + "=" * 70)
                    print("🎉 ALL TOOLS DISPLACED SUCCESSFULLY!")
                    print("=" * 70)
                    for class_id, name in self.class_names.items():
                        print(f"  {name}: {self.displaced_counts[class_id]}/{self.target_count}")
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Move arm to neutral
        if hasattr(self, 'arm'):
            self.arm.go_to_neutral()
        
        print("\n📊 FINAL REPORT:")
        print("=" * 50)
        for class_id, name in self.class_names.items():
            count = self.displaced_counts[class_id]
            status = "✅ DONE" if count >= self.target_count else f"❌ {count}/{self.target_count}"
            print(f"  {name}: {status}")
        print("=" * 50)
        print("✅ System shutdown complete")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("🤖 Yahboom Robotic Arm Control System")
    print("=" * 70)
    
    # Create and run system
    system = YahboomVisionSystem()
    system.run()