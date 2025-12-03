"""
YAHBOOM AUTOMATIC OBJECT SORTING SYSTEM - SERVO 6 GRIPPER
Detects objects → Picks up → Rotates to drop zone → Places → Returns
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import threading
import sys
import os

print("🤖 YAHBOOM AUTOMATIC OBJECT SORTING SYSTEM")
print("=" * 70)

# ============================================
# YAHBOOM ROBOTIC ARM - FULL CONTROL
# ============================================
class YahboomSortingArm:
    def __init__(self):
        self.bus = None
        self.connected = False
        self.gripper_open = True
        self.current_angle = 90  # Neutral position
        
        # I2C Configuration
        self.I2C_BUS = 6
        self.I2C_ADDR = 0x15
        
        # Servo IDs
        self.SERVO_BASE = 1       # Rotation
        self.SERVO_SHOULDER = 2   # Up/Down
        self.SERVO_ELBOW = 3      # Arm bend
        self.SERVO_WRIST = 4      # Wrist angle
        self.SERVO_GRIP_ROT = 5   # Grip rotation
        self.SERVO_GRIPPER = 6    # Gripper open/close
        
        # POSITION CALIBRATION - ADJUST THESE!
        # ====================================
        # NEUTRAL (Scanning position)
        self.NEUTRAL_BASE = 90      # Center facing table
        self.NEUTRAL_SHOULDER = 60  # Arm lowered
        self.NEUTRAL_ELBOW = 120    # Elbow position
        self.NEUTRAL_WRIST = 90     # Wrist level
        
        # PICKUP position (over object)
        self.PICKUP_SHOULDER = 50   # Lower to object
        self.PICKUP_ELBOW = 130     # Reach forward
        self.PICKUP_WRIST = 100     # Angle for pickup
        
        # DROP ZONE position
        self.DROP_BASE = 210        # 120° from neutral (90 + 120)
        self.DROP_SHOULDER = 40     # Lower to drop
        self.DROP_ELBOW = 110       # Extend to drop
        self.DROP_WRIST = 80        # Angle for drop
        
        # Gripper angles
        self.GRIPPER_OPEN = 180
        self.GRIPPER_CLOSED = 50
        
        # Movement speeds (ms)
        self.BASE_SPEED = 1500
        self.ARM_SPEED = 1000
        self.GRIPPER_SPEED = 800
        
        # Object class to drop positions
        self.CLASS_DROP_POSITIONS = {
            0: {'base': 150, 'shoulder': 40, 'elbow': 110, 'wrist': 80},  # Bolt
            1: {'base': 170, 'shoulder': 35, 'elbow': 115, 'wrist': 85},  # Hammer
            2: {'base': 190, 'shoulder': 45, 'elbow': 105, 'wrist': 75},  # Measuring Tape
            3: {'base': 210, 'shoulder': 40, 'elbow': 110, 'wrist': 80},  # Plier
            4: {'base': 230, 'shoulder': 35, 'elbow': 115, 'wrist': 85},  # Screwdriver
            5: {'base': 250, 'shoulder': 45, 'elbow': 105, 'wrist': 75},  # Wrench
        }
        
        self.is_moving = False
        self.action_queue = []
        
        print(f"⚙️  Automatic Sorting Arm Initialized")
        print(f"   I2C Bus: {self.I2C_BUS}")
        print(f"   Drop Zone: {self.DROP_BASE}° (120° from neutral)")
        
        self.initialize_i2c()
    
    def initialize_i2c(self):
        """Initialize I2C connection"""
        print(f"\n🔌 Initializing I2C on bus {self.I2C_BUS}...")
        
        try:
            import smbus
            self.bus = smbus.SMBus(self.I2C_BUS)
            time.sleep(0.1)
            
            # Test communication
            try:
                self.bus.read_byte(self.I2C_ADDR)
                print(f"✅ I2C connected on bus {self.I2C_BUS}")
                self.connected = True
                
                # Move to neutral position
                self.go_to_neutral()
                print("✅ Arm in NEUTRAL position")
                
            except Exception as e:
                print(f"❌ I2C test failed: {e}")
                self.connected = False
                
        except ImportError:
            print("❌ smbus not installed")
            self.connected = False
        except Exception as e:
            print(f"❌ I2C error: {e}")
            self.connected = False
    
    def write_servo_angle(self, servo_id, angle, time_ms=1000):
        """Send angle command to servo"""
        if not self.connected or self.bus is None:
            servo_names = {1: "Base", 2: "Shoulder", 3: "Elbow", 
                         4: "Wrist", 5: "GripRot", 6: "Gripper"}
            print(f"    [SIM] {servo_names.get(servo_id, f'Servo {servo_id}')} → {angle}°")
            time.sleep(time_ms / 1000)
            return True
        
        try:
            # Convert angle to pulse width (500-2500µs)
            pulse_width = 500 + (angle * 2000 / 180)
            pulse_width = int(max(500, min(2500, pulse_width)))
            
            # Prepare I2C data
            data = [
                0x55, 0x55,
                servo_id,
                time_ms & 0xFF,
                (time_ms >> 8) & 0xFF,
                pulse_width & 0xFF,
                (pulse_width >> 8) & 0xFF
            ]
            
            # Send via I2C
            self.bus.write_i2c_block_data(self.I2C_ADDR, 0, data)
            
            servo_names = {1: "Base", 2: "Shoulder", 3: "Elbow", 
                         4: "Wrist", 5: "GripRot", 6: "Gripper"}
            servo_name = servo_names.get(servo_id, f"Servo {servo_id}")
            
            print(f"    [ACT] {servo_name} → {angle}° ({time_ms}ms)")
            
            time.sleep(time_ms / 1000)
            return True
            
        except Exception as e:
            print(f"    ❌ I2C write failed: {e}")
            return False
    
    def move_multiple_servos(self, servo_angles, move_time=1000):
        """Move multiple servos simultaneously"""
        if not self.connected:
            for servo_id, angle in servo_angles.items():
                print(f"    [SIM] Servo {servo_id} → {angle}°")
            time.sleep(move_time / 1000)
            return True
        
        try:
            # Move all servos
            for servo_id, angle in servo_angles.items():
                self.write_servo_angle(servo_id, angle, move_time)
            return True
        except Exception as e:
            print(f"❌ Multi-servo move failed: {e}")
            return False
    
    def go_to_neutral(self):
        """Move to neutral scanning position"""
        print("\n📡 Moving to NEUTRAL (scanning) position...")
        
        servo_angles = {
            self.SERVO_BASE: self.NEUTRAL_BASE,
            self.SERVO_SHOULDER: self.NEUTRAL_SHOULDER,
            self.SERVO_ELBOW: self.NEUTRAL_ELBOW,
            self.SERVO_WRIST: self.NEUTRAL_WRIST,
            self.SERVO_GRIPPER: self.GRIPPER_OPEN
        }
        
        if self.move_multiple_servos(servo_angles, self.ARM_SPEED):
            self.current_angle = self.NEUTRAL_BASE
            print(f"✅ Neutral position: Base={self.NEUTRAL_BASE}°")
            return True
        return False
    
    def go_to_pickup(self, x_position=320):
        """Move to pickup position based on object location"""
        print(f"\n🎯 Moving to PICKUP position (X={x_position})")
        
        # Calculate base angle based on object position (320 is center)
        # Map screen X position to base angle (0-640px to 60-120°)
        base_angle = self.NEUTRAL_BASE + ((x_position - 320) / 320 * 30)
        base_angle = max(60, min(120, base_angle))
        
        servo_angles = {
            self.SERVO_BASE: base_angle,
            self.SERVO_SHOULDER: self.PICKUP_SHOULDER,
            self.SERVO_ELBOW: self.PICKUP_ELBOW,
            self.SERVO_WRIST: self.PICKUP_WRIST
        }
        
        # Move arm to position
        self.move_multiple_servos(servo_angles, self.ARM_SPEED)
        self.current_angle = base_angle
        return True
    
    def go_to_drop_zone(self, class_id=None):
        """Move to drop zone (120° from neutral or class-specific)"""
        print(f"\n📦 Moving to DROP ZONE")
        
        if class_id is not None and class_id in self.CLASS_DROP_POSITIONS:
            # Class-specific drop position
            pos = self.CLASS_DROP_POSITIONS[class_id]
            servo_angles = {
                self.SERVO_BASE: pos['base'],
                self.SERVO_SHOULDER: pos['shoulder'],
                self.SERVO_ELBOW: pos['elbow'],
                self.SERVO_WRIST: pos['wrist']
            }
            print(f"    Class {class_id} specific drop position")
        else:
            # Default drop zone (120° from neutral)
            servo_angles = {
                self.SERVO_BASE: self.DROP_BASE,
                self.SERVO_SHOULDER: self.DROP_SHOULDER,
                self.SERVO_ELBOW: self.DROP_ELBOW,
                self.SERVO_WRIST: self.DROP_WRIST
            }
        
        self.move_multiple_servos(servo_angles, self.BASE_SPEED)
        self.current_angle = servo_angles[self.SERVO_BASE]
        return True
    
    def open_gripper(self):
        """Open gripper"""
        print("    🤖 Opening gripper...")
        success = self.write_servo_angle(self.SERVO_GRIPPER, self.GRIPPER_OPEN, self.GRIPPER_SPEED)
        if success:
            self.gripper_open = True
        return success
    
    def close_gripper(self):
        """Close gripper"""
        print("    🤖 Closing gripper...")
        success = self.write_servo_angle(self.SERVO_GRIPPER, self.GRIPPER_CLOSED, self.GRIPPER_SPEED)
        if success:
            self.gripper_open = False
        return success
    
    def pickup_object(self, x_position=320):
        """Complete pickup sequence"""
        print(f"\n{'='*50}")
        print(f"🔄 PICKUP SEQUENCE STARTED")
        print(f"{'='*50}")
        
        # 1. Go to pickup position
        self.go_to_pickup(x_position)
        time.sleep(0.5)
        
        # 2. Close gripper
        self.close_gripper()
        time.sleep(0.3)
        
        # 3. Lift object slightly
        servo_angles = {
            self.SERVO_SHOULDER: self.NEUTRAL_SHOULDER - 10,
            self.SERVO_ELBOW: self.NEUTRAL_ELBOW - 10
        }
        self.move_multiple_servos(servo_angles, 500)
        
        print("✅ Pickup complete")
        return True
    
    def drop_object(self, class_id=None):
        """Complete drop sequence"""
        print(f"\n📤 DROP SEQUENCE STARTED")
        
        # 1. Go to drop zone
        self.go_to_drop_zone(class_id)
        time.sleep(0.5)
        
        # 2. Lower to drop position
        servo_angles = {
            self.SERVO_SHOULDER: self.DROP_SHOULDER + 10,
            self.SERVO_ELBOW: self.DROP_ELBOW + 10
        }
        self.move_multiple_servos(servo_angles, 500)
        time.sleep(0.3)
        
        # 3. Open gripper
        self.open_gripper()
        time.sleep(0.3)
        
        # 4. Lift gripper
        servo_angles = {
            self.SERVO_SHOULDER: self.DROP_SHOULDER - 20,
            self.SERVO_ELBOW: self.DROP_ELBOW - 20
        }
        self.move_multiple_servos(servo_angles, 500)
        
        print("✅ Drop complete")
        return True
    
    def complete_sorting_cycle(self, class_id, x_position):
        """Complete sorting cycle: Pick → Drop → Return"""
        print(f"\n{'='*60}")
        print(f"🚀 STARTING SORTING CYCLE for Class {class_id}")
        print(f"{'='*60}")
        
        try:
            # Pick up object
            self.pickup_object(x_position)
            time.sleep(0.5)
            
            # Drop object
            self.drop_object(class_id)
            time.sleep(0.5)
            
            # Return to neutral
            self.go_to_neutral()
            time.sleep(0.5)
            
            print(f"\n✅ Sorting cycle COMPLETE for class {class_id}")
            return True
            
        except Exception as e:
            print(f"❌ Sorting cycle failed: {e}")
            return False
    
    def emergency_stop(self):
        """Emergency stop - all servos to safe position"""
        print("\n🛑 EMERGENCY STOP - Moving to safe position")
        
        safe_angles = {
            1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: 180
        }
        
        for servo_id in range(1, 7):
            try:
                self.write_servo_angle(servo_id, safe_angles[servo_id], 1000)
            except:
                pass
        
        time.sleep(1)
        return True

# ============================================
# AUTOMATIC SORTING SYSTEM
# ============================================
class AutomaticSortingSystem:
    def __init__(self):
        print("\n🤖 Initializing Automatic Sorting System...")
        
        # Initialize robotic arm
        print("🔧 Initializing Robotic Arm...")
        self.arm = YahboomSortingArm()
        
        # Setup camera
        print("\n📷 Setting up camera...")
        self.cap = self.setup_camera()
        
        # Load YOLO model
        print("\n🧠 Loading YOLO model...")
        self.model = self.load_yolo_model()
        
        # Detection settings
        self.confidence_threshold = 0.6
        self.detection_cooldown = 2.0
        self.last_detection_time = 0
        self.is_sorting = False
        self.auto_mode = True  # Start in automatic mode
        
        # Object tracking
        self.detected_objects = []
        self.sorted_count = 0
        self.total_objects = 0
        
        # Display settings
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.show_detections = True
        
        # Class names
        self.class_names = {
            0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
            3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
        }
        
        print("\n" + "="*70)
        print("✅ SYSTEM READY - AUTOMATIC MODE")
        print("="*70)
        print("\n🎯 AUTOMATIC SORTING PROCESS:")
        print("1. Arm starts in NEUTRAL position")
        print("2. Camera scans table for objects")
        print("3. When object detected → Picks it up")
        print("4. Rotates 120° to DROP ZONE")
        print("5. Places object in designated area")
        print("6. Returns to NEUTRAL to continue")
        print("7. Repeats until no objects detected")
        print("\n📋 CONTROLS:")
        print("  SPACE - Toggle auto/manual mode")
        print("  n     - Force return to NEUTRAL")
        print("  e     - Emergency stop")
        print("  d     - Toggle detection display")
        print("  q     - Quit")
        print("="*70)
    
    def setup_camera(self):
        """Setup camera"""
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if cap.isOpened():
                    print(f"✅ Camera found at index {i}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    return cap
            except:
                continue
        
        print("⚠️ No camera found, using simulated mode")
        return None
    
    def load_yolo_model(self):
        """Load YOLO model"""
        model_files = ['best.pt', 'yolov8n.pt', 'yolo11n.pt']
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    model = YOLO(model_file)
                    print(f"✅ Loaded model: {model_file}")
                    return model
                except Exception as e:
                    print(f"❌ Failed to load {model_file}: {e}")
        
        print("⚠️ No YOLO model found, using simulated detection")
        return None
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if self.model is None:
            # Simulate detection
            return []
        
        try:
            results = self.model(frame, 
                               conf=0.5,
                               iou=0.3,
                               imgsz=320,
                               verbose=False,
                               device='cpu')
            
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(confidences)):
                    if confidences[i] >= self.confidence_threshold:
                        x1, y1, x2, y2 = boxes[i]
                        center_x = (x1 + x2) / 2
                        
                        detections.append({
                            'class_id': class_ids[i],
                            'class_name': self.class_names.get(class_ids[i], f"Obj_{class_ids[i]}"),
                            'confidence': confidences[i] * 100,
                            'bbox': (x1, y1, x2, y2),
                            'center_x': center_x
                        })
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return []
    
    def get_best_object_to_sort(self, detections):
        """Select the best object to sort (closest to center)"""
        if not detections:
            return None
        
        # Find object closest to center of frame (320)
        best_obj = min(detections, key=lambda x: abs(x['center_x'] - 320))
        return best_obj
    
    def process_detections(self, detections):
        """Process detections and trigger sorting if in auto mode"""
        if not self.auto_mode or self.is_sorting:
            return
        
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_cooldown:
            return
        
        if detections:
            # Get best object to sort
            best_obj = self.get_best_object_to_sort(detections)
            
            if best_obj:
                self.last_detection_time = current_time
                self.total_objects = len(detections)
                
                # Start sorting in separate thread
                threading.Thread(target=self.execute_sorting_cycle, 
                               args=(best_obj,), 
                               daemon=True).start()
    
    def execute_sorting_cycle(self, object_info):
        """Execute complete sorting cycle for an object"""
        self.is_sorting = True
        
        try:
            print(f"\n{'='*60}")
            print(f"🎯 DETECTED: {object_info['class_name']} "
                  f"({object_info['confidence']:.1f}%)")
            print(f"   Position: X={object_info['center_x']:.0f}")
            print(f"{'='*60}")
            
            # Execute sorting cycle
            success = self.arm.complete_sorting_cycle(
                class_id=object_info['class_id'],
                x_position=object_info['center_x']
            )
            
            if success:
                self.sorted_count += 1
                print(f"\n📊 PROGRESS: Sorted {self.sorted_count} objects")
            
        except Exception as e:
            print(f"❌ Sorting error: {e}")
        
        finally:
            self.is_sorting = False
    
    def process_video_frame(self):
        """Process video frame"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.start_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.start_time)
            self.frame_count = 0
            self.start_time = current_time
        
        # Detect objects
        detections = self.detect_objects(frame)
        self.detected_objects = detections
        
        # Process detections (auto mode)
        self.process_detections(detections)
        
        # Draw on frame
        display_frame = self.draw_display(frame, detections)
        
        return display_frame
    
    def draw_display(self, frame, detections):
        """Draw interface on frame"""
        if frame is None:
            return None
        
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw detection boxes
        if self.show_detections:
            for obj in detections:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                
                # Different colors for different classes
                colors = [
                    (255, 0, 0),    # Blue - Bolt
                    (0, 255, 0),    # Green - Hammer
                    (0, 0, 255),    # Red - Measuring Tape
                    (255, 255, 0),  # Cyan - Plier
                    (255, 0, 255),  # Magenta - Screwdriver
                    (0, 255, 255)   # Yellow - Wrench
                ]
                color = colors[obj['class_id'] % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{obj['class_name']} {obj['confidence']:.1f}%"
                cv2.rectangle(display, (x1, y1-25), (x1+200, y1), color, -1)
                cv2.putText(display, label, (x1+5, y1-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw center point
                center_x = int(obj['center_x'])
                cv2.circle(display, (center_x, int((y1+y2)/2)), 5, color, -1)
        
        # Draw info panel
        cv2.rectangle(display, (10, 10), (450, 180), (0, 0, 0, 180), -1)
        cv2.rectangle(display, (10, 10), (450, 180), (255, 255, 255), 1)
        
        y = 35
        line = 25
        
        # Title
        mode_color = (0, 255, 0) if self.auto_mode else (255, 255, 0)
        mode_text = "AUTO" if self.auto_mode else "MANUAL"
        cv2.putText(display, f"🤖 AUTOMATIC SORTING - {mode_text}", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        y += line
        
        # Arm status
        if self.arm.connected:
            status = f"✅ ARM: Base={self.arm.current_angle}°, "
            status += f"Gripper={'OPEN' if self.arm.gripper_open else 'CLOSED'}"
            color = (0, 255, 0)
        else:
            status = "❌ ARM DISCONNECTED"
            color = (255, 0, 0)
        
        cv2.putText(display, status, (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += line
        
        # Object info
        cv2.putText(display, f"Objects Detected: {len(detections)}", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        y += line
        
        cv2.putText(display, f"Sorted: {self.sorted_count}", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        y += line
        
        # FPS
        cv2.putText(display, f"FPS: {self.fps:.1f}", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
        y += line
        
        # Sorting status
        if self.is_sorting:
            cv2.putText(display, "🔄 SORTING IN PROGRESS...", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif self.auto_mode:
            cv2.putText(display, "📡 SCANNING FOR OBJECTS...", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw center line
        cv2.line(display, (w//2, 0), (w//2, h), (255, 255, 255), 1)
        cv2.circle(display, (w//2, h//2), 3, (255, 255, 255), -1)
        
        # Draw drop zone indicator (right side)
        cv2.rectangle(display, (w-150, h-100), (w-50, h-20), (0, 0, 255), 2)
        cv2.putText(display, "DROP ZONE", (w-140, h-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return display
    
    def run(self):
        """Main loop"""
        print("\n🚀 Starting Automatic Sorting System...")
        time.sleep(2)
        
        if self.cap is None:
            print("📷 No camera - Running in manual mode")
            self.manual_mode()
            return
        
        try:
            while True:
                # Process frame
                display_frame = self.process_video_frame()
                
                if display_frame is not None:
                    cv2.imshow('Automatic Object Sorting System', display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                
                elif key == ord(' '):  # Space bar
                    self.auto_mode = not self.auto_mode
                    mode = "AUTO" if self.auto_mode else "MANUAL"
                    print(f"\n🔄 Mode changed to: {mode}")
                    
                    if self.auto_mode:
                        self.arm.go_to_neutral()
                
                elif key == ord('n'):
                    print("\n📡 Returning to NEUTRAL position...")
                    self.arm.go_to_neutral()
                
                elif key == ord('e'):
                    print("\n🛑 Emergency stop!")
                    self.arm.emergency_stop()
                    self.is_sorting = False
                
                elif key == ord('d'):
                    self.show_detections = not self.show_detections
                    status = "ON" if self.show_detections else "OFF"
                    print(f"👁️ Detection display: {status}")
                
                elif key == ord('1'):  # Test pickup
                    if not self.is_sorting:
                        print("\n🔧 Testing pickup...")
                        self.arm.pickup_object()
                
                elif key == ord('2'):  # Test drop
                    if not self.is_sorting:
                        print("\n📦 Testing drop...")
                        self.arm.drop_object()
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        
        finally:
            self.cleanup()
    
    def manual_mode(self):
        """Manual control mode"""
        print("\n🎮 MANUAL CONTROL MODE")
        print("="*50)
        
        try:
            while True:
                print(f"\n📊 Status: Objects detected={len(self.detected_objects)}, "
                      f"Sorted={self.sorted_count}")
                print("\nCommands:")
                print("  n - Go to NEUTRAL")
                print("  p - Pickup object")
                print("  d - Drop object")
                print("  t - Test sorting cycle")
                print("  e - Emergency stop")
                print("  q - Quit")
                
                cmd = input("\nEnter command: ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'n':
                    self.arm.go_to_neutral()
                elif cmd == 'p':
                    x = int(input("Enter X position (0-640): "))
                    self.arm.pickup_object(x)
                elif cmd == 'd':
                    class_id = int(input("Enter class ID (0-5): "))
                    self.arm.drop_object(class_id)
                elif cmd == 't':
                    class_id = int(input("Enter class ID (0-5): "))
                    x = int(input("Enter X position (0-640): "))
                    self.arm.complete_sorting_cycle(class_id, x)
                elif cmd == 'e':
                    self.arm.emergency_stop()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Return to neutral
        print("🤖 Returning to NEUTRAL position...")
        self.arm.go_to_neutral()
        
        print(f"\n📊 SESSION SUMMARY:")
        print(f"   Total objects sorted: {self.sorted_count}")
        print(f"   I2C connected: {'✅' if self.arm.connected else '❌'}")
        print("✅ Cleanup complete")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("🤖 YAHBOOM AUTOMATIC OBJECT SORTING SYSTEM")
    print("="*70)
    print("\n⚠️  WARNING: Ensure arm has clear space before starting!")
    
    # Create and run system
    system = AutomaticSortingSystem()
    system.run()