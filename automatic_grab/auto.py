"""
FULLY AUTOMATIC ROBOTIC ARM SYSTEM - 3 POSITION SNAPSHOTS ONLY
Takes 3 stable snapshots from different positions (front, right, left)
"""

import cv2
import time
import numpy as np
import math
from ultralytics import YOLO
import os
from datetime import datetime

print("🤖 3-POSITION SNAPSHOT SYSTEM - FRONT, RIGHT, LEFT")
print("=" * 70)

# ============================================
# ARM_LIB SERVO CONTROL
# ============================================
class ArmController:
    def __init__(self):
        """Initialize arm with specified neutral angles"""
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
        
         # Servo mapping
        self.SERVO_BASE = 1       # Servo 1: Base
        self.SERVO_SHOULDER = 2   # Servo 2: Shoulder  
        self.SERVO_ELBOW = 3      # Servo 3: Elbow
        self.SERVO_WRIST = 4      # Servo 4: Wrist
        self.SERVO_WRIST_ROT = 5  # Servo 5: Wrist rotation
        self.SERVO_GRIPPER = 6    # Servo 6: Gripper
        
        # Define neutral position based on your specifications
        self.NEUTRAL_ANGLES = {
            self.SERVO_BASE: 90,      # Base at 90 degrees
            self.SERVO_SHOULDER: 115, # Shoulder at 115 degrees
            self.SERVO_ELBOW: 45,     # Elbow at 45 degrees
            self.SERVO_WRIST: -35,    # Wrist at -35 degrees (note: negative value)
            self.SERVO_WRIST_ROT: 90, # Wrist rotation at 90 degrees
            self.SERVO_GRIPPER: 70    # Gripper open
        }
        
        # Different positions for 3 snapshots - only changing the base servo
        self.POSITION_ANGLES = {
            'front': {
                self.SERVO_BASE: 90,      # Center - looking forward
                self.SERVO_SHOULDER: 115, # Shoulder up
                self.SERVO_ELBOW: 45,     # Elbow at 45
                self.SERVO_WRIST: -35,    # Wrist at -35
                self.SERVO_WRIST_ROT: 90, # Wrist rotation neutral
            },
            'right': {
                self.SERVO_BASE: 120,     # Turned right - looking right side
                self.SERVO_SHOULDER: 115, # Same shoulder
                self.SERVO_ELBOW: 45,     # Same elbow
                self.SERVO_WRIST: -35,    # Same wrist
                self.SERVO_WRIST_ROT: 90, # Same wrist rotation
            },
            'left': {
                self.SERVO_BASE: 30,      # Turned left - looking left side
                self.SERVO_SHOULDER: 115, # Same shoulder
                self.SERVO_ELBOW: 45,     # Same elbow
                self.SERVO_WRIST: -35,    # Same wrist
                self.SERVO_WRIST_ROT: 90, # Same wrist rotation
            }
        }
        
        # Current state
        self.current_angles = self.NEUTRAL_ANGLES.copy()
        self.current_position = 'front'
        
        # Initialize
        self.initialize_arm()
    
    def initialize_arm(self):
        """Initialize arm connection and set to neutral position"""
        try:
            self.arm = self.Arm_Device()
            time.sleep(2)
            
            # Move to neutral position
            print("📸 Setting to NEUTRAL position...")
            self.set_neutral_position()
            
            self.connected = True
            print("✅ Arm initialized to neutral position")
            
        except Exception as e:
            print(f"⚠️ Arm init failed: {e}")
            self.connected = False
    
    def set_neutral_position(self):
        """Set arm to the defined neutral position"""
        if self.connected:
            try:
                # Set each servo individually as per your specifications
                print("   Setting neutral position...")
                print("   Servo 1 (Base) → 90°")
                self.arm.Arm_serial_servo_write(1, 90, 1000)
                time.sleep(1.0)
                
                print("   Servo 2 (Shoulder) → 115°")
                self.arm.Arm_serial_servo_write(2, 115, 1000)
                time.sleep(1.0)
                
                print("   Servo 3 (Elbow) → 45°")
                self.arm.Arm_serial_servo_write(3, 45, 1000)
                time.sleep(1.0)
                
                print("   Servo 4 (Wrist) → -35°")
                # Note: Arm_Lib might not accept negative values directly
                # Adjusting for possible range 0-180
                wrist_angle = self.convert_wrist_angle(-35)
                self.arm.Arm_serial_servo_write(4, wrist_angle, 1000)
                time.sleep(1.0)
                
                print("   Servo 5 (Wrist Rotation) → 90°")
                self.arm.Arm_serial_servo_write(5, 90, 1000)
                time.sleep(1.0)
                
                # Gripper stays open
                print("   Servo 6 (Gripper) → 70° (open)")
                self.arm.Arm_serial_servo_write(6, 70, 500)
                time.sleep(0.5)
                
                # Update current angles
                self.current_angles = self.NEUTRAL_ANGLES.copy()
                print("✅ Neutral position set")
                
            except Exception as e:
                print(f"⚠️ Error setting neutral position: {e}")
                # Fallback to simulation
                self.simulate_neutral_position()
        else:
            self.simulate_neutral_position()
    
    def convert_wrist_angle(self, angle):
        """Convert wrist angle to 0-180 range if needed"""
        if angle < 0:
            # If negative, adjust to positive range
            return 180 + angle  # -35 becomes 145
        return angle
    
    def simulate_neutral_position(self):
        """Simulate setting neutral position"""
        print("   [SIM] Setting neutral position...")
        print("   [SIM] Servo 1 (Base) → 90°")
        print("   [SIM] Servo 2 (Shoulder) → 115°")
        print("   [SIM] Servo 3 (Elbow) → 45°")
        print("   [SIM] Servo 4 (Wrist) → -35° (adjusted to 145°)")
        print("   [SIM] Servo 5 (Wrist Rotation) → 90°")
        print("   [SIM] Servo 6 (Gripper) → 70° (open)")
        self.current_angles = self.NEUTRAL_ANGLES.copy()
        time.sleep(3.0)
    
    def go_to_position(self, position_name):
        """Move to specified position (front, right, left) for snapshot"""
        if position_name in self.POSITION_ANGLES:
            position_angles = self.POSITION_ANGLES[position_name].copy()
            position_angles[self.SERVO_GRIPPER] = 70  # Keep gripper open
            
            print(f"\n📸 Moving to {position_name.upper()} position...")
            print(f"   Target angles:")
            print(f"     Base: {position_angles[1]}°")
            print(f"     Shoulder: {position_angles[2]}°")
            print(f"     Elbow: {position_angles[3]}°")
            print(f"     Wrist: {position_angles[4]}°")
            print(f"   View type: {'Side' if position_name in ['right', 'left'] else 'Front'} view")
            
            success = self.set_multiple_angles(position_angles, 2000)  # Increased time
            
            if success:
                self.current_position = position_name
                print(f"✅ Now at {position_name} position")
                
                # EXTENDED stabilization time for camera
                print(f"   Waiting for camera stabilization...")
                
                # Different stabilization times based on position
                if position_name == 'front':
                    stabilization_time = 2.0
                elif position_name == 'right':
                    stabilization_time = 2.5  # Extra time for side view
                else:  # left
                    stabilization_time = 2.5
                    
                time.sleep(stabilization_time)
                print(f"   Stabilized for {stabilization_time} seconds")
            return success
        return False
    
    def return_to_neutral(self):
        """Return to neutral position after snapshots"""
        print("\n↩️  Returning to neutral position...")
        self.set_neutral_position()
    
    def set_servo_angle(self, servo_id, angle, move_time=1000):
        """Set single servo angle"""
        # Adjust wrist angle if needed
        if servo_id == self.SERVO_WRIST:
            angle = self.convert_wrist_angle(angle)
        
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
            # Prepare angles for all 6 servos
            all_angles = []
            for i in range(1, 7):
                angle = angles_dict.get(i, 90)
                # Adjust wrist angle
                if i == self.SERVO_WRIST:
                    angle = self.convert_wrist_angle(angle)
                all_angles.append(angle)
            
            if self.connected:
                self.arm.Arm_serial_servo_write6(*all_angles, move_time)
                
                for servo_id, angle in angles_dict.items():
                    self.current_angles[servo_id] = angle
                
                time.sleep(move_time / 1000)
                return True
        except Exception as e:
            print(f"    [ERROR] Setting multiple angles: {e}")
        
        # Fallback: Set individually
        for servo_id, angle in angles_dict.items():
            self.set_servo_angle(servo_id, angle, 0)
        
        time.sleep(move_time / 1000)
        return True

# ============================================
# 3-POSITION SNAPSHOT SYSTEM
# ============================================
class ThreePositionSnapshotSystem:
    def __init__(self):
        # Initialize components
        self.arm = ArmController()
        
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
        
        # Create output directory
        self.output_dir = self.create_output_directory()
        
        print("🔍 3-Position Snapshot system ready")
        print(f"📁 Output folder: {self.output_dir}")
        print("\nSEQUENCE: Neutral → Front → Right → Left → Neutral")
        print("=" * 70)
    
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
                
                # Test camera
                ret, frame = cap.read()
                if ret:
                    print(f"   Camera test OK - Frame size: {frame.shape}")
                else:
                    print(f"   Camera test failed")
                
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
    
    def capture_stable_snapshot(self, position_name):
        """
        Capture a STABLE snapshot at current position with anti-shake measures
        """
        print(f"\n📸 CAPTURING {position_name.upper()} SNAPSHOT")
        print(f"   Current arm position: {position_name}")
        
        # Get current angles for display
        angles_text = f"Base={self.arm.current_angles[1]}°, Shoulder={self.arm.current_angles[2]}°, Elbow={self.arm.current_angles[3]}°"
        print(f"   {angles_text}")
        
        # EXTENDED ANTI-SHAKE PROCEDURE
        
        # 1. Wait for arm to fully stabilize (position-specific time)
        stabilization_time = 2.0 if position_name == 'front' else 2.5
        print(f"   [1/4] Arm stabilization: {stabilization_time} seconds...")
        time.sleep(stabilization_time)
        
        # 2. Camera warm-up frames (discard initial frames)
        warmup_frames = 8  # Increased for side views
        print(f"   [2/4] Camera warm-up ({warmup_frames} frames)...")
        for i in range(warmup_frames):
            if self.cap is not None:
                ret, _ = self.cap.read()
                if ret:
                    if (i+1) % 4 == 0:
                        print(f"     Frame {i+1}/{warmup_frames} captured")
                time.sleep(0.05)  # Small delay between frames
        
        # 3. Capture multiple frames and select the sharpest
        num_frames = 5  # Increased for better selection
        print(f"   [3/4] Capturing {num_frames} frames for sharpest selection...")
        frames = []
        sharpness_scores = []
        
        for i in range(num_frames):
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    frames.append(frame)
                    # Calculate sharpness (variance of Laplacian)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    sharpness_scores.append(laplacian_var)
                    if (i+1) % 2 == 0:
                        print(f"     Frame {i+1}: Sharpness = {laplacian_var:.1f}")
                time.sleep(0.15)  # Increased delay between captures
            else:
                # Create test frame if no camera
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Different visual cues for different positions
                if position_name == 'front':
                    cv2.putText(test_frame, "FRONT VIEW", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.rectangle(test_frame, (200, 200), (440, 280), (0, 255, 0), 2)
                elif position_name == 'right':
                    cv2.putText(test_frame, "RIGHT SIDE VIEW", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.rectangle(test_frame, (100, 250), (300, 350), (255, 0, 0), 2)
                else:  # left
                    cv2.putText(test_frame, "LEFT SIDE VIEW", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.rectangle(test_frame, (340, 250), (540, 350), (0, 0, 255), 2)
                
                cv2.putText(test_frame, f"Base: {self.arm.current_angles[1]}°", 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(test_frame, f"Shoulder: {self.arm.current_angles[2]}°", 
                           (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(test_frame, f"Elbow: {self.arm.current_angles[3]}°", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                frames.append(test_frame)
                sharpness_scores.append(100 + i*10)  # Simulated increasing sharpness
                break
        
        # 4. Select the sharpest frame
        print(f"   [4/4] Selecting sharpest frame...")
        if frames and sharpness_scores:
            best_idx = np.argmax(sharpness_scores)
            best_frame = frames[best_idx]
            print(f"     Selected frame {best_idx+1} (Sharpness: {sharpness_scores[best_idx]:.1f})")
            
            # Check if frame is blurry
            if sharpness_scores[best_idx] < 100:  # Threshold for blur detection
                print(f"     ⚠️ Warning: Frame may be blurry (sharpness < 100)")
        else:
            best_frame = frames[0] if frames else None
        
        if best_frame is None:
            print(f"⚠️ Failed to capture {position_name} snapshot")
            return []
        
        # Save raw frame first (for debugging)
        raw_filename = f"{self.output_dir}/{position_name}_raw.jpg"
        cv2.imwrite(raw_filename, best_frame)
        
        # Detect objects
        detections = self.detect_objects(best_frame)
        
        # Annotate frame with detections
        annotated_frame = self.annotate_frame(best_frame, detections, position_name)
        
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
            det['base_angle'] = self.arm.current_angles[1]
            det['shoulder_angle'] = self.arm.current_angles[2]
            det['elbow_angle'] = self.arm.current_angles[3]
        
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
        
        # Different background colors for different positions
        if position_name == 'front':
            position_color = (0, 255, 255)  # Yellow
            view_text = "FRONT VIEW"
        elif position_name == 'right':
            position_color = (255, 100, 0)  # Orange
            view_text = "RIGHT SIDE VIEW"
        else:  # left
            position_color = (100, 255, 100)  # Light Green
            view_text = "LEFT SIDE VIEW"
        
        # Add position label with colored background
        cv2.rectangle(annotated, (5, 5), (250, 80), (40, 40, 40), -1)
        cv2.putText(annotated, view_text, 
                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, position_color, 2)
        
        # Add angle information
        angles_text = f"B:{self.arm.current_angles[1]}° S:{self.arm.current_angles[2]}° E:{self.arm.current_angles[3]}°"
        cv2.putText(annotated, angles_text, 
                   (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw bounding box
            color = position_color
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), (40, 40, 40), -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x, center_y = int(det['center_x']), int(det['center_y'])
            cv2.circle(annotated, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw coordinates text
            coord_text = f"({det['center_x']:.0f}, {det['center_y']:.0f})"
            cv2.putText(annotated, coord_text, (center_x - 30, center_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Add detection count
        cv2.putText(annotated, f"Detections: {len(detections)}", 
                   (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add stability indicator
        stability_text = "STABLE" if len(detections) > 0 else "NO OBJECTS"
        stability_color = (0, 255, 0) if len(detections) > 0 else (0, 0, 255)
        cv2.putText(annotated, stability_text, 
                   (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, timestamp, 
                   (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return annotated
    
    def save_detection_info(self, position_name, detections, image_filename):
        """Save detection information to text file"""
        info_filename = f"{self.output_dir}/{position_name}_detections.txt"
        
        with open(info_filename, 'w') as f:
            f.write(f"Position: {position_name}\n")
            f.write(f"View Type: {'Side' if position_name in ['right', 'left'] else 'Front'} view\n")
            f.write(f"Base Angle: {self.arm.current_angles[1]}°\n")
            f.write(f"Shoulder Angle: {self.arm.current_angles[2]}°\n")
            f.write(f"Elbow Angle: {self.arm.current_angles[3]}°\n")
            f.write(f"Wrist Angle: {self.arm.current_angles[4]}°\n")
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
    
    def scan_three_positions(self):
        """
        Scan only 3 positions and collect all detections
        Returns: List of all detected objects
        """
        print("\n" + "="*70)
        print("🔄 STARTING 3-POSITION SCAN (SNAPSHOTS ONLY)")
        print("="*70)
        
        self.all_detections = []
        positions = ['front', 'right', 'left']
        
        for position in positions:
            # Move to position (includes stabilization time)
            self.arm.go_to_position(position)
            
            # Capture stable snapshot at this position
            detections = self.capture_stable_snapshot(position)
            
            if detections:
                print(f"   ✅ Found {len(detections)} objects in {position} position")
                self.all_detections.extend(detections)
            else:
                print(f"   ⚠️ No objects found in {position} position")
            
            # Brief pause between positions
            if position != positions[-1]:  # Not after last position
                print(f"   Preparing for next position...")
                time.sleep(1)
        
        # Return to neutral position (NOT front)
        print("\n↩️  All snapshots complete, returning to neutral position...")
        self.arm.return_to_neutral()
        
        print(f"\n📊 TOTAL DETECTIONS: {len(self.all_detections)} objects")
        
        # Save summary file
        self.save_scan_summary()
        
        return self.all_detections
    
    def save_scan_summary(self):
        """Save summary of all detections"""
        summary_filename = f"{self.output_dir}/scan_summary.txt"
        
        with open(summary_filename, 'w') as f:
            f.write("3-POSITION SNAPSHOT SCAN SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Scan time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total objects detected: {len(self.all_detections)}\n")
            f.write(f"Neutral position: Base=90°, Shoulder=115°, Elbow=45°, Wrist=-35°\n\n")
            
            # Count by class and position
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
                f.write(f"   Base: {det.get('base_angle', 'N/A')}°, Shoulder: {det.get('shoulder_angle', 'N/A')}°, Elbow: {det.get('elbow_angle', 'N/A')}°\n")
                f.write(f"   Center: ({det['center_x']:.1f}, {det['center_y']:.1f})\n")
                f.write(f"   Snapshot: {os.path.basename(det['snapshot_file'])}\n")
                f.write("-"*40 + "\n")
            
            f.write("\nARM ANGLES USED FOR EACH VIEW:\n")
            f.write("-"*40 + "\n")
            for position in ['front', 'right', 'left']:
                angles = self.arm.POSITION_ANGLES[position]
                f.write(f"{position.upper()} VIEW:\n")
                f.write(f"  Base: {angles[1]}°\n")
                f.write(f"  Shoulder: {angles[2]}°\n")
                f.write(f"  Elbow: {angles[3]}°\n")
                f.write(f"  Wrist: {angles[4]}°\n")
                f.write(f"  View description: {'Side view from ' + position + ' side' if position in ['right', 'left'] else 'Front/top-down view'}\n\n")
    
    def run(self):
        """Main automatic sequence: Scan 3 positions and save snapshots ONLY"""
        print("\n🚀 Starting 3-POSITION SNAPSHOT SYSTEM...")
        print("   Sequence: Neutral → Front → Right → Left → Neutral")
        print("   Purpose: Capture stable snapshots only (NO GRABBING)")
        print("=" * 70)
        
        # Ensure arm is at neutral position
        print("\n🎯 Starting from NEUTRAL position...")
        self.arm.return_to_neutral()
        time.sleep(2.0)
        
        try:
            # STEP 1: Scan all 3 positions
            print("\n🔍 STEP 1: Scanning all 3 positions...")
            self.scan_three_positions()
            
            if not self.all_detections:
                print("\n⚠️ No objects detected.")
            else:
                # Show what was detected
                print("\n📋 DETECTED OBJECTS:")
                for i, det in enumerate(self.all_detections, 1):
                    print(f"   {i}. {det['class_name']} at {det['position']} position")
            
            print(f"\n🎉 SNAPSHOT MISSION COMPLETE!")
            print(f"📁 All snapshots saved in: {self.output_dir}")
            
            # Show final statistics
            print(f"\n📊 FINAL STATISTICS:")
            print(f"   Total objects detected: {len(self.all_detections)}")
            print(f"   Snapshots taken: 3 (front, right, left)")
            print(f"   Arm returned to neutral position")
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\n❌ System error: {e}")
        finally:
            # Cleanup
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n📁 All snapshots saved in: {self.output_dir}")
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
    system = ThreePositionSnapshotSystem()
    system.run()