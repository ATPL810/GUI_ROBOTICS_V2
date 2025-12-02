"""
COMPLETE ROBOTIC ARM CONTROL SYSTEM WITH YOLO11 AND YAHBOOM ARM
Real hardware control with displacement and priority management
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

# Try to import Yahboom libraries
try:
    # These are common Yahboom libraries - adjust based on your specific kit
    import Arm_Lib
    from Arm_Lib import Arm_Device
    import yahboom_arm_lib  # May have different name
    YAHBOOM_AVAILABLE = True
    print("✅ Yahboom libraries found")
except ImportError as e:
    print(f"⚠️ Yahboom libraries not found: {e}")
    print("Running in simulation mode")
    YAHBOOM_AVAILABLE = False

print("🤖 YAHBOOM ROBOTIC ARM CONTROL SYSTEM WITH PRIORITY MANAGEMENT")
print("=" * 70)

# ============================================
# REAL YAHBOOM ARM CONTROL CLASS
# ============================================
class YahboomRoboticArm:
    def __init__(self):
        """Initialize real Yahboom robotic arm"""
        self.connected = False
        self.arm = None
        self.current_angles = [90, 90, 90, 90, 90, 180]  # Initial servo angles
        self.gripper_open = True
        
        # Servo mapping (adjust based on your arm)
        self.SERVO_BASE = 1      # Servo 1: Base rotation
        self.SERVO_SHOULDER = 2  # Servo 2: Shoulder
        self.SERVO_ELBOW = 3     # Servo 3: Elbow
        self.SERVO_WRIST = 4     # Servo 4: Wrist
        self.SERVO_GRIPPER = 5   # Servo 5: Gripper
        self.SERVO_WRIST_ROT = 6 # Servo 6: Wrist rotation
        
        # Gripper positions (calibrate these)
        self.GRIPPER_OPEN = 70   # Angle for open gripper
        self.GRIPPER_CLOSED = 30 # Angle for closed gripper
        
        # Movement parameters
        self.move_speed = 1000   # Movement speed (ms)
        
        self.initialize_arm()
    
    def initialize_arm(self):
        """Initialize connection to Yahboom arm"""
        try:
            if YAHBOOM_AVAILABLE:
                # Initialize Yahboom arm - method depends on your specific library
                self.arm = Arm_Device()
                time.sleep(2)
                
                # Set initial position
                self.go_to_neutral()
                self.connected = True
                print("✅ Yahboom Arm Connected and Initialized")
            else:
                print("⚠️ Running in simulation mode - no real arm connected")
                self.connected = False
                
        except Exception as e:
            print(f"❌ Failed to initialize Yahboom arm: {e}")
            self.connected = False
    
    def go_to_neutral(self):
        """Move arm to neutral position"""
        if self.connected:
            # Send all servos to 90 degrees (neutral)
            for servo in range(1, 7):
                self.arm.Arm_serial_servo_write(servo, 90, 1000)
            time.sleep(1.5)
        else:
            print("    [SIM] Moving to neutral position")
            time.sleep(0.5)
        
        self.current_angles = [90, 90, 90, 90, 90, 180]
        self.gripper_open = True
    
    def set_servo_angle(self, servo_id, angle, move_time=None):
        """Set single servo angle"""
        if move_time is None:
            move_time = self.move_speed
            
        # Validate angle range
        angle = max(0, min(180, angle))
        
        if self.connected:
            try:
                self.arm.Arm_serial_servo_write(servo_id, angle, move_time)
                self.current_angles[servo_id-1] = angle
                time.sleep(move_time / 1000)
                return True
            except Exception as e:
                print(f"    [ERROR] Servo {servo_id} move failed: {e}")
                return False
        else:
            # Simulation
            print(f"    [SIM] Servo {servo_id} → {angle}° ({move_time}ms)")
            self.current_angles[servo_id-1] = angle
            time.sleep(move_time / 1000)
            return True
    
    def set_multiple_angles(self, angles_dict, move_time=None):
        """Set multiple servo angles simultaneously"""
        if move_time is None:
            move_time = self.move_speed
            
        if self.connected:
            try:
                for servo_id, angle in angles_dict.items():
                    angle = max(0, min(180, angle))
                    self.arm.Arm_serial_servo_write(servo_id, angle, move_time)
                    self.current_angles[servo_id-1] = angle
                time.sleep(move_time / 1000)
                return True
            except Exception as e:
                print(f"    [ERROR] Multi-servo move failed: {e}")
                return False
        else:
            # Simulation
            for servo_id, angle in angles_dict.items():
                angle = max(0, min(180, angle))
                print(f"    [SIM] Servo {servo_id} → {angle}°")
                self.current_angles[servo_id-1] = angle
            time.sleep(move_time / 1000)
            return True
    
    def open_gripper(self):
        """Open the gripper"""
        print("    [GRIPPER] Opening...")
        success = self.set_servo_angle(self.SERVO_GRIPPER, self.GRIPPER_OPEN, 800)
        if success:
            self.gripper_open = True
            print("    [GRIPPER] Opened")
        return success
    
    def close_gripper(self):
        """Close the gripper"""
        print("    [GRIPPER] Closing...")
        success = self.set_servo_angle(self.SERVO_GRIPPER, self.GRIPPER_CLOSED, 800)
        if success:
            self.gripper_open = False
            print("    [GRIPPER] Closed")
        return success
    
    def move_to_xyz(self, x, y, z):
        """
        Move end effector to x,y,z coordinates using inverse kinematics
        Simplified for 4-DOF arm
        """
        print(f"    [MOVE] Moving to ({x:.3f}, {y:.3f}, {z:.3f})...")
        
        # Convert coordinates to servo angles (simplified IK)
        # This needs calibration for your specific arm!
        
        # For simulation, use approximate angles
        # In real implementation, use proper IK or teach positions
        
        # Simple mapping for demonstration (CALIBRATE THESE!)
        base_angle = 90 + int(x * 60)  # Adjust multiplier based on workspace
        shoulder_angle = 90 - int(y * 40)
        elbow_angle = 90 + int(z * 30)
        
        # Ensure within limits
        base_angle = max(30, min(150, base_angle))
        shoulder_angle = max(40, min(140, shoulder_angle))
        elbow_angle = max(40, min(140, elbow_angle))
        
        # Set wrist to maintain orientation
        wrist_angle = 90
        
        angles = {
            self.SERVO_BASE: base_angle,
            self.SERVO_SHOULDER: shoulder_angle,
            self.SERVO_ELBOW: elbow_angle,
            self.SERVO_WRIST: wrist_angle
        }
        
        success = self.set_multiple_angles(angles)
        
        if success:
            print(f"    [MOVE] Arrived at ({x:.3f}, {y:.3f}, {z:.3f})")
            return True
        else:
            print(f"    [MOVE] Failed to reach position")
            return False
    
    def lift(self, height):
        """Lift the arm by adjusting Z coordinate"""
        print(f"    [LIFT] Lifting to height {height:.3f}...")
        # Get current position (simplified)
        current_z = (self.current_angles[self.SERVO_ELBOW-1] - 90) / 30
        
        # Create lift movement
        return self.move_to_xyz(0, 0, height)
    
    def rotate_wrist(self, degrees):
        """Rotate the wrist"""
        print(f"    [ROTATE] Wrist rotating {degrees}°...")
        
        current_rot = self.current_angles[self.SERVO_WRIST_ROT-1]
        target_rot = (current_rot + degrees) % 180
        
        success = self.set_servo_angle(self.SERVO_WRIST_ROT, target_rot, 1000)
        
        if success:
            print(f"    [ROTATE] Wrist at {target_rot}°")
            return True
        return False
    
    def get_position(self):
        """Get current position estimate"""
        return self.current_angles
    
    def emergency_stop(self):
        """Emergency stop - relax all servos"""
        print("    [EMERGENCY] Stopping arm!")
        if self.connected:
            try:
                self.arm.Arm_serial_servo_write6(90, 90, 90, 90, self.GRIPPER_OPEN, 90, 500)
            except:
                pass
        time.sleep(0.5)

# ============================================
# COORDINATE MAPPING SYSTEM (Calibrated)
# ============================================
class CoordinateMapper:
    def __init__(self, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # CALIBRATE THESE VALUES FOR YOUR SETUP!
        # Workspace boundaries in meters
        self.workspace_x_range = (-0.25, 0.25)   # Left/Right
        self.workspace_y_range = (0.15, 0.45)    # Forward/Back
        self.workspace_z_range = (-0.05, 0.10)   # Up/Down
        
        # Drop zone locations (calibrate these!)
        self.drop_zones = {
            0: (0.15, 0.35, 0.03),    # Bolt
            1: (0.20, 0.35, 0.03),    # Hammer
            2: (0.15, 0.30, 0.03),    # Measuring Tape
            3: (0.20, 0.30, 0.03),    # Plier
            4: (0.15, 0.25, 0.03),    # Screwdriver
            5: (0.20, 0.25, 0.03)     # Wrench
        }
        
        # Camera to robot transformation (calibrate!)
        self.camera_offset_x = 0.0    # Camera X offset from arm base
        self.camera_offset_y = 0.3     # Camera Y offset (forward)
        self.camera_height_z = 0.4     # Camera height above workspace
        
        print("📍 Coordinate Mapper Initialized")
        print(f"   Workspace: X{self.workspace_x_range}, Y{self.workspace_y_range}")
    
    def pixel_to_robot_coords(self, pixel_x, pixel_y, bbox_width, bbox_height):
        """
        Convert pixel coordinates to robot coordinates with calibration
        Returns: (x, y, z) in robot coordinate system (meters)
        """
        # Normalize pixel coordinates (0 to 1)
        norm_x = pixel_x / self.camera_width
        norm_y = pixel_y / self.camera_height
        
        # Flip Y axis (camera Y increases downward)
        norm_y = 1.0 - norm_y
        
        # Map to robot workspace (calibrated)
        robot_x = self.workspace_x_range[0] + norm_x * (self.workspace_x_range[1] - self.workspace_x_range[0])
        robot_y = self.workspace_y_range[0] + norm_y * (self.workspace_y_range[1] - self.workspace_y_range[0])
        
        # Estimate Z based on object size (calibrate this!)
        object_area = bbox_width * bbox_height
        max_area = self.camera_width * self.camera_height * 0.1  # 10% of frame
        area_ratio = min(1.0, object_area / max_area)
        
        # Larger objects are higher (already on surface), smaller objects need to go lower
        robot_z = self.workspace_z_range[0] + (1 - area_ratio) * (self.workspace_z_range[1] - self.workspace_z_range[0])
        
        # Adjust for camera position
        robot_x += self.camera_offset_x
        robot_y += self.camera_offset_y
        
        return (robot_x, robot_y, robot_z)
    
    def get_approach_height(self, target_z):
        """Get safe approach height above object"""
        return target_z + 0.08  # 8cm above object (adjust as needed)
    
    def get_drop_location(self, class_id):
        """Get drop location for specific tool class"""
        return self.drop_zones.get(class_id, (0.2, 0.3, 0.03))

# ============================================
# DISPLACEMENT MANAGER WITH REAL ARM CONTROL
# ============================================
class DisplacementManager:
    def __init__(self):
        # Initialize real Yahboom arm
        self.arm = YahboomRoboticArm()
        self.mapper = CoordinateMapper()
        
        # Tool classes
        self.class_names = {
            0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
            3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
        }
        
        # Displacement tracking
        self.displaced_counts = defaultdict(int)
        self.target_displace_count = 1  # Number of each tool to displace
        self.current_target_class = None
        self.is_displacing = False
        self.paused_class = None
        self.interrupt_requested = False
        
        # Detection buffer for stability
        self.detection_buffer = deque(maxlen=15)
        
        # Safety parameters
        self.safe_height = 0.15  # Safe Z for moving above objects
        self.grip_delay = 0.5    # Time to wait after gripping
        
        print("📋 Displacement Manager Initialized")
        print(f"🔧 Will displace {self.target_displace_count} of each tool")
        if not self.arm.connected:
            print("⚠️ WARNING: Running in simulation mode - no real arm movement")
    
    def add_detection(self, detection):
        """Add a new detection to buffer"""
        self.detection_buffer.append(detection)
    
    def get_stable_detection(self, min_votes=3):
        """Get most consistently detected object (voting system)"""
        if len(self.detection_buffer) < min_votes:
            return None
        
        # Get recent detections (last 2 seconds)
        recent_time = time.time() - 2.0
        recent_dets = [d for d in self.detection_buffer if d['timestamp'] > recent_time]
        
        if len(recent_dets) < min_votes:
            return None
        
        # Count occurrences of each class
        class_votes = defaultdict(int)
        for det in recent_dets:
            class_votes[det['class_id']] += 1
        
        # Find class with most votes (must have minimum votes)
        if class_votes:
            best_class, votes = max(class_votes.items(), key=lambda x: x[1])
            if votes >= min_votes:
                # Get most recent detection of this class
                for det in reversed(recent_dets):
                    if det['class_id'] == best_class:
                        return det
        
        return None
    
    def should_interrupt(self, detected_class_id):
        """
        Check if we should interrupt current displacement
        Returns: (should_interrupt, reason)
        """
        if not self.is_displacing:
            return False, "Not currently displacing"
        
        if self.interrupt_requested:
            return False, "Interrupt already requested"
        
        # Check if this class is already fully displaced
        if self.displaced_counts[detected_class_id] >= self.target_displace_count:
            return False, f"Class {self.class_names[detected_class_id]} already fully displaced"
        
        # If we're currently displacing a different class
        if self.current_target_class != detected_class_id:
            # Only interrupt if new object is clearly visible (high confidence)
            recent_det = self.get_stable_detection(min_votes=5)
            if recent_det and recent_det['confidence'] > 0.7:
                return True, f"Found clear {self.class_names[detected_class_id]} during displacement"
        
        return False, "Continuing with current class"
    
    def execute_displacement(self, detection):
        """Execute complete displacement sequence with real arm"""
        if self.is_displacing:
            print("    ⚠️ Already displacing, cannot start new displacement")
            return False
        
        self.is_displacing = True
        class_id = detection['class_id']
        class_name = self.class_names[class_id]
        
        print(f"\n{'='*60}")
        print(f"🚀 STARTING DISPLACEMENT: {class_name}")
        print(f"   Confidence: {detection['confidence_percent']:.1f}%")
        print(f"{'='*60}")
        
        try:
            # 1. Calculate object position
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Convert to robot coordinates
            target_pos = self.mapper.pixel_to_robot_coords(center_x, center_y, bbox_width, bbox_height)
            approach_height = self.mapper.get_approach_height(target_pos[2])
            drop_pos = self.mapper.get_drop_location(class_id)
            
            print(f"   📍 Object at: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})m")
            print(f"   📍 Drop zone: ({drop_pos[0]:.3f}, {drop_pos[1]:.3f}, {drop_pos[2]:.3f})m")
            print(f"   📏 Approach height: {approach_height:.3f}m")
            
            # 2. Open gripper (initial state)
            print("\n   [STEP 1] Opening gripper...")
            if not self.arm.open_gripper():
                raise Exception("Failed to open gripper")
            
            # 3. Move to safe approach position
            print("\n   [STEP 2] Moving to approach position...")
            approach_pos = (target_pos[0], target_pos[1], self.safe_height)
            if not self.arm.move_to_xyz(*approach_pos):
                raise Exception("Failed to move to approach position")
            
            # 4. Move down to object height
            print("\n   [STEP 3] Descending to object...")
            grasp_height = (target_pos[0], target_pos[1], target_pos[2] + 0.02)  # 2cm above surface
            if not self.arm.move_to_xyz(*grasp_height):
                raise Exception("Failed to descend to object")
            
            # 5. Close gripper
            print("\n   [STEP 4] Closing gripper...")
            if not self.arm.close_gripper():
                raise Exception("Failed to close gripper")
            
            # Wait for grip to stabilize
            time.sleep(self.grip_delay)
            
            # 6. Lift object
            print("\n   [STEP 5] Lifting object...")
            if not self.arm.lift(self.safe_height):
                raise Exception("Failed to lift object")
            
            # 7. Rotate wrist 180 degrees
            print("\n   [STEP 6] Rotating 180°...")
            if not self.arm.rotate_wrist(180):
                print("    ⚠️ Could not rotate wrist, continuing...")
            
            # 8. Move to drop zone approach
            print("\n   [STEP 7] Moving to drop zone...")
            drop_approach = (drop_pos[0], drop_pos[1], self.safe_height)
            if not self.arm.move_to_xyz(*drop_approach):
                raise Exception("Failed to move to drop zone")
            
            # 9. Lower to drop height
            print("\n   [STEP 8] Lowering to drop height...")
            if not self.arm.move_to_xyz(*drop_pos):
                raise Exception("Failed to lower to drop height")
            
            # 10. Open gripper to release
            print("\n   [STEP 9] Releasing object...")
            if not self.arm.open_gripper():
                raise Exception("Failed to open gripper for release")
            
            time.sleep(0.5)  # Let object settle
            
            # 11. Lift back to safe height
            print("\n   [STEP 10] Returning to safe height...")
            if not self.arm.lift(self.safe_height):
                raise Exception("Failed to return to safe height")
            
            # 12. Rotate back to original orientation
            print("\n   [STEP 11] Returning to original orientation...")
            self.arm.rotate_wrist(-180)
            
            # 13. Return to neutral position
            print("\n   [STEP 12] Returning to neutral...")
            self.arm.go_to_neutral()
            
            # Update displacement count
            self.displaced_counts[class_id] += 1
            print(f"\n   ✅ Successfully displaced {class_name}")
            print(f"   📊 Total {class_name}s displaced: {self.displaced_counts[class_id]}/{self.target_displace_count}")
            
            return True
            
        except Exception as e:
            print(f"\n   ❌ Displacement failed: {e}")
            # Emergency procedures
            print("   🚨 Executing emergency recovery...")
            self.arm.open_gripper()
            self.arm.lift(self.safe_height)
            self.arm.go_to_neutral()
            return False
        
        finally:
            self.is_displacing = False
            self.current_target_class = None
    
    def handle_interruption(self, new_detection):
        """Handle interruption for higher priority object"""
        if self.is_displacing:
            print(f"\n🚨 INTERRUPTION: Pausing current task...")
            
            # Store current state
            self.paused_class = self.current_target_class
            self.interrupt_requested = True
            
            # Wait for current movement to complete
            time.sleep(1.0)
            
            # Execute new displacement
            print(f"🚀 Executing interruption for {self.class_names[new_detection['class_id']]}")
            success = self.execute_displacement(new_detection)
            
            # Resume if needed
            if self.paused_class is not None and success:
                print(f"\n🔄 RESUMING paused task: {self.class_names[self.paused_class]}")
                self.current_target_class = self.paused_class
                self.paused_class = None
            
            self.interrupt_requested = False
            return success
        
        return False
    
    def check_completion(self):
        """Check if all displacement tasks are complete"""
        all_done = all(
            self.displaced_counts[class_id] >= self.target_displace_count 
            for class_id in self.class_names.keys()
        )
        return all_done
    
    def print_status(self):
        """Print current displacement status"""
        print("\n" + "=" * 60)
        print("📊 DISPLACEMENT STATUS")
        print("=" * 60)
        for class_id, class_name in self.class_names.items():
            count = self.displaced_counts[class_id]
            status = "✅ DONE" if count >= self.target_displace_count else f"🔄 {count}/{self.target_displace_count}"
            print(f"  {class_name}: {status}")
        
        if self.current_target_class is not None:
            print(f"\n🎯 Currently displacing: {self.class_names[self.current_target_class]}")
        
        if self.paused_class is not None:
            print(f"⏸️  Paused class: {self.class_names[self.paused_class]}")
        
        print(f"🤖 Arm connected: {'✅' if self.arm.connected else '❌'}")
        print("=" * 60)

# ============================================
# MAIN VISION AND CONTROL SYSTEM
# ============================================
class RoboticVisionSystem:
    def __init__(self):
        # Setup camera
        self.cap = self.setup_camera()
        if self.cap is None:
            print("❌ ERROR: No camera found!")
            exit()
        
        # Load YOLO model
        self.model = self.load_yolo_model()
        
        # Initialize displacement manager with real arm
        self.displacement_mgr = DisplacementManager()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Detection parameters
        self.detection_interval = 5
        self.frame_index = 0
        self.last_detections = []
        
        # Tool colors for display
        self.tool_colors = [
            (0, 255, 0),    # Green - Bolt
            (0, 0, 255),    # Red - Hammer  
            (255, 0, 0),    # Blue - Measuring Tape
            (255, 255, 0),  # Cyan - Plier
            (255, 0, 255),  # Magenta - Screwdriver
            (0, 255, 255)   # Yellow - Wrench
        ]
        
        # Display settings
        self.show_fps = True
        self.show_status = True
        self.highlight_tool = -1
        
        # Control flags
        self.paused = False
        self.emergency_stop = False
        
        print("\n✅ System initialized and ready!")
        print("Controls:")
        print("  s - Show/hide status")
        print("  p - Pause/resume system")
        print("  e - Emergency stop")
        print("  r - Reset arm to neutral")
        print("  q - Quit")
        print("=" * 70)
    
    def setup_camera(self):
        """Setup camera for maximum FPS"""
        print("📷 Setting up camera...")
        
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"✅ Found camera at index {i}")
                
                # Optimize camera settings
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure
                cap.set(cv2.CAP_PROP_EXPOSURE, 100)
                
                return cap
        
        return None
    
    def load_yolo_model(self):
        """Load YOLO model"""
        print("📦 Loading YOLO model...")
        
        try:
            model = YOLO('best.pt')
            
            # Apply optimizations
            model.overrides['conf'] = 0.35
            model.overrides['iou'] = 0.3
            model.overrides['agnostic_nms'] = True
            model.overrides['max_det'] = 6
            model.overrides['verbose'] = False
            
            print("✅ YOLO model loaded")
            return model
            
        except Exception as e:
            print(f"❌ ERROR loading YOLO: {e}")
            print("Running in camera-only mode...")
            return None
    
    def confidence_to_percent(self, confidence):
        """Convert confidence score to percentage"""
        return confidence * 100
    
    def process_detection(self, frame):
        """Process frame for object detection"""
        if self.model is None or self.frame_index % self.detection_interval != 0:
            return self.last_detections
        
        try:
            # Resize for faster inference
            inference_size = 256
            small_frame = cv2.resize(frame, (inference_size, int(inference_size * 0.75)))
            
            # Run inference
            results = self.model(small_frame, 
                               conf=0.45,  # Higher confidence for real operation
                               iou=0.3,
                               imgsz=inference_size,
                               max_det=6,
                               verbose=False,
                               device='cpu',
                               agnostic_nms=True)
            
            # Process results
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Scale boxes back to original size
                scale_x = frame.shape[1] / inference_size
                scale_y = frame.shape[0] / (inference_size * 0.75)
                
                detections = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    
                    # Scale coordinates
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    # Clamp to frame bounds
                    x1 = max(0, min(x1, frame.shape[1]-1))
                    y1 = max(0, min(y1, frame.shape[0]-1))
                    x2 = max(0, min(x2, frame.shape[1]-1))
                    y2 = max(0, min(y2, frame.shape[0]-1))
                    
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    confidence_percent = self.confidence_to_percent(confidence)
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confidence),
                        'confidence_percent': float(confidence_percent),
                        'class_id': int(class_id),
                        'class_name': self.displacement_mgr.class_names[class_id],
                        'timestamp': time.time()
                    }
                    
                    detections.append(detection)
                    self.displacement_mgr.add_detection(detection)
                
                self.last_detections = detections
                
                # Check for displacement opportunities (if not paused)
                if not self.paused and not self.displacement_mgr.is_displacing:
                    self.check_and_handle_displacement()
                
                return detections
            else:
                self.last_detections = []
                return []
                
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return []
    
    def check_and_handle_displacement(self):
        """Check if we should start or interrupt displacement"""
        if self.emergency_stop or self.paused:
            return
        
        # Get stable detection
        detection = self.displacement_mgr.get_stable_detection(min_votes=4)
        if detection is None:
            return
        
        class_id = detection['class_id']
        class_name = self.displacement_mgr.class_names[class_id]
        
        # Check if this class is already done
        if self.displacement_mgr.displaced_counts[class_id] >= self.displacement_mgr.target_displace_count:
            return
        
        # Check if we should interrupt
        should_interrupt, reason = self.displacement_mgr.should_interrupt(class_id)
        
        if should_interrupt:
            print(f"\n🚨 INTERRUPTION: {reason}")
            # Handle in a separate thread to not block video
            threading.Thread(target=self.displacement_mgr.handle_interruption, 
                           args=(detection,), daemon=True).start()
            
        elif not self.displacement_mgr.is_displacing:
            # Start new displacement
            if self.displacement_mgr.current_target_class is None or \
               self.displacement_mgr.current_target_class == class_id:
                
                self.displacement_mgr.current_target_class = class_id
                print(f"\n🎯 Starting displacement of {class_name}...")
                # Run in separate thread
                threading.Thread(target=self.displacement_mgr.execute_displacement, 
                               args=(detection,), daemon=True).start()
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence_percent = det['confidence_percent']
            
            # Skip if highlighting specific tool
            if self.highlight_tool != -1 and class_id != self.highlight_tool:
                continue
            
            # Get color
            color = self.tool_colors[class_id % len(self.tool_colors)]
            
            # Draw bounding box (thicker if this is current target)
            thickness = 3 if class_id == self.displacement_mgr.current_target_class else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with percentage
            label = f"{det['class_name']}: {confidence_percent:.1f}%"
            
            # Text background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            
            cv2.putText(frame, label, 
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
            cv2.circle(frame, (center_x, center_y), 6, (255, 255, 255), 1)
        
        return frame
    
    def draw_status(self, frame):
        """Draw system status on frame"""
        if not self.show_status:
            return frame
        
        y_offset = 30
        line_height = 22
        
        # Background for status
        cv2.rectangle(frame, (5, 5), (350, 220), (0, 0, 0, 180), -1)
        cv2.rectangle(frame, (5, 5), (350, 220), (255, 255, 255), 1)
        
        # System status header
        status_color = (0, 255, 0) if not self.emergency_stop else (0, 0, 255)
        status_text = "RUNNING" if not self.paused else "PAUSED"
        if self.emergency_stop:
            status_text = "EMERGENCY STOP"
        
        cv2.putText(frame, f"🤖 ROBOT STATUS: {status_text}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        y_offset += line_height
        
        # FPS and detection info
        cv2.putText