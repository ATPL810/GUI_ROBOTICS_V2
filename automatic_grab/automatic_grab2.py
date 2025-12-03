"""
COMPLETE ROBOTIC ARM CONTROL SYSTEM WITH YOLO11 AND ARM_LIB
Automatic search from neutral position with safe grabbing logic
Using ONLY Arm_Lib (no yahboom_arm_lib)
CORRECTED SERVO MAPPING
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
import json

print("🤖 ROBOTIC ARM CONTROL SYSTEM WITH AUTOMATIC SEARCH")
print("=" * 70)

# ============================================
# ARM_LIB SERVO CONTROL CLASS
# ============================================
class ArmLibServoController:
    def __init__(self):
        """Initialize using Arm_Lib library"""
        self.connected = False
        self.arm = None
        
        # Try to import Arm_Lib
        try:
            from Arm_Lib import Arm_Device
            self.Arm_Device = Arm_Device
            ARM_LIB_AVAILABLE = True
            print("✅ Arm_Lib found and imported")
        except ImportError as e:
            print(f"⚠️ Arm_Lib not found: {e}")
            print("Running in simulation mode")
            ARM_LIB_AVAILABLE = False
            self.connected = False
            return
        
        # ============================================
        # CORRECTED SERVO MAPPING FOR YAHBOOM ARM
        # ============================================
        # Arm_Lib uses servo IDs 1-6 (not channels 0-5)
        # Servo 1: Base (bottom-most)
        # Servo 2: Shoulder
        # Servo 3: Elbow
        # Servo 4: Wrist
        # Servo 5: Wrist rotation
        # Servo 6: Gripper (top-most)
        # ============================================
        
        self.SERVO_BASE = 1       # Servo 1: Base rotation (bottom)
        self.SERVO_SHOULDER = 2   # Servo 2: Shoulder
        self.SERVO_ELBOW = 3      # Servo 3: Elbow
        self.SERVO_WRIST = 4      # Servo 4: Wrist
        self.SERVO_WRIST_ROT = 5  # Servo 5: Wrist rotation
        self.SERVO_GRIPPER = 6    # Servo 6: Gripper (top)
        
        # Current servo positions (0-180 degrees)
        # Index 0 is unused (servo IDs start at 1)
        self.current_angles = [0, 90, 90, 90, 90, 90, 70]  # Index 0 unused
        self.gripper_open = True
        
        # Gripper positions (CALIBRATE THESE FOR YOUR ARM!)
        self.GRIPPER_OPEN = 70     # Angle for open gripper
        self.GRIPPER_CLOSED = 110  # Angle for closed gripper
        
        # Safe grabbing parameters
        self.gripping_pressure_threshold = 3  # Degrees change threshold
        self.max_grip_attempts = 5
        
        # Initialize arm connection
        self.initialize_arm()
    
    def initialize_arm(self):
        """Initialize connection to arm using Arm_Lib"""
        try:
            # Create Arm_Device instance
            self.arm = self.Arm_Device()
            time.sleep(2)  # Wait for arm to initialize
            
            # Test connection by reading current angles
            angles = self.arm.Arm_serial_servo_read_all()
            if angles:
                self.connected = True
                print("✅ Arm_Lib connected successfully")
                
                # Update current angles from arm
                for i in range(1, 7):
                    if i <= len(angles):
                        self.current_angles[i] = angles[i-1]
                
                # Set all servos to initial positions
                self.go_to_neutral()
                
                print("   Servo Mapping (Arm_Lib IDs):")
                print(f"     Servo {self.SERVO_BASE}: Base")
                print(f"     Servo {self.SERVO_SHOULDER}: Shoulder")
                print(f"     Servo {self.SERVO_ELBOW}: Elbow")
                print(f"     Servo {self.SERVO_WRIST}: Wrist")
                print(f"     Servo {self.SERVO_WRIST_ROT}: Wrist Rotation")
                print(f"     Servo {self.SERVO_GRIPPER}: Gripper")
                
            else:
                print("⚠️ Could not read arm angles, running in simulation")
                self.connected = False
                
        except Exception as e:
            print(f"⚠️ Arm_Lib initialization failed: {e}")
            print("Running in simulation mode")
            self.connected = False
    
    def set_servo_angle(self, servo_id, angle, move_time=1000):
        """Set servo to specific angle using Arm_Lib"""
        # Validate angle
        angle = max(0, min(180, float(angle)))
        
        # Get servo name for logging
        servo_names = {
            self.SERVO_BASE: "Base",
            self.SERVO_SHOULDER: "Shoulder",
            self.SERVO_ELBOW: "Elbow",
            self.SERVO_WRIST: "Wrist",
            self.SERVO_WRIST_ROT: "WristRot",
            self.SERVO_GRIPPER: "Gripper"
        }
        servo_name = servo_names.get(servo_id, f"Servo{servo_id}")
        
        if self.connected:
            try:
                # Arm_Lib method: Arm_serial_servo_write(servo_id, angle, time)
                self.arm.Arm_serial_servo_write(servo_id, angle, move_time)
                self.current_angles[servo_id] = angle
                
                # Wait for movement to complete
                time.sleep(move_time / 1000.0)
                
                print(f"    [ARM] {servo_name} → {angle}° ({move_time}ms)")
                return True
                
            except Exception as e:
                print(f"    [ERROR] Failed to move {servo_name}: {e}")
                return False
        else:
            # Simulation mode
            print(f"    [SIM] {servo_name} → {angle}° ({move_time}ms)")
            self.current_angles[servo_id] = angle
            
            if move_time > 0:
                time.sleep(move_time / 1000.0)
            
            return True
    
    def set_multiple_angles(self, angles_dict, move_time=1000):
        """Set multiple servos simultaneously using Arm_Lib"""
        results = []
        
        # Arm_Lib has different methods for multiple servos
        if self.connected:
            try:
                # Method 1: Set all 6 servos at once (if angles_dict has all 6)
                if len(angles_dict) == 6:
                    # Create array for all 6 servos
                    all_angles = [0] * 6  # Index 0-5 for servos 1-6
                    for servo_id, angle in angles_dict.items():
                        all_angles[servo_id-1] = angle
                    
                    # Use Arm_serial_servo_write6 for all servos
                    self.arm.Arm_serial_servo_write6(
                        all_angles[0], all_angles[1], all_angles[2],
                        all_angles[3], all_angles[4], all_angles[5],
                        move_time
                    )
                    
                    # Update all angles
                    for servo_id, angle in angles_dict.items():
                        self.current_angles[servo_id] = angle
                    
                    results.append(True)
                    
                else:
                    # Set servos individually
                    for servo_id, angle in angles_dict.items():
                        result = self.set_servo_angle(servo_id, angle, 0)
                        results.append(result)
                
                # Wait for movement to complete
                if move_time > 0:
                    time.sleep(move_time / 1000.0)
                
                return all(results)
                
            except Exception as e:
                print(f"    [ERROR] Multi-servo move failed: {e}")
                # Fall back to individual moves
                pass
        
        # Fallback: Set servos individually
        for servo_id, angle in angles_dict.items():
            result = self.set_servo_angle(servo_id, angle, 0)
            results.append(result)
        
        # Wait for movement to complete
        if move_time > 0:
            time.sleep(move_time / 1000.0)
        
        return all(results)
    
    def safe_close_gripper(self):
        """Close gripper with safe grabbing logic"""
        print("    [GRIPPER] Safe closing...")
        
        if not self.connected:
            # Simulation mode
            self.set_servo_angle(self.SERVO_GRIPPER, self.GRIPPER_CLOSED, 800)
            self.gripper_open = False
            return True
        
        # Real mode with safe closing
        current_position = self.current_angles[self.SERVO_GRIPPER]
        target_position = self.GRIPPER_CLOSED
        
        # Step-wise closing with monitoring
        step_size = 5  # Degrees per step
        steps = int(abs(target_position - current_position) / step_size)
        
        print(f"    [GRIPPER] Starting at {current_position}°, target {target_position}°")
        
        for step in range(steps):
            if target_position > current_position:
                next_position = current_position + step_size * (step + 1)
            else:
                next_position = current_position - step_size * (step + 1)
            
            # Check if we should stop (object grasped)
            if step > 0:
                position_change = abs(next_position - self.current_angles[self.SERVO_GRIPPER])
                if position_change < self.gripping_pressure_threshold:
                    print(f"    [GRIPPER] Object grasped at {next_position}°, stopping early")
                    break
            
            self.set_servo_angle(self.SERVO_GRIPPER, next_position, 100)
            time.sleep(0.1)  # Wait between steps
        
        self.gripper_open = False
        print("    [GRIPPER] Closed safely")
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
        """Close gripper (uses safe method)"""
        return self.safe_close_gripper()
    
    def go_to_neutral(self):
        """Move all servos to neutral position for camera view"""
        print("    [ARM] Moving to neutral/camera position...")
        
        # Neutral position optimized for camera view
        neutral_angles = {
            self.SERVO_BASE: 90,           # Base centered
            self.SERVO_SHOULDER: 70,       # Shoulder lowered for view
            self.SERVO_ELBOW: 100,         # Elbow extended
            self.SERVO_WRIST: 80,          # Wrist level
            self.SERVO_WRIST_ROT: 90,      # Wrist rotation straight
            self.SERVO_GRIPPER: self.GRIPPER_OPEN  # Gripper open
        }
        
        success = self.set_multiple_angles(neutral_angles, 1500)
        self.gripper_open = True
        
        if success:
            print("    [ARM] In neutral/camera position")
        return success
    
    def get_position(self):
        """Get current servo positions"""
        return self.current_angles.copy()
    
    def print_servo_status(self):
        """Print current status of all servos"""
        print("\n" + "=" * 40)
        print("SERVO STATUS (Arm_Lib IDs):")
        print("=" * 40)
        
        servo_names = {
            self.SERVO_BASE: "Base (Servo 1)",
            self.SERVO_SHOULDER: "Shoulder (Servo 2)", 
            self.SERVO_ELBOW: "Elbow (Servo 3)",
            self.SERVO_WRIST: "Wrist (Servo 4)",
            self.SERVO_WRIST_ROT: "WristRot (Servo 5)",
            self.SERVO_GRIPPER: "Gripper (Servo 6)"
        }
        
        for servo_id, name in servo_names.items():
            angle = self.current_angles[servo_id]
            status = "✅" if 0 <= angle <= 180 else "❌"
            print(f"  {name:20} → {angle:3}° {status}")
        
        print(f"  Gripper: {'OPEN' if self.gripper_open else 'CLOSED'}")
        print(f"  Connected: {'Yes' if self.connected else 'No (Simulation)'}")
        print("=" * 40)

# ============================================
# COORDINATE MAPPING SYSTEM
# ============================================
class CoordinateMapper:
    def __init__(self, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Camera calibration parameters (ADJUST THESE!)
        # Workspace in meters relative to arm base
        self.workspace_x_range = (-0.25, 0.25)   # Left/Right
        self.workspace_y_range = (0.15, 0.40)    # Forward/Back (arm reach)
        self.workspace_z_range = (-0.10, 0.05)   # Up/Down
        
        # Drop zones for each tool type
        self.drop_zones = {
            0: (0.15, 0.35, 0.02),    # Bolt - Zone 1
            1: (-0.15, 0.35, 0.02),   # Hammer - Zone 2
            2: (0.15, 0.25, 0.02),    # Measuring Tape - Zone 3
            3: (-0.15, 0.25, 0.02),   # Plier - Zone 4
            4: (0.15, 0.15, 0.02),    # Screwdriver - Zone 5
            5: (-0.15, 0.15, 0.02)    # Wrench - Zone 6
        }
        
        # Camera position relative to arm base (calibrate!)
        self.camera_offset_x = 0.0    # Side offset
        self.camera_offset_y = 0.25   # Forward offset
        self.camera_height_z = 0.35   # Height above workspace
        
        # Arm link lengths (ADJUST FOR YOUR ARM!)
        self.L1 = 0.05   # Base to shoulder height
        self.L2 = 0.12   # Upper arm length
        self.L3 = 0.10   # Forearm length
        self.L4 = 0.05   # Wrist to gripper
        
        print("📍 Coordinate Mapper Initialized")
        print(f"   Camera: {camera_width}x{camera_height}")
        print(f"   Workspace: X{self.workspace_x_range}, Y{self.workspace_y_range}")
    
    def pixel_to_robot_coords(self, pixel_x, pixel_y, bbox_width, bbox_height):
        """
        Convert pixel coordinates to robot coordinates
        Returns: (x, y, z) in meters relative to arm base
        """
        # Normalize pixel coordinates (0 to 1)
        norm_x = pixel_x / self.camera_width
        norm_y = pixel_y / self.camera_height
        
        # Flip Y axis (camera Y increases downward)
        norm_y = 1.0 - norm_y
        
        # Map to robot workspace
        robot_x = self.workspace_x_range[0] + norm_x * (self.workspace_x_range[1] - self.workspace_x_range[0])
        robot_y = self.workspace_y_range[0] + norm_y * (self.workspace_y_range[1] - self.workspace_y_range[0])
        
        # Estimate Z based on object size
        object_area = bbox_width * bbox_height
        max_area = self.camera_width * self.camera_height * 0.15
        area_ratio = min(1.0, object_area / max_area)
        
        # Larger objects appear higher (closer to camera), smaller are lower
        robot_z = self.workspace_z_range[0] + (1 - area_ratio) * (self.workspace_z_range[1] - self.workspace_z_range[0])
        
        # Apply camera offset
        robot_x += self.camera_offset_x
        robot_y += self.camera_offset_y
        
        # Add small random offset to avoid always grabbing center
        robot_x += np.random.uniform(-0.01, 0.01)
        robot_y += np.random.uniform(-0.01, 0.01)
        
        return (robot_x, robot_y, robot_z)
    
    def robot_to_servo_angles(self, x, y, z):
        """
        Convert robot coordinates to servo angles using inverse kinematics
        """
        # Calculate distance from base
        distance = math.sqrt(x**2 + y**2)
        
        # Base angle (rotation around Z axis)
        base_angle = 90 + math.degrees(math.atan2(x, y))
        
        # Target position for wrist (end of L3)
        wrist_x = distance - self.L4
        wrist_z = z - self.L1
        
        # Distance to wrist from shoulder
        D = math.sqrt(wrist_x**2 + wrist_z**2)
        
        # Check if reachable
        if D > (self.L2 + self.L3) or D < abs(self.L2 - self.L3):
            print(f"    ⚠️ Position ({x:.3f}, {y:.3f}, {z:.3f}) is unreachable (D={D:.3f})")
            return None
        
        try:
            # Elbow angle (theta3)
            cos_theta3 = (self.L2**2 + self.L3**2 - D**2) / (2 * self.L2 * self.L3)
            cos_theta3 = max(-1, min(1, cos_theta3))
            theta3 = math.acos(cos_theta3)
            
            # Shoulder angle (theta2)
            cos_theta2 = (self.L2**2 + D**2 - self.L3**2) / (2 * self.L2 * D)
            cos_theta2 = max(-1, min(1, cos_theta2))
            theta2 = math.acos(cos_theta2)
            
            alpha = math.atan2(wrist_z, wrist_x)
            
            # Convert to servo angles (degrees)
            shoulder_angle = 90 - math.degrees(alpha + theta2)
            elbow_angle = 90 - math.degrees(theta3)
            
            # Wrist angle to keep gripper level
            wrist_angle = 90 - (shoulder_angle - 90) - (elbow_angle - 90)
            
            # Wrist rotation (keep it straight for now)
            wrist_rot_angle = 90
            
        except Exception as e:
            print(f"    ⚠️ IK calculation error: {e}")
            return None
        
        # Angle constraints
        base_angle = max(30, min(150, base_angle))
        shoulder_angle = max(40, min(140, shoulder_angle))
        elbow_angle = max(40, min(140, elbow_angle))
        wrist_angle = max(40, min(140, wrist_angle))
        wrist_rot_angle = max(30, min(150, wrist_rot_angle))
        
        return {
            'base': base_angle,
            'shoulder': shoulder_angle,
            'elbow': elbow_angle,
            'wrist': wrist_angle,
            'wrist_rot': wrist_rot_angle,
            'gripper': 70  # Open
        }
    
    def get_drop_location(self, class_id):
        """Get drop location for specific tool class"""
        return self.drop_zones.get(class_id, (0.0, 0.3, 0.02))

# ============================================
# AUTOMATIC SEARCH MANAGER (WITH ARM_LIB)
# ============================================
class AutomaticSearchManager:
    def __init__(self):
        # Initialize arm controller with Arm_Lib
        self.arm = ArmLibServoController()
        self.mapper = CoordinateMapper()
        
        # Tool classes
        self.class_names = {
            0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
            3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
        }
        
        # Tool colors for visualization
        self.tool_colors = [
            (0, 255, 0),    # Green - Bolt
            (0, 0, 255),    # Red - Hammer  
            (255, 0, 0),    # Blue - Measuring Tape
            (255, 255, 0),  # Cyan - Plier
            (255, 0, 255),  # Magenta - Screwdriver
            (0, 255, 255)   # Yellow - Wrench
        ]
        
        # Search and displacement tracking
        self.target_counts = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
        self.displaced_counts = defaultdict(int)
        self.is_working = False
        
        # Detection buffer for stability
        self.detection_buffer = deque(maxlen=20)
        self.stable_detection_threshold = 5
        
        # Search parameters
        self.search_pattern = 'sequential'
        self.search_interval = 2.0
        
        print("🔍 Automatic Search Manager Initialized")
        print(f"   Using: {'Arm_Lib' if self.arm.connected else 'Simulation Mode'}")
    
    def add_detection(self, detection):
        """Add detection to buffer"""
        self.detection_buffer.append(detection)
    
    def get_stable_detections(self):
        """Get stable detections from buffer"""
        if len(self.detection_buffer) < self.stable_detection_threshold:
            return []
        
        recent_time = time.time() - 3.0
        recent_dets = [d for d in self.detection_buffer if d['timestamp'] > recent_time]
        
        if len(recent_dets) < self.stable_detection_threshold:
            return []
        
        # Group by class and calculate average positions
        class_groups = defaultdict(list)
        for det in recent_dets:
            class_groups[det['class_id']].append(det)
        
        # Filter for stable detections
        stable_dets = []
        for class_id, dets in class_groups.items():
            if len(dets) >= 3:
                avg_bbox = np.mean([d['bbox'] for d in dets], axis=0).astype(int)
                avg_confidence = np.mean([d['confidence'] for d in dets])
                
                stable_det = {
                    'bbox': tuple(avg_bbox),
                    'confidence': float(avg_confidence),
                    'confidence_percent': float(avg_confidence * 100),
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id],
                    'timestamp': time.time(),
                    'stable': True
                }
                stable_dets.append(stable_det)
        
        return stable_dets
    
    def analyze_workspace(self, detections):
        """Analyze workspace and choose target"""
        if not detections:
            return 'wait', None
        
        # Filter out completed tools
        valid_detections = []
        for det in detections:
            class_id = det['class_id']
            if self.displaced_counts[class_id] < self.target_counts[class_id]:
                valid_detections.append(det)
        
        if not valid_detections:
            return 'wait', None
        
        # Choose based on search pattern
        if self.search_pattern == 'sequential':
            for class_id in sorted(self.class_names.keys()):
                if self.displaced_counts[class_id] < self.target_counts[class_id]:
                    for det in valid_detections:
                        if det['class_id'] == class_id:
                            return 'pick', det
        
        elif self.search_pattern == 'closest':
            closest_dist = float('inf')
            closest_det = None
            for det in valid_detections:
                bbox = det['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                distance = math.sqrt((center_x - 320)**2 + (center_y - 240)**2)
                
                if distance < closest_dist:
                    closest_dist = distance
                    closest_det = det
            return 'pick', closest_det
        
        # Default: random
        import random
        return 'pick', random.choice(valid_detections)
    
    def execute_search_cycle(self):
        """Execute one complete search and pickup cycle"""
        if self.is_working:
            print("    ⚠️ Already working, skipping cycle")
            return False
        
        self.is_working = True
        
        try:
            print(f"\n{'='*60}")
            print(f"🔍 SEARCH CYCLE STARTED")
            print(f"{'='*60}")
            
            # Step 1: Move to neutral position
            print("\n   [STEP 1] Moving to neutral/camera position...")
            if not self.arm.go_to_neutral():
                raise Exception("Failed to move to neutral position")
            
            time.sleep(1.0)
            
            # Step 2: Get stable detections
            print("\n   [STEP 2] Analyzing workspace...")
            stable_dets = self.get_stable_detections()
            
            if not stable_dets:
                print("    ℹ️ No stable detections found, waiting...")
                self.is_working = False
                return False
            
            # Display detected tools
            print(f"    📊 Detected tools: {len(stable_dets)}")
            for det in stable_dets:
                print(f"      - {det['class_name']}: {det['confidence_percent']:.1f}%")
            
            # Step 3: Choose target
            action, target_det = self.analyze_workspace(stable_dets)
            
            if action != 'pick' or target_det is None:
                print("    ℹ️ No valid target found")
                self.is_working = False
                return False
            
            # Step 4: Execute pickup
            class_name = target_det['class_name']
            class_id = target_det['class_id']
            
            print(f"\n   [STEP 3] Selected target: {class_name}")
            print(f"    Confidence: {target_det['confidence_percent']:.1f}%")
            
            if self.displaced_counts[class_id] >= self.target_counts[class_id]:
                print(f"    ℹ️ Already displaced enough {class_name}s")
                self.is_working = False
                return False
            
            # Execute pickup
            success = self.execute_pickup(target_det)
            
            if success:
                self.displaced_counts[class_id] += 1
                print(f"\n   ✅ Successfully displaced {class_name}")
                print(f"    📊 Total: {self.displaced_counts[class_id]}/{self.target_counts[class_id]}")
            else:
                print(f"\n   ❌ Failed to displace {class_name}")
            
            # Step 5: Return to neutral
            print("\n   [STEP 4] Returning to neutral position...")
            self.arm.go_to_neutral()
            
            # Check completion
            all_done = all(
                self.displaced_counts[class_id] >= self.target_counts[class_id]
                for class_id in self.class_names.keys()
            )
            
            if all_done:
                print("\n🎉 ALL TASKS COMPLETED!")
            
            return success
            
        except Exception as e:
            print(f"\n   ❌ Search cycle failed: {e}")
            print("   🚨 Executing emergency recovery...")
            self.arm.open_gripper()
            self.arm.go_to_neutral()
            return False
        
        finally:
            self.is_working = False
    
    def execute_pickup(self, detection):
        """Execute pickup sequence for a single object"""
        try:
            class_id = detection['class_id']
            class_name = detection['class_name']
            
            print(f"\n   🤖 PICKUP SEQUENCE: {class_name}")
            print(f"   {'-'*40}")
            
            # Calculate object position
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Convert to robot coordinates
            target_pos = self.mapper.pixel_to_robot_coords(center_x, center_y, bbox_width, bbox_height)
            print(f"   📍 Object at: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})m")
            
            # Get servo angles
            servo_angles = self.mapper.robot_to_servo_angles(target_pos[0], target_pos[1], target_pos[2])
            if servo_angles is None:
                print("    ⚠️ Cannot reach this position")
                return False
            
            # Get drop location
            drop_pos = self.mapper.get_drop_location(class_id)
            print(f"   📍 Drop zone: ({drop_pos[0]:.3f}, {drop_pos[1]:.3f}, {drop_pos[2]:.3f})m")
            
            drop_angles = self.mapper.robot_to_servo_angles(drop_pos[0], drop_pos[1], drop_pos[2])
            if drop_angles is None:
                print("    ⚠️ Cannot reach drop location")
                return False
            
            # ----- PICKUP SEQUENCE -----
            
            # 1. Open gripper
            print("\n   [PICK 1] Ensuring gripper is open...")
            self.arm.open_gripper()
            time.sleep(0.5)
            
            # 2. Move to approach position
            print("\n   [PICK 2] Moving to approach position...")
            approach_angles = servo_angles.copy()
            approach_angles['elbow'] += 15
            approach_angles['shoulder'] -= 5
            
            success = self.arm.set_multiple_angles({
                self.arm.SERVO_BASE: approach_angles['base'],
                self.arm.SERVO_SHOULDER: approach_angles['shoulder'],
                self.arm.SERVO_ELBOW: approach_angles['elbow'],
                self.arm.SERVO_WRIST: approach_angles['wrist'],
                self.arm.SERVO_WRIST_ROT: approach_angles['wrist_rot']
            }, 1500)
            
            if not success:
                raise Exception("Failed to move to approach position")
            
            time.sleep(0.5)
            
            # 3. Move down to object
            print("\n   [PICK 3] Descending to object...")
            success = self.arm.set_multiple_angles({
                self.arm.SERVO_BASE: servo_angles['base'],
                self.arm.SERVO_SHOULDER: servo_angles['shoulder'],
                self.arm.SERVO_ELBOW: servo_angles['elbow'],
                self.arm.SERVO_WRIST: servo_angles['wrist']
            }, 1000)
            
            if not success:
                raise Exception("Failed to descend to object")
            
            time.sleep(0.3)
            
            # 4. Close gripper safely
            print("\n   [PICK 4] Closing gripper (SAFE MODE)...")
            if not self.arm.safe_close_gripper():
                raise Exception("Failed to close gripper safely")
            
            time.sleep(0.5)
            
            # 5. Lift object
            print("\n   [PICK 5] Lifting object...")
            success = self.arm.set_multiple_angles({
                self.arm.SERVO_SHOULDER: servo_angles['shoulder'] - 10,
                self.arm.SERVO_ELBOW: servo_angles['elbow'] + 15
            }, 1000)
            
            if not success:
                raise Exception("Failed to lift object")
            
            time.sleep(0.5)
            
            # ----- TRANSPORT SEQUENCE -----
            
            # 6. Move to drop approach
            print("\n   [DROP 1] Moving to drop zone...")
            drop_approach_angles = drop_angles.copy()
            drop_approach_angles['elbow'] += 10
            
            success = self.arm.set_multiple_angles({
                self.arm.SERVO_BASE: drop_approach_angles['base'],
                self.arm.SERVO_SHOULDER: drop_approach_angles['shoulder'],
                self.arm.SERVO_ELBOW: drop_approach_angles['elbow'],
                self.arm.SERVO_WRIST: drop_approach_angles['wrist']
            }, 1500)
            
            if not success:
                raise Exception("Failed to move to drop zone")
            
            time.sleep(0.5)
            
            # 7. Lower to drop height
            print("\n   [DROP 2] Lowering to drop height...")
            success = self.arm.set_multiple_angles({
                self.arm.SERVO_SHOULDER: drop_angles['shoulder'],
                self.arm.SERVO_ELBOW: drop_angles['elbow']
            }, 1000)
            
            if not success:
                raise Exception("Failed to lower to drop height")
            
            time.sleep(0.3)
            
            # 8. Open gripper to release
            print("\n   [DROP 3] Releasing object...")
            if not self.arm.open_gripper():
                raise Exception("Failed to open gripper for release")
            
            time.sleep(0.5)
            
            # 9. Lift away
            print("\n   [DROP 4] Moving away from drop zone...")
            success = self.arm.set_multiple_angles({
                self.arm.SERVO_SHOULDER: drop_angles['shoulder'] - 15,
                self.arm.SERVO_ELBOW: drop_angles['elbow'] + 20
            }, 1000)
            
            return True
            
        except Exception as e:
            print(f"    ❌ Pickup failed: {e}")
            self.arm.open_gripper()
            return False
    
    def print_status(self):
        """Print current system status"""
        print("\n" + "=" * 60)
        print("📊 SYSTEM STATUS")
        print("=" * 60)
        
        print("\n🎯 TARGET COUNTS:")
        for class_id, class_name in self.class_names.items():
            target = self.target_counts[class_id]
            displaced = self.displaced_counts[class_id]
            percentage = (displaced / target * 100) if target > 0 else 0
            
            status_bar = "█" * int(percentage / 10) + "░" * (10 - int(percentage / 10))
            
            if displaced >= target:
                status = f"✅ {status_bar} {displaced}/{target}"
            else:
                status = f"🔄 {status_bar} {displaced}/{target}"
            
            print(f"  {class_name:15} {status}")
        
        print(f"\n🔍 Search Pattern: {self.search_pattern}")
        print(f"🤖 Arm Connected: {'✅ Yes' if self.arm.connected else '❌ No (Simulation)'}")
        print(f"🔄 Working: {'Yes' if self.is_working else 'No'}")
        print(f"⏱️  Search Interval: {self.search_interval}s")
        
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
        
        # Initialize automatic search manager
        self.search_mgr = AutomaticSearchManager()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_search_time = time.time()
        
        # Detection parameters
        self.detection_interval = 3
        self.frame_index = 0
        
        # Display settings
        self.show_fps = True
        self.show_status = True
        self.show_detections = True
        
        # Control flags
        self.paused = False
        self.emergency_stop = False
        self.auto_search_enabled = True
        self.search_thread = None
        
        # FPS calculation
        self.fps_buffer = deque(maxlen=30)
        
        print("\n✅ System initialized and ready!")
        print("Controls:")
        print("  s - Toggle status display")
        print("  d - Toggle detections display")
        print("  p - Pause/resume system")
        print("  a - Toggle auto-search")
        print("  m - Manual search cycle")
        print("  e - Emergency stop")
        print("  n - Go to neutral position")
        print("  c - Print servo status")
        print("  q - Quit")
        print("=" * 70)
    
    def setup_camera(self):
        """Setup camera for maximum FPS"""
        print("📷 Setting up camera...")
        
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Found camera at index {i}")
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                cap.set(cv2.CAP_PROP_EXPOSURE, -5)
                
                return cap
        
        print("⚠️ No camera found, using simulated feed")
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
                    model.overrides['agnostic_nms'] = True
                    model.overrides['max_det'] = 10
                    model.overrides['verbose'] = False
                    
                    print(f"✅ YOLO model loaded from {path}")
                    return model
                    
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
        
        print("❌ No YOLO model found, running in camera-only mode")
        return None
    
    def process_frame(self, frame):
        """Process frame for object detection"""
        if self.model is None or self.frame_index % self.detection_interval != 0:
            return []
        
        try:
            results = self.model(frame, 
                               conf=0.35,
                               iou=0.3,
                               max_det=10,
                               verbose=False,
                               device='cpu')
            
            detections = []
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    x1 = max(0, min(x1, frame.shape[1]-1))
                    y1 = max(0, min(y1, frame.shape[0]-1))
                    x2 = max(0, min(x2, frame.shape[1]-1))
                    y2 = max(0, min(y2, frame.shape[0]-1))
                    
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    
                    if class_id not in self.search_mgr.class_names:
                        continue
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confidence),
                        'confidence_percent': float(confidence * 100),
                        'class_id': int(class_id),
                        'class_name': self.search_mgr.class_names[class_id],
                        'timestamp': time.time()
                    }
                    
                    detections.append(detection)
                    self.search_mgr.add_detection(detection)
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence_percent']
            
            color = self.search_mgr.tool_colors[class_id % len(self.search_mgr.tool_colors)]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']}: {confidence:.1f}%"
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
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
        
        return frame
    
    def draw_status(self, frame):
        """Draw system status on frame"""
        if not self.show_status:
            return frame
        
        panel_height = 200
        cv2.rectangle(frame, (5, 5), (400, panel_height), (0, 0, 0, 180), -1)
        cv2.rectangle(frame, (5, 5), (400, panel_height), (255, 255, 255), 1)
        
        y_offset = 30
        line_height = 22
        
        status_color = (0, 255, 0) if not self.emergency_stop else (0, 0, 255)
        status_text = "RUNNING" if not self.paused else "PAUSED"
        if self.emergency_stop:
            status_text = "EMERGENCY STOP"
        
        cv2.putText(frame, f"🤖 STATUS: {status_text}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y_offset += line_height
        
        cv2.putText(frame, f"📊 FPS: {self.fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        auto_color = (0, 255, 0) if self.auto_search_enabled else (100, 100, 100)
        auto_text = "ENABLED" if self.auto_search_enabled else "DISABLED"
        cv2.putText(frame, f"🔍 Auto-Search: {auto_text}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, auto_color, 1)
        y_offset += line_height
        
        arm_status = "CONNECTED" if self.search_mgr.arm.connected else "SIMULATION"
        arm_color = (0, 255, 0) if self.search_mgr.arm.connected else (255, 255, 0)
        cv2.putText(frame, f"🦾 Arm: {arm_status}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, arm_color, 1)
        y_offset += line_height
        
        completed = sum(1 for cid in self.search_mgr.class_names.keys()
                       if self.search_mgr.displaced_counts[cid] >= self.search_mgr.target_counts[cid])
        total = len(self.search_mgr.class_names)
        
        cv2.putText(frame, f"📦 Progress: {completed}/{total} tools", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        y_offset += 10
        cv2.putText(frame, "Controls:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += 15
        cv2.putText(frame, "s:Status d:Dets p:Pause a:Auto m:Manual", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += 15
        cv2.putText(frame, "e:Emergency n:Neutral c:ServoStatus q:Quit", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def run_auto_search(self):
        """Run automatic search cycle in background thread"""
        if not self.auto_search_enabled or self.paused or self.emergency_stop:
            return
        
        current_time = time.time()
        if current_time - self.last_search_time >= self.search_mgr.search_interval:
            if not self.search_mgr.is_working:
                print("\n🔄 Auto-search triggered")
                self.last_search_time = current_time
                
                self.search_thread = threading.Thread(
                    target=self.search_mgr.execute_search_cycle,
                    daemon=True
                )
                self.search_thread.start()
    
    def run(self):
        """Main system loop"""
        print("\n🚀 Starting main loop...")
        print("   Press 'q' to quit")
        
        while True:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Camera error, using blank frame")
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
            
            if not self.paused and not self.emergency_stop:
                detections = self.process_frame(frame)
                
                if self.auto_search_enabled:
                    self.run_auto_search()
            
            else:
                detections = []
            
            if self.show_detections:
                frame = self.draw_detections(frame, detections)
            
            if self.show_status:
                frame = self.draw_status(frame)
            
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.fps_buffer.append(self.fps)
                self.frame_count = 0
                self.start_time = current_time
                
                if self.fps_buffer:
                    self.fps = np.mean(self.fps_buffer)
            
            if self.show_fps:
                cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                           (frame.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Robotic Arm Control System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Quitting...")
                break
            elif key == ord('p'):
                self.paused = not self.paused
                print(f"\n⏸️  System {'PAUSED' if self.paused else 'RESUMED'}")
            elif key == ord('s'):
                self.show_status = not self.show_status
                print(f"\n📊 Status display {'ON' if self.show_status else 'OFF'}")
            elif key == ord('d'):
                self.show_detections = not self.show_detections
                print(f"\n🔍 Detection display {'ON' if self.show_detections else 'OFF'}")
            elif key == ord('a'):
                self.auto_search_enabled = not self.auto_search_enabled
                print(f"\n🔍 Auto-search {'ENABLED' if self.auto_search_enabled else 'DISABLED'}")
            elif key == ord('m'):
                print("\n🔍 Manual search triggered")
                if not self.search_mgr.is_working:
                    threading.Thread(
                        target=self.search_mgr.execute_search_cycle,
                        daemon=True
                    ).start()
            elif key == ord('e'):
                self.emergency_stop = not self.emergency_stop
                if self.emergency_stop:
                    print("\n🚨 EMERGENCY STOP ACTIVATED")
                    self.search_mgr.arm.open_gripper()
                else:
                    print("\n✅ Emergency stop deactivated")
            elif key == ord('n'):
                print("\n🏠 Moving to neutral position...")
                self.search_mgr.arm.go_to_neutral()
            elif key == ord('c'):
                print("\n📊 Current servo status:")
                self.search_mgr.arm.print_servo_status()
            
            self.frame_index += 1
        
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("FINAL STATUS:")
        self.search_mgr.print_status()
        print("=" * 70)

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    # Install required packages
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️ Installing ultralytics for YOLO...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO
    
    # Create and run system
    system = RoboticVisionSystem()
    system.run()