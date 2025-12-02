"""
COMPLETE ROBOTIC ARM CONTROL SYSTEM WITH YOLO11 AND DIRECT SERIAL CONTROL
Works with Yahboom DOFBOT/other arms via pyserial
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

# Try to import serial for direct arm control
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
    print("✅ PySerial available for direct arm control")
except ImportError:
    print("⚠️ PySerial not found, installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyserial"])
        import serial
        import serial.tools.list_ports
        SERIAL_AVAILABLE = True
        print("✅ PySerial installed successfully")
    except:
        print("❌ Failed to install PySerial")
        SERIAL_AVAILABLE = False

print("🤖 YAHBOOM ROBOTIC ARM CONTROL SYSTEM WITH AUTONOMOUS SEARCH")
print("=" * 70)

# ============================================
# DIRECT SERIAL ARM CONTROL (Works with Yahboom)
# ============================================
class YahboomSerialArm:
    def __init__(self, port=None, baudrate=115200):
        """Initialize direct serial connection to Yahboom arm"""
        self.connected = False
        self.ser = None
        self.current_angles = [90, 90, 90, 90, 90, 180]  # Initial servo angles
        self.gripper_open = True
        
        # Servo mapping (standard for Yahboom DOFBOT)
        self.SERVO_BASE = 1      # Servo 1: Base rotation
        self.SERVO_SHOULDER = 2  # Servo 2: Shoulder
        self.SERVO_ELBOW = 3     # Servo 3: Elbow
        self.SERVO_WRIST = 4     # Servo 4: Wrist
        self.SERVO_GRIPPER = 5   # Servo 5: Gripper
        self.SERVO_WRIST_ROT = 6 # Servo 6: Wrist rotation
        
        # Gripper positions (calibrate these for your arm)
        self.GRIPPER_OPEN = 70   # Angle for open gripper
        self.GRIPPER_CLOSED = 30 # Angle for closed gripper
        
        # Movement parameters
        self.move_speed = 1000   # Movement speed (ms)
        
        # Search pattern parameters
        self.search_positions = [
            (0.0, 0.3, 0.1),    # Center
            (-0.15, 0.25, 0.1), # Left
            (0.15, 0.25, 0.1),  # Right
            (-0.1, 0.35, 0.1),  # Left-Far
            (0.1, 0.35, 0.1),   # Right-Far
        ]
        
        self.initialize_serial(port, baudrate)
    
    def initialize_serial(self, port=None, baudrate=115200):
        """Initialize serial connection to arm"""
        try:
            if not SERIAL_AVAILABLE:
                print("⚠️ Running in simulation mode - PySerial not available")
                self.connected = False
                return
            
            # Find available serial ports
            ports = list(serial.tools.list_ports.comports())
            if not ports:
                print("❌ No serial ports found!")
                self.connected = False
                return
            
            print("🔌 Available serial ports:")
            for p in ports:
                print(f"  - {p.device}: {p.description}")
            
            # Try to connect
            if port:
                target_port = port
            else:
                # Try common Yahboom port patterns
                for p in ports:
                    if 'USB' in p.description or 'ACM' in p.device or 'ttyUSB' in p.device:
                        target_port = p.device
                        break
                else:
                    target_port = ports[0].device
            
            print(f"🔗 Attempting connection to {target_port} at {baudrate} baud...")
            
            # Open serial connection
            self.ser = serial.Serial(
                port=target_port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            
            time.sleep(2)  # Wait for connection to stabilize
            
            # Test connection
            self.ser.write(b'#1 P1500 T1000\r\n')
            time.sleep(1)
            
            self.connected = True
            print(f"✅ Connected to Yahboom arm on {target_port}")
            
            # Initialize arm to neutral position
            self.go_to_neutral()
            
        except Exception as e:
            print(f"❌ Failed to connect to arm: {e}")
            self.connected = False
    
    def send_serial_command(self, command):
        """Send command to serial port"""
        if not self.connected or self.ser is None:
            return False
        
        try:
            # Ensure command ends with carriage return
            if not command.endswith('\r\n'):
                command += '\r\n'
            
            self.ser.write(command.encode())
            time.sleep(0.01)  # Small delay for command processing
            return True
            
        except Exception as e:
            print(f"    [SERIAL ERROR] {e}")
            self.connected = False
            return False
    
    def set_servo_angle(self, servo_id, angle, move_time=None):
        """Set single servo angle via serial command"""
        if move_time is None:
            move_time = self.move_speed
        
        # Validate angle range
        angle = max(0, min(180, int(angle)))
        
        # Convert angle to pulse width (500-2500 microseconds)
        # Yahboom uses 500-2500µs for 0-180 degrees
        pulse_width = 500 + (angle * 2000 / 180)
        
        # Format: #<servo> P<pulse> T<time>\r\n
        command = f"#{servo_id} P{int(pulse_width)} T{move_time}"
        
        print(f"    [SERVO {servo_id}] {angle}° (Pulse: {pulse_width}µs)")
        
        if self.send_serial_command(command):
            self.current_angles[servo_id-1] = angle
            time.sleep(move_time / 1000)  # Wait for movement to complete
            return True
        else:
            print(f"    [ERROR] Failed to move servo {servo_id}")
            return False
    
    def set_multiple_angles(self, angles_dict, move_time=None):
        """Set multiple servo angles simultaneously"""
        if move_time is None:
            move_time = self.move_speed
        
        # Build combined command for all servos
        command_parts = []
        for servo_id, angle in angles_dict.items():
            angle = max(0, min(180, int(angle)))
            pulse_width = 500 + (angle * 2000 / 180)
            command_parts.append(f"#{servo_id} P{int(pulse_width)}")
        
        # Add time parameter
        command = " ".join(command_parts) + f" T{move_time}"
        
        print(f"    [MULTI SERVO] Command: {command}")
        
        if self.send_serial_command(command):
            for servo_id, angle in angles_dict.items():
                self.current_angles[servo_id-1] = angle
            time.sleep(move_time / 1000)
            return True
        else:
            return False
    
    def go_to_neutral(self):
        """Move arm to neutral position"""
        print("    [INIT] Moving to neutral position...")
        
        # Neutral position angles
        neutral_angles = {
            self.SERVO_BASE: 90,
            self.SERVO_SHOULDER: 90,
            self.SERVO_ELBOW: 90,
            self.SERVO_WRIST: 90,
            self.SERVO_GRIPPER: self.GRIPPER_OPEN,
            self.SERVO_WRIST_ROT: 180
        }
        
        if self.connected:
            success = self.set_multiple_angles(neutral_angles, 1500)
        else:
            # Simulation
            print("    [SIM] Moving to neutral")
            time.sleep(1)
            success = True
        
        if success:
            self.current_angles = [90, 90, 90, 90, self.GRIPPER_OPEN, 180]
            self.gripper_open = True
            print("    [INIT] Neutral position reached")
        
        return success
    
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
    
    def calculate_ik(self, x, y, z):
        """
        Calculate inverse kinematics for 4-DOF arm
        Returns servo angles for base, shoulder, elbow, wrist
        """
        # Simplified IK for demonstration
        # For real implementation, use proper IK for your arm
        
        # Base rotation (around Z axis)
        base_angle = 90 + int(math.degrees(math.atan2(x, y)))
        
        # Distance from base
        distance = math.sqrt(x**2 + y**2)
        
        # Height relative to base
        height = z
        
        # Simple triangle approximation (adjust for your arm lengths!)
        # Arm lengths (calibrate these!)
        L1 = 0.1  # Shoulder to elbow length
        L2 = 0.1  # Elbow to wrist length
        
        # Calculate shoulder and elbow angles
        try:
            # Distance in XY plane
            xy_distance = math.sqrt(distance**2 + height**2)
            
            # Law of cosines
            cos_shoulder = (L1**2 + xy_distance**2 - L2**2) / (2 * L1 * xy_distance)
            cos_shoulder = max(-1, min(1, cos_shoulder))
            shoulder_angle = math.degrees(math.acos(cos_shoulder))
            
            cos_elbow = (L1**2 + L2**2 - xy_distance**2) / (2 * L1 * L2)
            cos_elbow = max(-1, min(1, cos_elbow))
            elbow_angle = 180 - math.degrees(math.acos(cos_elbow))
            
            # Wrist angle to keep end effector level
            wrist_angle = 90 - (shoulder_angle + elbow_angle - 90)
            
            # Convert to servo angles (90 is neutral)
            shoulder_servo = 90 + shoulder_angle
            elbow_servo = 90 - elbow_angle
            wrist_servo = 90 + wrist_angle
            
            # Limit angles
            base_angle = max(30, min(150, base_angle))
            shoulder_servo = max(40, min(140, shoulder_servo))
            elbow_servo = max(40, min(140, elbow_servo))
            wrist_servo = max(60, min(120, wrist_servo))
            
            return {
                'base': int(base_angle),
                'shoulder': int(shoulder_servo),
                'elbow': int(elbow_servo),
                'wrist': int(wrist_servo)
            }
            
        except:
            # Fallback to simple mapping if IK fails
            print("    [WARNING] Using fallback IK")
            return {
                'base': 90 + int(x * 60),
                'shoulder': 90 - int(y * 40),
                'elbow': 90 + int(z * 30),
                'wrist': 90
            }
    
    def move_to_xyz(self, x, y, z):
        """
        Move end effector to x,y,z coordinates
        x: left/right (meters), y: forward/backward, z: up/down
        """
        print(f"    [MOVE] Moving to ({x:.3f}, {y:.3f}, {z:.3f})...")
        
        # Calculate inverse kinematics
        ik = self.calculate_ik(x, y, z)
        
        # Move servos
        angles = {
            self.SERVO_BASE: ik['base'],
            self.SERVO_SHOULDER: ik['shoulder'],
            self.SERVO_ELBOW: ik['elbow'],
            self.SERVO_WRIST: ik['wrist']
        }
        
        success = self.set_multiple_angles(angles, 1500)
        
        if success:
            print(f"    [MOVE] Arrived at ({x:.3f}, {y:.3f}, {z:.3f})")
            return True
        else:
            print(f"    [MOVE] Failed to reach position")
            return False
    
    def lift(self, height):
        """Lift the arm to specified height"""
        print(f"    [LIFT] Lifting to height {height:.3f}m...")
        
        # Get current position from angles
        current_angles = self.current_angles
        
        # Simple lift by adjusting shoulder and elbow
        lift_angles = {
            self.SERVO_SHOULDER: max(40, current_angles[self.SERVO_SHOULDER-1] - 20),
            self.SERVO_ELBOW: max(40, current_angles[self.SERVO_ELBOW-1] + 10)
        }
        
        success = self.set_multiple_angles(lift_angles, 1000)
        
        if success:
            print(f"    [LIFT] Lifted to {height:.3f}m")
            return True
        return False
    
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
    
    def execute_search_pattern(self, callback=None):
        """
        Execute search pattern to find objects
        Moves camera to different positions to scan workspace
        """
        print("\n🔍 Starting autonomous search pattern...")
        
        for i, pos in enumerate(self.search_positions):
            if callback:
                callback(f"Search position {i+1}/{len(self.search_positions)}")
            
            print(f"    [SEARCH] Position {i+1}: {pos}")
            
            # Move to search position (higher Z for overview)
            search_height = (pos[0], pos[1], pos[2] + 0.05)
            self.move_to_xyz(*search_height)
            
            # Wait for camera to stabilize
            time.sleep(1.5)
            
            # Give time for detection
            if callback:
                callback(f"Scanning at position {i+1}...")
            time.sleep(2.0)
        
        # Return to center
        self.move_to_xyz(0, 0.3, 0.15)
        print("🔍 Search pattern completed")
    
    def emergency_stop(self):
        """Emergency stop - relax all servos"""
        print("    [EMERGENCY] Stopping arm!")
        if self.connected:
            try:
                # Send stop command (varies by arm model)
                self.ser.write(b'\r\n')  # Sometimes just sending empty line stops
                time.sleep(0.1)
                self.ser.write(b'STOP\r\n')
            except:
                pass
        time.sleep(0.5)

# ============================================
# AUTONOMOUS SEARCH AND DISPLACEMENT MANAGER
# ============================================
class AutonomousDisplacementManager:
    def __init__(self):
        # Initialize arm with direct serial control
        self.arm = YahboomSerialArm()
        self.mapper = CoordinateMapper()
        
        # Tool classes
        self.class_names = {
            0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
            3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
        }
        
        # Displacement tracking
        self.displaced_counts = defaultdict(int)
        self.target_displace_count = 1
        self.current_target_class = None
        self.is_displacing = False
        self.is_searching = False
        self.search_in_progress = False
        
        # Detection buffer
        self.detection_buffer = deque(maxlen=20)
        self.last_detection_time = 0
        
        # Search parameters
        self.search_interval = 10  # Seconds between searches if no objects found
        self.min_confidence = 0.65  # Minimum confidence to act
        
        # Safety
        self.safe_height = 0.15
        self.grip_delay = 0.5
        
        print("🤖 Autonomous Displacement Manager Initialized")
        print(f"🔧 Target: {self.target_displace_count} of each tool")
        print(f"🔍 Auto-search enabled every {self.search_interval}s")
    
    def add_detection(self, detection):
        """Add detection to buffer and update timestamp"""
        self.detection_buffer.append(detection)
        self.last_detection_time = time.time()
    
    def get_best_detection(self):
        """Get the best detection from buffer (highest confidence, not yet displaced)"""
        if not self.detection_buffer:
            return None
        
        # Get recent detections (last 3 seconds)
        recent_time = time.time() - 3.0
        recent_dets = [d for d in self.detection_buffer if d['timestamp'] > recent_time]
        
        if not recent_dets:
            return None
        
        # Filter out already displaced classes
        available_dets = []
        for det in recent_dets:
            class_id = det['class_id']
            if self.displaced_counts[class_id] < self.target_displace_count:
                available_dets.append(det)
        
        if not available_dets:
            return None
        
        # Return detection with highest confidence
        return max(available_dets, key=lambda x: x['confidence'])
    
    def should_search(self):
        """Check if we should initiate search pattern"""
        if self.is_displacing or self.is_searching:
            return False
        
        # Search if no recent detections
        time_since_last_detection = time.time() - self.last_detection_time
        if time_since_last_detection > self.search_interval:
            return True
        
        # Check if all visible objects are already displaced
        recent_dets = [d for d in self.detection_buffer 
                      if d['timestamp'] > time.time() - 5.0]
        
        if not recent_dets:
            return True
        
        # Check if any recent detection is not yet displaced
        for det in recent_dets:
            if self.displaced_counts[det['class_id']] < self.target_displace_count:
                return False
        
        return True
    
    def execute_search(self, status_callback=None):
        """Execute autonomous search pattern"""
        if self.is_searching or self.is_displacing:
            return False
        
        self.is_searching = True
        self.search_in_progress = True
        
        try:
            if status_callback:
                status_callback("Starting autonomous search...")
            
            # Clear old detections
            self.detection_buffer.clear()
            
            # Execute search pattern
            self.arm.execute_search_pattern(status_callback)
            
            # Return to neutral
            self.arm.go_to_neutral()
            
            if status_callback:
                status_callback("Search completed")
            
            return True
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return False
            
        finally:
            self.is_searching = False
            time.sleep(1)  # Brief pause after search
            self.search_in_progress = False
    
    def execute_displacement(self, detection):
        """Execute complete displacement sequence"""
        if self.is_displacing:
            print("    ⚠️ Already displacing")
            return False
        
        # Check confidence threshold
        if detection['confidence'] < self.min_confidence:
            print(f"    ⚠️ Confidence too low: {detection['confidence']:.2f}")
            return False
        
        self.is_displacing = True
        class_id = detection['class_id']
        class_name = self.class_names[class_id]
        
        print(f"\n{'='*60}")
        print(f"🚀 DISPLACING: {class_name}")
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
            
            print(f"   📍 Object: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})m")
            print(f"   🎯 Drop: ({drop_pos[0]:.3f}, {drop_pos[1]:.3f}, {drop_pos[2]:.3f})m")
            
            # 2. Open gripper
            print("\n   [1/12] Opening gripper...")
            self.arm.open_gripper()
            
            # 3. Move to safe approach
            print("\n   [2/12] Moving to approach...")
            self.arm.move_to_xyz(target_pos[0], target_pos[1], self.safe_height)
            
            # 4. Move down to grasp height
            print("\n   [3/12] Descending...")
            grasp_height = max(0.02, target_pos[2] + 0.015)  # Slightly above surface
            self.arm.move_to_xyz(target_pos[0], target_pos[1], grasp_height)
            
            # 5. Close gripper
            print("\n   [4/12] Grasping...")
            self.arm.close_gripper()
            time.sleep(self.grip_delay)
            
            # 6. Lift object
            print("\n   [5/12] Lifting...")
            self.arm.lift(self.safe_height)
            
            # 7. Rotate 180 degrees
            print("\n   [6/12] Rotating...")
            self.arm.rotate_wrist(180)
            time.sleep(0.5)
            
            # 8. Move to drop zone
            print("\n   [7/12] Moving to drop zone...")
            self.arm.move_to_xyz(drop_pos[0], drop_pos[1], self.safe_height)
            
            # 9. Lower to drop height
            print("\n   [8/12] Lowering...")
            self.arm.move_to_xyz(drop_pos[0], drop_pos[1], drop_pos[2])
            
            # 10. Release object
            print("\n   [9/12] Releasing...")
            self.arm.open_gripper()
            time.sleep(0.5)
            
            # 11. Lift back up
            print("\n   [10/12] Lifting...")
            self.arm.lift(self.safe_height)
            
            # 12. Return to neutral
            print("\n   [11/12] Returning orientation...")
            self.arm.rotate_wrist(-180)
            
            print("\n   [12/12] Back to neutral...")
            self.arm.go_to_neutral()
            
            # Update counts
            self.displaced_counts[class_id] += 1
            print(f"\n   ✅ {class_name} displaced successfully!")
            print(f"   📊 Total: {self.displaced_counts[class_id]}/{self.target_displace_count}")
            
            # Clear buffer for this class
            self.detection_buffer = deque(
                [d for d in self.detection_buffer if d['class_id'] != class_id],
                maxlen=20
            )
            
            return True
            
        except Exception as e:
            print(f"\n   ❌ Displacement failed: {e}")
            print("   🚨 Emergency recovery...")
            self.arm.open_gripper()
            self.arm.go_to_neutral()
            return False
            
        finally:
            self.is_displacing = False
    
    def autonomous_operation(self, get_detections_callback, status_callback=None):
        """
        Main autonomous operation loop
        Runs in separate thread
        """
        print("\n🤖 Starting autonomous operation mode...")
        
        # Start in neutral
        self.arm.go_to_neutral()
        time.sleep(1)
        
        operation_start = time.time()
        last_search_time = time.time()
        
        while True:
            # Check if all tasks complete
            if self.check_completion():
                if status_callback:
                    status_callback("🎉 All tasks completed!")
                print("\n" + "="*60)
                print("🎉 ALL DISPLACEMENT TASKS COMPLETED!")
                print("="*60)
                self.print_status()
                break
            
            # Check if we should search
            if self.should_search() and not self.is_displacing:
                current_time = time.time()
                if current_time - last_search_time > self.search_interval:
                    if status_callback:
                        status_callback("🔍 Searching for objects...")
                    
                    self.execute_search(status_callback)
                    last_search_time = current_time
                    
                    # Wait a bit after search
                    time.sleep(2)
            
            # Get current detections from callback
            current_detections = get_detections_callback()
            if current_detections:
                for det in current_detections:
                    self.add_detection(det)
            
            # Try to displace if not already doing so
            if not self.is_displacing and not self.is_searching:
                best_detection = self.get_best_detection()
                if best_detection:
                    if status_callback:
                        status_callback(f"Found {best_detection['class_name']}...")
                    
                    # Execute displacement
                    success = self.execute_displacement(best_detection)
                    
                    if success:
                        # Brief pause after successful displacement
                        time.sleep(1)
                    else:
                        # Longer pause after failure
                        time.sleep(2)
            
            # Small delay to prevent CPU overload
            time.sleep(0.1)
    
    def check_completion(self):
        """Check if all displacement tasks are complete"""
        for class_id in self.class_names.keys():
            if self.displaced_counts[class_id] < self.target_displace_count:
                return False
        return True
    
    def print_status(self):
        """Print current status"""
        print("\n" + "="*60)
        print("📊 AUTONOMOUS OPERATION STATUS")
        print("="*60)
        for class_id, class_name in self.class_names.items():
            count = self.displaced_counts[class_id]
            status = "✅ DONE" if count >= self.target_displace_count else f"🔄 {count}/{self.target_displace_count}"
            print(f"  {class_name}: {status}")
        
        print(f"\n🤖 Arm connected: {'✅' if self.arm.connected else '❌'}")
        print(f"🔍 Searching: {'✅ Active' if self.is_searching else '❌ Inactive'}")
        print(f"🎯 Current target: {self.class_names[self.current_target_class] if self.current_target_class else 'None'}")
        print("="*60)

# ============================================
# COORDINATE MAPPING SYSTEM
# ============================================
class CoordinateMapper:
    def __init__(self, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Calibrate these for your setup!
        self.workspace_x_range = (-0.2, 0.2)    # Left/Right
        self.workspace_y_range = (0.2, 0.4)     # Forward/Back
        self.workspace_z_range = (0.0, 0.1)     # Up/Down
        
        # Drop zones for each tool
        self.drop_zones = {
            0: (0.15, 0.35, 0.03),    # Bolt
            1: (0.18, 0.35, 0.03),    # Hammer
            2: (0.15, 0.32, 0.03),    # Measuring Tape
            3: (0.18, 0.32, 0.03),    # Plier
            4: (0.15, 0.29, 0.03),    # Screwdriver
            5: (0.18, 0.29, 0.03)     # Wrench
        }
        
        print(f"📍 Workspace: X{self.workspace_x_range}, Y{self.workspace_y_range}")
    
    def pixel_to_robot_coords(self, pixel_x, pixel_y, bbox_width, bbox_height):
        """Convert pixels to robot coordinates"""
        # Normalize
        norm_x = pixel_x / self.camera_width
        norm_y = 1.0 - (pixel_y / self.camera_height)  # Flip Y
        
        # Map to workspace
        robot_x = self.workspace_x_range[0] + norm_x * (self.workspace_x_range[1] - self.workspace_x_range[0])
        robot_y = self.workspace_y_range[0] + norm_y * (self.workspace_y_range[1] - self.workspace_y_range[0])
        
        # Estimate Z from object size
        object_area = bbox_width * bbox_height
        max_area = self.camera_width * self.camera_height * 0.15
        area_ratio = min(1.0, object_area / max_area)
        robot_z = self.workspace_z_range[0] + (1 - area_ratio) * (self.workspace_z_range[1] - self.workspace_z_range[0])
        
        return (robot_x, robot_y, robot_z)
    
    def get_approach_height(self, target_z):
        return target_z + 0.07
    
    def get_drop_location(self, class_id):
        return self.drop_zones.get(class_id, (0.17, 0.3, 0.03))

# ============================================
# MAIN VISION SYSTEM WITH AUTONOMOUS CONTROL
# ============================================
class AutonomousVisionSystem:
    def __init__(self):
        # Setup camera
        self.cap = self.setup_camera()
        if self.cap is None:
            print("❌ ERROR: No camera found!")
            exit()
        
        # Load YOLO
        self.model = self.load_yolo_model()
        
        # Initialize autonomous manager
        self.manager = AutonomousDisplacementManager()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Detection
        self.detection_interval = 3
        self.frame_index = 0
        self.current_detections = []
        
        # Colors
        self.tool_colors = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        # UI
        self.show_status = True
        self.autonomous_mode = False
        self.auto_thread = None
        
        print("\n✅ System ready for autonomous operation!")
        print("Controls:")
        print("  a - Start autonomous mode")
        print("  s - Stop autonomous mode")
        print("  m - Manual displacement of current detection")
        print("  r - Reset arm to neutral")
        print("  p - Print status")
        print("  q - Quit")
        print("="*70)
    
    def setup_camera(self):
        """Setup camera"""
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"✅ Camera found at index {i}")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
        return None
    
    def load_yolo_model(self):
        """Load YOLO"""
        try:
            model = YOLO('best.pt')
            model.overrides['conf'] = 0.4
            model.overrides['iou'] = 0.3
            model.overrides['agnostic_nms'] = True
            model.overrides['max_det'] = 6
            model.overrides['verbose'] = False
            print("✅ YOLO loaded")
            return model
        except:
            print("❌ YOLO failed, using simulation")
            return None
    
    def get_current_detections(self):
        """Get current detections for autonomous thread"""
        return self.current_detections.copy()
    
    def update_status_callback(self, message):
        """Callback for status updates from autonomous thread"""
        print(f"🤖 {message}")
    
    def start_autonomous_mode(self):
        """Start autonomous operation"""
        if self.autonomous_mode:
            print("⚠️ Already in autonomous mode")
            return
        
        if self.auto_thread and self.auto_thread.is_alive():
            print("⚠️ Autonomous thread already running")
            return
        
        print("\n🤖 STARTING AUTONOMOUS MODE...")
        self.autonomous_mode = True
        
        # Start autonomous thread
        self.auto_thread = threading.Thread(
            target=self.manager.autonomous_operation,
            args=(self.get_current_detections, self.update_status_callback),
            daemon=True
        )
        self.auto_thread.start()
        
        print("✅ Autonomous mode started")
    
    def stop_autonomous_mode(self):
        """Stop autonomous operation"""
        if not self.autonomous_mode:
            print("⚠️ Not in autonomous mode")
            return
        
        print("\n🤖 STOPPING AUTONOMOUS MODE...")
        self.autonomous_mode = False
        
        # Wait for thread to finish
        if self.auto_thread and self.auto_thread.is_alive():
            self.auto_thread.join(timeout=2.0)
        
        # Return arm to neutral
        self.manager.arm.go_to_neutral()
        print("✅ Autonomous mode stopped")
    
    def process_detection(self, frame):
        """Process frame for detection"""
        if self.model is None or self.frame_index % self.detection_interval != 0:
            return self.current_detections
        
        try:
            # Resize for speed
            inference_size = 256
            small_frame = cv2.resize(frame, (inference_size, int(inference_size * 0.75)))
            
            # Inference
            results = self.model(small_frame, 
                               conf=0.4,
                               iou=0.3,
                               imgsz=inference_size,
                               max_det=6,
                               verbose=False,
                               device='cpu')
            
            # Process results
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Scale
                scale_x = frame.shape[1] / inference_size
                scale_y = frame.shape[0] / (inference_size * 0.75)
                
                detections = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
                    
                    # Clamp
                    x1 = max(0, min(x1, frame.shape[1]-1))
                    y1 = max(0, min(y1, frame.shape[0]-1))
                    x2 = max(0, min(x2, frame.shape[1]-1))
                    y2 = max(0, min(y2, frame.shape[0]-1))
                    
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    confidence_percent = confidence * 100
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confidence),
                        'confidence_percent': float(confidence_percent),
                        'class_id': int(class_id),
                        'class_name': self.manager.class_names[class_id],
                        'timestamp': time.time()
                    }
                    
                    detections.append(detection)
                
                self.current_detections = detections
                return detections
                
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
        
        return []
    
    def draw_ui(self, frame, detections):
        """Draw UI overlay"""
        if not self.show_status:
            return frame
        
        # Background
        cv2.rectangle(frame, (5, 5), (400, 180), (0, 0, 0, 180), -1)
        cv2.rectangle(frame, (5, 5), (400, 180), (255, 255, 255), 1)
        
        y = 30
        line = 22
        
        # Mode
        mode_color = (0, 255, 0) if self.autonomous_mode else (0, 165, 255)
        mode_text = "AUTONOMOUS" if self.autonomous_mode else "MANUAL"
        cv2.putText(frame, f"🤖 MODE: {mode_text}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        y += line
        
        # FPS
        cv2.putText(frame, f"📊 FPS: {self.fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += line
        
        # Arm status
        arm_status = "CONNECTED" if self.manager.arm.connected else "DISCONNECTED"
        arm_color = (0, 255, 0) if self.manager.arm.connected else (0, 0, 255)
        cv2.putText(frame, f"🤖 ARM: {arm_status}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, arm_color, 1)
        y += line
        
        # Current action
        if self.manager.is_displacing:
            action = "DISPLACING"
            color = (0, 165, 255)
        elif self.manager.is_searching:
            action = "SEARCHING"
            color = (255, 255, 0)
        else:
            action = "READY"
            color = (0, 255, 0)
        
        cv2.putText(frame, f"⚡ STATUS: {action}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += line
        
        # Detections
        cv2.putText(frame, f"🔍 DETECTIONS: {len(detections)}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += line
        
        # Progress
        cv2.putText(frame, "📈 PROGRESS:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        y += line
        
        # Tool progress (first row)
        for i in range(3):
            class_id = i
            count = self.manager.displaced_counts[class_id]
            total = self.manager.target_displace_count
            name = self.manager.class_names[class_id][:3]
            color = (0, 255, 0) if count >= total else (255, 255, 255)
            cv2.putText(frame, f"{name}:{count}/{total}", 
                       (15 + i*65, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += line
        
        # Second row
        for i in range(3, 6):
            class_id = i
            count = self.manager.displaced_counts[class_id]
            total = self.manager.target_displace_count
            name = self.manager.class_names[class_id][:3]
            color = (0, 255, 0) if count >= total else (255, 255, 255)
            cv2.putText(frame, f"{name}:{count}/{total}", 
                       (15 + (i-3)*65, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def draw_detections(self, frame, detections):
        """Draw detections"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            conf = det['confidence_percent']
            
            color = self.tool_colors[class_id]
            
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{det['class_name']} {conf:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            # Center dot
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(frame, (cx, cy), 4, color, -1)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("\n🎬 Starting vision system...")
        time.sleep(1)
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Can't receive frame")
                    break
                
                # Mirror
                frame = cv2.flip(frame, 1)
                
                # Update FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Process detection
                self.frame_index += 1
                detections = self.process_detection(frame)
                
                # Draw
                display = frame.copy()
                display = self.draw_detections(display, detections)
                display = self.draw_ui(display, detections)
                
                # Show
                cv2.imshow('Autonomous Robotic Arm - YOLO + Yahboom', display)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                
                elif key == ord('a'):
                    self.start_autonomous_mode()
                
                elif key == ord('s'):
                    self.stop_autonomous_mode()
                
                elif key == ord('m') and detections:
                    # Manual displacement of first detection
                    if not self.autonomous_mode and not self.manager.is_displacing:
                        best = max(detections, key=lambda x: x['confidence'])
                        print(f"\n🎯 Manual displacement: {best['class_name']}")
                        threading.Thread(
                            target=self.manager.execute_displacement,
                            args=(best,),
                            daemon=True
                        ).start()
                
                elif key == ord('r'):
                    print("\n🔄 Resetting arm to neutral...")
                    self.manager.arm.go_to_neutral()
                
                elif key == ord('p'):
                    self.manager.print_status()
                
                elif key == ord(' '):
                    self.show_status = not self.show_status
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup"""
        self.stop_autonomous_mode()
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Cleanup complete")
        self.manager.print_status()

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("🤖 YAHBOOM AUTONOMOUS ROBOTIC ARM CONTROL SYSTEM")
    print("="*70)
    print("\n⚠️  IMPORTANT: Before running:")
    print("   1. Connect Yahboom arm via USB")
    print("   2. Ensure power is ON")
    print("   3. Place objects in camera view")
    print("   4. Press 'a' to start autonomous mode")
    print("="*70)
    
    # Create and run system
    system = AutonomousVisionSystem()
    system.run()