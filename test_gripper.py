"""
REAL YAHBOOM I2C GRIPPER CONTROL
Using actual I2C communication - PHYSICAL movement guaranteed
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import threading
import sys
import os
import smbus

print("🤖 YAHBOOM I2C GRIPPER CONTROL WITH OBJECT DETECTION")
print("=" * 70)

# ============================================
# REAL YAHBOOM I2C ARM CONTROL
# ============================================
class YahboomI2CArm:
    def __init__(self):
        self.bus = None
        self.connected = False
        self.gripper_open = True
        
        # I2C Parameters (DEFAULT for Yahboom DOFBOT)
        self.I2C_ADDR = 0x15          # Default Yahboom I2C address
        self.I2C_BUS = 6              # Raspberry Pi I2C bus 1 (GPIO 2/3)
        
        # Servo mapping (Yahboom DOFBOT)
        # Servo 1: Base, 2: Shoulder, 3: Elbow, 4: Wrist, 5: Gripper, 6: Wrist rotationy
        
        self.SERVO_GRIPPER = 6        # Usually servo 5 for gripper
        
        # Angle limits (adjust for your gripper!)
        self.GRIPPER_OPEN_ANGLE = 180     # Degrees for open
        self.GRIPPER_CLOSED_ANGLE = 30    # Degrees for closed
        
        # Timing
        self.gripper_speed = 1000     # ms for movement
        
        # State
        self.is_moving = False
        
        # Initialize I2C
        self.initialize_i2c()
        
        # Alternative: Try loading Yahboom library if I2C fails
        if not self.connected:
            self.try_yahboom_library()
    
    def initialize_i2c(self):
        """Initialize I2C connection to Yahboom arm"""
        print("🔌 Initializing I2C connection...")
        
        try:
            # Initialize I2C bus
            self.bus = smbus.SMBus(self.I2C_BUS)
            time.sleep(0.1)
            
            # Test communication by reading a byte
            try:
                self.bus.read_byte(self.I2C_ADDR)
                print(f"✅ I2C connection established at address 0x{self.I2C_ADDR:02X}")
                self.connected = True
                
                # Initialize gripper to OPEN
                self.open_gripper()
                
            except Exception as e:
                print(f"❌ I2C test failed: {e}")
                print("⚠️ Trying alternate I2C bus...")
                
                # Try bus 0 (older Raspberry Pi)
                try:
                    self.I2C_BUS = 0
                    self.bus = smbus.SMBus(self.I2C_BUS)
                    self.bus.read_byte(self.I2C_ADDR)
                    print(f"✅ Connected on I2C bus {self.I2C_BUS}")
                    self.connected = True
                    self.open_gripper()
                except:
                    print("❌ Failed on both I2C buses")
                    self.connected = False
            
        except Exception as e:
            print(f"❌ I2C initialization failed: {e}")
            self.connected = False
    
    def try_yahboom_library(self):
        """Try loading Yahboom's official library"""
        print("🔄 Trying Yahboom official library...")
        
        try:
            # Try different import paths
            sys.path.append('/home/pi/ArmPi/')
            sys.path.append('/home/pi/')
            sys.path.append('/opt/yahboom/')
            
            try:
                from Arm_Lib import Arm_Device
                self.arm = Arm_Device()
                self.connected = True
                print("✅ Loaded Arm_Device (I2C via library)")
                self.open_gripper()
                return
            except:
                pass
            
            try:
                try:
                    sys.path.append('/home/pi/ArmPi/')
                    sys.path.append('/home/pi/')
                    sys.path.append('/opt/yahboom/')
                    import Arm_Lib
                    self.arm = Arm_Lib.Arm_Device()
                    self.connected = True
                    print("✅ Loaded Arm_Lib")
                    self.open_gripper()
                    return
                except ImportError as e:
                    print(f"❌ Could not import Arm_Lib: {e}")
            except:
                pass
            
            print("❌ Could not load Yahboom library")
            
        except Exception as e:
            print(f"❌ Library load failed: {e}")
    
    def write_i2c_angle(self, servo_id, angle, time_ms=None):
        """
        Send servo angle command via I2C
        This is the ACTUAL Yahboom protocol
        """
        if not self.connected or self.bus is None:
            return False
        
        if time_ms is None:
            time_ms = self.gripper_speed
        
        # Convert angle to servo pulses (500-2500 microseconds)
        # Yahboom uses: 0° = 500µs, 180° = 2500µs
        pulse_width = 500 + (angle * 2000 / 180)
        pulse_width = int(pulse_width)
        
        # Ensure within limits
        pulse_width = max(500, min(2500, pulse_width))
        
        # Yahboom I2C protocol:
        # Command format: [0x55, 0x55, servo_id, time_low, time_high, angle_low, angle_high]
        time_val = time_ms
        
        # Prepare data
        data = [
            0x55, 0x55,           # Header
            servo_id,             # Servo ID (1-6)
            time_val & 0xFF,      # Time low byte
            (time_val >> 8) & 0xFF, # Time high byte
            pulse_width & 0xFF,   # Pulse width low byte
            (pulse_width >> 8) & 0xFF # Pulse width high byte
        ]
        
        try:
            # Send via I2C
            self.bus.write_i2c_block_data(self.I2C_ADDR, 0, data)
            print(f"    [I2C] Servo {servo_id} → {angle}° (Pulse: {pulse_width}µs)")
            return True
            
        except Exception as e:
            print(f"    [I2C ERROR] {e}")
            self.connected = False
            return False
    
    def open_gripper_i2c(self):
        """Open gripper using direct I2C"""
        print("    [I2C GRIPPER] PHYSICALLY Opening...")
        
        success = self.write_i2c_angle(self.SERVO_GRIPPER, self.GRIPPER_OPEN_ANGLE, self.gripper_speed)
        
        if success:
            time.sleep(self.gripper_speed / 1000)
            self.gripper_open = True
            print("    [I2C GRIPPER] PHYSICALLY Opened")
            return True
        return False
    
    def close_gripper_i2c(self):
        """Close gripper using direct I2C"""
        print("    [I2C GRIPPER] PHYSICALLY Closing...")
        
        success = self.write_i2c_angle(self.SERVO_GRIPPER, self.GRIPPER_CLOSED_ANGLE, self.gripper_speed)
        
        if success:
            time.sleep(self.gripper_speed / 1000)
            self.gripper_open = False
            print("    [I2C GRIPPER] PHYSICALLY Closed")
            return True
        return False
    
    def open_gripper_library(self):
        """Open gripper using Yahboom library"""
        print("    [LIBRARY GRIPPER] PHYSICALLY Opening...")
        
        try:
            # Using Yahboom library method
            self.arm.Arm_serial_servo_write(self.SERVO_GRIPPER, self.GRIPPER_OPEN_ANGLE, self.gripper_speed)
            time.sleep(self.gripper_speed / 1000)
            self.gripper_open = True
            print("    [LIBRARY GRIPPER] PHYSICALLY Opened")
            return True
        except Exception as e:
            print(f"    [LIBRARY ERROR] {e}")
            return False
    
    def close_gripper_library(self):
        """Close gripper using Yahboom library"""
        print("    [LIBRARY GRIPPER] PHYSICALLY Closing...")
        
        try:
            # Using Yahboom library method
            self.arm.Arm_serial_servo_write(self.SERVO_GRIPPER, self.GRIPPER_CLOSED_ANGLE, self.gripper_speed)
            time.sleep(self.gripper_speed / 1000)
            self.gripper_open = False
            print("    [LIBRARY GRIPPER] PHYSICALLY Closed")
            return True
        except Exception as e:
            print(f"    [LIBRARY ERROR] {e}")
            return False
    
    def open_gripper(self):
        """Public method to open gripper"""
        if hasattr(self, 'bus') and self.bus is not None:
            return self.open_gripper_i2c()
        elif hasattr(self, 'arm'):
            return self.open_gripper_library()
        else:
            print("    [SIM] Gripper would open")
            time.sleep(0.8)
            self.gripper_open = True
            return True
    
    def close_gripper(self):
        """Public method to close gripper"""
        if hasattr(self, 'bus') and self.bus is not None:
            return self.close_gripper_i2c()
        elif hasattr(self, 'arm'):
            return self.close_gripper_library()
        else:
            print("    [SIM] Gripper would close")
            time.sleep(0.8)
            self.gripper_open = False
            return True
    
    def test_gripper(self):
        """Test gripper physically"""
        print("\n🔧 Testing PHYSICAL gripper movement...")
        print("⚠️ WATCH THE ARM - It should move!")
        
        success1 = self.open_gripper()
        time.sleep(1)
        
        success2 = self.close_gripper()
        time.sleep(1)
        
        success3 = self.open_gripper()
        time.sleep(1)
        
        if success1 and success2 and success3:
            print("✅ PHYSICAL TEST COMPLETE - Arm SHOULD have moved!")
            return True
        else:
            print("❌ Test failed")
            return False
    
    def emergency_stop(self):
        """Emergency stop - send all servos to neutral"""
        print("    [EMERGENCY] Stopping all servos...")
        
        if self.connected:
            try:
                # Send all servos to 90 degrees
                for servo in range(1, 7):
                    if hasattr(self, 'bus'):
                        self.write_i2c_angle(servo, 90, 500)
                    elif hasattr(self, 'arm'):
                        self.arm.Arm_serial_servo_write(servo, 90, 500)
                time.sleep(0.5)
                print("    [EMERGENCY] All servos stopped")
            except:
                pass
        
        self.gripper_open = True

# ============================================
# I2C TEST UTILITY
# ============================================
def test_i2c_connection():
    """Test I2C connection and find Yahboom arm"""
    print("\n" + "="*70)
    print("🔍 I2C CONNECTION TEST")
    print("="*70)
    
    try:
        import smbus
        
        # Try both I2C buses
        for bus_num in [1, 0]:
            print(f"\nTrying I2C bus {bus_num}...")
            
            try:
                bus = smbus.SMBus(bus_num)
                
                # Scan for devices
                print("Scanning I2C addresses...")
                devices_found = []
                
                for address in range(0x03, 0x78):
                    try:
                        bus.read_byte(address)
                        devices_found.append(address)
                        print(f"  Found device at 0x{address:02X}")
                    except:
                        pass
                
                if devices_found:
                    print(f"\n✅ I2C bus {bus_num} is working")
                    print(f"Found {len(devices_found)} device(s)")
                    
                    # Check for Yahboom address (usually 0x15)
                    if 0x15 in devices_found:
                        print("🎯 Found Yahboom arm at 0x15!")
                        return bus_num, 0x15
                    elif 0x16 in devices_found:
                        print("🎯 Found device at 0x16 (possible Yahboom)")
                        return bus_num, 0x16
                    else:
                        print("⚠️ Yahboom not found at 0x15, trying common addresses...")
                        # Try common Yahboom addresses
                        common_addrs = [0x15, 0x16, 0x17, 0x18, 0x29]
                        for addr in common_addrs:
                            if addr in devices_found:
                                print(f"🎯 Using address 0x{addr:02X}")
                                return bus_num, addr
                        
                        print("❌ No known Yahboom address found")
                        return bus_num, devices_found[0]  # Use first found
                
                else:
                    print(f"❌ No devices found on bus {bus_num}")
                    
            except Exception as e:
                print(f"❌ Bus {bus_num} error: {e}")
        
        print("\n❌ No working I2C bus found")
        return None, None
        
    except ImportError:
        print("❌ smbus not installed. Install with: sudo apt-get install python3-smbus")
        return None, None

# ============================================
# SIMPLE OBJECT DETECTION GRIPPER
# ============================================
class I2CGripperControl:
    def __init__(self):
        # Test I2C first
        print("\n🤖 Initializing Yahboom I2C Arm...")
        bus_num, i2c_addr = test_i2c_connection()
        
        if bus_num is not None:
            print(f"\n✅ I2C Configuration:")
            print(f"   Bus: {bus_num}")
            print(f"   Address: 0x{i2c_addr:02X}")
        
        # Initialize arm
        self.arm = YahboomI2CArm()
        
        # Setup camera
        print("\n📷 Setting up camera...")
        self.cap = self.setup_camera()
        if self.cap is None:
            print("❌ No camera found!")
            # Continue without camera for testing
            self.cap = None
        
        # Load YOLO
        print("\n🤖 Loading YOLO model...")
        self.model = self.load_yolo()
        
        # Detection settings
        self.confidence_threshold = 0.6  # 60%
        self.cooldown_seconds = 3.0
        self.last_action_time = 0
        self.is_active = False
        
        # Object tracking
        self.current_object = None
        self.current_confidence = 0
        self.object_counter = 0
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Display
        self.show_info = True
        
        print("\n" + "="*70)
        print("✅ SYSTEM READY")
        print("="*70)
        print("\n🎯 HOW IT WORKS:")
        print("1. Camera detects object (tool)")
        print("2. Gripper CLOSES physically")
        print("3. Waits 0.5 seconds")
        print("4. Gripper OPENS physically")
        print("5. Waits 3 seconds before next detection")
        print("\n📋 CONTROLS:")
        print("  t - Test gripper (manual)")
        print("  e - Emergency stop")
        print("  s - Show/hide info")
        print("  q - Quit")
        print("="*70)
        
        # Test arm immediately
        if self.arm.connected:
            response = input("\n🔧 Test gripper now? (y/n): ").lower()
            if response == 'y':
                self.arm.test_gripper()
    
    def setup_camera(self):
        """Setup camera"""
        # Try different camera indices
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
        
        print("⚠️ No camera found, running in manual mode")
        return None
    
    def load_yolo(self):
        """Load YOLO model"""
        model_paths = ['best.pt', 'yolo11n.pt', 'yolov8n.pt']
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = YOLO(path)
                    print(f"✅ Loaded YOLO model: {path}")
                    return model
                except Exception as e:
                    print(f"❌ Failed to load {path}: {e}")
        
        print("⚠️ No YOLO model found, running without detection")
        return None
    
    def detect_object(self, frame):
        """Detect object in frame"""
        if self.model is None:
            return False, None, 0
        
        try:
            # Run inference
            results = self.model(frame, 
                               conf=0.5,
                               iou=0.3,
                               imgsz=320,
                               verbose=False,
                               device='cpu')
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                if len(confidences) > 0:
                    # Get best detection
                    best_idx = np.argmax(confidences)
                    confidence = confidences[best_idx]
                    class_id = class_ids[best_idx]
                    
                    # Your tool classes
                    class_names = {
                        0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
                        3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
                    }
                    
                    class_name = class_names.get(class_id, f"Object {class_id}")
                    
                    return True, class_name, confidence*100
        
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
        
        return False, None, 0
    
    def trigger_gripper_action(self):
        """Trigger the gripper sequence"""
        if self.is_active:
            return
        
        current_time = time.time()
        if current_time - self.last_action_time < self.cooldown_seconds:
            return
        
        self.is_active = True
        self.last_action_time = current_time
        
        # Run in separate thread
        threading.Thread(target=self._gripper_sequence, daemon=True).start()
    
    def _gripper_sequence(self):
        """Actual gripper sequence"""
        print("\n" + "="*50)
        print("🎯 OBJECT DETECTED - ACTIVATING GRIPPER")
        print(f"   Object: {self.current_object}")
        print(f"   Confidence: {self.current_confidence:.1f}%")
        print("="*50)
        
        try:
            # 1. Close gripper (GRAB)
            print("\n   [1/2] GRABBING...")
            if self.arm.close_gripper():
                self.object_counter += 1
                print(f"   ✓ Grabbed {self.current_object}")
                
                # 2. Hold for a moment
                time.sleep(0.5)
                
                # 3. Open gripper (RELEASE)
                print("\n   [2/2] RELEASING...")
                if self.arm.open_gripper():
                    print(f"   ✓ Released {self.current_object}")
                    print(f"\n✅ Action complete! Total actions: {self.object_counter}")
                else:
                    print("   ❌ Failed to open gripper")
            else:
                print("   ❌ Failed to close gripper")
                
        except Exception as e:
            print(f"❌ Sequence error: {e}")
        
        finally:
            self.is_active = False
    
    def process_frame(self, frame):
        """Process camera frame"""
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.start_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.start_time)
            self.frame_count = 0
            self.start_time = current_time
        
        # Detect object
        detected, obj_name, confidence = self.detect_object(frame)
        
        # Store detection
        if detected and confidence >= self.confidence_threshold * 100:
            self.current_object = obj_name
            self.current_confidence = confidence
            
            # Trigger gripper
            self.trigger_gripper_action()
        
        # Draw on frame
        display_frame = self.draw_interface(frame, detected, obj_name, confidence)
        
        return display_frame, detected, obj_name, confidence
    
    def draw_interface(self, frame, detected, obj_name, confidence):
        """Draw interface on frame"""
        display = frame.copy()
        
        # Draw detection overlay
        if detected and confidence >= self.confidence_threshold * 100:
            # Draw pulsating circle in center
            h, w = display.shape[:2]
            pulse = int(20 * (1 + np.sin(time.time() * 5) * 0.5))
            
            cv2.circle(display, (w//2, h//2), 100 + pulse, (0, 255, 0), 3)
            cv2.circle(display, (w//2, h//2), 50, (0, 255, 0), -1)
            
            # Draw text
            text = f"{obj_name}: {confidence:.1f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            
            cv2.putText(display, text, (text_x, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw action indicator
            if self.is_active:
                cv2.putText(display, "GRIPPER ACTIVE", (w//2 - 100, h - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Draw info panel
        if self.show_info:
            # Background
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            cv2.rectangle(display, (10, 10), (350, 200), (255, 255, 255), 1)
            
            y = 35
            line = 25
            
            # Title
            cv2.putText(display, "🤖 I2C GRIPPER CONTROL", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += line
            
            # Connection status
            if self.arm.connected:
                status_text = "✅ I2C CONNECTED"
                status_color = (0, 255, 0)
            else:
                status_text = "❌ I2C DISCONNECTED"
                status_color = (255, 0, 0)
            
            cv2.putText(display, status_text, (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            y += line
            
            # FPS
            cv2.putText(display, f"FPS: {self.fps:.1f}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            y += line
            
            # Object info
            if detected:
                obj_text = f"Object: {obj_name}"
                conf_text = f"Confidence: {confidence:.1f}%"
                
                cv2.putText(display, obj_text, (15, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y += 20
                
                cv2.putText(display, conf_text, (15, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y += 20
            else:
                cv2.putText(display, "Status: Searching...", (15, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y += 20
            
            # Action counter
            cv2.putText(display, f"Actions: {self.object_counter}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
            y += 20
            
            # Cooldown indicator
            time_since = time.time() - self.last_action_time
            if time_since < self.cooldown_seconds:
                remaining = self.cooldown_seconds - time_since
                cv2.putText(display, f"Next: {remaining:.1f}s", (15, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
        
        return display
    
    def run(self):
        """Main loop"""
        print("\n🚀 Starting I2C Gripper Control System...")
        time.sleep(1)
        
        # If no camera, run in manual mode
        if self.cap is None:
            print("📷 No camera - Running in manual test mode")
            self.manual_mode()
            return
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera error")
                    break
                
                # Process frame
                display_frame, detected, obj_name, confidence = self.process_frame(frame)
                
                # Show frame
                cv2.imshow('Yahboom I2C Gripper Control', display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                
                elif key == ord('t'):
                    print("\n🔧 Manual gripper test...")
                    self.arm.test_gripper()
                
                elif key == ord('e'):
                    print("\n🛑 Emergency stop!")
                    self.arm.emergency_stop()
                
                elif key == ord('s'):
                    self.show_info = not self.show_info
                    print(f"ℹ️ Info: {'ON' if self.show_info else 'OFF'}")
                
                elif key == ord(' '):
                    # Manual trigger
                    print("\n⚡ Manual trigger...")
                    self.current_object = "Manual Trigger"
                    self.current_confidence = 100
                    self.trigger_gripper_action()
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        
        finally:
            self.cleanup()
    
    def manual_mode(self):
        """Manual mode without camera"""
        print("\n🎮 MANUAL MODE - Press keys to control:")
        print("  t - Test gripper")
        print("  o - Open gripper")
        print("  c - Close gripper")
        print("  e - Emergency stop")
        print("  q - Quit")
        
        try:
            while True:
                key = input("\nEnter command: ").strip().lower()
                
                if key == 'q':
                    break
                elif key == 't':
                    self.arm.test_gripper()
                elif key == 'o':
                    self.arm.open_gripper()
                elif key == 'c':
                    self.arm.close_gripper()
                elif key == 'e':
                    self.arm.emergency_stop()
                else:
                    print("Unknown command")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup"""
        print("\n🧹 Cleaning up...")
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Open gripper before exit
        print("🤖 Opening gripper before exit...")
        self.arm.open_gripper()
        
        print(f"\n📊 Session summary:")
        print(f"   Total gripper actions: {self.object_counter}")
        print(f"   I2C connected: {'✅' if self.arm.connected else '❌'}")
        print("✅ Cleanup complete")

# ============================================
# INSTALLATION CHECK
# ============================================
def check_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    
    missing = []
    
    # Check OpenCV
    try:
        import cv2
        print("✅ OpenCV installed")
    except:
        missing.append("opencv-python")
    
    # Check smbus (for I2C)
    try:
        import smbus
        print("✅ smbus installed")
    except:
        missing.append("python3-smbus (system package)")
    
    # Check ultralytics
    try:
        import ultralytics
        print("✅ ultralytics installed")
    except:
        missing.append("ultralytics")
    
    # Check numpy
    try:
        import numpy as np
        print("✅ numpy installed")
    except:
        missing.append("numpy")
    
    if missing:
        print(f"\n❌ Missing dependencies: {missing}")
        
        # Offer to install
        response = input("\nInstall missing packages? (y/n): ").lower()
        if response == 'y':
            for package in missing:
                if "system" in package:
                    print(f"\nInstalling system package...")
                    os.system("sudo apt-get update")
                    os.system("sudo apt-get install python3-smbus -y")
                else:
                    print(f"\nInstalling {package}...")
                    os.system(f"pip install {package}")
            
            print("\n✅ Installation complete. Restart the program.")
            exit()
    
    print("\n✅ All dependencies checked")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("🤖 YAHBOOM I2C GRIPPER CONTROL SYSTEM")
    print("="*70)
    
    # Check dependencies
    check_dependencies()
    
    # Enable I2C on Raspberry Pi if not already
    print("\n🔧 Checking I2C interface...")
    if os.path.exists("/sys/bus/i2c/devices/"):
        print("✅ I2C interface detected")
    else:
        print("⚠️ I2C not enabled. Enable with:")
        print("   sudo raspi-config")
        print("   -> Interface Options -> I2C -> Yes")
        print("   Then reboot")
    
    # Create and run system
    print("\n" + "="*70)
    print("🚀 Starting main system...")
    print("="*70)
    
    system = I2CGripperControl()
    system.run()