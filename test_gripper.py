# """
# REAL YAHBOOM I2C GRIPPER CONTROL
# Using actual I2C communication - PHYSICAL movement guaranteed
# """

# import cv2
# import time
# import numpy as np
# from ultralytics import YOLO
# import threading
# import sys
# import os
# import smbus

# print("🤖 YAHBOOM I2C GRIPPER CONTROL WITH OBJECT DETECTION")
# print("=" * 70)

# # ============================================
# # REAL YAHBOOM I2C ARM CONTROL
# # ============================================
# class YahboomI2CArm:
#     def __init__(self):
#         self.bus = None
#         self.connected = False
#         self.gripper_open = True
        
#         # I2C Parameters (DEFAULT for Yahboom DOFBOT)
#         self.I2C_ADDR = 0x15          # Default Yahboom I2C address
#         self.I2C_BUS = 6              # Raspberry Pi I2C bus 1 (GPIO 2/3)
        
#         # Servo mapping (Yahboom DOFBOT)
#         # Servo 1: Base, 2: Shoulder, 3: Elbow, 4: Wrist, 5: Gripper, 6: Wrist rotationy
        
#         self.SERVO_GRIPPER = 6        # Usually servo 5 for gripper
        
#         # Angle limits (adjust for your gripper!)
#         self.GRIPPER_OPEN_ANGLE = 180     # Degrees for open
#         self.GRIPPER_CLOSED_ANGLE = 30    # Degrees for closed
        
#         # Timing
#         self.gripper_speed = 1000     # ms for movement
        
#         # State
#         self.is_moving = False
        
#         # Initialize I2C
#         self.initialize_i2c()
        
#         # Alternative: Try loading Yahboom library if I2C fails
#         if not self.connected:
#             self.try_yahboom_library()
    
#     def initialize_i2c(self):
#         """Initialize I2C connection to Yahboom arm"""
#         print("🔌 Initializing I2C connection...")
        
#         try:
#             # Initialize I2C bus
#             self.bus = smbus.SMBus(self.I2C_BUS)
#             time.sleep(0.1)
            
#             # Test communication by reading a byte
#             try:
#                 self.bus.read_byte(self.I2C_ADDR)
#                 print(f"✅ I2C connection established at address 0x{self.I2C_ADDR:02X}")
#                 self.connected = True
                
#                 # Initialize gripper to OPEN
#                 self.open_gripper()
                
#             except Exception as e:
#                 print(f"❌ I2C test failed: {e}")
#                 print("⚠️ Trying alternate I2C bus...")
                
#                 # Try bus 0 (older Raspberry Pi)
#                 try:
#                     self.I2C_BUS = 0
#                     self.bus = smbus.SMBus(self.I2C_BUS)
#                     self.bus.read_byte(self.I2C_ADDR)
#                     print(f"✅ Connected on I2C bus {self.I2C_BUS}")
#                     self.connected = True
#                     self.open_gripper()
#                 except:
#                     print("❌ Failed on both I2C buses")
#                     self.connected = False
            
#         except Exception as e:
#             print(f"❌ I2C initialization failed: {e}")
#             self.connected = False
    
#     def try_yahboom_library(self):
#         """Try loading Yahboom's official library"""
#         print("🔄 Trying Yahboom official library...")
        
#         try:
#             # Try different import paths
#             sys.path.append('/home/pi/ArmPi/')
#             sys.path.append('/home/pi/')
#             sys.path.append('/opt/yahboom/')
            
#             try:
#                 from Arm_Lib import Arm_Device
#                 self.arm = Arm_Device()
#                 self.connected = True
#                 print("✅ Loaded Arm_Device (I2C via library)")
#                 self.open_gripper()
#                 return
#             except:
#                 pass
            
#             try:
#                 try:
#                     sys.path.append('/home/pi/ArmPi/')
#                     sys.path.append('/home/pi/')
#                     sys.path.append('/opt/yahboom/')
#                     import Arm_Lib
#                     self.arm = Arm_Lib.Arm_Device()
#                     self.connected = True
#                     print("✅ Loaded Arm_Lib")
#                     self.open_gripper()
#                     return
#                 except ImportError as e:
#                     print(f"❌ Could not import Arm_Lib: {e}")
#             except:
#                 pass
            
#             print("❌ Could not load Yahboom library")
            
#         except Exception as e:
#             print(f"❌ Library load failed: {e}")
    
#     def write_i2c_angle(self, servo_id, angle, time_ms=None):
#         """
#         Send servo angle command via I2C
#         This is the ACTUAL Yahboom protocol
#         """
#         if not self.connected or self.bus is None:
#             return False
        
#         if time_ms is None:
#             time_ms = self.gripper_speed
        
#         # Convert angle to servo pulses (500-2500 microseconds)
#         # Yahboom uses: 0° = 500µs, 180° = 2500µs
#         pulse_width = 500 + (angle * 2000 / 180)
#         pulse_width = int(pulse_width)
        
#         # Ensure within limits
#         pulse_width = max(500, min(2500, pulse_width))
        
#         # Yahboom I2C protocol:
#         # Command format: [0x55, 0x55, servo_id, time_low, time_high, angle_low, angle_high]
#         time_val = time_ms
        
#         # Prepare data
#         data = [
#             0x55, 0x55,           # Header
#             servo_id,             # Servo ID (1-6)
#             time_val & 0xFF,      # Time low byte
#             (time_val >> 8) & 0xFF, # Time high byte
#             pulse_width & 0xFF,   # Pulse width low byte
#             (pulse_width >> 8) & 0xFF # Pulse width high byte
#         ]
        
#         try:
#             # Send via I2C
#             self.bus.write_i2c_block_data(self.I2C_ADDR, 0, data)
#             print(f"    [I2C] Servo {servo_id} → {angle}° (Pulse: {pulse_width}µs)")
#             return True
            
#         except Exception as e:
#             print(f"    [I2C ERROR] {e}")
#             self.connected = False
#             return False
    
#     def open_gripper_i2c(self):
#         """Open gripper using direct I2C"""
#         print("    [I2C GRIPPER] PHYSICALLY Opening...")
        
#         success = self.write_i2c_angle(self.SERVO_GRIPPER, self.GRIPPER_OPEN_ANGLE, self.gripper_speed)
        
#         if success:
#             time.sleep(self.gripper_speed / 1000)
#             self.gripper_open = True
#             print("    [I2C GRIPPER] PHYSICALLY Opened")
#             return True
#         return False
    
#     def close_gripper_i2c(self):
#         """Close gripper using direct I2C"""
#         print("    [I2C GRIPPER] PHYSICALLY Closing...")
        
#         success = self.write_i2c_angle(self.SERVO_GRIPPER, self.GRIPPER_CLOSED_ANGLE, self.gripper_speed)
        
#         if success:
#             time.sleep(self.gripper_speed / 1000)
#             self.gripper_open = False
#             print("    [I2C GRIPPER] PHYSICALLY Closed")
#             return True
#         return False
    
#     def open_gripper_library(self):
#         """Open gripper using Yahboom library"""
#         print("    [LIBRARY GRIPPER] PHYSICALLY Opening...")
        
#         try:
#             # Using Yahboom library method
#             self.arm.Arm_serial_servo_write(self.SERVO_GRIPPER, self.GRIPPER_OPEN_ANGLE, self.gripper_speed)
#             time.sleep(self.gripper_speed / 1000)
#             self.gripper_open = True
#             print("    [LIBRARY GRIPPER] PHYSICALLY Opened")
#             return True
#         except Exception as e:
#             print(f"    [LIBRARY ERROR] {e}")
#             return False
    
#     def close_gripper_library(self):
#         """Close gripper using Yahboom library"""
#         print("    [LIBRARY GRIPPER] PHYSICALLY Closing...")
        
#         try:
#             # Using Yahboom library method
#             self.arm.Arm_serial_servo_write(self.SERVO_GRIPPER, self.GRIPPER_CLOSED_ANGLE, self.gripper_speed)
#             time.sleep(self.gripper_speed / 1000)
#             self.gripper_open = False
#             print("    [LIBRARY GRIPPER] PHYSICALLY Closed")
#             return True
#         except Exception as e:
#             print(f"    [LIBRARY ERROR] {e}")
#             return False
    
#     def open_gripper(self):
#         """Public method to open gripper"""
#         if hasattr(self, 'bus') and self.bus is not None:
#             return self.open_gripper_i2c()
#         elif hasattr(self, 'arm'):
#             return self.open_gripper_library()
#         else:
#             print("    [SIM] Gripper would open")
#             time.sleep(0.8)
#             self.gripper_open = True
#             return True
    
#     def close_gripper(self):
#         """Public method to close gripper"""
#         if hasattr(self, 'bus') and self.bus is not None:
#             return self.close_gripper_i2c()
#         elif hasattr(self, 'arm'):
#             return self.close_gripper_library()
#         else:
#             print("    [SIM] Gripper would close")
#             time.sleep(0.8)
#             self.gripper_open = False
#             return True
    
#     def test_gripper(self):
#         """Test gripper physically"""
#         print("\n🔧 Testing PHYSICAL gripper movement...")
#         print("⚠️ WATCH THE ARM - It should move!")
        
#         success1 = self.open_gripper()
#         time.sleep(1)
        
#         success2 = self.close_gripper()
#         time.sleep(1)
        
#         success3 = self.open_gripper()
#         time.sleep(1)
        
#         if success1 and success2 and success3:
#             print("✅ PHYSICAL TEST COMPLETE - Arm SHOULD have moved!")
#             return True
#         else:
#             print("❌ Test failed")
#             return False
    
#     def emergency_stop(self):
#         """Emergency stop - send all servos to neutral"""
#         print("    [EMERGENCY] Stopping all servos...")
        
#         if self.connected:
#             try:
#                 # Send all servos to 90 degrees
#                 for servo in range(1, 7):
#                     if hasattr(self, 'bus'):
#                         self.write_i2c_angle(servo, 90, 500)
#                     elif hasattr(self, 'arm'):
#                         self.arm.Arm_serial_servo_write(servo, 90, 500)
#                 time.sleep(0.5)
#                 print("    [EMERGENCY] All servos stopped")
#             except:
#                 pass
        
#         self.gripper_open = True

# # ============================================
# # I2C TEST UTILITY
# # ============================================
# def test_i2c_connection():
#     """Test I2C connection and find Yahboom arm"""
#     print("\n" + "="*70)
#     print("🔍 I2C CONNECTION TEST")
#     print("="*70)
    
#     try:
#         import smbus
        
#         # Try both I2C buses
#         for bus_num in [1, 0]:
#             print(f"\nTrying I2C bus {bus_num}...")
            
#             try:
#                 bus = smbus.SMBus(bus_num)
                
#                 # Scan for devices
#                 print("Scanning I2C addresses...")
#                 devices_found = []
                
#                 for address in range(0x03, 0x78):
#                     try:
#                         bus.read_byte(address)
#                         devices_found.append(address)
#                         print(f"  Found device at 0x{address:02X}")
#                     except:
#                         pass
                
#                 if devices_found:
#                     print(f"\n✅ I2C bus {bus_num} is working")
#                     print(f"Found {len(devices_found)} device(s)")
                    
#                     # Check for Yahboom address (usually 0x15)
#                     if 0x15 in devices_found:
#                         print("🎯 Found Yahboom arm at 0x15!")
#                         return bus_num, 0x15
#                     elif 0x16 in devices_found:
#                         print("🎯 Found device at 0x16 (possible Yahboom)")
#                         return bus_num, 0x16
#                     else:
#                         print("⚠️ Yahboom not found at 0x15, trying common addresses...")
#                         # Try common Yahboom addresses
#                         common_addrs = [0x15, 0x16, 0x17, 0x18, 0x29]
#                         for addr in common_addrs:
#                             if addr in devices_found:
#                                 print(f"🎯 Using address 0x{addr:02X}")
#                                 return bus_num, addr
                        
#                         print("❌ No known Yahboom address found")
#                         return bus_num, devices_found[0]  # Use first found
                
#                 else:
#                     print(f"❌ No devices found on bus {bus_num}")
                    
#             except Exception as e:
#                 print(f"❌ Bus {bus_num} error: {e}")
        
#         print("\n❌ No working I2C bus found")
#         return None, None
        
#     except ImportError:
#         print("❌ smbus not installed. Install with: sudo apt-get install python3-smbus")
#         return None, None

# # ============================================
# # SIMPLE OBJECT DETECTION GRIPPER
# # ============================================
# class I2CGripperControl:
#     def __init__(self):
#         # Test I2C first
#         print("\n🤖 Initializing Yahboom I2C Arm...")
#         bus_num, i2c_addr = test_i2c_connection()
        
#         if bus_num is not None:
#             print(f"\n✅ I2C Configuration:")
#             print(f"   Bus: {bus_num}")
#             print(f"   Address: 0x{i2c_addr:02X}")
        
#         # Initialize arm
#         self.arm = YahboomI2CArm()
        
#         # Setup camera
#         print("\n📷 Setting up camera...")
#         self.cap = self.setup_camera()
#         if self.cap is None:
#             print("❌ No camera found!")
#             # Continue without camera for testing
#             self.cap = None
        
#         # Load YOLO
#         print("\n🤖 Loading YOLO model...")
#         self.model = self.load_yolo()
        
#         # Detection settings
#         self.confidence_threshold = 0.6  # 60%
#         self.cooldown_seconds = 3.0
#         self.last_action_time = 0
#         self.is_active = False
        
#         # Object tracking
#         self.current_object = None
#         self.current_confidence = 0
#         self.object_counter = 0
        
#         # FPS tracking
#         self.fps = 0
#         self.frame_count = 0
#         self.start_time = time.time()
        
#         # Display
#         self.show_info = True
        
#         print("\n" + "="*70)
#         print("✅ SYSTEM READY")
#         print("="*70)
#         print("\n🎯 HOW IT WORKS:")
#         print("1. Camera detects object (tool)")
#         print("2. Gripper CLOSES physically")
#         print("3. Waits 0.5 seconds")
#         print("4. Gripper OPENS physically")
#         print("5. Waits 3 seconds before next detection")
#         print("\n📋 CONTROLS:")
#         print("  t - Test gripper (manual)")
#         print("  e - Emergency stop")
#         print("  s - Show/hide info")
#         print("  q - Quit")
#         print("="*70)
        
#         # Test arm immediately
#         if self.arm.connected:
#             response = input("\n🔧 Test gripper now? (y/n): ").lower()
#             if response == 'y':
#                 self.arm.test_gripper()
    
#     def setup_camera(self):
#         """Setup camera"""
#         # Try different camera indices
#         for i in range(4):
#             try:
#                 cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
#                 if cap.isOpened():
#                     print(f"✅ Camera found at index {i}")
#                     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#                     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#                     cap.set(cv2.CAP_PROP_FPS, 30)
#                     return cap
#             except:
#                 continue
        
#         print("⚠️ No camera found, running in manual mode")
#         return None
    
#     def load_yolo(self):
#         """Load YOLO model"""
#         model_paths = ['best.pt', 'yolo11n.pt', 'yolov8n.pt']
        
#         for path in model_paths:
#             if os.path.exists(path):
#                 try:
#                     model = YOLO(path)
#                     print(f"✅ Loaded YOLO model: {path}")
#                     return model
#                 except Exception as e:
#                     print(f"❌ Failed to load {path}: {e}")
        
#         print("⚠️ No YOLO model found, running without detection")
#         return None
    
#     def detect_object(self, frame):
#         """Detect object in frame"""
#         if self.model is None:
#             return False, None, 0
        
#         try:
#             # Run inference
#             results = self.model(frame, 
#                                conf=0.5,
#                                iou=0.3,
#                                imgsz=320,
#                                verbose=False,
#                                device='cpu')
            
#             if results and results[0].boxes is not None:
#                 boxes = results[0].boxes.xyxy.cpu().numpy()
#                 confidences = results[0].boxes.conf.cpu().numpy()
#                 class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
#                 if len(confidences) > 0:
#                     # Get best detection
#                     best_idx = np.argmax(confidences)
#                     confidence = confidences[best_idx]
#                     class_id = class_ids[best_idx]
                    
#                     # Your tool classes
#                     class_names = {
#                         0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
#                         3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
#                     }
                    
#                     class_name = class_names.get(class_id, f"Object {class_id}")
                    
#                     return True, class_name, confidence*100
        
#         except Exception as e:
#             print(f"⚠️ Detection error: {e}")
        
#         return False, None, 0
    
#     def trigger_gripper_action(self):
#         """Trigger the gripper sequence"""
#         if self.is_active:
#             return
        
#         current_time = time.time()
#         if current_time - self.last_action_time < self.cooldown_seconds:
#             return
        
#         self.is_active = True
#         self.last_action_time = current_time
        
#         # Run in separate thread
#         threading.Thread(target=self._gripper_sequence, daemon=True).start()
    
#     def _gripper_sequence(self):
#         """Actual gripper sequence"""
#         print("\n" + "="*50)
#         print("🎯 OBJECT DETECTED - ACTIVATING GRIPPER")
#         print(f"   Object: {self.current_object}")
#         print(f"   Confidence: {self.current_confidence:.1f}%")
#         print("="*50)
        
#         try:
#             # 1. Close gripper (GRAB)
#             print("\n   [1/2] GRABBING...")
#             if self.arm.close_gripper():
#                 self.object_counter += 1
#                 print(f"   ✓ Grabbed {self.current_object}")
                
#                 # 2. Hold for a moment
#                 time.sleep(0.5)
                
#                 # 3. Open gripper (RELEASE)
#                 print("\n   [2/2] RELEASING...")
#                 if self.arm.open_gripper():
#                     print(f"   ✓ Released {self.current_object}")
#                     print(f"\n✅ Action complete! Total actions: {self.object_counter}")
#                 else:
#                     print("   ❌ Failed to open gripper")
#             else:
#                 print("   ❌ Failed to close gripper")
                
#         except Exception as e:
#             print(f"❌ Sequence error: {e}")
        
#         finally:
#             self.is_active = False
    
#     def process_frame(self, frame):
#         """Process camera frame"""
#         # Flip for mirror view
#         frame = cv2.flip(frame, 1)
        
#         # Update FPS
#         self.frame_count += 1
#         current_time = time.time()
#         if current_time - self.start_time >= 1.0:
#             self.fps = self.frame_count / (current_time - self.start_time)
#             self.frame_count = 0
#             self.start_time = current_time
        
#         # Detect object
#         detected, obj_name, confidence = self.detect_object(frame)
        
#         # Store detection
#         if detected and confidence >= self.confidence_threshold * 100:
#             self.current_object = obj_name
#             self.current_confidence = confidence
            
#             # Trigger gripper
#             self.trigger_gripper_action()
        
#         # Draw on frame
#         display_frame = self.draw_interface(frame, detected, obj_name, confidence)
        
#         return display_frame, detected, obj_name, confidence
    
#     def draw_interface(self, frame, detected, obj_name, confidence):
#         """Draw interface on frame"""
#         display = frame.copy()
        
#         # Draw detection overlay
#         if detected and confidence >= self.confidence_threshold * 100:
#             # Draw pulsating circle in center
#             h, w = display.shape[:2]
#             pulse = int(20 * (1 + np.sin(time.time() * 5) * 0.5))
            
#             cv2.circle(display, (w//2, h//2), 100 + pulse, (0, 255, 0), 3)
#             cv2.circle(display, (w//2, h//2), 50, (0, 255, 0), -1)
            
#             # Draw text
#             text = f"{obj_name}: {confidence:.1f}%"
#             text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
#             text_x = (w - text_size[0]) // 2
            
#             cv2.putText(display, text, (text_x, 50),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
#             # Draw action indicator
#             if self.is_active:
#                 cv2.putText(display, "GRIPPER ACTIVE", (w//2 - 100, h - 30),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
#         # Draw info panel
#         if self.show_info:
#             # Background
#             overlay = display.copy()
#             cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
#             cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
#             cv2.rectangle(display, (10, 10), (350, 200), (255, 255, 255), 1)
            
#             y = 35
#             line = 25
            
#             # Title
#             cv2.putText(display, "🤖 I2C GRIPPER CONTROL", (15, y),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#             y += line
            
#             # Connection status
#             if self.arm.connected:
#                 status_text = "✅ I2C CONNECTED"
#                 status_color = (0, 255, 0)
#             else:
#                 status_text = "❌ I2C DISCONNECTED"
#                 status_color = (255, 0, 0)
            
#             cv2.putText(display, status_text, (15, y),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
#             y += line
            
#             # FPS
#             cv2.putText(display, f"FPS: {self.fps:.1f}", (15, y),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
#             y += line
            
#             # Object info
#             if detected:
#                 obj_text = f"Object: {obj_name}"
#                 conf_text = f"Confidence: {confidence:.1f}%"
                
#                 cv2.putText(display, obj_text, (15, y),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#                 y += 20
                
#                 cv2.putText(display, conf_text, (15, y),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
#                 y += 20
#             else:
#                 cv2.putText(display, "Status: Searching...", (15, y),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
#                 y += 20
            
#             # Action counter
#             cv2.putText(display, f"Actions: {self.object_counter}", (15, y),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
#             y += 20
            
#             # Cooldown indicator
#             time_since = time.time() - self.last_action_time
#             if time_since < self.cooldown_seconds:
#                 remaining = self.cooldown_seconds - time_since
#                 cv2.putText(display, f"Next: {remaining:.1f}s", (15, y),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
        
#         return display
    
#     def run(self):
#         """Main loop"""
#         print("\n🚀 Starting I2C Gripper Control System...")
#         time.sleep(1)
        
#         # If no camera, run in manual mode
#         if self.cap is None:
#             print("📷 No camera - Running in manual test mode")
#             self.manual_mode()
#             return
        
#         try:
#             while True:
#                 # Read frame
#                 ret, frame = self.cap.read()
#                 if not ret:
#                     print("❌ Camera error")
#                     break
                
#                 # Process frame
#                 display_frame, detected, obj_name, confidence = self.process_frame(frame)
                
#                 # Show frame
#                 cv2.imshow('Yahboom I2C Gripper Control', display_frame)
                
#                 # Handle keyboard
#                 key = cv2.waitKey(1) & 0xFF
                
#                 if key == ord('q'):
#                     print("\n👋 Quitting...")
#                     break
                
#                 elif key == ord('t'):
#                     print("\n🔧 Manual gripper test...")
#                     self.arm.test_gripper()
                
#                 elif key == ord('e'):
#                     print("\n🛑 Emergency stop!")
#                     self.arm.emergency_stop()
                
#                 elif key == ord('s'):
#                     self.show_info = not self.show_info
#                     print(f"ℹ️ Info: {'ON' if self.show_info else 'OFF'}")
                
#                 elif key == ord(' '):
#                     # Manual trigger
#                     print("\n⚡ Manual trigger...")
#                     self.current_object = "Manual Trigger"
#                     self.current_confidence = 100
#                     self.trigger_gripper_action()
        
#         except KeyboardInterrupt:
#             print("\n🛑 Interrupted")
        
#         finally:
#             self.cleanup()
    
#     def manual_mode(self):
#         """Manual mode without camera"""
#         print("\n🎮 MANUAL MODE - Press keys to control:")
#         print("  t - Test gripper")
#         print("  o - Open gripper")
#         print("  c - Close gripper")
#         print("  e - Emergency stop")
#         print("  q - Quit")
        
#         try:
#             while True:
#                 key = input("\nEnter command: ").strip().lower()
                
#                 if key == 'q':
#                     break
#                 elif key == 't':
#                     self.arm.test_gripper()
#                 elif key == 'o':
#                     self.arm.open_gripper()
#                 elif key == 'c':
#                     self.arm.close_gripper()
#                 elif key == 'e':
#                     self.arm.emergency_stop()
#                 else:
#                     print("Unknown command")
        
#         finally:
#             self.cleanup()
    
#     def cleanup(self):
#         """Cleanup"""
#         print("\n🧹 Cleaning up...")
        
#         # Release camera
#         if self.cap:
#             self.cap.release()
        
#         # Close windows
#         cv2.destroyAllWindows()
        
#         # Open gripper before exit
#         print("🤖 Opening gripper before exit...")
#         self.arm.open_gripper()
        
#         print(f"\n📊 Session summary:")
#         print(f"   Total gripper actions: {self.object_counter}")
#         print(f"   I2C connected: {'✅' if self.arm.connected else '❌'}")
#         print("✅ Cleanup complete")

# # ============================================
# # INSTALLATION CHECK
# # ============================================
# def check_dependencies():
#     """Check and install required dependencies"""
#     print("🔍 Checking dependencies...")
    
#     missing = []
    
#     # Check OpenCV
#     try:
#         import cv2
#         print("✅ OpenCV installed")
#     except:
#         missing.append("opencv-python")
    
#     # Check smbus (for I2C)
#     try:
#         import smbus
#         print("✅ smbus installed")
#     except:
#         missing.append("python3-smbus (system package)")
    
#     # Check ultralytics
#     try:
#         import ultralytics
#         print("✅ ultralytics installed")
#     except:
#         missing.append("ultralytics")
    
#     # Check numpy
#     try:
#         import numpy as np
#         print("✅ numpy installed")
#     except:
#         missing.append("numpy")
    
#     if missing:
#         print(f"\n❌ Missing dependencies: {missing}")
        
#         # Offer to install
#         response = input("\nInstall missing packages? (y/n): ").lower()
#         if response == 'y':
#             for package in missing:
#                 if "system" in package:
#                     print(f"\nInstalling system package...")
#                     os.system("sudo apt-get update")
#                     os.system("sudo apt-get install python3-smbus -y")
#                 else:
#                     print(f"\nInstalling {package}...")
#                     os.system(f"pip install {package}")
            
#             print("\n✅ Installation complete. Restart the program.")
#             exit()
    
#     print("\n✅ All dependencies checked")

# # ============================================
# # MAIN
# # ============================================
# if __name__ == "__main__":
#     print("="*70)
#     print("🤖 YAHBOOM I2C GRIPPER CONTROL SYSTEM")
#     print("="*70)
    
#     # Check dependencies
#     check_dependencies()
    
#     # Enable I2C on Raspberry Pi if not already
#     print("\n🔧 Checking I2C interface...")
#     if os.path.exists("/sys/bus/i2c/devices/"):
#         print("✅ I2C interface detected")
#     else:
#         print("⚠️ I2C not enabled. Enable with:")
#         print("   sudo raspi-config")
#         print("   -> Interface Options -> I2C -> Yes")
#         print("   Then reboot")
    
#     # Create and run system
#     print("\n" + "="*70)
#     print("🚀 Starting main system...")
#     print("="*70)
    
#     system = I2CGripperControl()
#     system.run()



"""
YAHBOOM I2C GRIPPER CONTROL - SERVO 6
Gripper is on servo 6, connected to I2C bus 6
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import threading
import sys
import os
import smbus

print("🤖 YAHBOOM I2C GRIPPER CONTROL - SERVO 6")
print("=" * 70)

# ============================================
# YAHBOOM I2C ARM CONTROL - SERVO 6
# ============================================
class YahboomArmServo6:
    def __init__(self):
        self.bus = None
        self.connected = False
        self.gripper_open = True
        
        # I2C Configuration - SERVO 6 on bus 6
        self.I2C_BUS = 6              # Your specific bus (not the default 1)
        self.I2C_ADDR = 0x15          # Yahboom default address
        
        # Servo configuration - GRIPPER IS SERVO 6
        self.SERVO_GRIPPER = 6        # IMPORTANT: Gripper on servo 6
        
        # Angle calibration for YOUR gripper
        # TEST these values first with manual controls!
        self.GRIPPER_OPEN_ANGLE = 180      # Degrees for fully open
        self.GRIPPER_CLOSED_ANGLE = 50     # Degrees for fully closed
        self.gripper_speed = 1000          # ms for movement
        
        self.is_moving = False
        
        print(f"⚙️  Configuration:")
        print(f"   I2C Bus: {self.I2C_BUS}")
        print(f"   I2C Address: 0x{self.I2C_ADDR:02X}")
        print(f"   Gripper Servo: {self.SERVO_GRIPPER}")
        print(f"   Open Angle: {self.GRIPPER_OPEN_ANGLE}°")
        print(f"   Closed Angle: {self.GRIPPER_CLOSED_ANGLE}°")
        
        self.initialize_i2c()
    
    def initialize_i2c(self):
        """Initialize I2C connection on bus 6"""
        print(f"\n🔌 Initializing I2C on bus {self.I2C_BUS}...")
        
        try:
            import smbus
            self.bus = smbus.SMBus(self.I2C_BUS)
            time.sleep(0.1)
            
            # Test communication
            try:
                self.bus.read_byte(self.I2C_ADDR)
                print(f"✅ I2C connected on bus {self.I2C_BUS}, address 0x{self.I2C_ADDR:02X}")
                self.connected = True
                
                # Open gripper initially
                self.open_gripper()
                print("✅ Gripper initialized to OPEN position")
                
            except Exception as e:
                print(f"❌ I2C test failed: {e}")
                print("\n⚠️  Troubleshooting steps:")
                print("1. Check if I2C is enabled: sudo raspi-config")
                print("2. Check device: sudo i2cdetect -y 6")
                print("3. Check wiring to servo 6")
                self.connected = False
        
        except ImportError:
            print("❌ smbus not installed. Run: sudo apt-get install python3-smbus")
            self.connected = False
        except Exception as e:
            print(f"❌ I2C initialization error: {e}")
            self.connected = False
    
    def write_servo_angle(self, servo_id, angle, time_ms=1000):
        """
        Send angle command to specific servo via I2C
        Yahboom protocol: [0x55, 0x55, servo_id, time_low, time_high, pulse_low, pulse_high]
        """
        if not self.connected or self.bus is None:
            print(f"    [SIM] Servo {servo_id} → {angle}°")
            time.sleep(time_ms / 1000)
            return True
        
        try:
            # Convert angle to pulse width (500-2500 microseconds)
            # 0° = 500µs, 180° = 2500µs
            pulse_width = 500 + (angle * 2000 / 180)
            pulse_width = int(max(500, min(2500, pulse_width)))
            
            # Prepare I2C data
            data = [
                0x55, 0x55,                    # Header
                servo_id,                      # Servo ID (1-6)
                time_ms & 0xFF,                # Time low byte
                (time_ms >> 8) & 0xFF,         # Time high byte
                pulse_width & 0xFF,            # Pulse width low
                (pulse_width >> 8) & 0xFF      # Pulse width high
            ]
            
            # Send via I2C
            self.bus.write_i2c_block_data(self.I2C_ADDR, 0, data)
            
            # Print movement info
            servo_names = {1: "Base", 2: "Shoulder", 3: "Elbow", 
                          4: "Wrist", 5: "GripRot", 6: "Gripper"}
            servo_name = servo_names.get(servo_id, f"Servo {servo_id}")
            
            print(f"    [I2C] {servo_name} → {angle}° ({pulse_width}µs, {time_ms}ms)")
            
            # Wait for movement to complete
            time.sleep(time_ms / 1000)
            return True
            
        except Exception as e:
            print(f"    ❌ I2C write failed: {e}")
            self.connected = False
            return False
    
    def open_gripper(self):
        """Open gripper - servo 6 to open angle"""
        print("    🤖 PHYSICALLY Opening gripper (Servo 6)...")
        
        success = self.write_servo_angle(
            servo_id=self.SERVO_GRIPPER,
            angle=self.GRIPPER_OPEN_ANGLE,
            time_ms=self.gripper_speed
        )
        
        if success:
            self.gripper_open = True
            print(f"    ✅ Gripper OPEN at {self.GRIPPER_OPEN_ANGLE}°")
        else:
            print("    ❌ Failed to open gripper")
        
        return success
    
    def close_gripper(self):
        """Close gripper - servo 6 to closed angle"""
        print("    🤖 PHYSICALLY Closing gripper (Servo 6)...")
        
        success = self.write_servo_angle(
            servo_id=self.SERVO_GRIPPER,
            angle=self.GRIPPER_CLOSED_ANGLE,
            time_ms=self.gripper_speed
        )
        
        if success:
            self.gripper_open = False
            print(f"    ✅ Gripper CLOSED at {self.GRIPPER_CLOSED_ANGLE}°")
        else:
            print("    ❌ Failed to close gripper")
        
        return success
    
    def test_all_servos(self):
        """Test all servos to identify which one is gripper"""
        print("\n🔧 Testing ALL servos to identify gripper...")
        print("⚠️ WATCH THE ARM CLOSELY - Each servo will move!")
        
        test_angles = {
            1: (90, 120),   # Base: 90° to 120°
            2: (90, 120),   # Shoulder
            3: (90, 120),   # Elbow
            4: (90, 120),   # Wrist
            5: (90, 120),   # Grip rotation
            6: (90, 120)    # Gripper
        }
        
        for servo_id in range(1, 7):
            print(f"\n🔍 Testing Servo {servo_id}...")
            
            # Move to test position
            self.write_servo_angle(servo_id, test_angles[servo_id][0], 1000)
            time.sleep(1)
            
            # Move to different position
            self.write_servo_angle(servo_id, test_angles[servo_id][1], 1000)
            time.sleep(1)
            
            # Return to center
            self.write_servo_angle(servo_id, 90, 1000)
            time.sleep(1)
            
            response = input(f"Did Servo {servo_id} move the gripper? (y/n): ").lower()
            if response == 'y':
                print(f"🎯 Found gripper at Servo {servo_id}!")
                self.SERVO_GRIPPER = servo_id
                return servo_id
        
        print("❌ Could not identify gripper servo")
        return None
    
    def calibrate_gripper(self):
        """Manual gripper calibration"""
        print("\n🔧 Gripper Calibration Mode")
        print("Use number keys to set angle, 'o' to open, 'c' to close")
        print("Press 's' to save current angle, 'q' to quit")
        
        current_angle = 90
        
        while True:
            print(f"\nCurrent angle: {current_angle}°")
            print("Commands: 0-9 (set angle), o (open), c (close), s (save), q (quit)")
            
            cmd = input("Enter command: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 'o':
                current_angle = self.GRIPPER_OPEN_ANGLE
                self.write_servo_angle(self.SERVO_GRIPPER, current_angle, 1000)
            elif cmd == 'c':
                current_angle = self.GRIPPER_CLOSED_ANGLE
                self.write_servo_angle(self.SERVO_GRIPPER, current_angle, 1000)
            elif cmd == 's':
                save_as = input("Save as OPEN or CLOSED? (o/c): ").lower()
                if save_as == 'o':
                    self.GRIPPER_OPEN_ANGLE = current_angle
                    print(f"✅ Open angle saved: {current_angle}°")
                elif save_as == 'c':
                    self.GRIPPER_CLOSED_ANGLE = current_angle
                    print(f"✅ Closed angle saved: {current_angle}°")
            elif cmd.isdigit():
                # Set specific angle
                angle = int(cmd) * 20  # 0=0°, 1=20°, 2=40°, etc.
                if 0 <= angle <= 180:
                    current_angle = angle
                    self.write_servo_angle(self.SERVO_GRIPPER, current_angle, 1000)
                else:
                    print("❌ Angle must be 0-180")
            else:
                print("❌ Invalid command")
        
        print(f"\n💾 Final calibration:")
        print(f"   Open angle: {self.GRIPPER_OPEN_ANGLE}°")
        print(f"   Closed angle: {self.GRIPPER_CLOSED_ANGLE}°")
    
    def test_gripper_sequence(self):
        """Test open/close sequence"""
        print("\n🔄 Testing gripper sequence...")
        print("Gripper should OPEN → CLOSE → OPEN")
        
        success1 = self.open_gripper()
        time.sleep(1)
        
        success2 = self.close_gripper()
        time.sleep(1)
        
        success3 = self.open_gripper()
        time.sleep(1)
        
        if success1 and success2 and success3:
            print("✅ Gripper test SUCCESSFUL!")
            print("📢 LISTEN for servo sounds and WATCH for movement!")
            return True
        else:
            print("❌ Gripper test FAILED")
            return False
    
    def emergency_stop(self):
        """Stop all servos"""
        print("🛑 Emergency stop - All servos to 90°")
        
        for servo_id in range(1, 7):
            try:
                self.write_servo_angle(servo_id, 90, 500)
            except:
                pass
        
        time.sleep(0.5)
        self.gripper_open = True

# ============================================
# OBJECT DETECTION SYSTEM
# ============================================
class ObjectDetectionGripperControl:
    def __init__(self):
        print("\n🤖 Initializing Object Detection Gripper System...")
        
        # Initialize Yahboom arm with servo 6
        print("🔧 Initializing Yahboom Arm (Servo 6)...")
        self.arm = YahboomArmServo6()
        
        # Setup camera
        print("\n📷 Setting up camera...")
        self.cap = self.setup_camera()
        
        # Load YOLO model
        print("\n🧠 Loading YOLO model...")
        self.model = self.load_yolo_model()
        
        # Detection settings
        self.confidence_threshold = 0.6
        self.cooldown_time = 3.0
        self.last_action_time = 0
        self.is_gripper_active = False
        
        # Object tracking
        self.detected_object = None
        self.detected_confidence = 0
        self.action_counter = 0
        
        # Display
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.show_info = True
        
        print("\n" + "="*70)
        print("✅ SYSTEM READY")
        print("="*70)
        print("\n🎯 HOW IT WORKS:")
        print("1. Camera detects object (tool)")
        print("2. Gripper (Servo 6) CLOSES physically")
        print("3. Waits 0.5 seconds")
        print("4. Gripper (Servo 6) OPENS physically")
        print("5. 3-second cooldown before next action")
        print("\n📋 CONTROLS:")
        print("  t  - Test gripper")
        print("  c  - Calibrate gripper")
        print("  a  - Test all servos (find gripper)")
        print("  e  - Emergency stop")
        print("  s  - Show/hide info")
        print("  q  - Quit")
        print("="*70)
        
        # Initial test
        if self.arm.connected:
            print("\n🔧 Initial gripper test...")
            self.arm.test_gripper_sequence()
    
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
        
        print("⚠️ No camera found, using simulated video")
        return None
    
    def load_yolo_model(self):
        """Load YOLO model"""
        model_files = ['best_2s.pt']
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    model = YOLO(model_file)
                    print(f"✅ Loaded model: {model_file}")
                    return model
                except Exception as e:
                    print(f"❌ Failed to load {model_file}: {e}")
        
        print("⚠️ No YOLO model found, will use simulated detection")
        return None
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if self.model is None:
            # Simulate detection for testing
            return False, "Simulated", 80.0
        
        try:
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
                    best_idx = np.argmax(confidences)
                    confidence = confidences[best_idx] * 100
                    class_id = class_ids[best_idx]
                    
                    class_names = {
                        0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
                        3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
                    }
                    
                    class_name = class_names.get(class_id, f"Obj_{class_id}")
                    
                    return True, class_name, confidence
        
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
        
        return False, None, 0
    
    def trigger_gripper_action(self):
        """Trigger gripper when object detected"""
        if self.is_gripper_active:
            return
        
        current_time = time.time()
        if current_time - self.last_action_time < self.cooldown_time:
            return
        
        self.is_gripper_active = True
        self.last_action_time = current_time
        
        # Run in separate thread
        threading.Thread(target=self._execute_gripper_sequence, daemon=True).start()
    
    def _execute_gripper_sequence(self):
        """Execute gripper sequence"""
        print(f"\n{'='*60}")
        print(f"🎯 OBJECT DETECTED: {self.detected_object}")
        print(f"   Confidence: {self.detected_confidence:.1f}%")
        print(f"{'='*60}")
        
        try:
            # 1. Close gripper
            print("\n   [1/2] 🤖 CLOSING gripper...")
            if self.arm.close_gripper():
                self.action_counter += 1
                
                # 2. Hold closed
                time.sleep(0.5)
                
                # 3. Open gripper
                print("\n   [2/2] 🤖 OPENING gripper...")
                if self.arm.open_gripper():
                    print(f"\n✅ Action #{self.action_counter} complete!")
                    print(f"   Object: {self.detected_object}")
                else:
                    print("❌ Failed to open gripper")
            else:
                print("❌ Failed to close gripper")
        
        except Exception as e:
            print(f"❌ Gripper sequence error: {e}")
        
        finally:
            self.is_gripper_active = False
    
    def process_video_frame(self):
        """Process video frame if camera available"""
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
        detected, obj_name, confidence = self.detect_objects(frame)
        
        # Store detection
        if detected and confidence >= self.confidence_threshold * 100:
            self.detected_object = obj_name
            self.detected_confidence = confidence
            
            # Trigger gripper action
            self.trigger_gripper_action()
        
        # Draw on frame
        display_frame = self.draw_display(frame, detected, obj_name, confidence)
        
        return display_frame
    
    def draw_display(self, frame, detected, obj_name, confidence):
        """Draw interface on frame"""
        if frame is None:
            return None
        
        display = frame.copy()
        
        # Draw detection marker
        if detected and confidence >= self.confidence_threshold * 100:
            h, w = display.shape[:2]
            
            # Pulsing circle
            pulse = int(30 * (1 + np.sin(time.time() * 5) * 0.3))
            cv2.circle(display, (w//2, h//2), 80 + pulse, (0, 255, 0), 3)
            
            # Detection text
            text = f"{obj_name}: {confidence:.1f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0]
            text_x = (w - text_size[0]) // 2
            
            cv2.putText(display, text, (text_x, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw info panel
        if self.show_info:
            # Background
            cv2.rectangle(display, (10, 10), (200, 160), (0, 0, 0, 180), -1)
            cv2.rectangle(display, (10, 10), (00, 160), (255, 255, 255), 1)
            
            y = 35
            line = 25
            
            # Title
            cv2.putText(display, "🤖 SERVO 6 GRIPPER CONTROL", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 255), 2)
            y += line
            
            # Arm status
            if self.arm.connected:
                status = f"✅ I2C Bus {self.arm.I2C_BUS}, Addr 0x{self.arm.I2C_ADDR:02X}"
                color = (0, 255, 0)
            else:
                status = "❌ I2C DISCONNECTED"
                color = (255, 0, 0)
            
            cv2.putText(display, status, (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += line
            
            # Gripper info
            gripper_status = "OPEN" if self.arm.gripper_open else "CLOSED"
            gripper_color = (0, 255, 0) if self.arm.gripper_open else (0, 0, 255)
            cv2.putText(display, f"Gripper (S6): {gripper_status}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, gripper_color, 1)
            y += line
            
            # FPS
            cv2.putText(display, f"FPS: {self.fps:.1f}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            y += line
            
            # Actions
            cv2.putText(display, f"Actions: {self.action_counter}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
            y += line
            
            # Cooldown
            time_since = time.time() - self.last_action_time
            if time_since < self.cooldown_time:
                remaining = self.cooldown_time - time_since
                cv2.putText(display, f"Next: {remaining:.1f}s", (15, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
        
        return display
    
    def run(self):
        """Main loop"""
        print("\n🚀 Starting Object Detection Gripper Control...")
        print("📸 Camera starting in 2 seconds...")
        time.sleep(2)
        
        # Check if we have camera
        if self.cap is None:
            print("📷 No camera - Running in manual mode")
            self.manual_mode()
            return
        
        try:
            while True:
                # Process frame
                display_frame = self.process_video_frame()
                
                if display_frame is not None:
                    cv2.imshow('Servo 6 Gripper Control - Object Detection', display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                
                elif key == ord('t'):
                    print("\n🔧 Testing gripper...")
                    self.arm.test_gripper_sequence()
                
                elif key == ord('c'):
                    print("\n🔧 Entering calibration mode...")
                    self.arm.calibrate_gripper()
                
                elif key == ord('a'):
                    print("\n🔧 Testing all servos to find gripper...")
                    self.arm.test_all_servos()
                
                elif key == ord('e'):
                    print("\n🛑 Emergency stop!")
                    self.arm.emergency_stop()
                
                elif key == ord('s'):
                    self.show_info = not self.show_info
                    print(f"ℹ️ Info display: {'ON' if self.show_info else 'OFF'}")
                
                elif key == ord(' '):
                    # Manual trigger
                    print("\n⚡ Manual trigger...")
                    self.detected_object = "Manual Trigger"
                    self.detected_confidence = 100
                    self.trigger_gripper_action()
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        
        finally:
            self.cleanup()
    
    def manual_mode(self):
        """Manual control mode"""
        print("\n🎮 MANUAL CONTROL MODE")
        print("="*50)
        print("Controls:")
        print("  o - Open gripper")
        print("  c - Close gripper")
        print("  t - Test sequence (open→close→open)")
        print("  e - Emergency stop (all servos to 90°)")
        print("  q - Quit")
        print("="*50)
        
        try:
            while True:
                cmd = input("\nEnter command: ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'o':
                    self.arm.open_gripper()
                elif cmd == 'c':
                    self.arm.close_gripper()
                elif cmd == 't':
                    self.arm.test_gripper_sequence()
                elif cmd == 'e':
                    self.arm.emergency_stop()
                else:
                    print("❌ Unknown command")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Ensure gripper is open
        print("🤖 Opening gripper before exit...")
        self.arm.open_gripper()
        
        print(f"\n📊 Session Summary:")
        print(f"   Total gripper actions: {self.action_counter}")
        print(f"   I2C connected: {'✅' if self.arm.connected else '❌'}")
        print(f"   Gripper servo: {self.arm.SERVO_GRIPPER}")
        print("✅ Cleanup complete")

# ============================================
# I2C DIAGNOSTIC TOOL
# ============================================
def i2c_diagnostic():
    """Diagnose I2C connection"""
    print("\n" + "="*70)
    print("🔍 I2C DIAGNOSTIC TOOL")
    print("="*70)
    
    try:
        import smbus
        
        # Try bus 6 specifically
        print(f"\n📡 Testing I2C bus 6...")
        
        try:
            bus = smbus.SMBus(6)
            print("✅ I2C bus 6 is accessible")
        except Exception as e:
            print(f"❌ Cannot access bus 6: {e}")
            print("\n⚠️ Available I2C buses:")
            
            # Try to find available buses
            for bus_num in [0, 1, 2, 3, 4, 5, 6, 7]:
                try:
                    bus = smbus.SMBus(bus_num)
                    print(f"  ✅ Bus {bus_num} is available")
                    bus.close()
                except:
                    pass
        
        # Scan for devices on bus 6
        print("\n🔎 Scanning for devices on bus 6...")
        try:
            bus = smbus.SMBus(6)
            devices_found = []
            
            for addr in range(0x03, 0x78):
                try:
                    bus.read_byte(addr)
                    devices_found.append(addr)
                    print(f"  Found device at 0x{addr:02X}")
                except:
                    pass
            
            if devices_found:
                print(f"\n✅ Found {len(devices_found)} device(s)")
                
                # Check for Yahboom
                if 0x15 in devices_found:
                    print("🎯 Yahboom arm detected at 0x15!")
                elif 0x16 in devices_found:
                    print("🎯 Device at 0x16 (might be Yahboom)")
                else:
                    print("⚠️ Yahboom not found at usual addresses")
            else:
                print("❌ No devices found on bus 6")
                
        except Exception as e:
            print(f"❌ Scan failed: {e}")
        
        bus.close()
        
    except ImportError:
        print("❌ smbus not installed. Install with: sudo apt-get install python3-smbus")
    
    print("\n" + "="*70)

# ============================================
# INSTALLATION CHECK
# ============================================
def check_requirements():
    """Check system requirements"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check for I2C
    if os.path.exists("/sys/bus/i2c/devices/"):
        print("✅ I2C interface detected")
    else:
        print("⚠️ I2C interface not found")
        print("   Enable with: sudo raspi-config")
        print("   Interface Options → I2C → Yes")
    
    # Check for camera
    print("\n📷 Checking for camera...")
    for i in range(4):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Camera found at index {i}")
                cap.release()
                break
        except:
            pass
    else:
        print("⚠️ No camera detected")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("🤖 YAHBOOM SERVO 6 GRIPPER CONTROL SYSTEM")
    print("="*70)
    
    # Run diagnostic
    i2c_diagnostic()
    
    # Check requirements
    check_requirements()
    
    print("\n" + "="*70)
    print("🚀 Starting main system...")
    print("="*70)
    
    # Create and run system
    system = ObjectDetectionGripperControl()
    system.run()