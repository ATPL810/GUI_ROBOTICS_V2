"""
SIMPLE GRIPPER CONTROL WITH YAHBOOM ARM LIBRARY
REAL physical movement when object detected
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import threading
import sys
import os

print("🤖 SIMPLE GRIPPER CONTROL WITH REAL YAHBOOM ARM")
print("=" * 60)

# ============================================
# REAL YAHBOOM ARM CONTROL (Using actual library)
# ============================================
class RealYahboomArm:
    def __init__(self):
        self.arm = None
        self.connected = False
        self.gripper_open = True
        
        # Try to import the actual Yahboom library
        try:
            # First, add common Yahboom library paths
            sys.path.append('/home/pi/ArmPi/')
            sys.path.append('/home/pi/yahboom/')
            sys.path.append('/home/pi/')
            
            # Try different import methods
            try:
                from Arm_Lib import Arm_Device
                self.Arm = Arm_Device
                self.arm = Arm_Device()
                print("✅ Loaded Arm_Device from Arm_Lib")
            except:
                try:
                    import Arm_Lib
                    self.Arm = Arm_Lib
                    self.arm = Arm_Lib.Arm_Device()
                    print("✅ Loaded Arm_Lib")
                except:
                    try:
                        import DOFBOT
                        self.Arm = DOFBOT
                        self.arm = DOFBOT.Arm_Device()
                        print("✅ Loaded DOFBOT")
                    except:
                        raise ImportError("No Yahboom library found")
            
            time.sleep(2)  # Wait for arm to initialize
            self.connected = True
            print("✅ Yahboom Arm CONNECTED and READY")
            
            # Initialize gripper to OPEN
            self.open_gripper()
            
        except Exception as e:
            print(f"❌ Could not load Yahboom library: {e}")
            print("⚠️  Trying alternative import methods...")
            self.try_alternative_import()
    
    def try_alternative_import(self):
        """Try alternative ways to import Yahboom library"""
        try:
            # Method 1: Direct import if installed
            import importlib.util
            
            # Common Yahboom library names
            library_names = ['Arm_Lib', 'dofbot', 'yahboom_arm', 'ArmPi']
            
            for lib_name in library_names:
                try:
                    spec = importlib.util.find_spec(lib_name)
                    if spec is not None:
                        module = importlib.import_module(lib_name)
                        if hasattr(module, 'Arm_Device'):
                            self.Arm = module
                            self.arm = module.Arm_Device()
                            self.connected = True
                            print(f"✅ Found and loaded {lib_name}")
                            return
                except:
                    continue
            
            # Method 2: Check common paths
            common_paths = [
                '/home/pi/ArmPi/Arm_Lib.py',
                '/home/pi/yahboom/Arm_Lib.py',
                '/home/pi/dofbot.py',
                '/opt/yahboom/arm_lib.py'
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    try:
                        # Add the directory to path
                        dir_path = os.path.dirname(path)
                        sys.path.append(dir_path)
                        
                        # Import based on filename
                        module_name = os.path.basename(path).replace('.py', '')
                        module = __import__(module_name)
                        
                        if hasattr(module, 'Arm_Device'):
                            self.Arm = module
                            self.arm = module.Arm_Device()
                            self.connected = True
                            print(f"✅ Loaded from {path}")
                            return
                    except:
                        continue
            
            print("❌ No Yahboom library found. Running in simulation mode.")
            self.connected = False
            
        except Exception as e:
            print(f"❌ Alternative import failed: {e}")
            self.connected = False
    
    def open_gripper(self):
        """REALLY open the gripper - PHYSICAL MOVEMENT"""
        print("    [GRIPPER] PHYSICALLY Opening...")
        
        try:
            if self.connected and self.arm:
                # Method 1: Using Arm_Device
                if hasattr(self.arm, 'Arm_serial_servo_write'):
                    # Servo 6 is usually gripper on Yahboom DOFBOT
                    # Adjust servo number based on your arm!
                    self.arm.Arm_serial_servo_write(6, 180, 800)  # 180 = open
                    time.sleep(0.8)
                    self.gripper_open = True
                    print("    [GRIPPER] PHYSICALLY Opened (servo 6)")
                    return True
                
                # Method 2: Different library method
                elif hasattr(self.arm, 'set_servo_angle'):
                    self.arm.set_servo_angle(6, 180, 800)
                    time.sleep(0.8)
                    self.gripper_open = True
                    print("    [GRIPPER] PHYSICALLY Opened")
                    return True
                
                else:
                    print("    ❌ Unknown arm control method")
                    return False
            
            else:
                # Simulation mode
                print("    [SIM] Gripper would open")
                time.sleep(0.8)
                self.gripper_open = True
                return True
                
        except Exception as e:
            print(f"    ❌ Failed to open gripper: {e}")
            return False
    
    def close_gripper(self):
        """REALLY close the gripper - PHYSICAL MOVEMENT"""
        print("    [GRIPPER] PHYSICALLY Closing...")
        
        try:
            if self.connected and self.arm:
                # Method 1: Using Arm_Device
                if hasattr(self.arm, 'Arm_serial_servo_write'):
                    # Servo 6, angle 0-30 = closed (adjust based on your arm)
                    self.arm.Arm_serial_servo_write(6, 30, 800)  # 30 = closed
                    time.sleep(0.8)
                    self.gripper_open = False
                    print("    [GRIPPER] PHYSICALLY Closed (servo 6)")
                    return True
                
                # Method 2: Different library method
                elif hasattr(self.arm, 'set_servo_angle'):
                    self.arm.set_servo_angle(6, 30, 800)
                    time.sleep(0.8)
                    self.gripper_open = False
                    print("    [GRIPPER] PHYSICALLY Closed")
                    return True
                
                else:
                    print("    ❌ Unknown arm control method")
                    return False
            
            else:
                # Simulation mode
                print("    [SIM] Gripper would close")
                time.sleep(0.8)
                self.gripper_open = False
                return True
                
        except Exception as e:
            print(f"    ❌ Failed to close gripper: {e}")
            return False
    
    def test_gripper(self):
        """Test gripper open/close sequence"""
        print("\n🔧 Testing REAL gripper movement...")
        success1 = self.open_gripper()
        time.sleep(1)
        success2 = self.close_gripper()
        time.sleep(1)
        success3 = self.open_gripper()
        
        if success1 and success2 and success3:
            print("✅ Gripper test SUCCESSFUL - Arm should have MOVED")
            return True
        else:
            print("❌ Gripper test FAILED")
            return False

# ============================================
# DETECTION AND CONTROL SYSTEM
# ============================================
class ObjectDetectionGripper:
    def __init__(self):
        # Initialize REAL Yahboom arm
        print("🔌 Initializing Yahboom Arm...")
        self.arm = RealYahboomArm()
        
        # Setup camera
        print("📷 Setting up camera...")
        self.cap = self.setup_camera()
        if self.cap is None:
            print("❌ ERROR: No camera found!")
            exit()
        
        # Load YOLO
        print("🤖 Loading YOLO model...")
        self.model = self.load_yolo()
        
        # Detection settings
        self.detection_threshold = 0.6  # 60% confidence
        self.cooldown_time = 3.0  # seconds between actions
        self.last_action_time = 0
        self.is_gripper_active = False
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Display
        self.show_info = True
        
        print("\n" + "="*60)
        print("✅ SYSTEM READY!")
        print("="*60)
        print("\n🎯 HOW TO USE:")
        print("1. Point camera at a tool (screwdriver, hammer, etc.)")
        print("2. When detected, gripper will CLOSE then OPEN")
        print("3. Move to another tool to repeat")
        print("\n📋 CONTROLS:")
        print("  t - Test gripper manually")
        print("  s - Show/hide info")
        print("  q - Quit")
        print("="*60)
    
    def setup_camera(self):
        """Setup webcam"""
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if cap.isOpened():
                    print(f"✅ Found camera at index {i}")
                    # Set resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    return cap
            except:
                continue
        return None
    
    def load_yolo(self):
        """Load YOLO model"""
        try:
            model = YOLO('best.pt')  # Your trained model
            print("✅ YOLO model loaded successfully")
            return model
        except Exception as e:
            print(f"⚠️ Could not load YOLO: {e}")
            print("⚠️ Running in camera-only mode")
            return None
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if self.model is None:
            return [], []
        
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
                
                # Class names from your model
                class_names = {
                    0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
                    3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
                }
                
                return boxes, confidences, class_ids, class_names
        
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
        
        return [], [], [], {}
    
    def trigger_gripper_sequence(self):
        """Trigger the gripper sequence in a separate thread"""
        if self.is_gripper_active:
            return
        
        current_time = time.time()
        if current_time - self.last_action_time < self.cooldown_time:
            return
        
        self.is_gripper_active = True
        
        # Run in separate thread to not block video
        threading.Thread(target=self._gripper_sequence, daemon=True).start()
    
    def _gripper_sequence(self):
        """Actual gripper sequence"""
        try:
            self.last_action_time = time.time()
            
            # 1. Close gripper
            success1 = self.arm.close_gripper()
            
            if success1:
                # 2. Hold for a moment
                time.sleep(0.5)
                
                # 3. Open gripper
                success2 = self.arm.open_gripper()
                
                if success2:
                    print("✅ Gripper sequence COMPLETED - Arm MOVED")
                else:
                    print("❌ Failed to open gripper")
            else:
                print("❌ Failed to close gripper")
                
        except Exception as e:
            print(f"❌ Gripper sequence error: {e}")
        
        finally:
            self.is_gripper_active = False
    
    def process_frame(self, frame):
        """Process each frame"""
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = current_time
        
        # Detect objects
        boxes, confidences, class_ids, class_names = self.detect_objects(frame)
        
        object_detected = False
        best_confidence = 0
        best_class = ""
        
        # Check if any detection meets threshold
        for i, confidence in enumerate(confidences):
            if confidence >= self.detection_threshold:
                object_detected = True
                if confidence > best_confidence:
                    best_confidence = confidence
                    class_id = class_ids[i]
                    best_class = class_names.get(class_id, f"Object {class_id}")
        
        # Trigger gripper if object detected
        if object_detected:
            print(f"🎯 Detected: {best_class} ({best_confidence*100:.1f}%)")
            self.trigger_gripper_sequence()
        
        # Draw on frame
        display_frame = self.draw_display(frame, boxes, confidences, class_ids, 
                                         class_names, object_detected, best_class, best_confidence)
        
        return display_frame, object_detected, best_class, best_confidence
    
    def draw_display(self, frame, boxes, confidences, class_ids, class_names,
                    object_detected, best_class, best_confidence):
        """Draw everything on frame"""
        display = frame.copy()
        
        # Draw detection boxes
        for i, box in enumerate(boxes):
            if confidences[i] >= 0.4:  # Lower threshold for display
                x1, y1, x2, y2 = map(int, box)
                class_id = class_ids[i]
                confidence = confidences[i]
                
                # Color based on class
                colors = [
                    (0, 255, 0),    # Green - Bolt
                    (0, 0, 255),    # Red - Hammer
                    (255, 0, 0),    # Blue - Measuring Tape
                    (255, 255, 0),  # Cyan - Plier
                    (255, 0, 255),  # Magenta - Screwdriver
                    (0, 255, 255)   # Yellow - Wrench
                ]
                color = colors[class_id % len(colors)]
                
                # Draw box
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_names.get(class_id, 'Unknown')}: {confidence*100:.1f}%"
                cv2.putText(display, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw info panel
        if self.show_info:
            # Semi-transparent background
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            cv2.rectangle(display, (10, 10), (350, 180), (255, 255, 255), 1)
            
            y = 35
            line_height = 25
            
            # Title
            cv2.putText(display, "🤖 YAHBOOM GRIPPER CONTROL", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += line_height
            
            # Status
            if object_detected:
                status_color = (0, 255, 0)
                status_text = f"DETECTED: {best_class}"
            else:
                status_color = (255, 255, 0)
                status_text = "SEARCHING..."
            
            cv2.putText(display, status_text, (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
            y += line_height
            
            # FPS
            cv2.putText(display, f"FPS: {self.fps:.1f}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            y += line_height
            
            # Arm status
            arm_status = "✅ CONNECTED" if self.arm.connected else "❌ SIMULATION"
            arm_color = (0, 255, 0) if self.arm.connected else (255, 0, 0)
            cv2.putText(display, f"ARM: {arm_status}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, arm_color, 1)
            y += line_height
            
            # Gripper status
            gripper_status = "OPEN" if self.arm.gripper_open else "CLOSED"
            gripper_color = (0, 255, 0) if self.arm.gripper_open else (0, 0, 255)
            cv2.putText(display, f"GRIPPER: {gripper_status}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, gripper_color, 1)
            y += line_height
            
            # Cooldown indicator
            time_since = time.time() - self.last_action_time
            if time_since < self.cooldown_time:
                remaining = self.cooldown_time - time_since
                cv2.putText(display, f"Next action in: {remaining:.1f}s", 
                           (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        
        # Draw center marker
        h, w = display.shape[:2]
        cv2.circle(display, (w//2, h//2), 8, (255, 255, 255), 1)
        cv2.circle(display, (w//2, h//2), 4, (0, 255, 0), -1)
        
        return display
    
    def run(self):
        """Main loop"""
        print("\n🚀 Starting detection system...")
        print("📸 Camera feed starting in 2 seconds...")
        time.sleep(2)
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Can't read from camera")
                    break
                
                # Process frame
                display_frame, detected, obj_class, confidence = self.process_frame(frame)
                
                # Show frame
                cv2.imshow('Yahboom Gripper Control - Object Detection', display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                
                elif key == ord('t'):
                    print("\n🔧 Manual gripper test...")
                    self.arm.test_gripper()
                
                elif key == ord('s'):
                    self.show_info = not self.show_info
                    print(f"ℹ️ Info display: {'ON' if self.show_info else 'OFF'}")
                
                elif key == ord(' '):
                    # Quick action on spacebar
                    print("\n⚡ Manual trigger...")
                    self.trigger_gripper_sequence()
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Make sure gripper is open
        print("🤖 Opening gripper before exit...")
        self.arm.open_gripper()
        
        print("✅ Cleanup complete")

# ============================================
# INSTALLATION CHECK SCRIPT
# ============================================
def check_installation():
    """Check if Yahboom libraries are installed"""
    print("🔍 Checking Yahboom installation...")
    
    # Common Yahboom library paths
    common_paths = [
        '/home/pi/ArmPi/',
        '/home/pi/yahboom/',
        '/home/pi/',
        '/opt/yahboom/'
    ]
    
    print("\n📁 Checking common paths:")
    for path in common_paths:
        if os.path.exists(path):
            print(f"  ✅ Found: {path}")
            # List files
            try:
                files = os.listdir(path)
                for file in files:
                    if 'arm' in file.lower() or 'dof' in file.lower():
                        print(f"    📄 {file}")
            except:
                pass
        else:
            print(f"  ❌ Not found: {path}")
    
    # Check Python imports
    print("\n🐍 Checking Python imports:")
    
    import importlib.util
    
    libraries_to_check = ['Arm_Lib', 'dofbot', 'yahboom_arm', 'ArmPi']
    
    for lib in libraries_to_check:
        spec = importlib.util.find_spec(lib)
        if spec is not None:
            print(f"  ✅ {lib} is importable")
            print(f"     Location: {spec.origin}")
        else:
            print(f"  ❌ {lib} not found")
    
    # Ask user for library location
    print("\n" + "="*60)
    print("❓ Can't find Yahboom library?")
    print("="*60)
    print("Please provide the path to your Yahboom library.")
    print("Common locations:")
    print("  /home/pi/ArmPi/Arm_Lib.py")
    print("  /home/pi/yahboom/Arm_Lib.py")
    print("  /home/pi/dofbot.py")
    print("\nOr enter 'skip' to continue with simulation mode.")
    
    user_path = input("\nEnter library path (or 'skip'): ").strip()
    
    if user_path.lower() != 'skip' and os.path.exists(user_path):
        # Add to path
        dir_path = os.path.dirname(user_path)
        sys.path.append(dir_path)
        print(f"✅ Added {dir_path} to Python path")
        
        # Try to import
        lib_name = os.path.basename(user_path).replace('.py', '')
        print(f"Attempting to import {lib_name}...")
        
        try:
            module = __import__(lib_name)
            print(f"✅ Successfully imported {lib_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to import: {e}")
            return False
    else:
        print("⚠️ Continuing with simulation mode")
        return False

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("🤖 REAL YAHBOOM GRIPPER CONTROL SYSTEM")
    print("="*70)
    
    # Check installation first
    check_installation()
    
    print("\n" + "="*70)
    print("🚀 Starting main system...")
    print("="*70)
    
    # Create and run system
    system = ObjectDetectionGripper()
    system.run()