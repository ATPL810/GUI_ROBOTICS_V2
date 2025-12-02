"""
SIMPLE GRIPPER CONTROL WITH YOLO DETECTION
When object detected → Open/Close Gripper
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import threading
from collections import deque

print("🤖 SIMPLE GRIPPER CONTROL WITH YOLO")
print("=" * 60)

# ============================================
# SIMPLE GRIPPER CONTROL (Open/Close Only)
# ============================================
class SimpleGripperControl:
    def __init__(self):
        """Simple gripper control via serial"""
        self.connected = False
        self.ser = None
        self.gripper_open = True
        
        # Gripper servo (usually servo 5 on Yahboom)
        self.SERVO_GRIPPER = 5
        
        # Calibrate these for your arm!
        self.GRIPPER_OPEN_PULSE = 700    # Pulse width for OPEN (500-2500)
        self.GRIPPER_CLOSED_PULSE = 1300 # Pulse width for CLOSED
        
        # Movement timing
        self.gripper_speed = 800  # ms for gripper movement
        
        # State
        self.last_action_time = 0
        self.action_cooldown = 2.0  # Seconds between actions
        self.is_acting = False
        
        self.initialize_serial()
    
    def initialize_serial(self):
        """Initialize serial connection"""
        try:
            import serial
            import serial.tools.list_ports
            
            # Find Yahboom port
            ports = list(serial.tools.list_ports.comports())
            if not ports:
                print("❌ No serial ports found!")
                return
            
            print("🔌 Available ports:")
            for p in ports:
                print(f"  - {p.device}: {p.description}")
            
            # Try to find Yahboom port
            target_port = None
            for p in ports:
                if 'USB' in p.description or 'ACM' in p.device or 'ttyUSB' in p.device:
                    target_port = p.device
                    break
            
            if target_port is None:
                target_port = ports[0].device
            
            print(f"🔗 Connecting to {target_port}...")
            
            # Open connection
            self.ser = serial.Serial(
                port=target_port,
                baudrate=115200,
                timeout=1
            )
            time.sleep(2)  # Wait for connection
            
            self.connected = True
            print(f"✅ Connected to Yahboom arm")
            
            # Initialize gripper to OPEN
            self.open_gripper()
            
        except Exception as e:
            print(f"❌ Serial connection failed: {e}")
            print("⚠️ Running in SIMULATION mode")
            self.connected = False
    
    def send_command(self, command):
        """Send serial command"""
        if not self.connected or self.ser is None:
            return False
        
        try:
            if not command.endswith('\r\n'):
                command += '\r\n'
            self.ser.write(command.encode())
            time.sleep(0.01)
            return True
        except Exception as e:
            print(f"    [SERIAL ERROR] {e}")
            self.connected = False
            return False
    
    def open_gripper(self):
        """Open the gripper"""
        print("    [GRIPPER] Opening...")
        
        command = f"#{self.SERVO_GRIPPER} P{self.GRIPPER_OPEN_PULSE} T{self.gripper_speed}"
        
        if self.send_command(command):
            time.sleep(self.gripper_speed / 1000)
            self.gripper_open = True
            print("    [GRIPPER] Open")
            return True
        else:
            # Simulation
            print("    [SIM] Gripper opened")
            time.sleep(0.8)
            self.gripper_open = True
            return False
    
    def close_gripper(self):
        """Close the gripper"""
        print("    [GRIPPER] Closing...")
        
        command = f"#{self.SERVO_GRIPPER} P{self.GRIPPER_CLOSED_PULSE} T{self.gripper_speed}"
        
        if self.send_command(command):
            time.sleep(self.gripper_speed / 1000)
            self.gripper_open = False
            print("    [GRIPPER] Closed")
            return True
        else:
            # Simulation
            print("    [SIM] Gripper closed")
            time.sleep(0.8)
            self.gripper_open = False
            return False
    
    def execute_grip_sequence(self):
        """Execute open→close→open sequence"""
        if self.is_acting:
            return
        
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return
        
        self.is_acting = True
        
        try:
            # Close gripper
            self.close_gripper()
            time.sleep(0.5)  # Hold closed
            
            # Open gripper
            self.open_gripper()
            
            print("✅ Grip sequence completed")
            
        except Exception as e:
            print(f"❌ Grip sequence failed: {e}")
        
        finally:
            self.is_acting = False
            self.last_action_time = time.time()
    
    def quick_test(self):
        """Quick test of gripper"""
        print("\n🔧 Testing gripper...")
        self.open_gripper()
        time.sleep(1)
        self.close_gripper()
        time.sleep(1)
        self.open_gripper()
        print("✅ Test complete")

# ============================================
# SIMPLE VISION SYSTEM
# ============================================
class SimpleVisionSystem:
    def __init__(self):
        # Setup camera
        self.cap = self.setup_camera()
        if self.cap is None:
            print("❌ ERROR: No camera found!")
            exit()
        
        # Load YOLO
        self.model = self.load_yolo()
        
        # Initialize gripper
        self.gripper = SimpleGripperControl()
        
        # Detection tracking
        self.last_detection_time = 0
        self.detection_cooldown = 3.0  # Min seconds between actions
        self.object_detected = False
        self.current_class = None
        self.confidence = 0
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Display
        self.show_info = True
        
        print("\n✅ System ready!")
        print("Controls:")
        print("  t - Test gripper (open/close)")
        print("  q - Quit")
        print("=" * 60)
    
    def setup_camera(self):
        """Setup camera"""
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"✅ Camera {i} found")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
        return None
    
    def load_yolo(self):
        """Load YOLO model"""
        try:
            model = YOLO('best.pt')
            model.overrides['conf'] = 0.5
            model.overrides['iou'] = 0.3
            model.overrides['max_det'] = 1  # Only care about first detection
            model.overrides['verbose'] = False
            print("✅ YOLO loaded")
            return model
        except Exception as e:
            print(f"⚠️ YOLO failed: {e}")
            return None
    
    def process_frame(self, frame):
        """Process frame for detection"""
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
                
                if len(boxes) > 0:
                    # Get best detection
                    best_idx = np.argmax(confidences)
                    bbox = boxes[best_idx]
                    confidence = confidences[best_idx]
                    class_id = class_ids[best_idx]
                    
                    # Class names
                    class_names = {
                        0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
                        3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
                    }
                    
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                    
                    return True, class_name, confidence*100
                    
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
        
        return False, None, 0
    
    def handle_detection(self, detected, class_name, confidence):
        """Handle object detection - trigger gripper"""
        current_time = time.time()
        
        if detected:
            self.object_detected = True
            self.current_class = class_name
            self.confidence = confidence
            
            # Check if enough time has passed since last action
            if current_time - self.last_detection_time > self.detection_cooldown:
                print(f"\n🎯 OBJECT DETECTED: {class_name} ({confidence:.1f}%)")
                print("   🤖 Triggering gripper sequence...")
                
                # Run gripper in separate thread
                threading.Thread(target=self.gripper.execute_grip_sequence, 
                               daemon=True).start()
                
                self.last_detection_time = current_time
        
        else:
            self.object_detected = False
    
    def draw_display(self, frame, detected, class_name, confidence):
        """Draw information on frame"""
        display = frame.copy()
        
        # Draw detection box if object detected
        if detected:
            # Draw a rectangle around the center
            h, w = frame.shape[:2]
            center_x, center_y = w//2, h//2
            box_size = 100
            
            # Draw pulsing box
            pulse = int(50 * (1 + np.sin(time.time() * 5) * 0.3))
            cv2.rectangle(display, 
                         (center_x - box_size, center_y - box_size),
                         (center_x + box_size, center_y + box_size),
                         (0, 255, 0), pulse//10)
            
            # Draw text
            text = f"{class_name}: {confidence:.1f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            
            cv2.putText(display, text, (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw "ACTIVE" indicator
            if self.gripper.is_acting:
                cv2.putText(display, "GRIPPER ACTIVE", (w//2 - 100, h - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Draw info panel
        if self.show_info:
            # Background
            cv2.rectangle(display, (10, 10), (300, 130), (0, 0, 0, 180), -1)
            cv2.rectangle(display, (10, 10), (300, 130), (255, 255, 255), 1)
            
            y = 35
            line = 25
            
            # Status
            status = "OBJECT DETECTED" if detected else "SEARCHING"
            color = (0, 255, 0) if detected else (255, 255, 0)
            cv2.putText(display, f"Status: {status}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y += line
            
            # FPS
            cv2.putText(display, f"FPS: {self.fps:.1f}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += line
            
            # Arm connection
            arm_status = "CONNECTED" if self.gripper.connected else "SIMULATION"
            arm_color = (0, 255, 0) if self.gripper.connected else (255, 0, 0)
            cv2.putText(display, f"Arm: {arm_status}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, arm_color, 1)
            y += line
            
            # Last action
            time_since = time.time() - self.last_detection_time
            if time_since < 10:
                cv2.putText(display, f"Last action: {time_since:.1f}s ago", 
                           (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        
        return display
    
    def run(self):
        """Main loop"""
        print("\n🎬 Starting simple gripper control...")
        print("📸 Point camera at an object to trigger gripper")
        print("⏱️  Cooldown: 3 seconds between actions")
        time.sleep(1)
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Can't receive frame")
                    break
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Update FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Process detection
                detected, class_name, confidence = self.process_frame(frame)
                
                # Handle detection (trigger gripper)
                self.handle_detection(detected, class_name, confidence)
                
                # Draw display
                display = self.draw_display(frame, detected, class_name, confidence)
                
                # Show frame
                cv2.imshow('Simple Gripper Control - Object Detection', display)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                
                elif key == ord('t'):
                    print("\n🔧 Manual gripper test...")
                    self.gripper.quick_test()
                
                elif key == ord(' '):
                    self.show_info = not self.show_info
                    print(f"ℹ️ Info display: {'ON' if self.show_info else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup"""
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Make sure gripper is open
        if hasattr(self, 'gripper'):
            self.gripper.open_gripper()
        
        print("\n✅ Cleanup complete")
        print("🤖 Gripper should be open")

# ============================================
# CALIBRATION TEST SCRIPT
# ============================================
def calibrate_gripper():
    """First, calibrate your gripper"""
    print("\n" + "="*60)
    print("🔧 GRIPPER CALIBRATION")
    print("="*60)
    
    try:
        import serial
        import serial.tools.list_ports
        
        # Find port
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("❌ No serial ports found!")
            return None, None
        
        print("Available ports:")
        for p in ports:
            print(f"  {p.device}: {p.description}")
        
        # Try to auto-select
        target_port = None
        for p in ports:
            if 'USB' in p.description or 'ACM' in p.device:
                target_port = p.device
                break
        
        if target_port is None:
            target_port = ports[0].device
        
        print(f"\n🔗 Using port: {target_port}")
        print("⚠️ Make sure Yahboom arm is powered ON!")
        
        # Connect
        ser = serial.Serial(target_port, 115200, timeout=1)
        time.sleep(2)
        
        print("✅ Connected!")
        print("\n🎮 Manual calibration controls:")
        print("  o - Open gripper")
        print("  c - Close gripper")
        print("  t - Test sequence (open→close→open)")
        print("  s - Save current pulse values")
        print("  q - Quit calibration")
        
        # Test values
        open_pulse = 700
        closed_pulse = 1300
        
        while True:
            print(f"\nCurrent: OPEN={open_pulse}, CLOSED={closed_pulse}")
            cmd = input("Enter command (o/c/t/s/q): ").strip().lower()
            
            if cmd == 'o':
                ser.write(f"#5 P{open_pulse} T1000\r\n".encode())
                time.sleep(1)
                print("✅ Sent OPEN command")
                
            elif cmd == 'c':
                ser.write(f"#5 P{closed_pulse} T1000\r\n".encode())
                time.sleep(1)
                print("✅ Sent CLOSE command")
                
            elif cmd == 't':
                ser.write(f"#5 P{open_pulse} T1000\r\n".encode())
                time.sleep(1.2)
                ser.write(f"#5 P{closed_pulse} T1000\r\n".encode())
                time.sleep(1.2)
                ser.write(f"#5 P{open_pulse} T1000\r\n".encode())
                time.sleep(1.2)
                print("✅ Test sequence complete")
                
            elif cmd == 's':
                print(f"\n💾 Save these values in the main script:")
                print(f"   self.GRIPPER_OPEN_PULSE = {open_pulse}")
                print(f"   self.GRIPPER_CLOSED_PULSE = {closed_pulse}")
                
            elif cmd == 'q':
                # Open before quitting
                ser.write(f"#5 P{open_pulse} T1000\r\n".encode())
                time.sleep(1)
                ser.close()
                print("✅ Calibration complete")
                return open_pulse, closed_pulse
            
            else:
                # Try to parse pulse value
                try:
                    new_pulse = int(cmd)
                    if 500 <= new_pulse <= 2500:
                        ser.write(f"#5 P{new_pulse} T1000\r\n".encode())
                        time.sleep(1)
                        print(f"✅ Set to {new_pulse}")
                        
                        # Ask which value this is
                        which = input("Is this OPEN or CLOSED position? (o/c): ").lower()
                        if which == 'o':
                            open_pulse = new_pulse
                        elif which == 'c':
                            closed_pulse = new_pulse
                    else:
                        print("❌ Pulse must be 500-2500")
                except:
                    print("❌ Invalid command")
        
    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        return None, None

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("🤖 SIMPLE GRIPPER CONTROL SYSTEM")
    print("="*60)
    
    # Ask if user wants to calibrate first
    calibrate = input("\n🔧 Calibrate gripper first? (y/n): ").strip().lower()
    
    if calibrate == 'y':
        open_pulse, closed_pulse = calibrate_gripper()
        if open_pulse and closed_pulse:
            print(f"\n📝 Update these in SimpleGripperControl class:")
            print(f"   self.GRIPPER_OPEN_PULSE = {open_pulse}")
            print(f"   self.GRIPPER_CLOSED_PULSE = {closed_pulse}")
            input("\nPress Enter to continue after updating the code...")
    
    # Run the main system
    print("\n" + "="*60)
    print("🚀 Starting main system...")
    print("="*60)
    
    system = SimpleVisionSystem()
    system.run()