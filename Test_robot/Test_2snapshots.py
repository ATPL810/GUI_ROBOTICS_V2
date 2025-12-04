"""
AUTOMATIC ROBOTIC ARM SNAPSHOT SYSTEM
Takes 2 snapshots from fixed positions and saves detections
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from Arm_Lib import Arm_Device  # Make sure Arm_Lib is installed

print("🤖 AUTOMATIC ROBOTIC ARM SNAPSHOT SYSTEM")
print("=" * 70)

# ============================================
# ROBOT ARM CONTROLLER
# ============================================
class RobotArmController:
    def __init__(self):
        """Initialize arm with exact specified angles"""
        print("🤖 Initializing robot arm...")
        
        try:
            # Initialize Arm_Device
            self.arm = Arm_Device()
            time.sleep(2)  # Wait for arm to initialize
            
            # Servo mapping
            self.SERVO_BASE = 1       # Servo 1: Base
            self.SERVO_SHOULDER = 2   # Servo 2: Shoulder  
            self.SERVO_ELBOW = 3      # Servo 3: Elbow
            self.SERVO_WRIST = 4      # Servo 4: Wrist
            self.SERVO_WRIST_ROT = 5  # Servo 5: Wrist rotation
            self.SERVO_GRIPPER = 6    # Servo 6: Gripper
            
            # EXACT INITIAL POSITION AS SPECIFIED (DO NOT ALTER)
            self.INITIAL_POSITION = {
                self.SERVO_BASE: 90,      # 90 degrees
                self.SERVO_SHOULDER: 105, # 115 degrees
                self.SERVO_ELBOW: 45,     # 45 degrees
                self.SERVO_WRIST: -35,    # -35 degrees (converted to servo range)
                self.SERVO_WRIST_ROT: 90, # 90 degrees
                self.SERVO_GRIPPER: 90    # 90 degrees
            }
            
            # Second position: base moves 30 degrees to robot's right
            self.SECOND_POSITION = {
                self.SERVO_BASE: 40,     # 90 + 30 = 120 degrees
                self.SERVO_SHOULDER: 105, # Same
                self.SERVO_ELBOW: 45,     # Same
                self.SERVO_WRIST: -35,    # Same
                self.SERVO_WRIST_ROT: 90, # Same
                self.SERVO_GRIPPER: 90    # Same
            }

            # third position: base moves 1 degrees to robot's right
            self.THIRD_POSITION = {
                self.SERVO_BASE: 1,     
                self.SERVO_SHOULDER: 105, # Same
                self.SERVO_ELBOW: 45,     # Same
                self.SERVO_WRIST: -35,    # Same
                self.SERVO_WRIST_ROT: 90, # Same
                self.SERVO_GRIPPER: 90    # Same
            }
            
            # Move to initial position
            print("📸 Moving to initial position...")
            self.go_to_initial_position()
            
            print("✅ Robot arm initialized successfully")
            print(f"   Initial position: {self.INITIAL_POSITION}")
            
        except Exception as e:
            print(f"❌ Failed to initialize robot arm: {e}")
            raise
    
    def convert_angle(self, angle):
        """Convert negative angles to servo range (0-180)"""
        if angle < 0:
            return 180 + angle  # -35 becomes 145
        return angle
    
    def go_to_initial_position(self):
        """Move to exact initial position"""
        angles_dict = self.INITIAL_POSITION.copy()
        # Convert negative wrist angle
        angles_dict[self.SERVO_WRIST] = (angles_dict[self.SERVO_WRIST])
        
        # Move all servos at once
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000  # Move time in ms
        )
        time.sleep(2.5)  # Wait for movement to complete
        print("   ✅ At initial position")
    
    def go_to_second_position(self):
        """Move to second position (base at 40 degrees)"""
        angles_dict = self.SECOND_POSITION.copy()
        # Convert negative wrist angle
        angles_dict[self.SERVO_WRIST] = (angles_dict[self.SERVO_WRIST])
        
        # Move all servos at once
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000  # Move time in ms
        )
        time.sleep(2.5)  # Wait for movement to complete
        print("   ✅ At second position (base at 40°)")
    
    def get_current_position_name(self, servo1_angle):
        """Get position name based on base servo angle"""
        if servo1_angle == self.INITIAL_POSITION[self.SERVO_BASE]:
            return "initial_position"
        elif servo1_angle == self.SECOND_POSITION[self.SERVO_BASE]:
            return "second_position"
        return "unknown_position"

# ============================================
# CAMERA AND DETECTION SYSTEM
# ============================================
class CameraDetectionSystem:
    def __init__(self):
        """Initialize camera and YOLO model"""
        print("📷 Setting up camera and detection system...")
        
        # Setup camera
        self.cap = self.setup_camera()
        
        # Load YOLO model
        self.model = self.load_yolo_model()
        
        # Tool classes
        self.TOOL_CLASSES = ['Bolt', 'Hammer', 'Measuring Tape', 'Plier', 'Screwdriver', 'Wrench']
        self.TOOL_COLORS = [
            (0, 255, 0),    # Green - Bolt
            (0, 0, 255),    # Red - Hammer  
            (255, 0, 0),    # Blue - Measuring Tape
            (255, 255, 0),  # Cyan - Plier
            (255, 0, 255),  # Magenta - Screwdriver
            (0, 255, 255)   # Yellow - Wrench
        ]
        
        print("✅ Camera and detection system ready")
    
    def setup_camera(self):
        """Setup camera for maximum FPS"""
        print("   Initializing camera...")
        
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"   ✅ Found camera at index {i}")
                
                # Optimize camera settings
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                cap.set(cv2.CAP_PROP_EXPOSURE, 100)
                
                return cap
        
        raise Exception("No camera found!")
    
    def load_yolo_model(self):
        """Load YOLO model"""
        print("   Loading YOLO model...")
        
        model_paths = ['./best_2s.pt']
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = YOLO(path)
                    model.overrides['conf'] = 0.35
                    model.overrides['iou'] = 0.3
                    model.overrides['agnostic_nms'] = True
                    model.overrides['max_det'] = 6
                    model.overrides['verbose'] = False
                    print(f"   ✅ Model loaded: {path}")
                    return model
                except Exception as e:
                    print(f"   ⚠️ Failed to load {path}: {e}")
                    continue
        
        raise Exception("No YOLO model found!")
    
    def detect_objects(self, frame):
        """Detect objects in frame and return detections"""
        if self.model is None:
            return []
        
        try:
            # Resize for faster inference
            inference_size = 256
            small_frame = cv2.resize(frame, (inference_size, int(inference_size * 0.75)))
            
            # Run detection
            results = self.model(small_frame, 
                               conf=0.35,
                               iou=0.3,
                               imgsz=inference_size,
                               max_det=6,
                               verbose=False,
                               half=False,
                               device='cpu',
                               agnostic_nms=True)
            
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Scale boxes back to original size
                scale_x = frame.shape[1] / inference_size
                scale_y = frame.shape[0] / (inference_size * 0.75)
                
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
                    confidence = float(confidences[i])
                    
                    if class_id < len(self.TOOL_CLASSES):
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'class_id': int(class_id),
                            'class_name': self.TOOL_CLASSES[class_id],
                            'confidence': confidence,
                            'confidence_percentage': f"{confidence * 100:.1f}%",
                            'center_x': (x1 + x2) / 2,
                            'center_y': (y1 + y2) / 2,
                            'width': x2 - x1,
                            'height': y2 - y1
                        })
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return []
    
    def annotate_frame(self, frame, detections, position_name, base_angle):
        """Annotate frame with detections and position info"""
        annotated = frame.copy()
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence_percentage']
            color = self.TOOL_COLORS[det['class_id'] % len(self.TOOL_COLORS)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with class and confidence
            label = f"{class_name}: {confidence}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background for label
            cv2.rectangle(annotated, 
                         (x1, y1 - label_height - 10),
                         (x1 + label_width, y1),
                         color, -1)
            
            # Draw text
            cv2.putText(annotated, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add position information at top
        position_text = f"Position: {position_name} (Base: {base_angle}°)"
        cv2.putText(annotated, position_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add detection count
        detection_text = f"Objects detected: {len(detections)}"
        cv2.putText(annotated, detection_text,
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, timestamp,
                   (10, annotated.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return annotated
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# ============================================
# MAIN AUTOMATIC SNAPSHOT SYSTEM
# ============================================
class AutomaticSnapshotSystem:
    def __init__(self):
        """Initialize the complete system"""
        print("🚀 Initializing Automatic Snapshot System...")
        
        # Create output directory
        self.output_dir = self.create_output_directory()
        print(f"📁 Output directory: {self.output_dir}")
        
        # Initialize components
        self.arm = RobotArmController()
        self.detector = CameraDetectionSystem()
        
        # Store all snapshots and detections
        self.all_snapshots = []
        
        print("✅ System initialized and ready to run automatically")
        print("=" * 70)
    
    def create_output_directory(self):
        """Create timestamped directory for saving snapshots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"robot_snapshots_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def wait_for_stabilization(self, seconds=3, position_name=""):
        """Wait for robot to stabilize before taking snapshot"""
        print(f"\n⏳ Waiting {seconds} seconds for {position_name} stabilization...")
        for i in range(seconds, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        print("   ✅ Robot stabilized")
    
    def capture_snapshot(self, position_name, base_angle):
        """Capture a snapshot, detect objects, and save results"""
        print(f"\n📸 CAPTURING SNAPSHOT: {position_name}")
        print(f"   Base servo angle: {base_angle}°")
        
        # Clear camera buffer by reading and discarding a few frames
        print("   Clearing camera buffer...")
        for _ in range(3):
            self.detector.cap.read()
            time.sleep(0.1)
        
        # Small delay to ensure arm has completely stopped moving
        time.sleep(0.5)
        
        # Capture frame
        ret, frame = self.detector.cap.read()
        if not ret:
            print(f"❌ Failed to capture frame at {position_name}")
            return None
        
        # Mirror the frame (like a mirror view)
        frame = cv2.flip(frame, 1)
        
        # Detect objects
        print("   Detecting objects...")
        detections = self.detector.detect_objects(frame)
        
        if detections:
            print(f"   ✅ Found {len(detections)} objects:")
            for det in detections:
                print(f"      • {det['class_name']}: {det['confidence_percentage']}")
        else:
            print("   ⚠️ No objects detected")
        
        # Annotate frame
        annotated_frame = self.detector.annotate_frame(frame, detections, position_name, base_angle)
        
        # Save snapshot
        filename = f"{self.output_dir}/{position_name}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"   💾 Saved snapshot: {filename}")
        
        # Save detection data to text file
        self.save_detection_data(position_name, detections, filename, base_angle)
        
        # Store snapshot info
        snapshot_info = {
            'position_name': position_name,
            'base_angle': base_angle,
            'filename': filename,
            'detections': detections,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.all_snapshots.append(snapshot_info)
        
        return snapshot_info
    
    def save_detection_data(self, position_name, detections, image_filename, base_angle):
        """Save detection information to text file"""
        txt_filename = f"{self.output_dir}/{position_name}_detections.txt"
        
        with open(txt_filename, 'w') as f:
            f.write(f"ROBOT ARM SNAPSHOT DETECTION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Position: {position_name}\n")
            f.write(f"Base Servo Angle: {base_angle}°\n")
            f.write(f"Image File: {os.path.basename(image_filename)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Objects Detected: {len(detections)}\n")
            f.write("=" * 60 + "\n\n")
            
            if detections:
                f.write("DETECTED OBJECTS:\n")
                f.write("-" * 40 + "\n")
                
                for i, det in enumerate(detections, 1):
                    f.write(f"Object #{i}:\n")
                    f.write(f"  Class: {det['class_name']}\n")
                    f.write(f"  Confidence: {det['confidence_percentage']}\n")
                    f.write(f"  Bounding Box: {det['bbox']}\n")
                    f.write(f"  Center Coordinates: ({det['center_x']:.1f}, {det['center_y']:.1f})\n")
                    f.write(f"  Size: {det['width']}x{det['height']} pixels\n")
                    f.write("-" * 30 + "\n")
            else:
                f.write("No objects detected in this snapshot.\n")
        
        print(f"   📝 Saved detection report: {txt_filename}")
    
    def save_summary_report(self):
        """Save summary report of all snapshots"""
        summary_filename = f"{self.output_dir}/summary_report.txt"
        
        with open(summary_filename, 'w') as f:
            f.write("ROBOT ARM AUTOMATIC SNAPSHOT SYSTEM - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total snapshots taken: {len(self.all_snapshots)}\n")
            f.write("=" * 70 + "\n\n")
            
            # Count all detections
            total_detections = sum(len(snap['detections']) for snap in self.all_snapshots)
            f.write(f"TOTAL OBJECTS DETECTED: {total_detections}\n\n")
            
            # Summary by position
            f.write("SNAPSHOT SUMMARY BY POSITION:\n")
            f.write("-" * 50 + "\n")
            
            for snap in self.all_snapshots:
                f.write(f"\n{snap['position_name'].upper()} (Base: {snap['base_angle']}°):\n")
                f.write(f"  Time: {snap['timestamp']}\n")
                f.write(f"  Image: {os.path.basename(snap['filename'])}\n")
                f.write(f"  Objects detected: {len(snap['detections'])}\n")
                
                if snap['detections']:
                    f.write("  Detected objects:\n")
                    for det in snap['detections']:
                        f.write(f"    • {det['class_name']}: {det['confidence_percentage']}\n")
                else:
                    f.write("  No objects detected\n")
            
            # Count by tool class
            f.write("\n" + "=" * 70 + "\n")
            f.write("DETECTION STATISTICS BY TOOL TYPE:\n")
            f.write("-" * 40 + "\n")
            
            tool_counts = {}
            for snap in self.all_snapshots:
                for det in snap['detections']:
                    tool_name = det['class_name']
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            
            if tool_counts:
                for tool_name, count in sorted(tool_counts.items()):
                    f.write(f"{tool_name}: {count} detection(s)\n")
            else:
                f.write("No tools detected in any snapshot.\n")
        
        print(f"\n📋 Saved summary report: {summary_filename}")
    
    def run_automatic_sequence(self):
        """Run the complete automatic sequence"""
        print("\n" + "=" * 70)
        print("🚀 STARTING AUTOMATIC SNAPSHOT SEQUENCE")
        print("=" * 70)
        
        try:
            # ============================================
            # STEP 1: INITIAL POSITION SNAPSHOT
            # ============================================
            print("\n🔹 STEP 1: INITIAL POSITION")
            
            # Robot is already at initial position from initialization
            initial_base_angle = self.arm.INITIAL_POSITION[self.arm.SERVO_BASE]
            
            # Wait 3 seconds for stabilization
            self.wait_for_stabilization(3, "Initial Position")
            
            # Capture first snapshot
            snapshot1 = self.capture_snapshot(
                position_name="initial_position",
                base_angle=initial_base_angle
            )
            
            # ============================================
            # STEP 2: MOVE TO SECOND POSITION
            # ============================================
            print("\n🔹 STEP 2: MOVING TO SECOND POSITION")
            
            # Move base servo +30 degrees
            print("   Moving base servo +30 degrees...")
            self.arm.go_to_second_position()
            
            second_base_angle = self.arm.SECOND_POSITION[self.arm.SERVO_BASE]
            
            # Wait 3 seconds for stabilization
            self.wait_for_stabilization(3, "Second Position")
            
            # Capture second snapshot
            snapshot2 = self.capture_snapshot(
                position_name="second_position",
                base_angle=second_base_angle
            )
            
            # ============================================
            # STEP 3: RETURN TO INITIAL POSITION
            # ============================================
            print("\n🔹 STEP 3: RETURNING TO INITIAL POSITION")
            
            # Wait 3 seconds before moving back
            print("   Waiting 3 seconds before returning...")
            time.sleep(3)
            
            # Return to initial position
            print("   Returning to initial position...")
            self.arm.go_to_initial_position()
            
            # ============================================
            # STEP 4: SAVE REPORTS AND CLEANUP
            # ============================================
            print("\n🔹 STEP 4: FINALIZING")
            
            # Save summary report
            self.save_summary_report()
            
            # Release resources
            self.detector.release()
            
            # Final status
            print("\n" + "=" * 70)
            print("✅ AUTOMATIC SNAPSHOT SEQUENCE COMPLETE")
            print("=" * 70)
            
            total_detections = sum(len(snap['detections']) for snap in self.all_snapshots)
            print(f"\n📊 FINAL STATISTICS:")
            print(f"   Snapshots taken: {len(self.all_snapshots)}")
            print(f"   Total objects detected: {total_detections}")
            print(f"   All files saved in: {self.output_dir}")
            
            # Show what was detected
            print(f"\n📋 DETECTED OBJECTS SUMMARY:")
            for snap in self.all_snapshots:
                print(f"\n   {snap['position_name'].upper()} (Base: {snap['base_angle']}°):")
                if snap['detections']:
                    for det in snap['detections']:
                        print(f"     • {det['class_name']}: {det['confidence_percentage']}")
                else:
                    print(f"     No objects detected")
            
            print(f"\n🎉 Program completed successfully!")
            
        except Exception as e:
            print(f"\n❌ ERROR during automatic sequence: {e}")
            raise
        
        finally:
            # Always ensure robot returns to initial position
            try:
                print("\n🔄 Ensuring robot returns to initial position...")
                self.arm.go_to_initial_position()
                print("✅ Robot safely returned to initial position")
            except:
                print("⚠️ Could not return robot to initial position")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("🤖 ROBOTIC ARM AUTOMATIC SNAPSHOT SYSTEM")
    print("=" * 70)
    print("This program will run automatically:")
    print("1. Start at initial position (servo1:90°, servo2:115°, servo3:45°, servo4:-35°, servo5:90°, servo6:90°)")
    print("2. Wait 3 seconds for stabilization")
    print("3. Take snapshot 1, detect and classify objects")
    print("4. Move servo1 +30° to right position")
    print("5. Wait 3 seconds for stabilization")
    print("6. Take snapshot 2, detect and classify objects")
    print("7. Wait 3 seconds")
    print("8. Return to initial position")
    print("9. Save all snapshots and reports")
    print("=" * 70)
    
    # Countdown to start
    print("\n⏱️ Starting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    # Create and run system
    try:
        system = AutomaticSnapshotSystem()
        system.run_automatic_sequence()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Program error: {e}")
    finally:
        print("\n✅ Program ended")