"""
COMPLETE ROBOTIC ARM CONTROL SYSTEM WITH YOLO11
Manages detection, grabbing, and displacement with priority interruption
"""

import cv2
import time
import numpy as np
import math
from ultralytics import YOLO
import threading
from collections import deque, defaultdict

print("🤖 ROBOTIC ARM CONTROL SYSTEM WITH PRIORITY MANAGEMENT")
print("=" * 60)

# ============================================
# ROBOT ARM CONTROL FUNCTIONS (Simulated)
# ============================================
class RoboticArmSimulator:
    def __init__(self):
        self.current_position = (0, 0, 0)  # x, y, z coordinates
        self.gripper_open = True
        self.is_moving = False
        self.current_rotation = 0  # degrees
        print("🤖 Robotic Arm Simulator Initialized")
    
    def open_gripper(self):
        """Open the gripper"""
        if not self.gripper_open:
            print("    [GRIPPER] Opening gripper...")
            time.sleep(0.5)  # Simulate delay
            self.gripper_open = True
            print("    [GRIPPER] Gripper opened")
        return True
    
    def close_gripper(self):
        """Close the gripper"""
        if self.gripper_open:
            print("    [GRIPPER] Closing gripper...")
            time.sleep(0.5)  # Simulate delay
            self.gripper_open = False
            print("    [GRIPPER] Gripper closed")
        return True
    
    def move_to(self, x, y, z):
        """Move arm to specific coordinates"""
        print(f"    [MOVE] Moving to ({x}, {y}, {z})...")
        self.is_moving = True
        time.sleep(1.0)  # Simulate movement time
        self.current_position = (x, y, z)
        self.is_moving = False
        print(f"    [MOVE] Arrived at ({x}, {y}, {z})")
        return True
    
    def rotate(self, degrees):
        """Rotate the arm"""
        print(f"    [ROTATE] Rotating {degrees}°...")
        time.sleep(0.8)  # Simulate rotation time
        self.current_rotation = (self.current_rotation + degrees) % 360
        print(f"    [ROTATE] Now at {self.current_rotation}°")
        return True
    
    def lift(self, height):
        """Lift the object"""
        print(f"    [LIFT] Lifting to height {height}...")
        x, y, z = self.current_position
        self.move_to(x, y, height)
        return True
    
    def get_position(self):
        """Get current position"""
        return self.current_position

# ============================================
# COORDINATE MAPPING SYSTEM
# ============================================
class CoordinateMapper:
    def __init__(self, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        # Define workspace boundaries (calibrate these for your setup)
        self.workspace_x_range = (-0.3, 0.3)  # meters
        self.workspace_y_range = (0.1, 0.5)   # meters
        self.drop_zone = (0.2, 0.3, 0.05)     # x, y, z for drop location
        
        print("📍 Coordinate Mapper Initialized")
    
    def pixel_to_robot_coords(self, pixel_x, pixel_y, bbox_width, bbox_height):
        """
        Convert pixel coordinates to robot coordinates
        pixel_x, pixel_y: Center of bounding box in pixels
        Returns: (x, y, z) in robot coordinate system
        """
        # Normalize pixel coordinates to 0-1 range
        norm_x = pixel_x / self.camera_width
        norm_y = pixel_y / self.camera_height
        
        # Map to robot workspace
        robot_x = self.workspace_x_range[0] + norm_x * (self.workspace_x_range[1] - self.workspace_x_range[0])
        robot_y = self.workspace_y_range[0] + (1 - norm_y) * (self.workspace_y_range[1] - self.workspace_y_range[0])
        
        # Z coordinate based on object size (simplified)
        # Smaller objects need lower Z, larger objects need higher Z
        object_size = bbox_width * bbox_height
        robot_z = 0.02 + (object_size / (self.camera_width * self.camera_height)) * 0.1
        
        return (robot_x, robot_y, robot_z)
    
    def get_approach_height(self, target_z):
        """Get approach height above object"""
        return target_z + 0.05  # 5cm above object
    
    def get_drop_location(self, class_id):
        """Get drop location for specific tool class"""
        # Different drop locations for each tool type
        drop_offsets = {
            0: (0.0, 0.0, 0.0),      # Bolt
            1: (0.05, 0.0, 0.0),     # Hammer
            2: (0.1, 0.0, 0.0),      # Measuring Tape
            3: (0.0, 0.05, 0.0),     # Plier
            4: (0.05, 0.05, 0.0),    # Screwdriver
            5: (0.1, 0.05, 0.0)      # Wrench
        }
        
        offset = drop_offsets.get(class_id, (0, 0, 0))
        drop_x = self.drop_zone[0] + offset[0]
        drop_y = self.drop_zone[1] + offset[1]
        drop_z = self.drop_zone[2] + offset[2]
        
        return (drop_x, drop_y, drop_z)

# ============================================
# DISPLACEMENT MANAGER WITH PRIORITY SYSTEM
# ============================================
class DisplacementManager:
    def __init__(self):
        self.arm = RoboticArmSimulator()
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
        self.displacement_queue = deque()
        self.is_displacing = False
        self.paused_class = None
        
        # Detection buffer for stability
        self.detection_buffer = deque(maxlen=10)
        
        print("📋 Displacement Manager Initialized")
        print(f"🔧 Will displace {self.target_displace_count} of each tool")
    
    def add_detection(self, detection):
        """Add a new detection to buffer"""
        self.detection_buffer.append(detection)
    
    def get_stable_detection(self):
        """Get most consistently detected object (voting system)"""
        if not self.detection_buffer:
            return None
        
        # Count occurrences of each class
        class_votes = defaultdict(int)
        for det in self.detection_buffer:
            class_votes[det['class_id']] += 1
        
        # Find class with most votes
        if class_votes:
            best_class = max(class_votes.items(), key=lambda x: x[1])[0]
            # Get most recent detection of this class
            for det in reversed(self.detection_buffer):
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
        
        # Check if this class is already fully displaced
        if self.displaced_counts[detected_class_id] >= self.target_displace_count:
            return False, f"Class {self.class_names[detected_class_id]} already fully displaced"
        
        # If we're currently displacing a different class
        if self.current_target_class != detected_class_id:
            # Check if current class still needs more displacement
            if self.displaced_counts[self.current_target_class] >= self.target_displace_count:
                return True, f"Current class {self.class_names[self.current_target_class]} is done"
            else:
                # We could interrupt if new class has higher priority
                # For now, we interrupt for any new class that's not fully displaced
                return True, f"Found {self.class_names[detected_class_id]} during displacement of {self.class_names[self.current_target_class]}"
        
        return False, "Continuing with current class"
    
    def execute_displacement(self, detection):
        """Execute the complete displacement sequence"""
        self.is_displacing = True
        class_id = detection['class_id']
        class_name = self.class_names[class_id]
        
        print(f"\n🚀 STARTING DISPLACEMENT: {class_name}")
        print(f"   Confidence: {detection['confidence_percent']:.1f}%")
        
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
        
        print(f"   📍 Object at: {target_pos}")
        print(f"   📍 Drop zone: {drop_pos}")
        
        try:
            # 2. Open gripper (initial state)
            if not self.arm.gripper_open:
                self.arm.open_gripper()
            
            # 3. Move to approach position
            self.arm.move_to(target_pos[0], target_pos[1], approach_height)
            
            # 4. Move down to object
            self.arm.move_to(target_pos[0], target_pos[1], target_pos[2])
            
            # 5. Close gripper
            self.arm.close_gripper()
            time.sleep(0.3)  # Let gripper stabilize
            
            # 6. Lift object
            self.arm.lift(approach_height)
            
            # 7. Rotate 180 degrees
            self.arm.rotate(180)
            
            # 8. Move to drop location
            self.arm.move_to(drop_pos[0], drop_pos[1], approach_height)
            
            # 9. Lower to drop height
            self.arm.move_to(drop_pos[0], drop_pos[1], drop_pos[2])
            
            # 10. Open gripper to release
            self.arm.open_gripper()
            
            # 11. Lift back up
            self.arm.lift(approach_height)
            
            # 12. Rotate back to original orientation
            self.arm.rotate(180)
            
            # Update displacement count
            self.displaced_counts[class_id] += 1
            print(f"   ✅ Successfully displaced {class_name}")
            print(f"   📊 Total {class_name}s displaced: {self.displaced_counts[class_id]}/{self.target_displace_count}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Displacement failed: {e}")
            # Emergency open gripper
            self.arm.open_gripper()
            return False
        
        finally:
            self.is_displacing = False
    
    def check_completion(self):
        """Check if all displacement tasks are complete"""
        all_done = all(
            self.displaced_counts[class_id] >= self.target_displace_count 
            for class_id in self.class_names.keys()
        )
        return all_done
    
    def print_status(self):
        """Print current displacement status"""
        print("\n" + "=" * 50)
        print("📊 DISPLACEMENT STATUS")
        print("=" * 50)
        for class_id, class_name in self.class_names.items():
            count = self.displaced_counts[class_id]
            status = "✅ DONE" if count >= self.target_displace_count else f"🔄 {count}/{self.target_displace_count}"
            print(f"  {class_name}: {status}")
        
        if self.current_target_class is not None:
            print(f"\n🎯 Currently displacing: {self.class_names[self.current_target_class]}")
        
        if self.paused_class is not None:
            print(f"⏸️  Paused class: {self.class_names[self.paused_class]}")
        
        print("=" * 50)

# ============================================
# MAIN DETECTION AND CONTROL SYSTEM
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
        
        # Initialize displacement manager
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
        
        print("\n✅ System initialized and ready!")
        print("Press 's' to show/hide status")
        print("Press 'q' to quit")
    
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
                               conf=0.35,
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
                
                # Check for displacement opportunities
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
        # Get stable detection
        detection = self.displacement_mgr.get_stable_detection()
        if detection is None:
            return
        
        class_id = detection['class_id']
        class_name = self.displacement_mgr.class_names[class_id]
        
        # Check if this class is already done
        if self.displacement_mgr.displaced_counts[class_id] >= self.displacement_mgr.target_displace_count:
            return
        
        # Check if we should interrupt current displacement
        should_interrupt, reason = self.displacement_mgr.should_interrupt(class_id)
        
        if should_interrupt:
            print(f"\n🚨 INTERRUPTION REQUESTED: {reason}")
            print(f"   Detected: {class_name} ({detection['confidence_percent']:.1f}%)")
            
            # Store current state and pause
            if self.displacement_mgr.current_target_class is not None:
                self.displacement_mgr.paused_class = self.displacement_mgr.current_target_class
                print(f"   ⏸️ Pausing: {self.displacement_mgr.class_names[self.displacement_mgr.paused_class]}")
            
            # Start displacement of new class
            self.displacement_mgr.current_target_class = class_id
            self.displacement_mgr.execute_displacement(detection)
            
            # Resume paused class if any
            if self.displacement_mgr.paused_class is not None:
                print(f"\n🔄 RESUMING: {self.displacement_mgr.class_names[self.displacement_mgr.paused_class]}")
                self.displacement_mgr.current_target_class = self.displacement_mgr.paused_class
                self.displacement_mgr.paused_class = None
                
        elif not self.displacement_mgr.is_displacing:
            # Start new displacement if not currently displacing
            if self.displacement_mgr.current_target_class is None or \
               self.displacement_mgr.current_target_class == class_id:
                
                self.displacement_mgr.current_target_class = class_id
                print(f"\n🎯 Starting displacement of {class_name}...")
                self.displacement_mgr.execute_displacement(detection)
    
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
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with percentage
            label = f"{det['class_name']}: {confidence_percent:.1f}%"
            
            # Text background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            
            cv2.putText(frame, label, 
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        return frame
    
    def draw_status(self, frame):
        """Draw system status on frame"""
        if not self.show_status:
            return frame
        
        y_offset = 30
        line_height = 20
        
        # Background for status
        cv2.rectangle(frame, (5, 5), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (300, 150), (255, 255, 255), 1)
        
        # System status
        cv2.putText(frame, "🤖 ROBOT STATUS", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += line_height
        
        # Displacement status
        status_color = (0, 255, 0) if not self.displacement_mgr.is_displacing else (0, 165, 255)
        status_text = "IDLE" if not self.displacement_mgr.is_displacing else "DISPLACING"
        cv2.putText(frame, f"Arm: {status_text}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        y_offset += line_height
        
        # Current target
        if self.displacement_mgr.current_target_class is not None:
            class_name = self.displacement_mgr.class_names[self.displacement_mgr.current_target_class]
            cv2.putText(frame, f"Target: {class_name}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += line_height
        
        # Displacement counts
        cv2.putText(frame, "Displaced:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        y_offset += line_height
        
        for class_id in range(3):  # First row
            class_name = self.displacement_mgr.class_names[class_id][:4]
            count = self.displacement_mgr.displaced_counts[class_id]
            total = self.displacement_mgr.target_displace_count
            color = (0, 255, 0) if count >= total else (255, 255, 255)
            cv2.putText(frame, f"{class_name}:{count}/{total}", 
                       (10 + 70 * (class_id % 3), y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        y_offset += line_height
        
        for class_id in range(3, 6):  # Second row
            class_name = self.displacement_mgr.class_names[class_id][:4]
            count = self.displacement_mgr.displaced_counts[class_id]
            total = self.displacement_mgr.target_displace_count
            color = (0, 255, 0) if count >= total else (255, 255, 255)
            cv2.putText(frame, f"{class_name}:{count}/{total}", 
                       (10 + 70 * ((class_id-3) % 3), y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("\n🎬 Starting robotic vision system...")
        time.sleep(2)
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Can't receive frame")
                    break
                
                # Mirror view
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
                
                # Draw detections
                display_frame = frame.copy()
                display_frame = self.draw_detections(display_frame, detections)
                
                # Draw status
                if self.show_fps:
                    display_frame = self.draw_status(display_frame)
                
                # Display
                cv2.imshow('Robotic Vision System - Displacement Control', display_frame)
                
                # Check for completion
                if self.displacement_mgr.check_completion():
                    print("\n" + "=" * 60)
                    print("🎉 ALL DISPLACEMENT TASKS COMPLETED!")
                    print("=" * 60)
                    self.displacement_mgr.print_status()
                    break
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('s'):
                    self.show_status = not self.show_status
                    print(f"📊 Status display: {'ON' if self.show_status else 'OFF'}")
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                elif key == ord('+'):
                    self.detection_interval = max(2, self.detection_interval - 1)
                    print(f"⚡ Detection frequency: 1/{self.detection_interval}")
                elif key == ord('-'):
                    self.detection_interval += 1
                    print(f"🐢 Detection frequency: 1/{self.detection_interval}")
                elif ord('1') <= key <= ord('6'):
                    tool_idx = key - ord('1')
                    if self.highlight_tool == tool_idx:
                        self.highlight_tool = -1
                        print(f"🎯 Showing ALL tools")
                    else:
                        self.highlight_tool = tool_idx
                        print(f"🎯 Highlighting: {self.displacement_mgr.class_names[tool_idx]}")
                elif key == ord('p'):
                    self.displacement_mgr.print_status()
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ System cleanup complete")
        print("\n📊 FINAL DISPLACEMENT REPORT:")
        self.displacement_mgr.print_status()

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    # Initialize system
    system = RoboticVisionSystem()
    
    # Run main loop
    system.run()