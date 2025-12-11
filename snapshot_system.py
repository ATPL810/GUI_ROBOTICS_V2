import os
import time
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

class SnapshotSystem:
    def __init__(self, arm_controller, camera, log_callback=None):
        self.arm = arm_controller
        self.camera = camera
        self.log_callback = log_callback
        self.model = None
        self.output_dir = None
        self.all_snapshots = []
        
        # Tool classes
        self.TOOL_CLASSES = ['Bolt', 'Hammer', 'Measuring Tape', 'Plier', 'Screwdriver', 'Wrench']
        self.TOOL_COLORS = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def load_yolo_model(self):
        """Load YOLO model"""
        model_paths = ['best_best.pt']
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = YOLO(path)
                    model.overrides['conf'] = 0.35
                    model.overrides['iou'] = 0.3
                    model.overrides['agnostic_nms'] = True
                    model.overrides['max_det'] = 6
                    model.overrides['verbose'] = False
                    self.log(f"Model loaded: {path}")
                    return model
                except Exception as e:
                    self.log(f"Failed to load {path}: {e}", "warning")
                    continue
        
        raise Exception("No YOLO model found!")
    
    def create_output_directory(self):
        """Create timestamped directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/snapshots/robot_snapshots_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def capture_frame(self):
        """Capture a single frame from camera"""
        return self.camera.capture_frame()
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if self.model is None:
            self.model = self.load_yolo_model()
        
        try:
            inference_size = 256
            small_frame = cv2.resize(frame, (inference_size, int(inference_size * 0.75)))
            
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
                
                scale_x = frame.shape[1] / inference_size
                scale_y = frame.shape[0] / (inference_size * 0.75)
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
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
                            'center_y': (y1 + y2) / 2
                        })
            
            return detections
            
        except Exception as e:
            self.log(f"Detection error: {e}", "error")
            return []
    
    def take_snapshots_sequence(self):
        """Run the complete automatic snapshot sequence"""
        self.log("🚀 Starting automatic snapshot sequence...", "success")
        self.output_dir = self.create_output_directory()
        self.log(f"Output directory: {self.output_dir}")
        
        try:
            # Step 1: Initial Position
            self.log("\n🔹 STEP 1: INITIAL POSITION", "info")
            self.wait_for_stabilization(3, "Initial Position")
            snapshot1 = self.capture_and_save("initial_position", self.arm.INITIAL_POSITION[self.arm.SERVO_BASE])
            
            # Step 2: Second Position
            self.log("\n🔹 STEP 2: MOVING TO SECOND POSITION", "info")
            self.arm.go_to_second_position()
            self.wait_for_stabilization(3, "Second Position")
            snapshot2 = self.capture_and_save("second_position", self.arm.SECOND_POSITION[self.arm.SERVO_BASE])
            
            # Step 3: Third Position
            self.log("\n🔹 STEP 3: MOVING TO THIRD POSITION", "info")
            self.arm.go_to_third_position()
            self.wait_for_stabilization(3, "Third Position")
            snapshot3 = self.capture_and_save("third_position", self.arm.THIRD_POSITION[self.arm.SERVO_BASE])
            
            # Step 4: Return to Initial
            self.log("\n🔹 STEP 4: RETURNING TO INITIAL POSITION", "info")
            time.sleep(3)
            self.arm.go_to_initial_position()
            
            # Step 5: Finalize
            self.save_summary_report()
            
            self.log("✅ Automatic snapshot sequence COMPLETE!", "success")
            total_detections = sum(len(snap['detections']) for snap in self.all_snapshots)
            self.log(f"Snapshots taken: {len(self.all_snapshots)}", "info")
            self.log(f"Total objects detected: {total_detections}", "info")
            
            return self.output_dir
            
        except Exception as e:
            self.log(f"❌ Error during snapshot sequence: {e}", "error")
            raise
    
    def wait_for_stabilization(self, seconds, position_name):
        """Wait for robot to stabilize"""
        self.log(f"Waiting {seconds} seconds for {position_name} stabilization...")
        for i in range(seconds, 0, -1):
            self.log(f"   {i}...", "info")
            time.sleep(1)
    
    def capture_and_save(self, position_name, base_angle):
        """Capture and save a snapshot"""
        self.log(f"Capturing snapshot: {position_name}")
        
        time.sleep(0.5)
        frame = self.capture_frame()
        if frame is None:
            self.log(f"Failed to capture frame at {position_name}", "error")
            return None
        
        detections = self.detect_objects(frame)
        
        if detections:
            self.log(f"Found {len(detections)} objects:", "success")
            for det in detections:
                self.log(f"  • {det['class_name']}: {det['confidence_percentage']}", "info")
        else:
            self.log("No objects detected", "warning")
        
        # Annotate and save frame
        annotated_frame = self.annotate_frame(frame, detections, position_name, base_angle)
        filename = f"{self.output_dir}/{position_name}.jpg"
        cv2.imwrite(filename, annotated_frame)
        self.log(f"Saved snapshot: {filename}")
        
        # Save detection data
        self.save_detection_data(position_name, detections, filename, base_angle)
        
        snapshot_info = {
            'position_name': position_name,
            'base_angle': base_angle,
            'filename': filename,
            'detections': detections,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.all_snapshots.append(snapshot_info)
        
        return snapshot_info
    
    def annotate_frame(self, frame, detections, position_name, base_angle):
        """Annotate frame with detections"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence_percentage']
            color = self.TOOL_COLORS[det['class_id'] % len(self.TOOL_COLORS)]
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(annotated, 
                         (x1, y1 - label_height - 10),
                         (x1 + label_width, y1),
                         color, -1)
            
            cv2.putText(annotated, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add position info
        position_text = f"Position: {position_name} (Base: {base_angle}°)"
        cv2.putText(annotated, position_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        detection_text = f"Objects detected: {len(detections)}"
        cv2.putText(annotated, detection_text,
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return annotated
    
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
                    f.write("-" * 30 + "\n")
        
        self.log(f"Saved detection report: {txt_filename}")
    
    def save_summary_report(self):
        """Save summary report of all snapshots"""
        summary_filename = f"{self.output_dir}/summary_report.txt"
        
        with open(summary_filename, 'w') as f:
            f.write("ROBOT ARM AUTOMATIC SNAPSHOT SYSTEM - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total snapshots taken: {len(self.all_snapshots)}\n")
            f.write("=" * 70 + "\n\n")
            
            total_detections = sum(len(snap['detections']) for snap in self.all_snapshots)
            f.write(f"TOTAL OBJECTS DETECTED: {total_detections}\n\n")
            
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
        
        self.log(f"Saved summary report: {summary_filename}")