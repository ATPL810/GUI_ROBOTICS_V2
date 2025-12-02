import cv2
import time
import numpy as np
from ultralytics import YOLO

class SmoothToolDetector:
    def __init__(self):
        print("🚀 Initializing Smooth Tool Detector...")
        
        # Initialize camera with DIRECT access (no threading)
        self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        
        # CRITICAL: Camera settings for MAXIMUM smoothness
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Standard HD
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)           # Match your display
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # Small buffer
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # Disable auto-focus
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual exposure
        
        # Check camera
        if not self.cap.isOpened():
            print("❌ ERROR: Cannot open camera")
            exit()
        
        print("✅ Camera initialized successfully")
        
        # Load YOLO model (but we'll use it LIGHTLY)
        print("📦 Loading YOLO model...")
        self.model = YOLO('best.pt')
        
        # Tool classes and colors
        self.class_names = ['Bolt', 'Hammer', 'Measuring Tape', 'Plier', 'Screwdriver', 'Wrench']
        self.colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red  
            (255, 0, 0),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Yellow
        ]
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Detection frequency (process every N frames)
        self.detect_every = 3  # Start with processing every 3rd frame
        self.frame_counter = 0
        self.last_detections = []
        
        print("✅ Ready! Press 'q' to quit, '+' to increase detection, '-' to decrease")

    def process_frame_simple(self, frame):
        """Process frame with minimal lag"""
        self.frame_counter += 1
        
        # Only run YOLO every N frames (adjustable)
        if self.frame_counter % self.detect_every == 0:
            try:
                # Run YOLO on a SMALLER version of the frame
                small_frame = cv2.resize(frame, (320, 240))
                
                # Quick inference
                results = self.model(small_frame, 
                                    conf=0.4,
                                    imgsz=320,
                                    verbose=False,
                                    max_det=6)
                
                if results and len(results) > 0:
                    self.last_detections = []
                    result = results[0]
                    
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        
                        # Scale back to original size
                        scale_x = frame.shape[1] / 320
                        scale_y = frame.shape[0] / 240
                        
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box
                            
                            # Scale coordinates
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)
                            
                            self.last_detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': float(confidences[i]),
                                'class_id': int(class_ids[i])
                            })
            except Exception as e:
                print(f"Detection error: {e}")
                self.last_detections = []
        
        return frame

    def draw_detections(self, frame, detections):
        """Draw detections on frame (fast)"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']
            
            # Get color
            color = self.colors[class_id % len(self.colors)]
            
            # Draw rectangle (thin for speed)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{self.class_names[class_id]}: {confidence:.1f}"
            cv2.putText(frame, label, (x1, max(y1-10, 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame

    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed >= 1.0:  # Update every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        
        return self.fps

    def run(self):
        """Main loop - SIMPLE and SMOOTH"""
        print("\n🎬 Starting detection loop...")
        print("="*50)
        
        try:
            while True:
                # 1. Read frame (DIRECT, no delay)
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Can't receive frame")
                    break
                
                # 2. Flip for mirror view
                frame = cv2.flip(frame, 1)
                
                # 3. Process with YOLO (lightweight)
                processed_frame = self.process_frame_simple(frame)
                
                # 4. Draw last detections
                if self.last_detections:
                    processed_frame = self.draw_detections(processed_frame, self.last_detections)
                
                # 5. Update FPS
                fps = self.update_fps()
                
                # 6. Display FPS and info
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(processed_frame, f"Detect: 1/{self.detect_every}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(processed_frame, f"Tools: {len(self.last_detections)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 7. Show controls info
                cv2.putText(processed_frame, "q: quit  +/-: speed", (10, processed_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 8. Show the frame
                cv2.imshow('Smooth Tool Detector', processed_frame)
                
                # 9. Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('+'):
                    # Increase detection frequency (more lag, more accurate)
                    self.detect_every = max(1, self.detect_every - 1)
                    print(f"🔍 Increased detection frequency: 1/{self.detect_every}")
                elif key == ord('-'):
                    # Decrease detection frequency (less lag, less frequent detection)
                    self.detect_every += 1
                    print(f"⚡ Decreased detection frequency: 1/{self.detect_every}")
                elif key == ord(' '):
                    # Force detection on next frame
                    self.frame_counter = self.detect_every - 1
                    print("🔍 Force detection on next frame")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera released")
            print("🎉 Goodbye!")


# Run it directly
if __name__ == "__main__":
    detector = SmoothToolDetector()
    detector.run()