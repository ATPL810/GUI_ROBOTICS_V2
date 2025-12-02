"""
EXTREME OPTIMIZATION - MAXIMUM FPS on Raspberry Pi
Cuts EVERY possible corner for speed
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO

print("⚡ EXTREME OPTIMIZATION MODE")
print("Camera + YOLO at MAXIMUM possible FPS")

# 1. Camera - DIRECT
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

# 2. YOLO - MINIMAL
model = YOLO('best.pt')
model.overrides['conf'] = 0.25  # VERY low confidence
model.overrides['iou'] = 0.2    # VERY low IOU
model.overrides['max_det'] = 3  # Limit to 3 detections max
model.overrides['verbose'] = False

# 3. Run detection in SEPARATE thread (simplified)
import threading

detections = []
running = True

def detection_worker():
    """Run YOLO in background"""
    global detections
    while running:
        try:
            # Get frame for detection
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Tiny inference size
            tiny_frame = cv2.resize(frame, (192, 144))  # SUPER SMALL
            
            # Run YOLO
            results = model(tiny_frame, imgsz=192, verbose=False, device='cpu')
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Scale boxes
                scale_x = 640 / 192
                scale_y = 480 / 144
                
                current_detections = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                    x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                    
                    current_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': int(classes[i])
                    })
                
                detections = current_detections
                
        except:
            pass
        time.sleep(0.05)  # 20 FPS detection rate

# Start detection thread
detection_thread = threading.Thread(target=detection_worker, daemon=True)
detection_thread.start()

# Colors
colors = [(0,255,0), (0,0,255), (255,0,0), (255,255,0), (255,0,255), (0,255,255)]

# Main display loop
fps_counter = 0
fps_time = time.time()

print("\nStarting... Press 'q' to quit")

while True:
    # Get fresh frame for display
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mirror
    frame = cv2.flip(frame, 1)
    
    # Draw current detections
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_id = det['class']
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_id % 6], 1)
    
    # FPS
    fps_counter += 1
    if time.time() - fps_time >= 1.0:
        print(f"FPS: {fps_counter}")
        fps_counter = 0
        fps_time = time.time()
    
    # Show
    cv2.imshow('MAX FPS Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
running = False
detection_thread.join(timeout=1)
cap.release()
cv2.destroyAllWindows()
print("Done")