"""
ULTRA-SMOOTH Tool Detector for Raspberry Pi DofBot
This prioritizes SMOOTH camera display with light detection
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO

print("🎯 Starting ULTRA-SMOOTH Tool Detector")

# ==================== CAMERA SETUP ====================
# Try different camera indices
camera_index = 0
cap = None

for idx in [0, 1, 2, 3]:
    print(f"Trying camera {idx}...")
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    
    if cap.isOpened():
        # CRITICAL SETTINGS FOR SMOOTHNESS:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # MINIMAL BUFFER = LESS LAG
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        # Test read
        ret, test = cap.read()
        if ret:
            camera_index = idx
            print(f"✅ Camera {idx} works!")
            break
        else:
            cap.release()
            cap = None

if cap is None:
    print("❌ ERROR: No camera found!")
    exit()

print(f"📹 Using camera index: {camera_index}")

# ==================== YOLO SETUP (LIGHTWEIGHT) ====================
print("📦 Loading YOLO model (lightweight mode)...")
try:
    model = YOLO('best.pt')
    # Use tiny inference size for speed
    print("✅ YOLO loaded")
except:
    print("⚠️  YOLO not loaded, running camera-only mode")
    model = None

# Tool info
tool_names = ['Bolt', 'Hammer', 'Measuring Tape', 'Plier', 'Screwdriver', 'Wrench']
tool_colors = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]

# ==================== MAIN LOOP ====================
print("\n" + "="*50)
print("🎬 STARTING - Press:")
print("  q = Quit")
print("  d = Toggle detection (on/off)")
print("  1-6 = Force detect specific tool")
print("="*50 + "\n")

# State variables
detection_enabled = True
detection_counter = 0
detect_every = 2  # Process every 2nd frame
last_detections = []
fps_counter = 0
fps_timer = time.time()
current_fps = 0

while True:
    loop_start = time.time()
    
    # 1. READ FRAME
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame read error")
        break
    
    # 2. MIRROR VIEW
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    
    # 3. DETECTION (only if enabled and it's time)
    detection_counter += 1
    
    if detection_enabled and model and detection_counter >= detect_every:
        detection_counter = 0
        
        try:
            # Run YOLO on SMALL image (320x240) for SPEED
            small = cv2.resize(frame, (320, 240))
            results = model(small, conf=0.4, imgsz=320, verbose=False, max_det=6)
            
            if results and results[0].boxes is not None:
                last_detections = []
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Scale boxes back to original size
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    last_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': float(confs[i]),
                        'class': int(classes[i])
                    })
        except Exception as e:
            print(f"Detection error: {e}")
            last_detections = []
    
    # 4. DRAW DETECTIONS
    if last_detections:
        for det in last_detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class']
            conf = det['conf']
            
            color = tool_colors[class_id % len(tool_colors)]
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{tool_names[class_id]}: {conf:.1f}"
            cv2.putText(display_frame, label, (x1, max(y1-10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 5. UPDATE FPS
    fps_counter += 1
    if time.time() - fps_timer >= 1.0:
        current_fps = fps_counter
        fps_counter = 0
        fps_timer = time.time()
        print(f"📊 FPS: {current_fps} | Detections: {len(last_detections)}")
    
    # 6. DISPLAY INFO
    # FPS
    cv2.putText(display_frame, f"FPS: {current_fps}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Detection status
    status_color = (0, 255, 0) if detection_enabled else (0, 0, 255)
    status_text = "DETECT ON" if detection_enabled else "DETECT OFF"
    cv2.putText(display_frame, status_text, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Tools detected
    cv2.putText(display_frame, f"Tools: {len(last_detections)}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 7. SHOW FRAME
    cv2.imshow('SMOOTH Tool Detector', display_frame)
    
    # 8. HANDLE KEYS
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n👋 Quitting...")
        break
    elif key == ord('d'):
        detection_enabled = not detection_enabled
        print(f"🔍 Detection: {'ENABLED' if detection_enabled else 'DISABLED'}")
    elif key == ord('+'):
        detect_every = max(1, detect_every - 1)
        print(f"⚡ More frequent detection: every {detect_every} frame(s)")
    elif key == ord('-'):
        detect_every += 1
        print(f"🐢 Less frequent detection: every {detect_every} frame(s)")
    elif ord('1') <= key <= ord('6'):
        tool_idx = key - ord('1')
        print(f"🎯 Looking for {tool_names[tool_idx]}")

# ==================== CLEANUP ====================
cap.release()
cv2.destroyAllWindows()
print("\n✅ Camera released")
print("🎉 Thank you for using the SMOOTH Tool Detector!")