"""
ULTRA-OPTIMIZED YOLO11 for Raspberry Pi
Only camera + detection, MAXIMUM FPS possible
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO

print("🚀 ULTRA-OPTIMIZED YOLO11 on Raspberry Pi")
print("=" * 50)

# ============================================
# STEP 1: CAMERA SETUP - MAXIMUM FPS
# ============================================
print("📷 Setting up camera for MAXIMUM FPS...")

# Try different camera indices
cap = None
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        print(f"✅ Found camera at index {i}")
        
        # CRITICAL: These settings give MAXIMUM FPS on Pi
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # MJPEG encoding
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Don't go lower or YOLO won't detect well
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # MINIMAL buffer = minimal lag
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)       # Disable auto-focus
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # 3 = manual exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, 100)      # Adjust if too dark/bright
        
        break
    cap = None

if cap is None:
    print("❌ ERROR: No camera found!")
    exit()

# ============================================
# STEP 2: YOLO SETUP - EXTREME OPTIMIZATION
# ============================================
print("📦 Loading YOLO11 with EXTREME optimizations...")

try:
    # Load YOLO11s model (small version)
    model = YOLO('best.pt')  # Make sure this is YOLO11s model
    
    # Apply MASSIVE optimizations
    model.overrides['conf'] = 0.35      # Lower confidence = faster
    model.overrides['iou'] = 0.3        # Lower IOU = faster
    model.overrides['agnostic_nms'] = True  # Faster NMS
    model.overrides['max_det'] = 6      # Exactly your 6 tools
    model.overrides['verbose'] = False  # No prints
    
    print("✅ YOLO loaded with optimizations")
    
except Exception as e:
    print(f"❌ ERROR loading YOLO: {e}")
    print("Running in camera-only mode...")
    model = None

# Tool information
TOOL_CLASSES = ['Bolt', 'Hammer', 'Measuring Tape', 'Plier', 'Screwdriver', 'Wrench']
TOOL_COLORS = [
    (0, 255, 0),    # Green - Bolt
    (0, 0, 255),    # Red - Hammer  
    (255, 0, 0),    # Blue - Measuring Tape
    (255, 255, 0),  # Cyan - Plier
    (255, 0, 255),  # Magenta - Screwdriver
    (0, 255, 255)   # Yellow - Wrench
]

# ============================================
# STEP 3: PERFORMANCE VARIABLES
# ============================================
fps = 0
frame_count = 0
start_time = time.time()
detection_fps = 0
detection_counter = 0
detection_start = time.time()

# We'll process detection in a smart way
last_detections = []
detection_interval = 5  # Process every 5th frame initially
frame_index = 0

print("\n" + "=" * 50)
print("🎬 STARTING DETECTION")
print("=" * 50)
print("Controls:")
print("  • Press 'q' to quit")
print("  • Press 'f' to show/hide FPS")
print("  • Press '+' to increase detection frequency")
print("  • Press '-' to decrease detection frequency")
print("  • Press '1-6' to highlight specific tools")
print("=" * 50 + "\n")

# State variables
show_fps = True
highlight_tool = -1  # -1 = show all

# ============================================
# STEP 4: MAIN LOOP - OPTIMIZED FOR PI
# ============================================
print("Starting in 2 seconds...")
time.sleep(2)

try:
    while True:
        # 1. READ FRAME
        ret, frame = cap.read()
        if not ret:
            print("❌ Can't receive frame")
            break
        
        # 2. MIRROR VIEW (like a mirror)
        frame = cv2.flip(frame, 1)
        
        # 3. UPDATE FPS COUNTER
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
            
            # Print FPS every second
            print(f"📊 FPS: {fps:.1f} | Detection FPS: {detection_fps:.1f} | Tools: {len(last_detections)}")
        
        # 4. DETECTION LOGIC (SMART PROCESSING)
        frame_index += 1
        display_frame = frame.copy()
        
        # Only run YOLO every N frames
        if model and frame_index % detection_interval == 0:
            detection_counter += 1
            detection_elapsed = time.time() - detection_start
            
            if detection_elapsed >= 1.0:
                detection_fps = detection_counter / detection_elapsed
                detection_counter = 0
                detection_start = time.time()
            
            try:
                # CRITICAL: Use TINY inference size for MAXIMUM speed
                # 256x192 is 1/8 the pixels of 640x480 = 8x faster!
                inference_size = 256
                
                # Resize to tiny size for inference
                small_frame = cv2.resize(frame, (inference_size, int(inference_size * 0.75)))
                
                # Run YOLO with minimal settings
                results = model(small_frame, 
                              conf=0.35,      # Low confidence threshold
                              iou=0.3,        # Low IOU threshold  
                              imgsz=inference_size,
                              max_det=6,
                              verbose=False,
                              half=False,     # Keep as False on Pi
                              device='cpu',   # Force CPU on Pi
                              agnostic_nms=True)  # Faster NMS
                
                # Process results
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    # Scale boxes back to original frame size
                    scale_x = frame.shape[1] / inference_size
                    scale_y = frame.shape[0] / (inference_size * 0.75)
                    
                    last_detections = []
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
                        
                        last_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(confidence),
                            'class_id': int(class_id)
                        })
                else:
                    last_detections = []
                    
            except Exception as e:
                print(f"⚠️ Detection error: {e}")
                last_detections = []
        
        # 5. DRAW DETECTIONS (FAST)
        if last_detections:
            for det in last_detections:
                x1, y1, x2, y2 = det['bbox']
                class_id = det['class_id']
                confidence = det['confidence']
                
                # Skip if we're highlighting a specific tool and this isn't it
                if highlight_tool != -1 and class_id != highlight_tool:
                    continue
                
                # Get color
                color = TOOL_COLORS[class_id % len(TOOL_COLORS)]
                
                # Draw bounding box (THIN for speed)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)
                
                # Draw simple label (NO background box for speed)
                label = f"{TOOL_CLASSES[class_id]}:{confidence:.1f}"
                cv2.putText(display_frame, label, 
                           (x1, max(y1 - 5, 15)),  # Position above box
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.35,  # Small font
                           color, 
                           1)     # Thin line
        
        # 6. DISPLAY INFO (OPTIONAL - can disable for even more FPS)
        if show_fps:
            # Show FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Show detection info
            cv2.putText(display_frame, f"Detect: 1/{detection_interval}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show tools detected
            cv2.putText(display_frame, f"Tools: {len(last_detections)}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Show current mode
            mode_text = "ALL" if highlight_tool == -1 else TOOL_CLASSES[highlight_tool]
            cv2.putText(display_frame, f"Mode: {mode_text}", (10, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 7. SHOW FRAME
        cv2.imshow('YOLO11 Tool Detector - Raspberry Pi', display_frame)
        
        # 8. KEYBOARD CONTROLS
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Quitting...")
            break
        elif key == ord('f'):
            show_fps = not show_fps
            print(f"📊 FPS display: {'ON' if show_fps else 'OFF'}")
        elif key == ord('+'):
            detection_interval = max(2, detection_interval - 1)  # Minimum every 2nd frame
            print(f"⚡ Increased detection to: 1/{detection_interval}")
        elif key == ord('-'):
            detection_interval += 1
            print(f"🐢 Decreased detection to: 1/{detection_interval}")
        elif ord('1') <= key <= ord('6'):
            tool_idx = key - ord('1')
            if highlight_tool == tool_idx:
                highlight_tool = -1  # Show all
                print(f"🎯 Showing ALL tools")
            else:
                highlight_tool = tool_idx
                print(f"🎯 Highlighting: {TOOL_CLASSES[tool_idx]}")
        elif key == ord('0'):
            highlight_tool = -1
            print("🎯 Showing ALL tools")
        
except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Camera released")
    print("🎉 Thank you for using the ULTRA-OPTIMIZED YOLO11 Tool Detector!")