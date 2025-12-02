import cv2
import time

print("🚀 Starting ULTRA-SMOOTH Camera with Light Detection")

# 1. Open camera with DIRECT settings
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

# 2. SETTINGS FOR MAXIMUM SMOOTHNESS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # MINIMAL BUFFER = LESS LAG
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

print("📹 Camera ready. Press 'q' to quit")

# FPS tracking
fps_start = time.time()
fps_frames = 0
current_fps = 0

# Main loop - AS SIMPLE AS POSSIBLE
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mirror the frame (like a mirror)
    frame = cv2.flip(frame, 1)
    
    # FPS calculation
    fps_frames += 1
    if time.time() - fps_start >= 1.0:
        current_fps = fps_frames
        fps_frames = 0
        fps_start = time.time()
        print(f"📊 FPS: {current_fps}")
    
    # Display FPS on frame
    cv2.putText(frame, f"FPS: {current_fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Live Camera', frame)
    
    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Done")