import cv2
import threading
import time

class CameraSystem:
    def __init__(self, log_callback=None, frame_callback=None):
        self.log_callback = log_callback
        self.frame_callback = frame_callback
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_lock = threading.Lock()
        self.current_frame = None
        
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def setup_camera(self):
        """Setup camera for maximum FPS"""
        # Try multiple camera indices
        camera_indices = [0, 1, 2, 3]
        
        for i in camera_indices:
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, test_frame = cap.read()
                    if ret:
                        self.log(f"Found camera at index {i}")
                        
                        # Try to set camera properties (may not work on all cameras)
                        try:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            cap.set(cv2.CAP_PROP_FPS, 30)
                        except:
                            self.log(f"Could not set camera properties for index {i}", "warning")
                        
                        return cap
                    else:
                        cap.release()
            except Exception as e:
                self.log(f"Camera index {i} error: {e}", "warning")
                continue
        
        raise Exception("No working camera found!")
    
    def start(self):
        """Start camera thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def _run(self):
        try:
            self.cap = self.setup_camera()
            self.log("Camera started successfully")
            
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Mirror the frame horizontally
                    frame = cv2.flip(frame, 1)
                    
                    # Store frame with lock
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    
                    # Callback should handle thread-safe updates
                    if self.frame_callback:
                        self.frame_callback(frame)
                else:
                    self.log("Failed to read frame from camera", "warning")
                    break
                
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            self.log(f"Camera error: {e}", "error")
        finally:
            self.stop()
    
    def get_frame(self):
        """Thread-safe method to get current frame"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def capture_frame(self):
        """Capture a single frame from camera"""
        if self.cap and self.cap.isOpened():
            # Clear buffer
            for _ in range(2):
                self.cap.grab()
            
            ret, frame = self.cap.read()
            if ret:
                return cv2.flip(frame, 1)
        return None
    
    def stop(self):
        """Stop camera"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.log("Camera stopped")