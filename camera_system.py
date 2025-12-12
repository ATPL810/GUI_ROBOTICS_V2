import cv2
import threading
import time

class CameraSystem:
    def __init__(self, log_callback=None, frame_callback=None):
        self.log_callback = log_callback
        self.frame_callback = frame_callback
        self.cap = None
        self.running = False
        
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def setup_camera(self):
        """Setup camera for maximum FPS"""
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                self.log(f"Found camera at index {i}")
                
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
    
    def start(self):
        """Start camera thread"""
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
    
    def run(self):
        self.running = True
        try:
            self.cap = self.setup_camera()
            self.log("Camera started")
            
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    if self.frame_callback:
                        self.frame_callback(frame)
                else:
                    break
                time.sleep(0.03)
                
        except Exception as e:
            self.log(f"Camera error: {e}", "error")
        finally:
            self.stop()
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.log("Camera stopped")