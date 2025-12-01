import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO
import numpy as np
import threading
import time


class OptimizedYOLODetector:
    def __init__(self, model_path, class_names):
        # Use GPU if available, otherwise CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load model with optimizations
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.fuse()  # Fuse layers for better performance
        self.class_names = class_names
        self.confidence = 0.0
        self.fps = 0.0
        self.running = False

    def detect_objects(self, frame):
        start_time = time.time()

        # Run YOLO inference with optimized settings
        results = self.model.track(
            frame,
            persist=True,
            conf=0.5,
            verbose=False,  # Disable verbose output
            half=False,  # Keep as False for stability
            device=self.device,
            imgsz=640,  # Fixed inference size
            max_det=10  # Limit maximum detections
        )

        # Calculate FPS
        end_time = time.time()
        self.fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0

        # Process results
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                # Get tracking IDs if available
                track_ids = None
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)

                # Update confidence (average of all detections)
                self.confidence = np.mean(confidences) if len(confidences) > 0 else 0

                # Draw bounding boxes and labels
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    confidence = confidences[i]
                    class_id = class_ids[i]

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Create label with class name and confidence
                    label = f"{self.class_names[class_id]}: {confidence:.2f}"

                    # Add tracking ID if available
                    if track_ids is not None:
                        label += f" ID:{track_ids[i]}"

                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                  (x1 + label_size[0], y1), (0, 255, 0), -1)

                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame


class StableCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO11 Object Detection & Tracking - Mirrored View")
        self.root.geometry("900x700")

        # Class names
        self.class_names = {
            0: 'Bolt', 1: 'Hammer', 2: 'Measuring Tape',
            3: 'Plier', 4: 'Screwdriver', 5: 'Wrench'
        }

        # Camera setup
        self.cap = None
        self.current_camera = 0
        self.is_camera_on = False

        # Performance tracking
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.smoothed_fps = 0.0

        # Create GUI
        self.setup_gui()

        # Variables for video display - CRITICAL: Keep persistent reference
        self.current_frame = None
        self.photo_reference = None  # Persistent reference to prevent garbage collection

        # Initialize YOLO detector
        self.detector = OptimizedYOLODetector('best.pt', self.class_names)

        # Start camera automatically
        self.root.after(100, self.initialize_camera)

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(main_frame, text="YOLO11 Object Detection & Tracking",
                                font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)

        # Camera frame
        camera_frame = ttk.LabelFrame(main_frame, text="Camera View - Detection Active",
                                      padding=10)
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Video display label - FIXED: Use a fixed size to prevent resizing flicker
        self.video_label = ttk.Label(camera_frame, background='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=10)

        # Left side - Performance info
        performance_frame = ttk.Frame(status_frame)
        performance_frame.pack(side=tk.LEFT, padx=10)

        # Confidence display
        self.confidence_label = ttk.Label(performance_frame, text="Confidence: 0.00",
                                          font=('Arial', 12, 'bold'))
        self.confidence_label.pack(anchor=tk.W)

        # FPS display
        self.fps_label = ttk.Label(performance_frame, text="FPS: 0.00",
                                   font=('Arial', 12, 'bold'))
        self.fps_label.pack(anchor=tk.W)

        # Device info
        self.device_label = ttk.Label(performance_frame, text="Device: Initializing...",
                                      font=('Arial', 10))
        self.device_label.pack(anchor=tk.W)



        # Right side - Instructions
        instructions_frame = ttk.Frame(status_frame)
        instructions_frame.pack(side=tk.RIGHT, padx=10)

        instructions = ttk.Label(instructions_frame,
                                 text="Camera started. Close window to stop.",
                                 font=('Arial', 10), foreground='blue')
        instructions.pack(anchor=tk.E)

        # Performance info
        info_frame = ttk.Frame(status_frame)
        info_frame.pack(side=tk.RIGHT, padx=20)

        info_text = ttk.Label(info_frame,
                              text="Real-time Processing",
                              font=('Arial', 10), foreground='green')
        info_text.pack(anchor=tk.E)

        # Class information
        class_frame = ttk.LabelFrame(main_frame, text="Detected Objects", padding=10)
        class_frame.pack(fill=tk.X, pady=10)

        class_text = " | ".join([f"{k}: {v}" for k, v in self.class_names.items()])
        class_label = ttk.Label(class_frame, text=class_text, font=('Arial', 10, 'bold'))
        class_label.pack()

    def initialize_camera(self):
        """Initialize camera and start detection automatically"""
        self.cap = cv2.VideoCapture(self.current_camera)

        # Optimize camera settings for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Balanced resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Small buffer for low latency
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        if not self.cap.isOpened():
            # Try different camera indices
            for i in range(3):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    self.current_camera = i
                    break

            if not self.cap.isOpened():
                error_label = ttk.Label(self.video_label, text="Cannot open camera. Please check camera connection.",
                                        foreground='red', font=('Arial', 14))
                error_label.pack(expand=True)
                return

        self.is_camera_on = True
        self.detector.running = True

        # Update device info
        device_info = f"Device: {self.detector.device.upper()} | Camera: {self.current_camera}"
        self.device_label.config(text=device_info)

        # Start video processing in a separate thread
        self.video_thread = threading.Thread(target=self.stable_video_processing)
        self.video_thread.daemon = True
        self.video_thread.start()

    def stable_video_processing(self):
        """Stable video processing without flickering"""
        last_time = time.time()
        fps_counter = 0

        while self.is_camera_on and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                continue

            # FLIP THE FRAME HORIZONTALLY (Right to Left)
            frame = cv2.flip(frame, 1)  # 1 means horizontal flip

            # Resize frame for consistent processing
            frame = cv2.resize(frame, (640, 480))

            # Detect objects on EVERY frame
            processed_frame = self.detector.detect_objects(frame)

            # Convert to RGB for tkinter
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Convert to ImageTk format - CRITICAL: Keep reference to prevent garbage collection
            pil_image = Image.fromarray(processed_frame)
            photo = ImageTk.PhotoImage(image=pil_image)

            # Update GUI in main thread with the new frame
            self.root.after(0, self.update_display, photo, pil_image)

            # Calculate smoothed FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:  # Update every second
                self.smoothed_fps = fps_counter / (current_time - last_time)
                fps_counter = 0
                last_time = current_time

    def update_display(self, photo, pil_image):
        """Update display with current frame and metrics - FIXED: No flickering"""
        # CRITICAL: Keep reference to prevent garbage collection
        self.photo_reference = photo
        self.current_frame = pil_image

        # Update the label with the new image
        self.video_label.configure(image=photo)

        # Update performance labels with smoothed values
        current_fps = self.smoothed_fps if self.smoothed_fps > 0 else self.detector.fps

        self.confidence_label.config(text=f"Confidence: {self.detector.confidence:.2f}")
        self.fps_label.config(text=f"FPS: {current_fps:.1f}")

    def on_closing(self):
        """Clean up when closing the application"""
        self.is_camera_on = False
        self.detector.running = False

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = StableCameraApp(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    root.mainloop()


if __name__ == "__main__":
    main()