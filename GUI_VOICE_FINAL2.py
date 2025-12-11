import sys
import os
import time
import threading
import json
import re
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from Arm_Lib import Arm_Device

# Tkinter imports
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from PIL import Image, ImageTk

# Voice recognition imports
import pyaudio
import pyttsx3
from vosk import Model, KaldiRecognizer

# ============================================
# VOICE RECOGNITION SYSTEM (Updated from Voice_Commands.py)
# ============================================
class VoiceRecognitionSystem:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.running = False
        self.voice_enabled = False
        self.is_speaking = False
        self.command_queue = []
        self.last_command_time = 0
        self.command_callback = None
        self.paused = False  # NEW: Add paused state
        
        # Tools dictionary from Voice_Commands.py
        self.tools = {
            "screwdriver": ["screwdriver", "screw", "driver", "screwdriv", "skrewdriver", "skrewed", "skewed"],
            "bolt": ["bolt", "bold", "board", "boat", "bult", "bolte", "bohlt", "boy", "bowl", "ball", "pool", "paul", "bullets", "bullet"],
            "wrench": ["wrench", "rench", "range", "french", "spanner", "rensh", "right","branch", "ranch", "wench", "trench"],
            "measuring tape": ["measuring tape", "measuring tap", "tape", "measure", "measuring the", "measuring", "measur", "measure tape"],
            "hammer": ["hammer", "armor", "amor", "ammer", "hamer", "hamr", "mallet", "hummer", "however", "harvard", "homo", "rough"],
            "plier": ["plier", "players", "pliers", "pryer", "player", "plyer", "pincher", "apply", "flyer", "flier", "liar", "lawyer"],
        }
        
        
        # Initialize Text-to-Speech
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 170)
        self.tts.setProperty('volume', 0.9)
        
        # Vosk model
        self.model = None
        self.recognizer = None
        self.stream = None
        self.p = None
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[VOICE {level.upper()}] {message}")
    
    def set_command_callback(self, callback):
        self.command_callback = callback
    
    def pause_listening(self):
        """Pause voice listening"""
        self.paused = True
    
    def resume_listening(self):
        """Resume voice listening"""
        self.paused = False
    
    def initialize_voice(self):
        """Initialize voice recognition system"""
        try:
            model_path = "./vosk-model-small-en-us-0.15"
            if not os.path.exists(model_path):
                self.log(f"Vosk model not found at {model_path}", "error")
                return False
            
            self.log("Loading Vosk model...", "info")
            self.model = Model(model_path)
            
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000,
                input_device_index=3
            )
            
            self.recognizer = KaldiRecognizer(self.model, 16000)
            
            self.log("✅ Voice recognition ready!", "success")
            return True
            
        except Exception as e:
            self.log(f"Failed to initialize voice: {e}", "error")
            return False
    
    def speak(self, text):
        """Make the robot speak through speakers - FROM Voice_Commands.py"""
        if not text or self.is_speaking:
            return
        
        self.is_speaking = True
        self.log(f"Speaking: {text}", "info")
        
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            self.log(f"Speaking error: {e}", "error")
            # If speech fails, at least print it
            self.log(f"[SPEAKING]: {text}", "info")
            time.sleep(len(text.split()) * 0.3)  # Simulate speaking time
        
        self.is_speaking = False
    
    def get_fetching_message(self, tool):
        """Generate fetching message - ALWAYS USED when fetching any tool - FROM Voice_Commands.py"""
        return f"Fetching the {tool} for you!"
    
    def get_confirmation_message(self, tool):
        """Generate confirmation message after fetching - FROM Voice_Commands.py"""
        return f"Here is your {tool}! What else can I get for you?"
    
    def find_tool(self, text):
        """Find which tool was mentioned - FROM Voice_Commands.py"""
        text_lower = text.lower()
        
        # Check each tool's keywords
        for tool, keywords in self.tools.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return tool
        
        # Check for similar words (first 3 letters)
        words = text_lower.split()
        for word in words:
            if len(word) >= 3:
                for tool, keywords in self.tools.items():
                    for keyword in keywords:
                        if word[:3] == keyword[:3]:
                            return tool
        
        return None
    
    def process_voice_command(self, text):
        """Process a voice command - MODIFIED to skip confirmation"""
        # self.log(f"Voice command: '{text}'", "info")
        
        text_lower = text.lower()
        
        # Check for help first
        if any(word in text_lower for word in ["help", "what can you do", "instructions"]):
            help_msg = "I can fetch tools for you. Just say the name of the tool, like 'hammer' or 'wrench'."
            self.speak(help_msg)
            return None
        
        # Check for tool request - primary action
        tool = self.find_tool(text_lower)
        
        if tool:
            # MODIFIED: Skip speaking fetching message and return tool directly
            # This allows immediate grabbing without confirmation
            return tool
        
        # Check for greeting if no tool or help command was found
        if any(greet in text_lower for greet in ["hello", "hi", "hey", "greetings"]):
            greeting_msg = "Hello! I'm here to help. Just say the name of a tool you need."
            self.speak(greeting_msg)
            return None
        
        # If nothing was understood
        error_msg = "I didn't catch that. Please say the name of a tool, like 'hammer' or 'screwdriver'."
        self.speak(error_msg)
        return None
    
    def run(self):
        """Main voice recognition loop - UPDATED to respect paused state"""
        self.running = True
        
        if not self.initialize_voice():
            self.log("Failed to initialize voice recognition", "error")
            return
        
        self.log("🔊 Voice recognition started", "success")
        welcome_msg = "Hello! I am your Arrange Assistant. I can fetch tools for you. Just say the name of a tool you need, like 'wrench' or 'hammer'."
        self.speak(welcome_msg)
        
        while self.running and self.voice_enabled:
            try:
                # Skip processing if paused
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                if self.is_speaking:
                    time.sleep(0.1)
                    continue
                
                data = self.stream.read(4000, exception_on_overflow=False)
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    
                    if text and len(text) > 2:
                        current_time = time.time()
                        if current_time - self.last_command_time > 1.5:
                            tool = self.process_voice_command(text)
                            if tool and self.command_callback:
                                self.command_callback(tool)
                            self.last_command_time = current_time
                
                time.sleep(0.01)
                
            except Exception as e:
                self.log(f"Voice recognition error: {e}", "error")
                time.sleep(1)
    
    def stop(self):
        """Stop voice recognition"""
        self.running = False
        self.voice_enabled = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.p:
            self.p.terminate()
        
        self.log("Voice recognition stopped", "info")

# ============================================
# ROBOT ARM CONTROLLER
# ============================================
class RobotArmController:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.log("Initializing robot arm...")
        
        try:
            self.arm = Arm_Device()
            time.sleep(2)
            
            self.SERVO_BASE = 1
            self.SERVO_SHOULDER = 2  
            self.SERVO_ELBOW = 3
            self.SERVO_WRIST = 4
            self.SERVO_WRIST_ROT = 5
            self.SERVO_GRIPPER = 6
            
            self.INITIAL_POSITION = {
                self.SERVO_BASE: 90,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }
            
            self.SECOND_POSITION = {
                self.SERVO_BASE: 40,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }

            self.THIRD_POSITION = {
                self.SERVO_BASE: 1,
                self.SERVO_SHOULDER: 105,
                self.SERVO_ELBOW: 45,
                self.SERVO_WRIST: -35,
                self.SERVO_WRIST_ROT: 90,
                self.SERVO_GRIPPER: 90
            }
            
            self.go_to_initial_position()
            self.log("✅ Robot arm initialized successfully")
            
        except Exception as e:
            self.log(f"❌ Failed to initialize robot arm: {e}", "error")
            raise
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def go_to_initial_position(self):
        """Move to exact initial position"""
        angles_dict = self.INITIAL_POSITION.copy()
        angles_dict[self.SERVO_WRIST] = self.convert_angle(angles_dict[self.SERVO_WRIST])
        
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000
        )
        time.sleep(2.5)
        self.log("At initial position (Base: 90°)")
    
    def go_to_second_position(self):
        """Move to second position (base at 40 degrees)"""
        angles_dict = self.SECOND_POSITION.copy()
        angles_dict[self.SERVO_WRIST] = self.convert_angle(angles_dict[self.SERVO_WRIST])
        
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000
        )
        time.sleep(2.5)
        self.log("At second position (Base: 40°)")
    
    def go_to_third_position(self):
        """Move to third position (base at 1 degree)"""
        angles_dict = self.THIRD_POSITION.copy()
        angles_dict[self.SERVO_WRIST] = self.convert_angle(angles_dict[self.SERVO_WRIST])
        
        self.arm.Arm_serial_servo_write6(
            angles_dict[1], angles_dict[2], angles_dict[3],
            angles_dict[4], angles_dict[5], angles_dict[6],
            2000
        )
        time.sleep(2.5)
        self.log("At third position (Base: 1°)")
    
    def convert_angle(self, angle):
        """Convert negative angles to servo range (0-180)"""
        if angle < 0:
            return angle
        return angle
    
    def go_to_home(self):
        """Go to safe home position"""
        self.log("Moving to home position...")
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
        time.sleep(2)
        self.log("✅ At home position")

# ============================================
# CAMERA SYSTEM
# ============================================
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

# ============================================
# SNAPSHOT SYSTEM
# ============================================
class SnapshotSystem:
    def __init__(self, arm_controller, camera, log_callback=None):
        self.arm = arm_controller
        self.camera = camera
        self.log_callback = log_callback
        self.model = None
        self.output_dir = None
        self.all_snapshots = []
        
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
        if self.camera.cap and self.camera.cap.isOpened():
            for _ in range(3):
                self.camera.cap.read()
                time.sleep(0.1)
            
            ret, frame = self.camera.cap.read()
            if ret:
                return cv2.flip(frame, 1)
        return None
    
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
            self.log("\n🔹 STEP 1: INITIAL POSITION", "info")
            self.wait_for_stabilization(3, "Initial Position")
            snapshot1 = self.capture_and_save("initial_position", self.arm.INITIAL_POSITION[self.arm.SERVO_BASE])
            
            self.log("\n🔹 STEP 2: MOVING TO SECOND POSITION", "info")
            self.arm.go_to_second_position()
            self.wait_for_stabilization(3, "Second Position")
            snapshot2 = self.capture_and_save("second_position", self.arm.SECOND_POSITION[self.arm.SERVO_BASE])
            
            self.log("\n🔹 STEP 3: MOVING TO THIRD POSITION", "info")
            self.arm.go_to_third_position()
            self.wait_for_stabilization(3, "Third Position")
            snapshot3 = self.capture_and_save("third_position", self.arm.THIRD_POSITION[self.arm.SERVO_BASE])
            
            self.log("\n🔹 STEP 4: RETURNING TO INITIAL POSITION", "info")
            time.sleep(3)
            self.arm.go_to_initial_position()
            
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
        
        annotated_frame = self.annotate_frame(frame, detections, position_name, base_angle)
        filename = f"{self.output_dir}/{position_name}.jpg"
        cv2.imwrite(filename, annotated_frame)
        self.log(f"Saved snapshot: {filename}")
        
        self.save_detection_data(position_name, detections, filename, base_angle)
        
        snapshot_info = {
            'position_name': position_name,
            'base_angle': base_angle,
            'filename': filename,
            'detections': detections,
            'timestamp': datetime.now().strftime("%Y-%m-d %H:%M:%S")
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

# ============================================
# GRAB SYSTEM WITH DUPLICATE HANDLING
# ============================================
class GrabSystem:
    def __init__(self, arm_controller, snapshot_system, camera_system, log_callback=None):
        self.arm = arm_controller
        self.snapshot_system = snapshot_system
        self.camera_system = camera_system
        self.log_callback = log_callback
        self.tool_mapping = {}
        self.all_tool_locations = {}
        self.master_report_path = None
        
        self.grab_scripts_dir = "movements"
        self.fetched_tools_file = "data/fetched_tools.json"
        self.fetched_locations = {}
        
        # Load fetched locations history
        self.load_fetched_locations()
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def load_fetched_locations(self):
        """Load previously fetched tool locations from file"""
        self.fetched_locations = {}
        
        if os.path.exists(self.fetched_tools_file):
            try:
                with open(self.fetched_tools_file, 'r') as f:
                    self.fetched_locations = json.load(f)
                self.log(f"Loaded {len(self.fetched_locations)} fetched locations from history", "info")
            except:
                self.log("Could not load fetched locations history", "warning")
                self.fetched_locations = {}
    
    def save_fetched_location(self, tool_name, point):
        """Save a fetched location to history"""
        location_key = f"{tool_name}_{point}"
        self.fetched_locations[location_key] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool_name,
            "point": point
        }
        
        # Save to file
        try:
            os.makedirs(os.path.dirname(self.fetched_tools_file), exist_ok=True)
            with open(self.fetched_tools_file, 'w') as f:
                json.dump(self.fetched_locations, f, indent=2)
        except Exception as e:
            self.log(f"Warning: Could not save fetched location: {e}", "warning")
    
    def find_latest_master_report(self):
        """Find the latest master report"""
        import glob
        pattern = "data/mappings/mapping_*/master_report.txt"
        matches = glob.glob(pattern)
        
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
        
        raise FileNotFoundError("No master report found! Please run analysis first.")
    
    def load_mapping(self, master_report_path=None):
        """Load tool mapping from master report with duplicate handling"""
        if master_report_path:
            self.master_report_path = master_report_path
        else:
            self.master_report_path = self.find_latest_master_report()
        
        self.log(f"Using tool mapping from: {self.master_report_path}", "info")
        self.tool_mapping, self.all_tool_locations = self.parse_master_report()
        
        # Sync fetched status
        self.sync_fetched_status()
        
        self.log(f"Loaded {len(self.tool_mapping)} unique tool types", "success")
        if self.all_tool_locations:
            total_tools = sum(len(locs) for locs in self.all_tool_locations.values())
            self.log(f"Total tool instances: {total_tools}", "info")
        
        return self.tool_mapping
    
    def parse_master_report(self):
        """
        Parse tool mapping from master report with duplicate handling
        Returns:
            - tool_mapping: Dict of tool_name -> list of available locations (sorted by confidence)
            - all_tool_locations: Complete mapping of all tool instances
        """
        self.log("Parsing tool mapping (with duplicate handling)...", "info")
        
        tool_mapping = {}
        all_tool_locations = {}
        
        with open(self.master_report_path, 'r') as f:
            content = f.read()
        
        # Look for "ALL TOOL LOCATIONS (INCLUDING DUPLICATES):" section
        if "ALL TOOL LOCATIONS (INCLUDING DUPLICATES):" in content:
            self.log("Found duplicate locations section", "info")
            
            # Extract the entire section
            start_idx = content.find("ALL TOOL LOCATIONS (INCLUDING DUPLICATES):")
            next_section_idx = content.find("ROBOT ACTION PLAN", start_idx)
            
            if next_section_idx > start_idx:
                all_locations_section = content[start_idx:next_section_idx]
            else:
                all_locations_section = content[start_idx:]
            
            # Split into lines and parse
            lines = all_locations_section.strip().split('\n')
            
            current_tool = None
            
            for line in lines:
                line = line.strip()
                
                # Check for tool header (e.g., "BOLT (3 locations):")
                if "locations):" in line or "(1 location):" in line:
                    # Extract tool name
                    match = re.search(r'^([A-Z\s]+)\s*\(', line)
                    if match:
                        current_tool = match.group(1).strip().lower()
                        tool_mapping[current_tool] = []
                        all_tool_locations[current_tool] = []
                        self.log(f"Found tool: {current_tool}", "info")
                
                # Parse location lines
                elif "Point" in line and current_tool and ":" in line and not line.startswith("="):
                    if line.startswith("-") or not line:
                        continue
                    
                    # Try multiple patterns
                    patterns = [
                        r'[⭐\s]*Point\s+([A-I]):\s*([\d.]+)%\s*\((.*?)\)',
                        r'[⭐\s]*Point\s+([A-I]):\s*([\d.]+)%',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, line)
                        if match:
                            point = match.group(1)
                            confidence = float(match.group(2))
                            
                            # Extract position description
                            position_desc = "unknown"
                            if len(match.groups()) > 2 and match.group(3):
                                position_desc = match.group(3)
                            else:
                                if "initial" in line.lower():
                                    position_desc = "initial position"
                                elif "second" in line.lower():
                                    position_desc = "second position"
                                elif "third" in line.lower():
                                    position_desc = "third position"
                            
                            # Determine position
                            if "initial" in position_desc.lower():
                                position = "initial_position"
                            elif "second" in position_desc.lower():
                                position = "second_position"
                            elif "third" in position_desc.lower():
                                position = "third_position"
                            else:
                                position = "unknown"
                            
                            # Skip if point E is being added to bolt (E is hammer)
                            if current_tool == "bolt" and point == "E":
                                self.log(f"Skipping Point E for bolt (E is hammer)", "info")
                                continue
                            
                            location_data = {
                                "point": point,
                                "confidence": confidence / 100,
                                "position": position,
                                "position_desc": position_desc,
                                "fetched": False
                            }
                            
                            tool_mapping[current_tool].append(location_data)
                            all_tool_locations[current_tool].append(location_data)
                            
                            self.log(f"  Point {point}: {confidence}% confidence", "info")
                            break
        
        # Clean up: Remove any tool entries with no locations
        tools_to_remove = [tool for tool, locations in tool_mapping.items() if not locations]
        for tool in tools_to_remove:
            del tool_mapping[tool]
            del all_tool_locations[tool]
        
        # Sort each tool's locations by confidence (highest first)
        for tool_name in tool_mapping:
            tool_mapping[tool_name].sort(key=lambda x: x["confidence"], reverse=True)
        
        return tool_mapping, all_tool_locations
    
    def sync_fetched_status(self):
        """Sync fetched status from fetched_locations to tool_mapping"""
        for tool_name, locations in self.tool_mapping.items():
            for location in locations:
                location_key = f"{tool_name}_{location['point']}"
                if location_key in self.fetched_locations:
                    location["fetched"] = True
                    self.log(f"Marked {tool_name.upper()} at Point {location['point']} as fetched", "info")
                else:
                    location["fetched"] = False
    
    def get_next_available_location(self, tool_name):
        """
        Get the next available (not fetched) location for a tool
        Returns None if all instances are fetched
        """
        if tool_name not in self.tool_mapping:
            return None
        
        locations = self.tool_mapping[tool_name]
        
        # Find first available (not fetched) location
        for location in locations:
            if not location.get("fetched", False):
                return location
        
        # All locations fetched
        return None
    
    def check_grab_script_exists(self, grab_letter):
        """Check if grab script exists for a point"""
        script_path = os.path.join(self.grab_scripts_dir, f"grab_point_{grab_letter}.py")
        exists = os.path.exists(script_path)
        
        if not exists:
            self.log(f"Script missing: {script_path}", "warning")
        
        return exists
    
    def execute_grab_script(self, grab_letter, tool_name):
        """Execute the grab script for a specific point - UNIVERSAL VERSION"""
        script_path = os.path.join(self.grab_scripts_dir, f"grab_point_{grab_letter}.py")
        
        if not os.path.exists(script_path):
            self.log(f"ERROR: Grab script not found: {script_path}", "error")
            return False
        
        self.log(f"Executing grab script for Point {grab_letter} - {tool_name.upper()}...", "info")
        
        try:
            # Dynamically import the module
            import importlib.util
            script_name = f"grab_point_{grab_letter}"
            spec = importlib.util.spec_from_file_location(script_name, script_path)
            module = importlib.util.module_from_spec(spec)
            
            # Add arm object to module
            module.arm = self.arm.arm
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Get the grab function
            grab_func_name = f"grab_point_{grab_letter}"
            if not hasattr(module, grab_func_name):
                self.log(f"ERROR: Function {grab_func_name} not found in script", "error")
                return False
            
            grab_func = getattr(module, grab_func_name)
            
            # UNIVERSAL APPROACH: Try both calling patterns
            success = False
            
            # Pattern 1: Try with tool_type parameter (for scripts like B)
            try:
                self.log(f"  Trying Pattern 1: with tool_type='{tool_name}'", "info")
                grab_func(self.arm.arm, tool_type=tool_name)
                self.log(f"  Success with tool_type parameter", "success")
                success = True
            except TypeError as e1:
                # Pattern 2: Try without parameter (for scripts like A)
                try:
                    self.log(f"  Trying Pattern 2: without parameters", "info")
                    grab_func(self.arm.arm)
                    self.log(f"  Success without parameters", "success")
                    success = True
                except TypeError as e2:
                    # Pattern 3: Try with tool_name (alternative parameter name)
                    try:
                        self.log(f"  Trying Pattern 3: with tool_name='{tool_name}'", "info")
                        grab_func(self.arm.arm, tool_name=tool_name)
                        self.log(f"  Success with tool_name parameter", "success")
                        success = True
                    except TypeError as e3:
                        self.log(f"ERROR: Function doesn't accept expected parameters", "error")
                        success = False
            
            # Check if script actually grabbed the right tool
            if success:
                # Read the script to see what tool it's configured for
                with open(script_path, 'r') as f:
                    script_content = f.read()
                
                # Check if script mentions a specific tool
                if "BOLT" in script_content.upper() and "bolt" not in tool_name.lower():
                    self.log(f"WARNING: Script appears to be for BOLTS, but fetching {tool_name.upper()}", "warning")
                elif "WRENCH" in script_content.upper() and "wrench" not in tool_name.lower():
                    self.log(f"WARNING: Script appears to be for WRENCH, but fetching {tool_name.upper()}", "warning")
                    
        except Exception as e:
            self.log(f"ERROR executing grab script: {e}", "error")
            import traceback
            traceback.print_exc()
            return False
        
        return success
    
    def verify_object_exists(self, tool_name):
        """Verify if the requested object was actually detected in the last scan"""
        import glob
        snapshot_folders = glob.glob("data/snapshots/robot_snapshots_*")
        if not snapshot_folders:
            self.log("No scan data found!", "error")
            return False
        
        snapshot_folders.sort(key=os.path.getmtime, reverse=True)
        latest_folder = snapshot_folders[0]
        
        positions = ["initial_position", "second_position", "third_position"]
        
        for position in positions:
            report_file = os.path.join(latest_folder, f"{position}_detections.txt")
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    content = f.read()
                    if tool_name.lower() in content.lower():
                        self.log(f"Verified: {tool_name} found in {position}", "success")
                        return True
        
        self.log(f"Verification failed: {tool_name} was not detected in last scan", "error")
        return False
    
    def fetch_tool(self, tool_name, parent_window, voice_system=None, skip_confirmation=False):
        """Fetch a specific tool with duplicate handling"""
        tool_name_lower = tool_name.lower().strip()
        
        self.log(f"\n{'='*60}", "system")
        self.log(f"🤖 FETCH REQUEST: {tool_name.upper()}", "system")
        self.log(f"{'='*60}", "system")
        
        # Check if tool exists in mapping
        if tool_name_lower not in self.tool_mapping:
            self.log(f"Tool '{tool_name}' not found in tool mapping!", "error")
            self.show_available_tools()
            return False
        
        # Get next available location (considering fetched status)
        location = self.get_next_available_location(tool_name_lower)
        
        if location is None:
            self.log(f"All {tool_name_lower.upper()} instances have been fetched!", "error")
            total_count = len(self.tool_mapping[tool_name_lower])
            self.log(f"Total inventory: {total_count}, All fetched", "info")
            return False
        
        grab_letter = location["point"]
        confidence = location["confidence"]
        position_desc = location["position_desc"]
        
        self.log(f"Tool location: Point {grab_letter} ({position_desc})", "success")
        self.log(f"Confidence: {confidence*100:.1f}%", "info")
        
        # Show inventory status
        total_count = len(self.tool_mapping[tool_name_lower])
        fetched_count = sum(1 for loc in self.tool_mapping[tool_name_lower] if loc.get("fetched", False))
        remaining = total_count - fetched_count - 1  # Minus the one we're about to fetch
        
        if total_count > 1:
            self.log(f"Inventory: {remaining} more available after this fetch", "info")
        
        # Check if grab script exists
        if not self.check_grab_script_exists(grab_letter):
            self.log(f"No grab script found for Point {grab_letter}", "error")
            return False
        
        # SKIP CONFIRMATION FOR VOICE COMMANDS
        if not skip_confirmation:
            # Confirm with user (only for manual fetch)
            confirmation_message = f"Ready to fetch {tool_name.upper()} from Point {grab_letter}.\n\n"
            confirmation_message += f"Confidence: {confidence*100:.1f}%\n"
            confirmation_message += f"Position: {position_desc}\n"
            if total_count > 1:
                confirmation_message += f"Remaining after fetch: {remaining}\n"
            confirmation_message += "\nProceed with fetching?"
            
            reply = messagebox.askyesno("Fetch Confirmation", confirmation_message)
            if not reply:
                self.log("Fetch cancelled by user.", "info")
                return False
        
        self.log("Starting fetch sequence...", "success")
        
        # Verify object exists (optional)
        self.log("Verifying object detection...", "info")
        if not self.verify_object_exists(tool_name):
            self.log(f"WARNING: {tool_name} was not detected in the last scan!", "warning")
            
            # Only show warning for manual fetch, not for voice commands
            if not skip_confirmation:
                warning_reply = messagebox.askyesno(
                    "⚠️ Warning",
                    f"{tool_name.upper()} was not detected in last scan!\n"
                    f"The tool may have been moved or is not in the workspace.\n\n"
                    f"Do you still want to attempt fetching?"
                )
                
                if not warning_reply:
                    self.log("Fetch cancelled by user after warning.", "info")
                    return False
        
        # Execute grab sequence
        success = self.execute_grab_script(grab_letter, tool_name_lower)
        
        if success:
            # Mark as fetched
            self.save_fetched_location(tool_name_lower, grab_letter)
            location["fetched"] = True
            
            self.log(f"Successfully fetched {tool_name.upper()}!", "success")
            
            # Show remaining inventory
            if total_count > 1:
                new_remaining = total_count - (fetched_count + 1)
                if new_remaining > 0:
                    self.log(f"{new_remaining} more {tool_name.upper()} available in workshop", "info")
                else:
                    self.log(f"Last {tool_name.upper()} fetched!", "info")
            
            return True
        else:
            self.log(f"Failed to fetch {tool_name.upper()}", "error")
            return False
    
    def show_available_tools(self):
        """Show available tools from mapping with inventory status"""
        if not self.tool_mapping:
            self.log("No tools available in mapping!", "warning")
            return
        
        self.log("Available tools in mapping:", "info")
        for tool_name, locations in sorted(self.tool_mapping.items()):
            available_count = sum(1 for loc in locations if not loc.get("fetched", False))
            total_count = len(locations)
            
            status = "✅" if available_count > 0 else "⛔"
            
            if total_count > 1:
                self.log(f"  {status} {tool_name.upper():<15} → {available_count}/{total_count} available", "info")
            else:
                self.log(f"  {status} {tool_name.upper():<15} → {available_count}/{total_count} available", "info")

# ============================================
# ANALYSIS SYSTEM
# ============================================
class AnalysisSystem:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.grab_points = None
        self.detections_data = {}
        self.grab_point_assignments = {}
        self.tool_mapping = {}
        self.output_dir = None
    
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def analyze_grab_points(self, snapshot_folder):
        """Run complete grab point analysis with duplicate handling"""
        self.log("🔍 Starting grab point analysis...", "info")
        self.snapshot_folder = snapshot_folder
        
        self.load_grab_points()
        self.parse_all_snapshots()
        self.assign_tools_to_grab_points()
        self.generate_tool_mapping_with_duplicates()
        self.save_json_mapping()
        report_path = self.generate_master_report_with_duplicates()
        
        self.log("✅ Grab point analysis COMPLETE!", "success")
        return report_path
    
    def load_grab_points(self):
        """Load grab point coordinates"""
        self.grab_points = {
            "initial_position": {
                "A": {"x": 75, "y": 260},
                "B": {"x": 243, "y": 280},
                "C": {"x": 410, "y": 155},
                "D": {"x": 500, "y": 300}
            },
            "second_position": {
                "E": {"x": 260, "y": 240},
                "F": {"x": 300, "y": 155},
                "G": {"x": 425, "y": 420}
            },
            "third_position": {
                "H": {"x": 160, "y": 300},
                "I": {"x": 470, "y": 140}
            }
        }
        
        os.makedirs("config", exist_ok=True)
        with open("config/grab_points.json", "w") as f:
            json.dump(self.grab_points, f, indent=2)
        
        self.log(f"Loaded {sum(len(v) for v in self.grab_points.values())} grab points", "info")
    
    def parse_all_snapshots(self):
        """Parse detection data from all 3 snapshots"""
        positions = ["initial_position", "second_position", "third_position"]
        
        for position in positions:
            detections = self.parse_snapshot_report(position)
            self.detections_data[position] = {
                "detections": detections,
                "count": len(detections)
            }
            
            self.log(f"{position}: {len(detections)} tools detected", "info")
    
    def parse_snapshot_report(self, position_name):
        """Parse detection data from a snapshot's text report"""
        report_file = os.path.join(self.snapshot_folder, f"{position_name}_detections.txt")
        
        if not os.path.exists(report_file):
            self.log(f"Report file not found: {report_file}", "warning")
            return []
        
        detections = []
        
        with open(report_file, 'r') as f:
            content = f.read()
            
            if "DETECTED OBJECTS:" in content:
                objects_section = content.split("DETECTED OBJECTS:")[1]
                object_blocks = objects_section.split("Object #")
                
                for block in object_blocks[1:]:
                    detection = {}
                    
                    class_match = re.search(r"Class:\s*([^\n]+)", block)
                    if class_match:
                        detection["class_name"] = class_match.group(1).strip()
                    
                    conf_match = re.search(r"Confidence:\s*([^\n]+)", block)
                    if conf_match:
                        conf_str = conf_match.group(1).strip()
                        detection["confidence"] = float(conf_str.replace('%', '')) / 100
                    
                    bbox_match = re.search(r"Bounding Box:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", block)
                    if bbox_match:
                        x1, y1, x2, y2 = map(int, bbox_match.groups())
                        detection["bbox"] = (x1, y1, x2, y2)
                        detection["center"] = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    if detection:
                        detections.append(detection)
        
        return detections
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        import math
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def assign_tools_to_grab_points(self, max_distance=200):
        """Assign the closest tool to each grab point"""
        self.log("Assigning tools to grab points...", "info")
        
        assignments = {}
        
        for position_name, grab_points in self.grab_points.items():
            self.log(f"Processing {position_name}...", "info")
            
            if position_name not in self.detections_data:
                continue
            
            position_detections = self.detections_data[position_name]["detections"]
            
            if not position_detections:
                for point_id, point_coords in grab_points.items():
                    assignments[point_id] = {
                        "class_name": "none",
                        "confidence": 0.0,
                        "distance": float('inf'),
                        "grab_point": (point_coords["x"], point_coords["y"])
                    }
                continue
            
            assigned_tools = set()
            all_pairs = []
            
            for point_id, point_coords in grab_points.items():
                grab_point = (point_coords["x"], point_coords["y"])
                
                for i, detection in enumerate(position_detections):
                    if "center" not in detection:
                        continue
                    
                    tool_center = detection["center"]
                    distance = self.calculate_distance(grab_point, tool_center)
                    
                    if distance <= max_distance:
                        all_pairs.append({
                            "point_id": point_id,
                            "grab_point": grab_point,
                            "tool_index": i,
                            "distance": distance,
                            "detection": detection
                        })
            
            all_pairs.sort(key=lambda x: x["distance"])
            
            for pair in all_pairs:
                point_id = pair["point_id"]
                tool_index = pair["tool_index"]
                
                if point_id in assignments or tool_index in assigned_tools:
                    continue
                    
                detection = pair["detection"]
                assignments[point_id] = {
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "distance": pair["distance"],
                    "tool_center": detection["center"],
                    "grab_point": pair["grab_point"]
                }
                
                assigned_tools.add(tool_index)
                self.log(f"  Point {point_id}: {detection['class_name']} ({pair['distance']:.1f}px)", "info")
            
            for point_id, point_coords in grab_points.items():
                if point_id not in assignments:
                    assignments[point_id] = {
                        "class_name": "none",
                        "confidence": 0.0,
                        "distance": float('inf'),
                        "tool_center": None,
                        "grab_point": (point_coords["x"], point_coords["y"])
                    }
        
        self.grab_point_assignments = assignments
        return assignments
    
    def generate_tool_mapping_with_duplicates(self):
        """Generate mapping from tool name to ALL grab points (including duplicates)"""
        tool_mapping = {}
        
        for point_id, assignment in self.grab_point_assignments.items():
            if assignment["class_name"] != "none":
                tool_name = assignment["class_name"].lower()
                
                if tool_name not in tool_mapping:
                    tool_mapping[tool_name] = []
                
                tool_mapping[tool_name].append({
                    "grab_point": point_id,
                    "distance": assignment["distance"],
                    "confidence": assignment["confidence"],
                    "position": self.get_position_from_point_id(point_id),
                    "position_desc": self.get_position_desc(self.get_position_from_point_id(point_id))
                })
        
        # Sort each tool's locations by confidence (highest first)
        for tool_name in tool_mapping:
            tool_mapping[tool_name].sort(key=lambda x: (-x["confidence"], x["distance"]))
        
        self.tool_mapping = tool_mapping
        
        if tool_mapping:
            self.log("Tool mapping generated (with duplicates):", "success")
            for tool_name, locations in tool_mapping.items():
                if len(locations) > 1:
                    self.log(f"  {tool_name.upper():<15} → {len(locations)} locations", "info")
                else:
                    self.log(f"  {tool_name.upper():<15} → Point {locations[0]['grab_point']} ({locations[0]['confidence']*100:.1f}%)", "info")
        
        return tool_mapping
    
    def get_position_from_point_id(self, point_id):
        """Determine which position a grab point belongs to"""
        if point_id in ["A", "B", "C", "D"]:
            return "initial_position"
        elif point_id in ["E", "F", "G"]:
            return "second_position"
        elif point_id in ["H", "I"]:
            return "third_position"
        return "unknown"
    
    def get_position_desc(self, position):
        """Get human-readable position description"""
        if position == "initial_position":
            return "initial position"
        elif position == "second_position":
            return "second position"
        elif position == "third_position":
            return "third position"
        return position
    
    def save_json_mapping(self):
        """Save grab point assignments and tool mapping to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("data", "mappings", f"mapping_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        mapping_data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "snapshot_folder": self.snapshot_folder,
                "grab_points_count": len(self.grab_point_assignments)
            },
            "grab_point_assignments": self.grab_point_assignments,
            "tool_mapping": self.tool_mapping
        }
        
        json_file = os.path.join(self.output_dir, "tool_mapping.json")
        with open(json_file, 'w') as f:
            json.dump(mapping_data, f, indent=2, default=str)
        
        self.log(f"Saved JSON mapping to: {json_file}", "info")
        return json_file
    
    def generate_master_report_with_duplicates(self):
        """Generate comprehensive master report with duplicate handling"""
        total_points = len(self.grab_point_assignments)
        assigned_points = sum(1 for a in self.grab_point_assignments.values() if a["class_name"] != "none")
        
        report_lines = []
        report_lines.append("=" * 75)
        report_lines.append(" " * 20 + "GARAGE ASSISTANT - TOOL MAPPING REPORT")
        report_lines.append("=" * 75)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Grab Points Analyzed: {total_points}")
        report_lines.append(f"Points with Assigned Tools: {assigned_points}")
        report_lines.append(f"Points without Tools: {total_points - assigned_points}")
        report_lines.append(f"Unique Tools: {len(self.tool_mapping)}")
        report_lines.append("")
        
        report_lines.append("GRAB POINT ASSIGNMENTS:")
        report_lines.append("=" * 75)
        
        for position_name, points in [("INITIAL POSITION", ["A", "B", "C", "D"]),
                                      ("SECOND POSITION", ["E", "F", "G"]),
                                      ("THIRD POSITION", ["H", "I"])]:
            report_lines.append(f"\n{position_name}:")
            report_lines.append("-" * 40)
            for point_id in points:
                if point_id in self.grab_point_assignments:
                    assignment = self.grab_point_assignments[point_id]
                    if assignment["class_name"] != "none":
                        report_lines.append(f"  Point {point_id}: {assignment['class_name'].upper()} "
                                          f"({assignment['confidence']*100:.1f}%)")
                    else:
                        report_lines.append(f"  Point {point_id}: No tool assigned")
        
        report_lines.append("\nALL TOOL LOCATIONS (INCLUDING DUPLICATES):")
        report_lines.append("-" * 40)
        
        if self.tool_mapping:
            for tool_name, locations in sorted(self.tool_mapping.items()):
                if len(locations) > 1:
                    report_lines.append(f"\n{tool_name.upper()} ({len(locations)} locations):")
                    for i, loc in enumerate(locations, 1):
                        star = "⭐ " if i == 1 else "  "
                        position_desc = self.get_position_desc(loc["position"])
                        report_lines.append(f"  {star}Point {loc['grab_point']}: {loc['confidence']*100:.1f}% ({position_desc})")
                else:
                    report_lines.append(f"\n{tool_name.upper()} (1 location):")
                    position_desc = self.get_position_desc(locations[0]["position"])
                    report_lines.append(f"  ⭐ Point {locations[0]['grab_point']}: {locations[0]['confidence']*100:.1f}% ({position_desc})")
        else:
            report_lines.append("  No tools found")
        
        report_lines.append("\nROBOT ACTION PLAN (WITH DUPLICATE HANDLING):")
        report_lines.append("-" * 40)
        
        if self.tool_mapping:
            for tool_name, locations in sorted(self.tool_mapping.items()):
                if len(locations) > 1:
                    report_lines.append(f"\nWhen user requests '{tool_name.lower()}':")
                    report_lines.append(f"  Available at {len(locations)} locations:")
                    for i, loc in enumerate(locations, 1):
                        star = "⭐ " if i == 1 else "  "
                        position_desc = self.get_position_desc(loc["position"])
                        report_lines.append(f"  {star}Point {loc['grab_point']} ({position_desc})")
                    report_lines.append(f"  Will fetch from: Point {locations[0]['grab_point']} (highest confidence)")
                else:
                    report_lines.append(f"\nWhen user requests '{tool_name.lower()}':")
                    position_desc = self.get_position_desc(locations[0]["position"])
                    report_lines.append(f"  Will fetch from: Point {locations[0]['grab_point']} ({position_desc})")
        else:
            report_lines.append("  No action plan available (no tools mapped)")
        
        report_file = os.path.join(self.output_dir, "master_report.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.log(f"Saved master report to: {report_file}", "info")
        return report_file

# ============================================
# TOOL PROMPT DIALOG (Tkinter)
# ============================================
class ToolPromptDialog:
    def __init__(self, parent, tool_mapping):
        self.parent = parent
        self.tool_mapping = tool_mapping
        self.selected_tool = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("🤖 Fetch Tool")
        self.dialog.geometry("400x500")
        self.dialog.configure(bg="#1e1e2e")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.init_ui()
    
    def init_ui(self):
        # Title
        title_label = tk.Label(
            self.dialog,
            text="Select or Type Tool Name:",
            font=("Arial", 16, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e"
        )
        title_label.pack(pady=10)
        
        # Available tools list
        list_label = tk.Label(
            self.dialog,
            text="Available Tools:",
            font=("Arial", 12),
            fg="#cdd6f4",
            bg="#1e1e2e"
        )
        list_label.pack(anchor="w", padx=20)
        
        # Listbox for tools
        self.tools_listbox = tk.Listbox(
            self.dialog,
            bg="#313244",
            fg="#cdd6f4",
            selectbackground="#89b4fa",
            selectforeground="#1e1e2e",
            font=("Arial", 12),
            height=8
        )
        self.tools_listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=(5, 10))
        
        for tool_name in sorted(self.tool_mapping.keys()):
            self.tools_listbox.insert(tk.END, tool_name.upper())
        
        self.tools_listbox.bind('<<ListboxSelect>>', self.on_list_select)
        self.tools_listbox.bind('<Double-Button-1>', self.on_double_click)
        
        # Or type manually
        type_label = tk.Label(
            self.dialog,
            text="Or type tool name:",
            font=("Arial", 12),
            fg="#cdd6f4",
            bg="#1e1e2e"
        )
        type_label.pack(anchor="w", padx=20)
        
        self.tool_input = tk.Entry(
            self.dialog,
            bg="#313244",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            font=("Arial", 12),
            width=30
        )
        self.tool_input.pack(padx=20, pady=(5, 20), fill=tk.X)
        self.tool_input.bind('<KeyRelease>', self.check_input)
        
        # Buttons
        button_frame = tk.Frame(self.dialog, bg="#1e1e2e")
        button_frame.pack(pady=(0, 20))
        
        self.ok_button = tk.Button(
            button_frame,
            text="✅ Fetch",
            command=self.on_ok,
            bg="#585b70",
            fg="#cdd6f4",
            font=("Arial", 12, "bold"),
            width=12,
            state=tk.DISABLED
        )
        self.ok_button.pack(side=tk.LEFT, padx=10)
        
        self.cancel_button = tk.Button(
            button_frame,
            text="❌ Cancel",
            command=self.on_cancel,
            bg="#585b70",
            fg="#cdd6f4",
            font=("Arial", 12, "bold"),
            width=12
        )
        self.cancel_button.pack(side=tk.LEFT, padx=10)
    
    def on_list_select(self, event):
        selection = self.tools_listbox.curselection()
        if selection:
            tool_name = self.tools_listbox.get(selection[0])
            self.tool_input.delete(0, tk.END)
            self.tool_input.insert(0, tool_name.lower())
            self.ok_button.config(state=tk.NORMAL)
    
    def on_double_click(self, event):
        self.on_ok()
    
    def check_input(self, event):
        if self.tool_input.get().strip():
            self.ok_button.config(state=tk.NORMAL)
        else:
            self.ok_button.config(state=tk.DISABLED)
    
    def on_ok(self):
        tool_name = self.tool_input.get().strip().lower()
        if tool_name:
            self.selected_tool = tool_name
            self.dialog.destroy()
    
    def on_cancel(self):
        self.selected_tool = None
        self.dialog.destroy()
    
    def show(self):
        self.dialog.wait_window()
        return self.selected_tool

# ============================================
# MAIN GUI WINDOW (Tkinter) - UPDATED
# ============================================
class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🏗️ Garage Assistant Pro")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1e1e2e")
        
        # System components
        self.arm_controller = None
        self.camera_system = None
        self.snapshot_system = None
        self.analysis_system = None
        self.grab_system = None
        self.voice_system = None
        
        # Current state
        self.scanning = False
        self.fetching = False
        self.current_snapshot_folder = None
        self.tool_mapping = {}
        self.voice_enabled = False
        
        # Camera display
        self.current_frame = None
        self.camera_image = None
        
        # Initialize GUI first
        self.init_ui()
        
        # Now start systems after GUI is initialized
        self.initialize_systems()
        
        # Start periodic updates
        self.update_camera()
        self.root.after(100, self.periodic_update)
    
    def init_ui(self):
        # Main container
        main_container = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#1e1e2e")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel
        left_panel = tk.Frame(main_container, bg="#1e1e2e")
        main_container.add(left_panel, width=650)
        
        # Right panel
        right_panel = tk.Frame(main_container, bg="#1e1e2e")
        main_container.add(right_panel, width=400)
        
        # ========== LEFT PANEL ==========
        
        # Camera display
        camera_frame = tk.LabelFrame(
            left_panel,
            text="🎥 Live Camera Feed",
            font=("Arial", 12, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e",
            relief=tk.RAISED,
            borderwidth=2
        )
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.camera_label = tk.Label(
            camera_frame,
            bg="black",
            relief=tk.SUNKEN,
            borderwidth=3
        )
        self.camera_label.pack(padx=10, pady=10)
        
        # Control buttons
        control_frame = tk.LabelFrame(
            left_panel,
            text="⚙️ System Controls",
            font=("Arial", 12, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e",
            relief=tk.RAISED,
            borderwidth=2
        )
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Voice button
        self.voice_button = tk.Button(
            control_frame,
            text="🎤 Enable Voice",
            command=self.toggle_voice_recognition,
            bg="#74c7ec",
            fg="#1e1e2e",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        )
        self.voice_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Scan button
        self.scan_button = tk.Button(
            control_frame,
            text="🔍 Start Scan & Analysis",
            command=self.start_scan,
            bg="#a6e3a1",
            fg="#1e1e2e",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        )
        self.scan_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Home button
        self.home_button = tk.Button(
            control_frame,
            text="🏠 Go Home",
            command=self.go_home,
            bg="#585b70",
            fg="#cdd6f4",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        )
        self.home_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Fetch button
        self.fetch_prompt_button = tk.Button(
            control_frame,
            text="🤖 Fetch Tool...",
            command=self.prompt_for_tool,
            bg="#cba6f7",
            fg="#1e1e2e",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        )
        self.fetch_prompt_button.grid(row=0, column=3, padx=5, pady=5)
        
        # Clear log button
        clear_log_btn = tk.Button(
            control_frame,
            text="🗑️ Clear Log",
            command=self.clear_log,
            bg="#585b70",
            fg="#cdd6f4",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        )
        clear_log_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Exit button
        exit_button = tk.Button(
            control_frame,
            text="🚪 Exit",
            command=self.on_closing,
            bg="#f38ba8",
            fg="#1e1e2e",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        )
        exit_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Status labels
        status_frame = tk.Frame(control_frame, bg="#1e1e2e")
        status_frame.grid(row=2, column=0, columnspan=4, pady=(10, 5), sticky="ew")
        
        self.status_label = tk.Label(
            status_frame,
            text="Status: Ready",
            font=("Arial", 11, "bold"),
            fg="#a6e3a1",
            bg="#1e1e2e"
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        self.voice_status_label = tk.Label(
            status_frame,
            text="Voice: Disabled",
            font=("Arial", 11, "bold"),
            fg="#74c7ec",
            bg="#1e1e2e"
        )
        self.voice_status_label.pack(side=tk.RIGHT, padx=20)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            control_frame,
            length=400,
            mode='determinate'
        )
        self.progress_bar.grid(row=3, column=0, columnspan=4, padx=5, pady=(5, 10), sticky="ew")
        
        # Tools list
        tools_frame = tk.LabelFrame(
            left_panel,
            text="🛠 Mapped Tools",
            font=("Arial", 12, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e",
            relief=tk.RAISED,
            borderwidth=2
        )
        tools_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tools listbox
        self.tools_listbox = tk.Listbox(
            tools_frame,
            bg="#313244",
            fg="#cdd6f4",
            selectbackground="#89b4fa",
            selectforeground="#1e1e2e",
            font=("Arial", 11)
        )
        self.tools_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.tools_listbox.bind('<Double-Button-1>', lambda e: self.prompt_for_tool())
        
        # ========== RIGHT PANEL ==========
        
        # Logger
        logger_frame = tk.LabelFrame(
            right_panel,
            text="📋 System Logger",
            font=("Arial", 12, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e",
            relief=tk.RAISED,
            borderwidth=2
        )
        logger_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Text widget for logging
        self.logger_text = scrolledtext.ScrolledText(
            logger_frame,
            bg="#181825",
            fg="#a6adc8",
            font=("Courier New", 10),
            width=50,
            height=25
        )
        self.logger_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # System info
        info_frame = tk.LabelFrame(
            right_panel,
            text="ℹ️ System Information",
            font=("Arial", 12, "bold"),
            fg="#89b4fa",
            bg="#1e1e2e",
            relief=tk.RAISED,
            borderwidth=2
        )
        info_frame.pack(fill=tk.X)
        
        self.info_label = tk.Label(
            info_frame,
            text="Arm Status: Not connected\n"
                 "Camera Status: Not connected\n"
                 "Last Scan: None\n"
                 "Tools Mapped: 0\n"
                 "Last Fetch: None\n"
                 "Voice Status: Disabled",
            font=("Monospace", 10),
            fg="#cba6f7",
            bg="#1e1e2e",
            justify=tk.LEFT,
            anchor="w"
        )
        self.info_label.pack(fill=tk.X, padx=10, pady=10)
        
        # Add initial log message
        self.log("🏗️ Garage Assistant Pro Initializing...", "system")
    
    def initialize_systems(self):
        """Initialize all systems - called AFTER GUI is set up"""
        try:
            self.log("🚀 Initializing systems...", "system")
            
            self.log("🤖 Initializing robot arm...", "info")
            self.arm_controller = RobotArmController(log_callback=self.log)
            
            self.log("📷 Starting camera...", "info")
            self.camera_system = CameraSystem(
                log_callback=self.log,
                frame_callback=self.update_camera_frame
            )
            self.camera_system.start()
            
            self.log("🎤 Initializing voice recognition...", "info")
            self.voice_system = VoiceRecognitionSystem(log_callback=self.log)
            self.voice_system.set_command_callback(self.handle_voice_command)
            
            self.snapshot_system = SnapshotSystem(self.arm_controller, self.camera_system, self.log)
            self.analysis_system = AnalysisSystem(self.log)
            self.grab_system = GrabSystem(self.arm_controller, self.snapshot_system, self.camera_system, self.log)
            
            self.log("✅ All systems initialized successfully!", "success")
            self.update_info("Arm: Connected ✓", "Camera: Running ✓", "Voice: Ready ✓")
            
        except Exception as e:
            self.log(f"❌ Failed to initialize systems: {e}", "error")
    
    def log(self, message, level="info"):
        """Log message to logger and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding - MODIFIED: Removed red, made more intuitive
        colors = {
            "info": "#89b4fa",        # Blue for regular info
            "success": "#a6e3a1",     # Green for success
            "warning": "#f9e2af",     # Yellow for warnings
            "error": "#f5c2e7",       # REMOVED RED: Changed to pink (softer)
            "system": "#cba6f7"       # Purple for system messages
        }
        
        color = colors.get(level, "#cdd6f4")
        formatted_message = f"[{timestamp}] {message}\n"
        
        # Safely insert into logger_text (check if it exists)
        if hasattr(self, 'logger_text') and self.logger_text:
            self.logger_text.insert(tk.END, formatted_message)
            self.logger_text.tag_add(level, f"end-{len(formatted_message)+1}c", "end")
            self.logger_text.tag_config(level, foreground=color)
            self.logger_text.see(tk.END)  # Auto-scroll to bottom
        else:
            # Fallback to console if logger_text not ready
            print(f"[{timestamp}] {message}")
        
        # Update status for important messages
        if hasattr(self, 'status_label') and self.status_label:
            if level == "error":
                self.status_label.config(text=f"Status: Error - {message[:30]}...")
            elif level == "success":
                self.status_label.config(text=f"Status: {message[:40]}...")
    
    def update_camera_frame(self, frame):
        """Update camera frame from camera thread"""
        self.current_frame = frame
    
    def update_camera(self):
        """Update camera display in GUI thread - RESIZED TO HALF"""
        if self.current_frame is not None:
            try:
                rgb_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                h, w = rgb_image.shape[:2]
                
                # RESIZE TO HALF
                new_w = w // 2
                new_h = h // 2
                
                resized = cv2.resize(rgb_image, (new_w, new_h))
                
                # Convert to PhotoImage
                image = Image.fromarray(resized)
                self.camera_image = ImageTk.PhotoImage(image)
                
                self.camera_label.config(image=self.camera_image)
                
            except Exception as e:
                pass
        
        # Schedule next update
        self.root.after(33, self.update_camera)  # ~30 FPS
    
    def periodic_update(self):
        """Periodic updates for GUI"""
        self.root.after(100, self.periodic_update)
    
    def toggle_voice_recognition(self):
        """Toggle voice recognition on/off"""
        if not self.voice_enabled:
            self.enable_voice_recognition()
        else:
            self.disable_voice_recognition()
    
    def enable_voice_recognition(self):
        """Enable voice recognition"""
        if self.voice_system is None:
            self.log("Voice system not initialized!", "error")
            return
        
        self.log("🎤 Enabling voice recognition...", "system")
        self.voice_enabled = True
        
        # Update UI
        self.voice_button.config(
            text="🔴 Disable Voice",
            bg="#f38ba8",
            fg="#1e1e2e"
        )
        self.voice_status_label.config(
            text="Voice: Listening...",
            fg="#a6e3a1"
        )
        
        # Start voice recognition in separate thread
        self.voice_system.voice_enabled = True
        self.voice_thread = threading.Thread(target=self.voice_system.run)
        self.voice_thread.daemon = True
        self.voice_thread.start()
        
        self.log("✅ Voice recognition enabled. Speak commands now.", "success")
        self.update_info("Voice Status: Listening ✓")
    
    def disable_voice_recognition(self):
        """Disable voice recognition"""
        self.log("🎤 Disabling voice recognition...", "system")
        self.voice_enabled = False
        
        # Update UI
        self.voice_button.config(
            text="🎤 Enable Voice",
            bg="#74c7ec",
            fg="#1e1e2e"
        )
        self.voice_status_label.config(
            text="Voice: Disabled",
            fg="#74c7ec"
        )
        
        # Stop voice recognition
        if self.voice_system:
            self.voice_system.voice_enabled = False
            self.voice_system.stop()
        
        self.log("✅ Voice recognition disabled.", "success")
        self.update_info("Voice Status: Disabled")
    
    def handle_voice_command(self, tool_name):
        """Handle voice command received"""
        self.log(f"🎤 Voice command received: Fetch {tool_name}", "system")
        
        if self.scanning:
            self.log("Cannot process voice command during scan!", "warning")
            return
        
        if self.fetching:
            self.log("Already fetching a tool!", "warning")
            return
        
        # Schedule fetch in main thread
        self.root.after(0, lambda: self.start_fetch_tool_with_voice(tool_name))
    
    def start_fetch_tool_with_voice(self, tool_name):
        """Start fetching a tool with voice confirmation"""
        if self.fetching:
            self.log("Another fetch operation is already in progress!", "warning")
            return
        
        self.fetching = True
        self.fetch_prompt_button.config(state=tk.DISABLED)
        self.voice_button.config(state=tk.DISABLED)
        self.status_label.config(text=f"Status: Fetching {tool_name.upper()}...")
        
        # Pause voice listening
        if self.voice_system:
            self.voice_system.pause_listening()
            self.log("Voice listening paused during fetch operation", "info")
        
        # Run in separate thread
        self.fetch_thread = threading.Thread(target=self.run_fetch_sequence_with_voice, args=(tool_name,))
        self.fetch_thread.daemon = True
        self.fetch_thread.start()
    
    def run_fetch_sequence_with_voice(self, tool_name):
        """Run the fetch sequence with voice confirmation"""
        try:
            # Load mapping if not already loaded
            if not self.tool_mapping:
                self.grab_system.load_mapping()
                self.tool_mapping = self.grab_system.tool_mapping
            
            # Actually fetch the tool (skip confirmation for voice commands)
            success = self.grab_system.fetch_tool(tool_name, self.root, skip_confirmation=True)
            
            if success:
                self.log(f"✅ Successfully fetched {tool_name.upper()}!", "success")
                self.update_info(
                    f"Last Fetch: {datetime.now().strftime('%H:%M:%S')}",
                    f"Last Tool: {tool_name.upper()}"
                )
                
                # Speak confirmation message AFTER successful fetch
                confirmation_msg = f"Here is your {tool_name}! What else can I get for you?"
                self.voice_system.speak(confirmation_msg)
                
            else:
                self.log(f"❌ Failed to fetch {tool_name.upper()}", "error")
                
        except Exception as e:
            self.log(f"❌ Fetch error: {e}", "error")
        
        finally:
            self.fetching = False
            # Resume voice listening after fetch is complete
            if self.voice_system and self.voice_enabled:
                self.voice_system.resume_listening()
                self.log("Voice listening resumed", "info")
            
            self.root.after(0, self.enable_fetch_buttons)
    
    def prompt_for_tool(self):
        """Prompt user for which tool to fetch"""
        if not self.tool_mapping:
            self.log("❌ No tool mapping available! Please run scan first.", "error")
            return
        
        dialog = ToolPromptDialog(self.root, self.tool_mapping)
        tool_name = dialog.show()
        
        if tool_name:
            self.log(f"User requested to fetch: {tool_name.upper()}", "system")
            self.start_fetch_tool(tool_name)
    
    def start_fetch_tool(self, tool_name):
        """Start fetching a tool"""
        if self.fetching:
            self.log("Another fetch operation is already in progress!", "warning")
            return
        
        self.fetching = True
        self.fetch_prompt_button.config(state=tk.DISABLED)
        self.voice_button.config(state=tk.DISABLED)
        self.status_label.config(text=f"Status: Fetching {tool_name.upper()}...")
        
        # Run in separate thread
        self.fetch_thread = threading.Thread(target=self.run_fetch_sequence, args=(tool_name,))
        self.fetch_thread.daemon = True
        self.fetch_thread.start()
    
    def run_fetch_sequence(self, tool_name):
        """Run the fetch sequence"""
        try:
            # Load mapping if not already loaded
            if not self.tool_mapping:
                self.grab_system.load_mapping()
                self.tool_mapping = self.grab_system.tool_mapping
            
            success = self.grab_system.fetch_tool(tool_name, self.root)
            
            if success:
                self.log(f"✅ Successfully fetched {tool_name.upper()}!", "success")
                self.update_info(
                    f"Last Fetch: {datetime.now().strftime('%H:%M:%S')}",
                    f"Last Tool: {tool_name.upper()}"
                )
            else:
                self.log(f"❌ Failed to fetch {tool_name.upper()}", "error")
                
        except Exception as e:
            self.log(f"❌ Fetch error: {e}", "error")
        
        finally:
            self.fetching = False
            self.root.after(0, self.enable_fetch_buttons)
    
    def enable_fetch_buttons(self):
        """Re-enable fetch buttons"""
        self.fetch_prompt_button.config(state=tk.NORMAL)
        self.voice_button.config(state=tk.NORMAL)
        self.home_button.config(state=tk.NORMAL)
        self.scan_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Ready")
    
    def start_scan(self):
        """Start the automatic scan and analysis"""
        if self.scanning:
            self.log("Scan already in progress!", "warning")
            return
        
        self.scanning = True
        self.scan_button.config(state=tk.DISABLED)
        self.fetch_prompt_button.config(state=tk.DISABLED)
        self.voice_button.config(state=tk.DISABLED)
        self.home_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Scanning...")
        self.progress_bar['value'] = 0
        
        # Run in separate thread
        self.scan_thread = threading.Thread(target=self.run_scan_sequence)
        self.scan_thread.daemon = True
        self.scan_thread.start()
    
    def run_scan_sequence(self):
        """Run complete scan sequence"""
        try:
            self.log("\n" + "="*50, "system")
            self.log("STARTING AUTOMATIC SCAN SEQUENCE", "system")
            self.log("="*50, "system")
            
            self.log("📸 Step 1: Taking snapshots...", "info")
            self.update_progress(10)
            
            self.current_snapshot_folder = self.snapshot_system.take_snapshots_sequence()
            self.update_progress(50)
            
            self.log("\n🔍 Step 2: Analyzing grab points...", "info")
            report_path = self.analysis_system.analyze_grab_points(self.current_snapshot_folder)
            self.update_progress(80)
            
            self.log("\n🗺️ Step 3: Loading tool mapping...", "info")
            self.tool_mapping = self.grab_system.load_mapping(report_path)
            self.update_progress(90)
            
            self.root.after(0, self.update_tools_list)
            self.update_progress(100)
            
            self.log("\n✅ SCAN COMPLETE!", "success")
            self.log(f"📊 Tools mapped: {len(self.tool_mapping)}", "success")
            
            self.update_info(
                f"Last Scan: {datetime.now().strftime('%H:%M:%S')}",
                f"Tools Mapped: {len(self.tool_mapping)}"
            )
            
        except Exception as e:
            self.log(f"❌ Scan failed: {e}", "error")
        
        finally:
            self.scanning = False
            self.root.after(0, self.enable_scan_buttons)
    
    def update_progress(self, value):
        """Update progress bar from thread"""
        self.root.after(0, lambda: self.progress_bar.config(value=value))
    
    def enable_scan_buttons(self):
        """Re-enable scan buttons"""
        self.scan_button.config(state=tk.NORMAL)
        self.fetch_prompt_button.config(state=tk.NORMAL)
        self.voice_button.config(state=tk.NORMAL)
        self.home_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Ready")
    
    def update_tools_list(self):
        """Update the tools list widget"""
        self.tools_listbox.delete(0, tk.END)
        
        color_map = {
            "hammer": "#f38ba8",
            "screwdriver": "#89b4fa",
            "wrench": "#f9e2af",
            "plier": "#a6e3a1",
            "bolt": "#cba6f7",
            "measuring tape": "#f5c2e7",
            "tape": "#f5c2e7"
        }
        
        for tool_name in sorted(self.tool_mapping.keys()):
            self.tools_listbox.insert(tk.END, tool_name.upper())
            
            # Find the right color
            color = "#cdd6f4"  # Default color
            for key, col in color_map.items():
                if key in tool_name.lower():
                    color = col
                    break
            
            self.tools_listbox.itemconfig(tk.END, {'fg': color})
        
        if self.tool_mapping:
            self.log(f"Updated tools list with {len(self.tool_mapping)} tools", "success")
    
    def go_home(self):
        """Move arm to home position"""
        self.log("Moving arm to home position...", "info")
        try:
            self.arm_controller.go_to_home()
            self.log("✅ Arm at home position", "success")
        except Exception as e:
            self.log(f"❌ Failed to go home: {e}", "error")
    
    def update_info(self, *args):
        """Update system information"""
        text = "\n".join(args)
        self.info_label.config(text=text)
    
    def clear_log(self):
        """Clear the logger"""
        self.logger_text.delete(1.0, tk.END)
        self.log("Log cleared", "info")
    
    def run(self):
        """Start the main loop"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Cleanup on window close"""
        self.log("Shutting down systems...", "system")
        
        self.scanning = False
        self.fetching = False
        
        if self.voice_enabled:
            self.disable_voice_recognition()
        
        if self.camera_system:
            self.camera_system.stop()
        
        try:
            if self.arm_controller:
                self.arm_controller.go_to_home()
        except:
            pass
        
        self.log("✅ Goodbye!", "success")
        self.root.quit()
        self.root.destroy()

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # Create necessary directories
    os.makedirs("data/snapshots", exist_ok=True)
    os.makedirs("data/mappings", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("movements", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # os.remove("data/fetched_tools.json") if os.path.exists("data/fetched_tools.json") else None

    if os.path.exists("data/fetched_tools.json"):
        os.remove("data/fetched_tools.json")
        print("Removed existing fetched_tools.json file.")
    
    # Check for grab point scripts
    movements_dir = "movements"
    if not os.path.exists(movements_dir):
        print(f"Warning: {movements_dir} directory not found!")
        print("Please create grab point scripts (grab_point_A.py to grab_point_I.py) in the movements directory.")
    
    # Create and run main window
    window = MainWindow()
    window.run()

if __name__ == "__main__":
    main()