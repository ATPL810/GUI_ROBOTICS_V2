#!/usr/bin/env python3
# coding: utf-8
"""
DOFBOT-PI Garage Assistant
Tool Detection and Pick/Place System
"""

import cv2
import numpy as np
import time
import json
import os
from pathlib import Path
import threading
from Arm_Lib import Arm_Device
import torch
import torch.backends.cudnn as cudnn

class GarageAssistant:
    def __init__(self, yolo_model_path="best.pt", calibration_file="calibration.json"):
        # Initialize robot arm
        self.arm = Arm_Device()
        
        # Initialize camera
        self.camera = None
        self.camera_init()
        
        # Load YOLO model
        self.model = self.load_yolo_model(yolo_model_path)
        self.class_names = self.get_class_names()
        
        # Calibration parameters
        self.calibration_file = calibration_file
        self.calibration_params = self.load_calibration()
        
        # Workspace dimensions (in mm, adjust based on your setup)
        self.workspace_bounds = {
            'x_min': -200, 'x_max': 200,
            'y_min': -200, 'y_max': 200,
            'z_min': 0, 'z_max': 250
        }
        
        # Tool drop-off zone coordinates
        self.drop_zone = {'x': 150, 'y': 0, 'z': 50}
        
        # Arm home position (angles for all 6 servos)
        self.home_position = [90, 90, 90, 90, 90, 90]
        
        # Tool properties database
        self.tools_db = self.load_tools_database()
        
        print("Garage Assistant initialized")
    
    def camera_init(self):
        """Initialize Raspberry Pi camera"""
        try:
            # For Raspberry Pi camera module
            from picamera2 import Picamera2
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(main={"size": (640, 480)})
            self.camera.configure(config)
            self.camera.start()
            print("Camera initialized successfully")
        except ImportError:
            # Fallback to USB camera
            print("Using USB camera (Picamera2 not available)")
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def load_yolo_model(self, model_path):
        """Load YOLOv5 model"""
        try:
            # Use YOLOv5 from torch hub
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            model.conf = 0.6  # Confidence threshold
            model.iou = 0.45  # NMS IoU threshold
            if torch.cuda.is_available():
                model.cuda()
                cudnn.benchmark = True
            print(f"YOLO model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None
    
    def get_class_names(self):
        """Get class names from YOLO model"""
        if hasattr(self.model, 'names'):
            return self.model.names
        return {}
    
    def capture_image(self, save_path=None):
        """Capture a single image from camera"""
        try:
            if hasattr(self.camera, 'capture_file'):
                # Picamera2
                if save_path:
                    self.camera.capture_file(save_path)
                temp_file = "/tmp/temp_capture.jpg"
                self.camera.capture_file(temp_file)
                image = cv2.imread(temp_file)
                return image
            else:
                # USB camera
                ret, frame = self.camera.read()
                if ret:
                    if save_path:
                        cv2.imwrite(save_path, frame)
                    return frame
        except Exception as e:
            print(f"Error capturing image: {e}")
        return None
    
    def detect_tools(self, image):
        """Run YOLO detection on image"""
        if self.model is None or image is None:
            return []
        
        # Run inference
        results = self.model(image)
        
        # Parse results
        detections = []
        if hasattr(results, 'xyxy'):
            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': class_id,
                    'class_name': class_name,
                    'center_pixel': [(x1 + x2) // 2, (y1 + y2) // 2]
                }
                detections.append(detection)
        
        return detections
    
    def pixel_to_world_coordinates(self, pixel_x, pixel_y, depth=None):
        """
        Convert 2D pixel coordinates to 3D world coordinates.
        This uses camera calibration and workspace mapping.
        """
        if not self.calibration_params:
            print("Calibration not loaded. Using default mapping.")
            return self.default_pixel_to_world(pixel_x, pixel_y)
        
        # Get image dimensions
        img_height = self.calibration_params.get('image_height', 480)
        img_width = self.calibration_params.get('image_width', 640)
        
        # Normalize pixel coordinates
        norm_x = pixel_x / img_width
        norm_y = pixel_y / img_height
        
        # Use calibration matrix if available
        if 'homography_matrix' in self.calibration_params:
            # For planar workspace (tools on a table)
            pts_src = np.array([[norm_x, norm_y]], dtype='float32')
            h_matrix = np.array(self.calibration_params['homography_matrix'])
            pts_dst = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), h_matrix)
            world_x, world_y = pts_dst[0][0]
        else:
            # Simple linear mapping (calibrate this for your setup)
            world_x = self.workspace_bounds['x_min'] + \
                     norm_x * (self.workspace_bounds['x_max'] - self.workspace_bounds['x_min'])
            world_y = self.workspace_bounds['y_min'] + \
                     (1 - norm_y) * (self.workspace_bounds['y_max'] - self.workspace_bounds['y_min'])
        
        # Estimate Z based on tool type or use fixed height
        if depth is not None:
            world_z = depth
        else:
            world_z = self.calibration_params.get('default_z', 30)
        
        return {'x': float(world_x), 'y': float(world_y), 'z': float(world_z)}
    
    def default_pixel_to_world(self, pixel_x, pixel_y):
        """Default mapping (requires calibration for your setup)"""
        # These values need calibration for your specific camera setup
        world_x = -200 + (pixel_x / 640) * 400
        world_y = 200 - (pixel_y / 480) * 400  # Inverted Y axis
        world_z = 30  # Default height above table
        
        return {'x': world_x, 'y': world_y, 'z': world_z}
    
    def inverse_kinematics(self, target_position, tool_type=None):
        """
        Simple inverse kinematics for DOFBOT arm.
        This is a simplified version - you may need to adjust for your specific arm.
        """
        x, y, z = target_position['x'], target_position['y'], target_position['z']
        
        # Base rotation (servo 1)
        base_angle = np.degrees(np.arctan2(y, x))
        base_angle = np.clip(base_angle, -90, 90)
        
        # Distance in XY plane
        r = np.sqrt(x**2 + y**2)
        
        # Adjust r based on tool properties
        if tool_type and tool_type in self.tools_db:
            r += self.tools_db[tool_type].get('grip_offset', 0)
        
        # Simple 2-link arm approximation (adjust for your arm)
        # This is a placeholder - you need proper kinematics for your DOFBOT
        l1 = 100  # Shoulder to elbow length (mm)
        l2 = 100  # Elbow to wrist length (mm)
        
        # Calculate angles (simplified)
        try:
            # Law of cosines for 2-link planar arm
            D = (r**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
            D = np.clip(D, -1, 1)
            
            theta2 = np.arccos(D)
            theta1 = np.arctan2(z, r) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(D))
            
            shoulder_angle = 90 - np.degrees(theta1)
            elbow_angle = 90 - np.degrees(theta2)
            
            # Wrist angle to keep gripper level
            wrist_angle = 90 - (shoulder_angle + elbow_angle)
            
            angles = [
                90 + base_angle,          # Base
                np.clip(shoulder_angle, 0, 180),    # Shoulder
                np.clip(elbow_angle, 0, 180),       # Elbow
                np.clip(wrist_angle, 0, 180),       # Wrist
                90,                       # Gripper rotation (adjust as needed)
                45                        # Gripper open (will close later)
            ]
            
            return angles
            
        except Exception as e:
            print(f"IK error: {e}")
            return None
    
    def move_to_position(self, target_position, tool_type=None, move_time=2000):
        """Move arm to target position"""
        angles = self.inverse_kinematics(target_position, tool_type)
        if angles:
            print(f"Moving to angles: {angles}")
            self.arm.Arm_serial_servo_write6_array(angles, move_time)
            time.sleep(move_time / 1000 + 0.5)
            return True
        return False
    
    def pick_tool(self, tool_position, tool_type):
        """Execute pick sequence for a tool"""
        print(f"Picking {tool_type} at position {tool_position}")
        
        # Approach position (above tool)
        approach_pos = tool_position.copy()
        approach_pos['z'] += 50  # 50mm above tool
        
        # Move to approach
        if not self.move_to_position(approach_pos, tool_type):
            return False
        
        time.sleep(1)
        
        # Move down to tool
        if not self.move_to_position(tool_position, tool_type):
            return False
        
        time.sleep(1)
        
        # Close gripper
        print("Closing gripper")
        self.arm.Arm_serial_servo_write(6, 135, 500)  # Close gripper (adjust angle)
        time.sleep(0.5)
        
        # Lift tool
        lift_pos = tool_position.copy()
        lift_pos['z'] += 100
        self.move_to_position(lift_pos, tool_type)
        
        return True
    
    def place_tool(self):
        """Place tool at drop zone"""
        print("Placing tool at drop zone")
        
        # Approach drop zone
        approach_drop = self.drop_zone.copy()
        approach_drop['z'] += 50
        
        self.move_to_position(approach_drop, None)
        time.sleep(1)
        
        # Lower to drop position
        self.move_to_position(self.drop_zone, None)
        time.sleep(1)
        
        # Open gripper
        print("Opening gripper")
        self.arm.Arm_serial_servo_write(6, 45, 500)  # Open gripper (adjust angle)
        time.sleep(0.5)
        
        # Lift away
        self.move_to_position(approach_drop, None)
        
        return True
    
    def scan_workspace(self, num_images=3):
        """Take multiple images and detect tools"""
        all_detections = []
        
        print(f"Scanning workspace with {num_images} images...")
        
        for i in range(num_images):
            # Capture image
            image = self.capture_image(f"scan_{i}.jpg")
            if image is None:
                continue
            
            # Detect tools
            detections = self.detect_tools(image)
            
            # Convert to world coordinates
            for det in detections:
                center_x, center_y = det['center_pixel']
                world_pos = self.pixel_to_world_coordinates(center_x, center_y)
                det['world_position'] = world_pos
                all_detections.append(det)
            
            # Optional: move arm slightly for different viewpoint
            if i < num_images - 1:
                self.arm.Arm_serial_servo_write(1, 90 + (i+1)*10, 1000)
                time.sleep(1)
        
        # Return to home
        self.go_home()
        
        return all_detections
    
    def find_specific_tool(self, tool_name):
        """Find a specific tool in the workspace"""
        detections = self.scan_workspace()
        
        for det in detections:
            if det['class_name'].lower() == tool_name.lower():
                print(f"Found {tool_name} at position {det['world_position']}")
                return det
        
        print(f"Tool {tool_name} not found")
        return None
    
    def execute_tool_pickup(self, tool_name):
        """Complete sequence to find and pick up a specific tool"""
        # Find the tool
        tool_detection = self.find_specific_tool(tool_name)
        if not tool_detection:
            return False
        
        # Pick the tool
        success = self.pick_tool(tool_detection['world_position'], 
                                tool_detection['class_name'])
        
        if success:
            # Place at drop zone
            self.place_tool()
            print(f"Successfully delivered {tool_name}")
            return True
        
        return False
    
    def calibrate_camera(self):
        """Perform camera calibration for coordinate transformation"""
        print("Starting camera calibration...")
        
        # This should use a calibration pattern (checkerboard)
        # For simplicity, we'll use manual measurement
        
        calibration_points = []
        
        print("Please place a marker at known positions in the workspace.")
        print("We'll capture images and map pixel to world coordinates.")
        
        for i in range(4):  # At least 4 points for homography
            input(f"Place marker at corner {i+1} and press Enter...")
            image = self.capture_image(f"calib_{i}.jpg")
            
            # Show image and let user click on marker
            cv2.imshow("Calibration", image)
            print("Click on the marker in the image window...")
            
            # Simple mouse callback (in production, use proper UI)
            points = []
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                    print(f"Point selected: ({x}, {y})")
            
            cv2.setMouseCallback("Calibration", mouse_callback)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if points:
                pixel_point = points[0]
                world_x = float(input(f"Enter X coordinate (mm) for point {i+1}: "))
                world_y = float(input(f"Enter Y coordinate (mm) for point {i+1}: "))
                calibration_points.append({
                    'pixel': pixel_point,
                    'world': (world_x, world_y)
                })
        
        # Calculate homography matrix
        if len(calibration_points) >= 4:
            src_pts = np.array([p['pixel'] for p in calibration_points], dtype='float32')
            dst_pts = np.array([p['world'] for p in calibration_points], dtype='float32')
            
            # Calculate homography
            H, _ = cv2.findHomography(src_pts, dst_pts)
            
            self.calibration_params = {
                'homography_matrix': H.tolist(),
                'image_width': 640,
                'image_height': 480,
                'calibration_points': calibration_points,
                'calibration_date': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.save_calibration()
            print("Calibration completed and saved.")
        
        return self.calibration_params
    
    def load_calibration(self):
        """Load calibration from file"""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading calibration: {e}")
        return None
    
    def save_calibration(self):
        """Save calibration to file"""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration_params, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
        return False
    
    def load_tools_database(self):
        """Load tool properties database"""
        tools_db = {
            'wrench': {
                'grip_offset': 10,
                'gripper_width': 60,
                'weight': 100
            },
            'screwdriver': {
                'grip_offset': 5,
                'gripper_width': 40,
                'weight': 50
            },
            'pliers': {
                'grip_offset': 8,
                'gripper_width': 50,
                'weight': 80
            },
            'hammer': {
                'grip_offset': 15,
                'gripper_width': 70,
                'weight': 200
            }
        }
        
        # Load from file if exists
        if os.path.exists("tools_database.json"):
            try:
                with open("tools_database.json", 'r') as f:
                    tools_db.update(json.load(f))
            except:
                pass
        
        return tools_db
    
    def go_home(self):
        """Move arm to home position"""
        print("Moving to home position")
        self.arm.Arm_serial_servo_write6_array(self.home_position, 1500)
        time.sleep(2)
    
    def shutdown(self):
        """Clean shutdown"""
        print("Shutting down...")
        self.go_home()
        if hasattr(self.camera, 'stop'):
            self.camera.stop()
        else:
            self.camera.release()
        cv2.destroyAllWindows()

def main():
    """Main program loop"""
    assistant = GarageAssistant()
    
    try:
        # Go to home position
        assistant.go_home()
        
        # Check if calibration exists
        if not assistant.calibration_params:
            print("No calibration found. Please calibrate first.")
            response = input("Calibrate now? (y/n): ")
            if response.lower() == 'y':
                assistant.calibrate_camera()
        
        # Main menu
        while True:
            print("\n" + "="*50)
            print("GARAGE ASSISTANT MENU")
            print("="*50)
            print("1. Scan workspace for tools")
            print("2. Find and pick up specific tool")
            print("3. Calibrate camera")
            print("4. Test coordinate conversion")
            print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ")
            
            if choice == '1':
                # Scan workspace
                detections = assistant.scan_workspace()
                print(f"\nFound {len(detections)} tools:")
                for det in detections:
                    print(f"  - {det['class_name']} at {det['world_position']}")
            
            elif choice == '2':
                # Pick specific tool
                tool_name = input("Enter tool name to pick up: ")
                assistant.execute_tool_pickup(tool_name)
            
            elif choice == '3':
                # Calibrate
                assistant.calibrate_camera()
            
            elif choice == '4':
                # Test coordinate conversion
                image = assistant.capture_image("test.jpg")
                if image is not None:
                    # Let user click on a point
                    points = []
                    def mouse_callback(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            world = assistant.pixel_to_world_coordinates(x, y)
                            print(f"Pixel ({x}, {y}) -> World ({world['x']:.1f}, {world['y']:.1f}, {world['z']:.1f})")
                            points.append((x, y))
                            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                            cv2.imshow("Test", image)
                    
                    cv2.imshow("Test", image)
                    cv2.setMouseCallback("Test", mouse_callback)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            
            elif choice == '5':
                break
            
            else:
                print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        assistant.shutdown()

if __name__ == "__main__":
    main()