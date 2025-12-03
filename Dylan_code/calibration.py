#!/usr/bin/env python3
# calibration.py
"""
Camera calibration for DOFBOT-PI Garage Assistant
"""

import cv2
import numpy as np
import json
import os
from Arm_Lib import Arm_Device

class CameraCalibrator:
    def __init__(self):
        self.arm = Arm_Device()
        self.calibration_points = []
        self.world_points = []
        
        # Known calibration pattern (checkerboard)
        self.pattern_size = (7, 7)  # Number of inner corners
        self.square_size = 30  # mm
        
    def capture_calibration_images(self, num_images=20):
        """Capture multiple images of calibration pattern from different angles using OpenCV VideoCapture"""
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError("Unable to open camera (VideoCapture index 1)")

        print(f"Capturing {num_images} calibration images...")
        print("Move the calibration pattern to different positions/orientations")

        images = []
        for i in range(num_images):
            input(f"Position {i+1}/{num_images}. Press Enter to capture...")

            # Capture image from VideoCapture
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"  Failed to capture image {i+1}")
                continue

            temp_file = f"/tmp/calib_{i}.jpg"
            cv2.imwrite(temp_file, frame)
            images.append(frame)
            print(f"  Image {i+1} captured")

        cap.release()
        return images
    
    def find_checkerboard_corners(self, images):
        """Find checkerboard corners in images"""
        objp = np.zeros((self.pattern_size[0]*self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Refine corner locations
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # Draw and display corners
                cv2.drawChessboardCorners(img, self.pattern_size, corners2, ret)
                cv2.imshow('Calibration', img)
                cv2.waitKey(500)
        
        cv2.destroyAllWindows()
        return objpoints, imgpoints, gray.shape[::-1]
    
    def calibrate_camera(self, images):
        """Perform camera calibration"""
        objpoints, imgpoints, img_size = self.find_checkerboard_corners(images)
        
        if not objpoints:
            print("No calibration points found!")
            return None
        
        print(f"Calibrating with {len(objpoints)} images...")
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None
        )
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        print(f"Calibration error: {mean_error/len(objpoints):.3f} pixels")
        
        calibration_data = {
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.tolist(),
            'image_size': img_size,
            'reprojection_error': float(mean_error/len(objpoints))
        }
        
        return calibration_data
    
    def calibrate_workspace(self):
        """Calibrate workspace mapping (pixel to world coordinates)"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError("Unable to open camera (VideoCapture index 0)")

        print("\n=== WORKSPACE CALIBRATION ===")
        print("We'll now map pixel coordinates to world coordinates.")
        print("Please place a marker at known positions.")
        
        calibration_points = []
        
        # Pre-defined positions (adjust based on your workspace)
        positions = [
            {'name': 'Bottom-left', 'x': -200, 'y': -200},
            {'name': 'Bottom-right', 'x': 200, 'y': -200},
            {'name': 'Top-right', 'x': 200, 'y': 200},
            {'name': 'Top-left', 'x': -200, 'y': 200},
            {'name': 'Center', 'x': 0, 'y': 0}
        ]
        
        for pos in positions:
            print(f"\nPlace marker at {pos['name']} (X={pos['x']}mm, Y={pos['y']}mm)")
            input("Press Enter when ready...")
            # Capture image from VideoCapture
            ret, img = cap.read()
            if not ret or img is None:
                print("  Failed to capture image for this point")
                continue
            
            # Let user click on marker
            point = None
            def click_event(event, x, y, flags, params):
                nonlocal point
                if event == cv2.EVENT_LBUTTONDOWN:
                    point = (x, y)
                    print(f"  Selected pixel: ({x}, {y})")
            
            cv2.imshow('Select Marker', img)
            cv2.setMouseCallback('Select Marker', click_event)
            
            print("Click on the marker in the image...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if point:
                calibration_points.append({
                    'pixel': point,
                    'world': (pos['x'], pos['y'])
                })
        
        cap.release()
        
        # Calculate homography matrix
        if len(calibration_points) >= 4:
            src_pts = np.array([p['pixel'] for p in calibration_points], dtype='float32')
            dst_pts = np.array([p['world'] for p in calibration_points], dtype='float32')
            
            H, mask = cv2.findHomography(src_pts, dst_pts)
            
            calibration = {
                'homography_matrix': H.tolist(),
                'calibration_points': calibration_points,
                'workspace_bounds': {
                    'x_min': -200, 'x_max': 200,
                    'y_min': -200, 'y_max': 200
                }
            }
            
            # Test the calibration
            print("\nTesting calibration...")
            test_pixel = (320, 240)  # Center of image
            test_world = self.pixel_to_world(H, test_pixel)
            print(f"Image center ({test_pixel}) -> World ({test_world[0]:.1f}, {test_world[1]:.1f})")
            
            return calibration
        
        return None
    
    def pixel_to_world(self, homography, pixel_point):
        """Convert pixel to world coordinates using homography"""
        pts = np.array([[pixel_point]], dtype='float32')
        world_pts = cv2.perspectiveTransform(pts, homography)
        return world_pts[0][0]
    
    def save_calibration(self, camera_calib, workspace_calib, filename="calibration_full.json"):
        """Save complete calibration data"""
        calibration_data = {
            'camera_calibration': camera_calib,
            'workspace_calibration': workspace_calib,
            'calibration_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Calibration saved to {filename}")
        return True

def main():
    calibrator = CameraCalibrator()
    
    print("DOFBOT-PI Camera Calibration")
    print("="*50)
    
    # Step 1: Camera intrinsic calibration
    print("\nStep 1: Camera intrinsic calibration")
    images = calibrator.capture_calibration_images(15)
    camera_calib = calibrator.calibrate_camera(images)
    
    # Step 2: Workspace calibration
    print("\nStep 2: Workspace calibration")
    workspace_calib = calibrator.calibrate_workspace()
    
    # Save calibration
    if camera_calib and workspace_calib:
        calibrator.save_calibration(camera_calib, workspace_calib)
        print("\nCalibration completed successfully!")
    else:
        print("\nCalibration failed!")

if __name__ == "__main__":
    import time
    main()