import sys
import os
from datetime import datetime
import tkinter as tk

# Import split modules
from voice_system import VoiceRecognitionSystem
from robot_arm import RobotArmController
from camera_system import CameraSystem
from snapshot_system import SnapshotSystem
from grab_system import GrabSystem
from analysis_system import AnalysisSystem
from tool_prompt import ToolPromptDialog
from main_window import MainWindow

def main():
    # Create necessary directories
    os.makedirs("data/snapshots", exist_ok=True)
    os.makedirs("data/mappings", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("movements", exist_ok=True)
    os.makedirs("data", exist_ok=True)

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