#!/usr/bin/env python3
"""
Garage Assistant GUI - Tool Selection and Robot Control Interface
"""

import sys
import json
import time
import threading
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QGroupBox, QTextEdit, QListWidget, QListWidgetItem,
                             QMessageBox, QProgressBar, QTabWidget, QFrame)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from Arm_Lib import Arm_Device

# ============================================
# ROBOT WORKER THREAD (Prevents GUI Freezing)
# ============================================

class RobotWorker(QThread):
    """Thread for running robot operations without freezing GUI"""
    status_update = pyqtSignal(str)
    operation_complete = pyqtSignal(str, bool)  # tool_name, success
    progress_update = pyqtSignal(int)
    
    def __init__(self, movements_file="config/movements.json"):
        super().__init__()
        self.movements_file = movements_file
        self.current_operation = None
        self.tool_name = None
        self.grab_point = None
        self.running = True
        
    def load_movements(self):
        """Load movement sequences"""
        try:
            with open(self.movements_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.status_update.emit(f"Error loading movements: {str(e)}")
            return None
    
    def pick_and_deliver_tool(self, tool_name, grab_point):
        """Complete operation: pick tool and deliver to drop zone"""
        self.current_operation = "pick_and_deliver"
        self.tool_name = tool_name
        self.grab_point = grab_point
        self.start()
    
    def return_to_home(self):
        """Return robot to home position"""
        self.current_operation = "go_home"
        self.start()
    
    def stop_robot(self):
        """Stop current operation"""
        self.running = False
    
    def run(self):
        """Main thread execution"""
        try:
            if not self.running:
                return
            
            # Initialize arm
            self.status_update.emit("Initializing robot arm...")
            self.arm = Arm_Device()
            time.sleep(2)
            
            # Load movements
            movements = self.load_movements()
            if not movements:
                self.operation_complete.emit(self.tool_name, False)
                return
            
            if self.current_operation == "pick_and_deliver":
                self._execute_pick_and_deliver(movements)
            elif self.current_operation == "go_home":
                self._execute_go_home(movements)
            
        except Exception as e:
            self.status_update.emit(f"Robot error: {str(e)}")
            self.operation_complete.emit(self.tool_name, False)
    
    def _execute_pick_and_deliver(self, movements):
        """Execute complete pick and deliver operation"""
        # Phase 1: Go to home
        self.status_update.emit("Moving to home position...")
        self.progress_update.emit(10)
        self._go_to_position(movements["metadata"]["initial_position"], 2000)
        time.sleep(1)
        
        # Phase 2: Check if point exists
        if self.grab_point not in movements["grab_points"]:
            self.status_update.emit(f"Error: Point {self.grab_point} not found")
            self.operation_complete.emit(self.tool_name, False)
            return
        
        point_data = movements["grab_points"][self.grab_point]
        
        # Phase 3: Execute pickup sequence
        self.status_update.emit(f"Picking up {self.tool_name} from Point {self.grab_point}...")
        self.progress_update.emit(30)
        
        if not self._execute_sequence(point_data["movement_sequence"]):
            self.status_update.emit("Pickup failed")
            self.operation_complete.emit(self.tool_name, False)
            return
        
        # Phase 4: Return to home with tool
        self.status_update.emit(f"Returning to home with {self.tool_name}...")
        self.progress_update.emit(60)
        
        if not self._execute_sequence(point_data["return_to_home_sequence"]):
            self.status_update.emit("Return failed")
            self.operation_complete.emit(self.tool_name, False)
            return
        
        # Phase 5: Deliver to drop zone
        self.status_update.emit(f"Delivering {self.tool_name} to drop zone...")
        self.progress_update.emit(80)
        
        if "drop_sequence" in point_data:
            self._execute_sequence(point_data["drop_sequence"])
        else:
            # Default drop sequence
            self.status_update.emit("Using default drop sequence...")
            self.arm.Arm_serial_servo_write6(180, 90, 90, 90, 90, 135, 2000)
            time.sleep(2)
            self.arm.Arm_serial_servo_write(6, 90, 1000)  # Open gripper
            time.sleep(1)
        
        # Phase 6: Return to home (empty)
        self.status_update.emit("Returning to home position...")
        self.progress_update.emit(90)
        self._go_to_position(movements["metadata"]["initial_position"], 2000)
        time.sleep(1)
        
        self.progress_update.emit(100)
        self.status_update.emit(f"Successfully delivered {self.tool_name}!")
        self.operation_complete.emit(self.tool_name, True)
    
    def _execute_go_home(self, movements):
        """Return robot to home position"""
        self.status_update.emit("Returning robot to home position...")
        self._go_to_position(movements["metadata"]["initial_position"], 2000)
        time.sleep(2)
        self.status_update.emit("Robot at home position")
        self.operation_complete.emit("home", True)
    
    def _execute_sequence(self, sequence):
        """Execute a movement sequence"""
        for step in sequence:
            if not self.running:
                return False
            
            if "gripper_angle" in step:
                self.arm.Arm_serial_servo_write(6, step["gripper_angle"], step["move_time"])
            elif "servo_angles" in step:
                angles = step["servo_angles"]
                self.arm.Arm_serial_servo_write6(
                    angles[0], angles[1], angles[2],
                    angles[3], angles[4], angles[5],
                    step["move_time"]
                )
            
            time.sleep(step.get("stabilization_time", 0.5))
        
        return True
    
    def _go_to_position(self, angles, move_time):
        """Move arm to specific position"""
        self.arm.Arm_serial_servo_write6(
            angles[0], angles[1], angles[2],
            angles[3], angles[4], angles[5],
            move_time
        )

# ============================================
# MAIN GUI APPLICATION
# ============================================

class GarageAssistantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.robot_worker = None
        self.current_tool = None
        self.tool_mapping = {}
        
        self.setup_ui()
        self.load_tool_mapping()
        
        # Setup status timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_time)
        self.status_timer.start(1000)  # Update every second
        
    def setup_ui(self):
        """Setup the main GUI interface"""
        self.setWindowTitle("Garage Assistant - Robot Control Panel")
        self.setGeometry(100, 100, 1000, 700)
        
        # Set dark theme
        self.set_dark_theme()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Main content area (split)
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(15)
        
        # Left panel - Tool selection and control
        left_panel = self.create_left_panel()
        content_layout.addWidget(left_panel, 1)
        
        # Right panel - Status and logs
        right_panel = self.create_right_panel()
        content_layout.addWidget(right_panel, 1)
        
        main_layout.addWidget(content_widget)
        
        # Footer with progress bar
        footer = self.create_footer()
        main_layout.addWidget(footer)
    
    def set_dark_theme(self):
        """Apply dark theme to the application"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        self.setPalette(dark_palette)
        
        # Set style sheet for additional styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #555;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:pressed {
                background-color: #777;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #777;
            }
            QListWidget {
                background-color: #252525;
                border: 1px solid #444;
            }
            QTextEdit {
                background-color: #252525;
                border: 1px solid #444;
            }
            QComboBox {
                background-color: #555;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2a82da;
                border-radius: 3px;
            }
        """)
    
    def create_header(self):
        """Create the header with title and robot status"""
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_layout = QHBoxLayout(header_frame)
        
        # Title
        title_label = QLabel("🔧 GARAGE ASSISTANT ROBOT")
        title_font = QFont("Arial", 16, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2a82da;")
        
        # Status indicator
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: #4CAF50; font-size: 20px;")
        self.status_indicator.setFixedWidth(30)
        
        self.status_label = QLabel("Robot: READY")
        self.status_label.setStyleSheet("font-weight: bold;")
        
        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        # Time display
        self.time_label = QLabel()
        self.time_label.setStyleSheet("color: #888;")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(status_widget)
        header_layout.addSpacing(20)
        header_layout.addWidget(self.time_label)
        
        return header_frame
    
    def create_left_panel(self):
        """Create the left panel with tool selection and controls"""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        # Tool Selection Group
        tool_group = QGroupBox("🛠️ TOOL SELECTION")
        tool_layout = QVBoxLayout()
        
        # Tool list
        tool_list_label = QLabel("Available Tools:")
        tool_list_label.setStyleSheet("font-weight: bold;")
        tool_layout.addWidget(tool_list_label)
        
        self.tool_list = QListWidget()
        self.tool_list.setMinimumHeight(200)
        self.tool_list.itemClicked.connect(self.on_tool_selected)
        tool_layout.addWidget(self.tool_list)
        
        # Tool info
        tool_info_widget = QWidget()
        tool_info_layout = QHBoxLayout(tool_info_widget)
        
        self.tool_info_label = QLabel("Selected: None")
        self.tool_info_label.setStyleSheet("color: #2a82da; font-weight: bold;")
        
        self.tool_location_label = QLabel("Location: --")
        self.tool_location_label.setStyleSheet("color: #888;")
        
        tool_info_layout.addWidget(self.tool_info_label)
        tool_info_layout.addStretch()
        tool_info_layout.addWidget(self.tool_location_label)
        tool_layout.addWidget(tool_info_widget)
        
        tool_group.setLayout(tool_layout)
        left_layout.addWidget(tool_group)
        
        # Control Buttons Group
        control_group = QGroupBox("🎮 ROBOT CONTROLS")
        control_layout = QVBoxLayout()
        
        # Fetch button
        self.fetch_button = QPushButton("🤖 FETCH SELECTED TOOL")
        self.fetch_button.setStyleSheet("""
            QPushButton {
                background-color: #2a82da;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1a72ca;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self.fetch_button.clicked.connect(self.on_fetch_clicked)
        self.fetch_button.setEnabled(False)
        control_layout.addWidget(self.fetch_button)
        
        # Emergency stop button
        self.emergency_button = QPushButton("🛑 EMERGENCY STOP")
        self.emergency_button.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c32f2f;
            }
        """)
        self.emergency_button.clicked.connect(self.on_emergency_stop)
        control_layout.addWidget(self.emergency_button)
        
        # Utility buttons
        button_row = QHBoxLayout()
        
        self.home_button = QPushButton("🏠 GO HOME")
        self.home_button.clicked.connect(self.on_go_home)
        button_row.addWidget(self.home_button)
        
        self.scan_button = QPushButton("🔍 SCAN WORKSHOP")
        self.scan_button.clicked.connect(self.on_scan_workshop)
        button_row.addWidget(self.scan_button)
        
        control_layout.addLayout(button_row)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        left_layout.addStretch()
        
        return left_panel
    
    def create_right_panel(self):
        """Create the right panel with status and logs"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        
        # Status Display Group
        status_group = QGroupBox("📊 SYSTEM STATUS")
        status_layout = QVBoxLayout()
        
        # Current operation
        self.operation_label = QLabel("Current Operation: Idle")
        self.operation_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        status_layout.addWidget(self.operation_label)
        
        # Progress bar
        progress_label = QLabel("Operation Progress:")
        status_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        
        # Stats
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        
        self.tools_count_label = QLabel("Tools Available: 0")
        self.points_count_label = QLabel("Grab Points: 0")
        
        stats_layout.addWidget(self.tools_count_label)
        stats_layout.addWidget(self.points_count_label)
        stats_layout.addStretch()
        
        status_layout.addWidget(stats_widget)
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        # Activity Log Group
        log_group = QGroupBox("📝 ACTIVITY LOG")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        
        # Add log controls
        log_controls = QHBoxLayout()
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.on_clear_log)
        self.save_log_button = QPushButton("Save Log")
        self.save_log_button.clicked.connect(self.on_save_log)
        
        log_controls.addWidget(self.clear_log_button)
        log_controls.addWidget(self.save_log_button)
        log_controls.addStretch()
        
        log_layout.addWidget(self.log_text)
        log_layout.addLayout(log_controls)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # Workspace Map Group (simplified)
        map_group = QGroupBox("🗺️ WORKSPACE OVERVIEW")
        map_layout = QVBoxLayout()
        
        map_info = QLabel("Grab Points Configuration:")
        map_layout.addWidget(map_info)
        
        self.map_text = QTextEdit()
        self.map_text.setReadOnly(True)
        self.map_text.setMaximumHeight(150)
        map_layout.addWidget(self.map_text)
        
        map_group.setLayout(map_layout)
        right_layout.addWidget(map_group)
        
        right_layout.addStretch()
        
        return right_panel
    
    def create_footer(self):
        """Create footer with additional controls"""
        footer_frame = QFrame()
        footer_frame.setFrameShape(QFrame.StyledPanel)
        footer_layout = QHBoxLayout(footer_frame)
        
        # Version info
        version_label = QLabel("Garage Assistant v1.0 | Powered by Dofbot-Pi")
        version_label.setStyleSheet("color: #888;")
        
        # Connection status
        self.connection_label = QLabel("Arm: Disconnected")
        self.connection_label.setStyleSheet("color: #ff6b6b;")
        
        footer_layout.addWidget(version_label)
        footer_layout.addStretch()
        footer_layout.addWidget(self.connection_label)
        
        return footer_frame
    
    def load_tool_mapping(self):
        """Load tool-to-grab-point mapping"""
        try:
            with open("config/positions.json", "r") as f:
                data = json.load(f)
                self.tool_mapping = data.get("tool_mapping", {})
            
            # Update tool list
            self.tool_list.clear()
            for tool_name in sorted(self.tool_mapping.keys()):
                item = QListWidgetItem(f"🔧 {tool_name}")
                self.tool_list.addItem(item)
            
            # Update counts
            self.tools_count_label.setText(f"Tools Available: {len(self.tool_mapping)}")
            
            # Load grab points count from movements
            try:
                with open("config/movements.json", "r") as f:
                    movements = json.load(f)
                    points_count = len(movements.get("grab_points", {}))
                    self.points_count_label.setText(f"Grab Points: {points_count}")
                    
                    # Update workspace map
                    self.update_workspace_map(movements)
            except:
                self.points_count_label.setText("Grab Points: Not loaded")
            
            self.log_message("System initialized. Tool mapping loaded.")
            self.connection_label.setText("Arm: Ready to connect")
            self.connection_label.setStyleSheet("color: #4CAF50;")
            
        except FileNotFoundError:
            self.log_message("WARNING: Tool mapping not found. Run workshop scan first.")
            self.connection_label.setText("Arm: Configuration needed")
            self.connection_label.setStyleSheet("color: #ff9800;")
        except Exception as e:
            self.log_message(f"ERROR loading tool mapping: {str(e)}")
    
    def update_workspace_map(self, movements):
        """Update workspace map display"""
        grab_points = movements.get("grab_points", {})
        map_text = ""
        
        for position in ["initial_position", "second_position", "third_position"]:
            points_in_position = {}
            for point_id, data in grab_points.items():
                if data.get("position_name") == position:
                    points_in_position[point_id] = data
            
            if points_in_position:
                map_text += f"\n{position.replace('_', ' ').title()}:\n"
                for point_id in sorted(points_in_position.keys()):
                    data = points_in_position[point_id]
                    map_text += f"  Point {point_id}: ({data['image_coords'][0]}, {data['image_coords'][1]})\n"
        
        self.map_text.setText(map_text.strip())
    
    def on_tool_selected(self, item):
        """Handle tool selection"""
        tool_name = item.text().replace("🔧 ", "")
        self.current_tool = tool_name
        
        # Update display
        self.tool_info_label.setText(f"Selected: {tool_name}")
        
        # Show tool location
        if tool_name in self.tool_mapping:
            mapping = self.tool_mapping[tool_name]
            location = f"Point {mapping['grab_point']} ({mapping['position'].replace('_', ' ')})"
            self.tool_location_label.setText(f"Location: {location}")
            
            # Enable fetch button
            self.fetch_button.setEnabled(True)
            self.fetch_button.setText(f"🤖 FETCH {tool_name.upper()}")
        else:
            self.tool_location_label.setText("Location: Unknown")
            self.fetch_button.setEnabled(False)
    
    def on_fetch_clicked(self):
        """Handle fetch button click"""
        if not self.current_tool or self.current_tool not in self.tool_mapping:
            QMessageBox.warning(self, "Error", "No tool selected or tool not found in mapping.")
            return
        
        # Confirm operation
        reply = QMessageBox.question(
            self, "Confirm Fetch",
            f"Fetch {self.current_tool} from Point {self.tool_mapping[self.current_tool]['grab_point']}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Start robot operation
        self.start_robot_operation()
    
    def start_robot_operation(self):
        """Start the robot operation in a separate thread"""
        if self.robot_worker and self.robot_worker.isRunning():
            self.log_message("Robot is already busy. Please wait.")
            return
        
        # Get tool mapping
        mapping = self.tool_mapping[self.current_tool]
        grab_point = mapping['grab_point']
        
        # Disable controls
        self.set_controls_enabled(False)
        
        # Update UI
        self.status_indicator.setStyleSheet("color: #ff9800; font-size: 20px;")
        self.status_label.setText("Robot: BUSY")
        self.operation_label.setText(f"Current Operation: Fetching {self.current_tool}")
        self.progress_bar.setValue(0)
        
        # Create and start worker thread
        self.robot_worker = RobotWorker()
        self.robot_worker.status_update.connect(self.on_robot_status)
        self.robot_worker.operation_complete.connect(self.on_robot_complete)
        self.robot_worker.progress_update.connect(self.on_progress_update)
        
        self.log_message(f"Starting fetch operation: {self.current_tool} from Point {grab_point}")
        self.robot_worker.pick_and_deliver_tool(self.current_tool, grab_point)
    
    def on_robot_status(self, message):
        """Handle status updates from robot thread"""
        self.log_message(f"Robot: {message}")
    
    def on_robot_complete(self, tool_name, success):
        """Handle robot operation completion"""
        if success:
            self.log_message(f"SUCCESS: {tool_name} delivered to drop zone!")
            self.status_indicator.setStyleSheet("color: #4CAF50; font-size: 20px;")
            self.status_label.setText("Robot: READY")
            self.operation_label.setText("Current Operation: Idle")
            self.progress_bar.setValue(100)
            
            # Show success message
            QMessageBox.information(self, "Success", 
                                  f"{tool_name} successfully delivered to drop zone!")
        else:
            self.log_message(f"FAILED: Could not deliver {tool_name}")
            self.status_indicator.setStyleSheet("color: #f44336; font-size: 20px;")
            self.status_label.setText("Robot: ERROR")
            
            QMessageBox.warning(self, "Operation Failed", 
                              f"Failed to deliver {tool_name}. Check robot and try again.")
        
        # Re-enable controls
        self.set_controls_enabled(True)
        
        # Reset robot worker
        self.robot_worker = None
    
    def on_progress_update(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def on_emergency_stop(self):
        """Handle emergency stop"""
        if self.robot_worker:
            self.robot_worker.stop_robot()
            self.log_message("EMERGENCY STOP: Robot operation halted")
            self.status_indicator.setStyleSheet("color: #f44336; font-size: 20px;")
            self.status_label.setText("Robot: EMERGENCY STOP")
            
            QMessageBox.warning(self, "Emergency Stop", 
                              "Robot operation stopped. Check robot before continuing.")
            
            # Re-enable controls
            self.set_controls_enabled(True)
    
    def on_go_home(self):
        """Return robot to home position"""
        if self.robot_worker and self.robot_worker.isRunning():
            self.log_message("Robot is busy. Cannot go home now.")
            return
        
        self.robot_worker = RobotWorker()
        self.robot_worker.status_update.connect(self.on_robot_status)
        self.robot_worker.operation_complete.connect(self.on_go_home_complete)
        
        self.log_message("Returning robot to home position...")
        self.robot_worker.return_to_home()
    
    def on_go_home_complete(self, operation, success):
        """Handle go home completion"""
        if success:
            self.log_message("Robot returned to home position")
        self.robot_worker = None
    
    def on_scan_workshop(self):
        """Handle scan workshop button"""
        # This would run your take_snapshots.py and analyze_grab_points.py
        self.log_message("Starting workshop scan...")
        QMessageBox.information(self, "Scan Workshop", 
                              "This feature would run the workshop scan.\n"
                              "For now, please run:\n"
                              "1. python take_snapshots.py\n"
                              "2. python analyze_grab_points.py\n"
                              "Then restart this application.")
    
    def on_clear_log(self):
        """Clear the activity log"""
        self.log_text.clear()
        self.log_message("Log cleared")
    
    def on_save_log(self):
        """Save the activity log to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/operation_log_{timestamp}.txt"
            
            with open(filename, "w") as f:
                f.write(self.log_text.toPlainText())
            
            self.log_message(f"Log saved to {filename}")
            QMessageBox.information(self, "Log Saved", f"Log saved to {filename}")
        except Exception as e:
            self.log_message(f"Error saving log: {str(e)}")
            QMessageBox.warning(self, "Save Error", f"Could not save log: {str(e)}")
    
    def log_message(self, message):
        """Add message to log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def set_controls_enabled(self, enabled):
        """Enable or disable control buttons"""
        self.fetch_button.setEnabled(enabled and self.current_tool is not None)
        self.home_button.setEnabled(enabled)
        self.scan_button.setEnabled(enabled)
        self.tool_list.setEnabled(enabled)
    
    def update_status_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.robot_worker and self.robot_worker.isRunning():
            reply = QMessageBox.question(
                self, "Robot is Active",
                "Robot is currently operating. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # Stop robot worker if running
        if self.robot_worker:
            self.robot_worker.stop_robot()
            self.robot_worker.wait(1000)  # Wait up to 1 second
        
        event.accept()


# ============================================
# APPLICATION ENTRY POINT
# ============================================

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = GarageAssistantGUI()
    window.show()
    
    # Check for required files
    try:
        with open("config/movements.json", "r"):
            pass
    except FileNotFoundError:
        QMessageBox.warning(window, "Configuration Missing", 
                          "movements.json not found. Please calibrate robot movements first.")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()