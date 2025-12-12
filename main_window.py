import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
from datetime import datetime
from PIL import Image, ImageTk

# Import split modules
from voice_system import VoiceRecognitionSystem
from robot_arm import RobotArmController
from camera_system import CameraSystem
from snapshot_system import SnapshotSystem
from grab_system import GrabSystem
from analysis_system import AnalysisSystem
from tool_prompt import ToolPromptDialog

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GUIDO Garage Assistant")
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
            text="Live Camera Feed",
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
            text="System Controls",
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
            text="Enable Voice",
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
            text=" Scan & Analysis",
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
            text="Go Home",
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
            text="Fetch Tool...",
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
            text="Clear Log",
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
            text=" Exit",
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
            text="Mapped Tools",
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
            text="System Logger",
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
            text="System Information",
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
        self.log("Guido Garage Assistant Initializing...", "system")
    
    def initialize_systems(self):
        """Initialize all systems - called AFTER GUI is set up"""
        try:
            self.log("Initializing systems...", "system")
            
            self.log("Initializing robot arm...", "info")
            self.arm_controller = RobotArmController(log_callback=self.log)
            
            self.log("Starting camera...", "info")
            self.camera_system = CameraSystem(
                log_callback=self.log,
                frame_callback=self.update_camera_frame
            )
            self.camera_system.start()
            
            self.log("Initializing voice recognition...", "info")
            self.voice_system = VoiceRecognitionSystem(log_callback=self.log)
            self.voice_system.set_command_callback(self.handle_voice_command)
            
            self.snapshot_system = SnapshotSystem(self.arm_controller, self.camera_system, self.log)
            self.analysis_system = AnalysisSystem(self.log)
            self.grab_system = GrabSystem(self.arm_controller, self.snapshot_system, self.camera_system, self.log)
            
            self.log("All systems initialized successfully!", "success")
            self.update_info("Arm: Connected ✓", "Camera: Running ✓", "Voice: Ready ✓")
            
        except Exception as e:
            self.log(f"Failed to initialize systems: {e}", "error")
    
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
                import cv2
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
        
        self.log("Enabling voice recognition...", "system")
        self.voice_enabled = True
        
        # Update UI
        self.voice_button.config(
            text="Disable Voice",
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
        
        self.log("Voice recognition enabled. Speak commands now.", "success")
        self.update_info("Voice Status: Listening ✓")
    
    def disable_voice_recognition(self):
        """Disable voice recognition"""
        self.log(" Disabling voice recognition...", "system")
        self.voice_enabled = False
        
        # Update UI
        self.voice_button.config(
            text=" Enable Voice",
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
        
        self.log("Voice recognition disabled.", "success")
        self.update_info("Voice Status: Disabled")
    
    def handle_voice_command(self, tool_name):
        """Handle voice command received"""
        self.log(f"Voice command received: Fetch {tool_name}", "system")
        
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
                self.log(f"Successfully fetched {tool_name.upper()}!", "success")
                self.update_info(
                    f"Last Fetch: {datetime.now().strftime('%H:%M:%S')}",
                    f"Last Tool: {tool_name.upper()}"
                )
                
                # Update tools list to show new counts
                self.root.after(0, self.update_tools_list)
                
                # Speak confirmation message AFTER successful fetch
                confirmation_msg = f"Here is your {tool_name}! What else can I get for you?"
                self.voice_system.speak(confirmation_msg)
                
            else:
                self.log(f"Failed to fetch {tool_name.upper()}", "error")
                
        except Exception as e:
            self.log(f"Fetch error: {e}", "error")
        
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
            self.log("No tool mapping available! Please run scan first.", "error")
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
                self.log(f"Successfully fetched {tool_name.upper()}!", "success")
                self.update_info(
                    f"Last Fetch: {datetime.now().strftime('%H:%M:%S')}",
                    f"Last Tool: {tool_name.upper()}"
                )
                # Update tools list to show new counts
                # Ensure we have the latest mapping
                self.tool_mapping = self.grab_system.tool_mapping
                # Update tools list to show new counts
                self.root.after(0, self.update_tools_list)
            else:
                self.log(f"Failed to fetch {tool_name.upper()}", "error")
                
        except Exception as e:
            self.log(f"Fetch error: {e}", "error")
        
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
        if os.path.exists("data/fetched_tools.json"):
            os.remove("data/fetched_tools.json")
            print("Removed existing fetched_tools.json file.")
            self.log("Removed existing fetched_tools.json file.", "info")
        
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
            
            self.log("Step 1: Taking snapshots...", "info")
            self.update_progress(10)
            
            self.current_snapshot_folder = self.snapshot_system.take_snapshots_sequence()
            self.update_progress(50)
            
            self.log("\nStep 2: Analyzing grab points...", "info")
            report_path = self.analysis_system.analyze_grab_points(self.current_snapshot_folder)
            self.update_progress(80)
            
            self.log("\nStep 3: Loading tool mapping...", "info")
            self.tool_mapping = self.grab_system.load_mapping(report_path)
            self.update_progress(90)
            
            self.root.after(0, self.update_tools_list)
            self.update_progress(100)
            
            self.log("\nSCAN COMPLETE!", "success")
            self.log(f"Tools mapped: {len(self.tool_mapping)}", "success")
            
            self.update_info(
                f"Last Scan: {datetime.now().strftime('%H:%M:%S')}",
                f"Tools Mapped: {len(self.tool_mapping)}"
            )
            
        except Exception as e:
            self.log(f"Scan failed: {e}", "error")
        
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

    def refresh_tool_mapping(self):
        """Refresh tool mapping from grab system"""
        if self.grab_system:
            self.tool_mapping = self.grab_system.tool_mapping    
    
    def update_tools_list(self):
        """Update the tools list widget with counts"""
        self.tools_listbox.delete(0, tk.END)
        
        color_map = {
            "hammer": "#f38ba8",
            "screwdriver": "#89b4fa",
            "wrench": "#f9e2af",
            "plier": "#a6e3a1",
            "bolt": "#cba6f7",
            "measuring tape": "#f5c2e7"
        }
        
        # Get current tool counts
        if self.grab_system and self.tool_mapping:
            for tool_name in sorted(self.tool_mapping.keys()):
                # Calculate available and total counts
                locations = self.tool_mapping[tool_name]
                total_count = len(locations)
                available_count = sum(1 for loc in locations if not loc.get("fetched", False))
                
                # Format display with counts
                display_text = f"{tool_name.upper()} ({available_count}/{total_count})"
                
                self.tools_listbox.insert(tk.END, display_text)
                
                # Find the right color
                color = "#cdd6f4"  # Default color
                for key, col in color_map.items():
                    if key in tool_name.lower():
                        color = col
                        break
                
                # Color code based on availability
                if available_count == 0:
                    color = "#585b70"  # Grayed out when none available
                
                self.tools_listbox.itemconfig(tk.END, {'fg': color})
        
        if self.tool_mapping:
            total_tools = sum(len(locs) for locs in self.tool_mapping.values())
            available_tools = sum(1 for tool_name in self.tool_mapping 
                                for loc in self.tool_mapping[tool_name] 
                                if not loc.get("fetched", False))
            self.log(f"Updated tools list: {available_tools}/{total_tools} tools available", "success")
    
    def go_home(self):
        """Move arm to home position"""
        self.log("Moving arm to home position...", "info")
        try:
            self.arm_controller.go_to_home()
            self.log("Arm at home position", "success")
        except Exception as e:
            self.log(f"Failed to go home: {e}", "error")
    
    def update_info(self, *args):
        """Update system information with tool counts"""
        info_lines = list(args)
        
        # Add tool count information if available
        if self.grab_system and self.tool_mapping:
            total_tools = sum(len(locs) for locs in self.tool_mapping.values())
            available_tools = sum(1 for tool_name in self.tool_mapping 
                                for loc in self.tool_mapping[tool_name] 
                                if not loc.get("fetched", False))
            info_lines.append(f"Tools Available: {available_tools}/{total_tools}")
        
        text = "\n".join(info_lines)
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
        
        self.log("Goodbye!", "success")
        self.root.quit()
        self.root.destroy()