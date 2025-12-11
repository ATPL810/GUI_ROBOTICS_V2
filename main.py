import sys
import os
import threading
import cv2
from datetime import datetime
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

# Import the separated modules
from robot_arm import RobotArmController
from camera_system import CameraSystem
from snapshot_system import SnapshotSystem
from analysis_system import AnalysisSystem
from grab_system import GrabSystem
from gui_components import ToolPromptDialog

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
        
        # State variables
        self.scanning = False
        self.fetching = False
        self.current_snapshot_folder = None
        self.tool_mapping = {}
        
        # Camera frame
        self.current_frame = None
        self.photo = None
        
        # Initialize GUI
        self.init_ui()
        
        # Start systems
        self.initialize_systems()
        
        # Start periodic updates
        self.update_camera()
        
    def init_ui(self):
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Left panel (Camera + Controls)
        left_frame = tk.Frame(self.root, bg="#1e1e2e")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(0, weight=3)
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_rowconfigure(2, weight=2)
        
        # Camera display
        camera_frame = tk.LabelFrame(left_frame, text="🎥 Live Camera Feed", 
                                    font=("Arial", 12, "bold"),
                                    fg="#89b4fa", bg="#1e1e2e",
                                    borderwidth=2, relief="solid")
        camera_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        camera_frame.grid_columnconfigure(0, weight=1)
        camera_frame.grid_rowconfigure(0, weight=1)
        
        self.camera_label = tk.Label(camera_frame, bg="#000000", 
                                    borderwidth=3, relief="solid")
        self.camera_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Control buttons
        control_frame = tk.LabelFrame(left_frame, text="⚙️ System Controls", 
                                     font=("Arial", 12, "bold"),
                                     fg="#89b4fa", bg="#1e1e2e",
                                     borderwidth=2, relief="solid")
        control_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Control buttons grid
        self.scan_button = tk.Button(control_frame, text="🔍 Start Scan & Analysis", 
                                    font=("Arial", 11, "bold"),
                                    bg="#a6e3a1", fg="#1e1e2e",
                                    activebackground="#94e2d5",
                                    command=self.start_scan)
        self.scan_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.stop_button = tk.Button(control_frame, text="🛑 Emergency Stop", 
                                    font=("Arial", 11, "bold"),
                                    bg="#f38ba8", fg="#1e1e2e",
                                    activebackground="#f5c2e7",
                                    command=self.emergency_stop)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.home_button = tk.Button(control_frame, text="🏠 Go Home", 
                                    font=("Arial", 11, "bold"),
                                    bg="#585b70", fg="#cdd6f4",
                                    activebackground="#89b4fa",
                                    command=self.go_home)
        self.home_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.fetch_prompt_button = tk.Button(control_frame, text="🤖 Fetch Tool...", 
                                           font=("Arial", 11, "bold"),
                                           bg="#cba6f7", fg="#1e1e2e",
                                           activebackground="#f5c2e7",
                                           command=self.prompt_for_tool)
        self.fetch_prompt_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Status: Ready", 
                                    font=("Arial", 11, "bold"),
                                    fg="#a6e3a1", bg="#1e1e2e")
        self.status_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(control_frame, length=400, mode='determinate')
        self.progress_bar.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Tools list
        tools_frame = tk.LabelFrame(left_frame, text="🛠 Mapped Tools", 
                                   font=("Arial", 12, "bold"),
                                   fg="#89b4fa", bg="#1e1e2e",
                                   borderwidth=2, relief="solid")
        tools_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        tools_frame.grid_columnconfigure(0, weight=1)
        tools_frame.grid_rowconfigure(0, weight=5)
        tools_frame.grid_rowconfigure(1, weight=1)
        
        # Listbox with scrollbar
        list_container = tk.Frame(tools_frame, bg="#313244")
        list_container.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        list_container.grid_columnconfigure(0, weight=1)
        list_container.grid_rowconfigure(0, weight=1)
        
        scrollbar = tk.Scrollbar(list_container)
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.tools_list = tk.Listbox(list_container, bg="#313244", fg="#cdd6f4", 
                                    font=("Arial", 11), yscrollcommand=scrollbar.set,
                                    selectbackground="#585b70", selectforeground="#cdd6f4",
                                    borderwidth=2, relief="solid")
        self.tools_list.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.tools_list.yview)
        
        self.tools_list.bind("<Double-Button-1>", lambda e: self.fetch_selected_tool())
        
        self.fetch_button = tk.Button(tools_frame, text="🤖 Fetch Selected Tool", 
                                     font=("Arial", 11, "bold"),
                                     bg="#585b70", fg="#cdd6f4",
                                     activebackground="#89b4fa",
                                     command=self.fetch_selected_tool)
        self.fetch_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Right panel (Logger + Info)
        right_frame = tk.Frame(self.root, bg="#1e1e2e")
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=3)
        right_frame.grid_rowconfigure(1, weight=1)
        
        # Logger
        logger_frame = tk.LabelFrame(right_frame, text="📋 System Logger", 
                                    font=("Arial", 12, "bold"),
                                    fg="#89b4fa", bg="#1e1e2e",
                                    borderwidth=2, relief="solid")
        logger_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        logger_frame.grid_columnconfigure(0, weight=1)
        logger_frame.grid_rowconfigure(0, weight=5)
        logger_frame.grid_rowconfigure(1, weight=1)
        
        # Text widget with scrollbar
        text_container = tk.Frame(logger_frame, bg="#181825")
        text_container.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        text_container.grid_columnconfigure(0, weight=1)
        text_container.grid_rowconfigure(0, weight=1)
        
        text_scrollbar = tk.Scrollbar(text_container)
        text_scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.logger_text = tk.Text(text_container, bg="#181825", fg="#a6adc8", 
                                  font=("Courier New", 10), 
                                  yscrollcommand=text_scrollbar.set,
                                  borderwidth=2, relief="solid",
                                  wrap="word")
        self.logger_text.grid(row=0, column=0, sticky="nsew")
        text_scrollbar.config(command=self.logger_text.yview)
        
        # Clear button
        clear_button = tk.Button(logger_frame, text="🗑️ Clear Log", 
                                font=("Arial", 11, "bold"),
                                bg="#585b70", fg="#cdd6f4",
                                activebackground="#89b4fa",
                                command=self.clear_log)
        clear_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # System info
        info_frame = tk.LabelFrame(right_frame, text="ℹ️ System Information", 
                                  font=("Arial", 12, "bold"),
                                  fg="#89b4fa", bg="#1e1e2e",
                                  borderwidth=2, relief="solid")
        info_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        info_frame.grid_columnconfigure(0, weight=1)
        info_frame.grid_rowconfigure(0, weight=1)
        
        self.info_label = tk.Label(info_frame, 
                                  text="Arm Status: Not connected\n"
                                       "Camera Status: Not connected\n"
                                       "Last Scan: None\n"
                                       "Tools Mapped: 0\n"
                                       "Last Fetch: None",
                                  font=("Monospace", 10),
                                  fg="#cba6f7", bg="#1e1e2e",
                                  justify="left", anchor="nw")
        self.info_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    
    def initialize_systems(self):
        """Initialize all systems"""
        try:
            self.log("🚀 Initializing Garage Assistant Pro...", "system")
            
            # Initialize arm
            self.log("🤖 Initializing robot arm...", "info")
            self.arm_controller = RobotArmController(log_callback=self.log)
            
            # Initialize camera
            self.log("📷 Starting camera...", "info")
            self.camera_system = CameraSystem(log_callback=self.log, 
                                             frame_callback=self.set_camera_frame)
            self.camera_system.start()
            
            # Initialize other systems
            self.snapshot_system = SnapshotSystem(self.arm_controller, self.camera_system, self.log)
            self.analysis_system = AnalysisSystem(self.log)
            self.grab_system = GrabSystem(self.arm_controller, self.snapshot_system, 
                                         self.camera_system, self.log)
            
            self.log("✅ All systems initialized successfully!", "success")
            self.update_info("Arm: Connected ✓", "Camera: Running ✓")
            
        except Exception as e:
            self.log(f"❌ Failed to initialize systems: {e}", "error")
    
    def log(self, message, level="info"):
        """Log message to logger"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color tags
        self.logger_text.tag_config("info", foreground="#89b4fa")
        self.logger_text.tag_config("success", foreground="#a6e3a1")
        self.logger_text.tag_config("warning", foreground="#f9e2af")
        self.logger_text.tag_config("error", foreground="#f38ba8")
        self.logger_text.tag_config("system", foreground="#cba6f7")
        
        # Insert message
        self.logger_text.insert(tk.END, f"[{timestamp}] {message}\n", level)
        self.logger_text.see(tk.END)
        
        # Update status for important messages
        if level == "error":
            self.status_label.config(text=f"Status: Error - {message[:30]}...")
        elif level == "success":
            self.status_label.config(text=f"Status: {message[:40]}...")
    
    def set_camera_frame(self, frame):
        """Set the current camera frame (called from camera thread)"""
        self.current_frame = frame
    
    def update_camera(self):
        """Update camera display"""
        if self.current_frame is not None:
            try:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                
                # Resize to fit label
                h, w, _ = rgb_image.shape
                max_h = self.camera_label.winfo_height() or 480
                max_w = self.camera_label.winfo_width() or 640
                
                scale = min(max_w/w, max_h/h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                if new_w > 0 and new_h > 0:
                    resized = cv2.resize(rgb_image, (new_w, new_h))
                    
                    # Convert to ImageTk
                    image = Image.fromarray(resized)
                    self.photo = ImageTk.PhotoImage(image=image)
                    self.camera_label.config(image=self.photo)
                    
            except Exception as e:
                pass
        
        # Schedule next update
        self.root.after(30, self.update_camera)
    
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
        self.fetch_prompt_button.config(state="disabled")
        self.fetch_button.config(state="disabled")
        self.status_label.config(text=f"Status: Fetching {tool_name.upper()}...")
        
        # Run in separate thread
        threading.Thread(target=self.run_fetch_sequence, args=(tool_name,), daemon=True).start()
    
    def run_fetch_sequence(self, tool_name):
        """Run the fetch sequence"""
        try:
            # Fetch the tool using enhanced grab system
            success = self.grab_system.fetch_tool(tool_name)
            
            if success:
                self.log(f"✅ Successfully fetched {tool_name.upper()}!", "success")
                self.update_info(f"Last Fetch: {datetime.now().strftime('%H:%M:%S')}",
                               f"Last Tool: {tool_name.upper()}")
                
                # Update tools list to reflect fetched status
                self.update_tools_list()
            else:
                self.log(f"❌ Failed to fetch {tool_name.upper()}", "error")
                
        except Exception as e:
            self.log(f"❌ Fetch error: {e}", "error")
        
        finally:
            self.fetching = False
            self.fetch_prompt_button.config(state="normal")
            self.fetch_button.config(state="normal")
            self.status_label.config(text="Status: Ready")
    
    def fetch_selected_tool(self):
        """Fetch the selected tool from list"""
        selection = self.tools_list.curselection()
        if not selection:
            self.log("Please select a tool first!", "warning")
            return
        
        tool_with_info = self.tools_list.get(selection[0])
        # Extract just the tool name (remove inventory info)
        tool_name = tool_with_info.split()[0].lower()
        self.start_fetch_tool(tool_name)
    
    def start_scan(self):
        """Start the automatic scan and analysis"""
        if self.scanning:
            self.log("Scan already in progress!", "warning")
            return
        
        self.scanning = True
        self.scan_button.config(state="disabled")
        self.fetch_prompt_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Status: Scanning...")
        self.progress_bar["value"] = 0
        
        # Run in separate thread
        threading.Thread(target=self.run_scan_sequence, daemon=True).start()
    
    def run_scan_sequence(self):
        """Run complete scan sequence"""
        try:
            # Step 1: Take snapshots
            self.log("\n" + "="*50, "system")
            self.log("STARTING AUTOMATIC SCAN SEQUENCE", "system")
            self.log("="*50, "system")
            
            self.log("📸 Step 1: Taking snapshots...", "info")
            self.update_progress(10)
            
            self.current_snapshot_folder = self.snapshot_system.take_snapshots_sequence()
            self.update_progress(50)
            
            # Step 2: Analyze grab points (with duplicate handling)
            self.log("\n🔍 Step 2: Analyzing grab points with duplicate handling...", "info")
            report_path = self.analysis_system.analyze_grab_points(self.current_snapshot_folder)
            self.update_progress(80)
            
            # Step 3: Load mapping
            self.log("\n🗺️ Step 3: Loading tool mapping...", "info")
            self.tool_mapping = self.grab_system.load_mapping(report_path)
            self.update_progress(90)
            
            # Update tools list
            self.update_tools_list()
            self.update_progress(100)
            
            self.log("\n✅ SCAN COMPLETE!", "success")
            self.log(f"📊 Tools mapped: {len(self.tool_mapping)}", "success")
            
            # Log inventory status
            for tool_name, locations in self.tool_mapping.items():
                available = sum(1 for loc in locations if not loc.get("fetched", False))
                total = len(locations)
                if total > 1:
                    self.log(f"  {tool_name.upper()}: {available}/{total} available", "info")
            
            self.update_info(f"Last Scan: {datetime.now().strftime('%H:%M:%S')}",
                           f"Tools Mapped: {len(self.tool_mapping)}")
            
        except Exception as e:
            self.log(f"❌ Scan failed: {e}", "error")
        
        finally:
            self.scanning = False
            self.scan_button.config(state="normal")
            self.fetch_prompt_button.config(state="normal")
            self.stop_button.config(state="normal")
            self.status_label.config(text="Status: Ready")
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar["value"] = value
        self.root.update_idletasks()
    
    def update_tools_list(self):
        """Update the tools list widget with inventory info"""
        self.tools_list.delete(0, tk.END)
        
        for tool_name in sorted(self.tool_mapping.keys()):
            locations = self.tool_mapping[tool_name]
            available = sum(1 for loc in locations if not loc.get("fetched", False))
            total = len(locations)
            
            if available > 0:
                if total > 1:
                    self.tools_list.insert(tk.END, f"{tool_name.upper()} ({available}/{total} available)")
                else:
                    self.tools_list.insert(tk.END, f"{tool_name.upper()}")
        
        if self.tool_mapping:
            available_tools = sum(1 for tool in self.tool_mapping.values() 
                                if any(not loc.get("fetched", False) for loc in tool))
            self.log(f"Updated tools list: {available_tools} tools available", "success")
    
    def go_home(self):
        """Move arm to home position"""
        self.log("Moving arm to home position...", "info")
        try:
            self.arm_controller.go_to_home()
            self.log("✅ Arm at home position", "success")
        except Exception as e:
            self.log(f"❌ Failed to go home: {e}", "error")
    
    def emergency_stop(self):
        """Emergency stop all operations"""
        self.log("🛑 EMERGENCY STOP ACTIVATED!", "error")
        
        self.scanning = False
        self.fetching = False
        
        # Enable buttons
        self.scan_button.config(state="normal")
        self.fetch_prompt_button.config(state="normal")
        self.fetch_button.config(state="normal")
        self.status_label.config(text="Status: Emergency Stop")
        
        # Stop camera if running
        if self.camera_system:
            self.camera_system.stop()
        
        # Move arm to safe position
        try:
            self.log("Moving arm to safe position...", "info")
            self.arm_controller.go_to_home()
            self.log("✅ Arm at safe position", "success")
        except:
            self.log("⚠️ Could not move arm to safe position", "warning")
    
    def update_info(self, *args):
        """Update system information"""
        text = "\n".join(args)
        self.info_label.config(text=text)
    
    def clear_log(self):
        """Clear the logger"""
        self.logger_text.delete(1.0, tk.END)
        self.log("Log cleared", "info")
    
    def on_closing(self):
        """Cleanup on window close"""
        self.log("Shutting down systems...", "system")
        
        # Stop any ongoing operations
        self.scanning = False
        self.fetching = False
        
        # Stop camera
        if self.camera_system:
            self.camera_system.stop()
        
        # Move arm to home
        try:
            if self.arm_controller:
                self.arm_controller.go_to_home()
        except:
            pass
        
        self.log("✅ Goodbye!", "success")
        self.root.destroy()
    
    def run(self):
        """Run the main application"""
        # Set closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start main loop
        self.root.mainloop()

def main():
    # Create necessary directories
    os.makedirs("data/snapshots", exist_ok=True)
    os.makedirs("data/mappings", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("movements", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Clean fetched tools file on startup
    fetched_tools_file = "data/fetched_tools.json"
    if os.path.exists(fetched_tools_file):
        os.remove(fetched_tools_file)
        print(f"Deleted previous fetched_tools.json")
    
    # Create and run main window
    app = MainWindow()
    app.run()

if __name__ == "__main__":
    main()