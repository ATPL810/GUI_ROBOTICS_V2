import sys
import os
import threading
import cv2
import time
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
from voice_assistant import VoiceAssistant

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🏗️ Garage Assistant Pro")
        self.root.geometry("1200x800")  # Increased height
        self.root.configure(bg="#1e1e2e")
        
        # System components
        self.arm_controller = None
        self.camera_system = None
        self.snapshot_system = None
        self.analysis_system = None
        self.grab_system = None
        self.voice_assistant = None
        
        # State variables
        self.scanning = False
        self.fetching = False
        self.listening_voice = False
        self.current_snapshot_folder = None
        self.tool_mapping = {}
        
        # Camera frame
        self.current_frame = None
        self.photo = None
        self.frame_update_scheduled = False
        
        # Initialize GUI
        self.init_ui()
        
        # Start systems
        self.initialize_systems()
        
        # Start periodic updates
        self.update_camera()
        self.update_voice_commands()
        
    def init_ui(self):
        # Configure grid weights - adjust for larger camera
        self.root.grid_columnconfigure(0, weight=3)  # Increased weight for left panel
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Left panel (Camera + Controls)
        left_frame = tk.Frame(self.root, bg="#1e1e2e")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(0, weight=4)  # Increased weight for camera
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_rowconfigure(2, weight=2)
        
        # Camera display - Made taller
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
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        
        # Control buttons grid
        self.scan_button = tk.Button(control_frame, text="🔍 Start Scan & Analysis", 
                                    font=("Arial", 11, "bold"),
                                    bg="#a6e3a1", fg="#1e1e2e",
                                    activebackground="#94e2d5",
                                    command=self.start_scan)
        self.scan_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Voice control button instead of emergency stop
        self.voice_button = tk.Button(control_frame, text="🎤 Fetch Tool Using Voice", 
                                     font=("Arial", 11, "bold"),
                                     bg="#f5c2e7", fg="#1e1e2e",
                                     activebackground="#cba6f7",
                                     command=self.toggle_voice_control)
        self.voice_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.home_button = tk.Button(control_frame, text="🏠 Go Home", 
                                    font=("Arial", 11, "bold"),
                                    bg="#585b70", fg="#cdd6f4",
                                    activebackground="#89b4fa",
                                    command=self.go_home)
        self.home_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.fetch_prompt_button = tk.Button(control_frame, text="🤖 Fetch Tool (GUI)...", 
                                           font=("Arial", 11, "bold"),
                                           bg="#89b4fa", fg="#1e1e2e",
                                           activebackground="#74c7ec",
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
        
        # Voice status indicator
        self.voice_status_label = tk.Label(control_frame, text="🎤 Voice: OFF", 
                                         font=("Arial", 10),
                                         fg="#f38ba8", bg="#1e1e2e")
        self.voice_status_label.grid(row=4, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        
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
                                       "Voice Control: Disabled\n"
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
            
            # Initialize camera with thread-safe callback
            self.log("📷 Starting camera...", "info")
            self.camera_system = CameraSystem(
                log_callback=self.log, 
                frame_callback=self.handle_camera_frame
            )
            self.camera_system.start()
            
            # Initialize other systems
            self.snapshot_system = SnapshotSystem(self.arm_controller, self.camera_system, self.log)
            self.analysis_system = AnalysisSystem(self.log)
            self.grab_system = GrabSystem(self.arm_controller, self.snapshot_system, 
                                         self.camera_system, self.log)
            
            # Initialize voice assistant
            self.log("🎤 Initializing voice assistant...", "info")
            self.voice_assistant = VoiceAssistant(
                fetch_callback=self.handle_voice_fetch,
                log_callback=self.log
            )
            
            self.log("✅ All systems initialized successfully!", "success")
            self.update_info("Arm: Connected ✓", "Camera: Running ✓", "Voice Control: Ready")
            
        except Exception as e:
            self.log(f"❌ Failed to initialize systems: {e}", "error")
    
    def handle_camera_frame(self, frame):
        """Handle camera frames from background thread - DON'T update GUI here!"""
        # Just store the frame, GUI update happens in main thread
        self.current_frame = frame
    
    def log(self, message, level="info"):
        """Log message to logger - thread-safe"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Use after() to schedule GUI update in main thread
        def update_log():
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
        
        # Schedule the update in the main thread
        self.root.after(0, update_log)
    
    def toggle_voice_control(self):
        """Toggle voice control on/off"""
        if self.listening_voice:
            self.stop_voice_control()
        else:
            self.start_voice_control()
    
    def start_voice_control(self):
        """Start voice control"""
        if self.voice_assistant:
            success = self.voice_assistant.start_listening()
            if success:
                self.listening_voice = True
                self.voice_button.config(text="🎤 Stop Voice Control", bg="#f38ba8")
                self.voice_status_label.config(text="🎤 Voice: ON (Listening...)", fg="#a6e3a1")
                self.log("Voice control activated - Say 'hammer', 'wrench', etc.", "success")
                self.update_info("Voice Control: Active")
            else:
                self.log("Failed to start voice control", "error")
    
    def stop_voice_control(self):
        """Stop voice control"""
        if self.voice_assistant:
            self.voice_assistant.stop_listening()
            self.listening_voice = False
            self.voice_button.config(text="🎤 Fetch Tool Using Voice", bg="#f5c2e7")
            self.voice_status_label.config(text="🎤 Voice: OFF", fg="#f38ba8")
            self.log("Voice control deactivated", "info")
            self.update_info("Voice Control: Disabled")
    
    def handle_voice_fetch(self, tool_name):
        """Handle voice fetch request"""
        self.log(f"🎤 Voice command to fetch: {tool_name}", "system")
        self.start_fetch_tool(tool_name)
    
    def update_voice_commands(self):
        """Periodically check for voice commands"""
        if self.voice_assistant and self.listening_voice:
            # Process any queued voice commands
            self.voice_assistant.process_voice_commands()
            
            # Check for tool requests from voice assistant
            if hasattr(self.voice_assistant, 'command_queue') and self.voice_assistant.command_queue:
                while self.voice_assistant.command_queue:
                    text = self.voice_assistant.command_queue.pop(0)
                    success, tool_name = self.voice_assistant.process_command(text)
                    
                    if tool_name:
                        # Voice assistant will speak fetching message itself
                        self.log(f"🎤 Voice requested: {tool_name}", "info")
                        self.start_fetch_tool(tool_name)
        
        # Schedule next update
        self.root.after(100, self.update_voice_commands)
    
    def update_camera(self):
        """Update camera display - runs in main thread"""
        try:
            if self.current_frame is not None:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                
                # Resize to fit label with larger vertical size
                h, w, _ = rgb_image.shape
                max_h = self.camera_label.winfo_height() or 600  # Increased from 480
                max_w = self.camera_label.winfo_width() or 640
                
                if max_h > 0 and max_w > 0:
                    scale = min(max_w/w, max_h/h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    
                    if new_w > 0 and new_h > 0:
                        resized = cv2.resize(rgb_image, (new_w, new_h))
                        
                        # Convert to ImageTk
                        image = Image.fromarray(resized)
                        self.photo = ImageTk.PhotoImage(image=image)
                        self.camera_label.config(image=self.photo)
        except Exception as e:
            # Don't log every frame error to avoid spam
            if not self.frame_update_scheduled:
                self.log(f"Camera display error: {e}", "warning")
        
        # Schedule next update
        self.root.after(33, self.update_camera)  # ~30 FPS
    
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
        self.voice_button.config(state="disabled")
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
                
                # If voice control is active, give confirmation via voice
                if self.listening_voice and self.voice_assistant:
                    self.voice_assistant.speak(f"Here is your {tool_name}! What else can I get for you?")
            else:
                self.log(f"❌ Failed to fetch {tool_name.upper()}", "error")
                if self.listening_voice and self.voice_assistant:
                    self.voice_assistant.speak(f"I couldn't fetch the {tool_name}. Please try again or choose another tool.")
                
        except Exception as e:
            self.log(f"❌ Fetch error: {e}", "error")
            if self.listening_voice and self.voice_assistant:
                self.voice_assistant.speak("There was an error fetching the tool. Please check the system.")
        
        finally:
            self.fetching = False
            self.fetch_prompt_button.config(state="normal")
            self.fetch_button.config(state="normal")
            self.voice_button.config(state="normal")
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
        self.fetch_button.config(state="disabled")
        self.voice_button.config(state="disabled")
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
            
            # Announce via voice if enabled
            if self.listening_voice and self.voice_assistant:
                self.voice_assistant.speak(f"Scan complete! I've mapped {len(self.tool_mapping)} tools. You can now ask for any tool.")
            
        except Exception as e:
            self.log(f"❌ Scan failed: {e}", "error")
        
        finally:
            self.scanning = False
            self.scan_button.config(state="normal")
            self.fetch_prompt_button.config(state="normal")
            self.fetch_button.config(state="normal")
            self.voice_button.config(state="normal")
            self.status_label.config(text="Status: Ready")
    
    def update_progress(self, value):
        """Update progress bar - thread-safe"""
        def update():
            self.progress_bar["value"] = value
        
        self.root.after(0, update)
    
    def update_tools_list(self):
        """Update the tools list widget with inventory info - thread-safe"""
        def update():
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
        
        self.root.after(0, update)
    
    def go_home(self):
        """Move arm to home position"""
        self.log("Moving arm to home position...", "info")
        try:
            self.arm_controller.go_to_home()
            self.log("✅ Arm at home position", "success")
        except Exception as e:
            self.log(f"❌ Failed to go home: {e}", "error")
    
    def update_info(self, *args):
        """Update system information - thread-safe"""
        def update():
            text = "\n".join(args)
            self.info_label.config(text=text)
        
        self.root.after(0, update)
    
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
        
        # Stop voice control
        self.stop_voice_control()
        
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
    
    # Check for Vosk model
    vosk_model_path = "vosk-model-small-en-us-0.15"
    if not os.path.exists(vosk_model_path):
        print("⚠️  Warning: Vosk model not found!")
        print("Please download and extract the model:")
        print("1. Download from: https://alphacephei.com/vosk/models")
        print(f"2. Extract to: {vosk_model_path}")
        print("3. Or place the model folder in the current directory")
        print("\nVoice control will be disabled until model is installed.")
    
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