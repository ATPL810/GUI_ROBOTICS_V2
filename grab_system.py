import os
import time
import json
import re
import glob
from datetime import datetime

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
                        r'[\s]*Point\s+([A-I]):\s*([\d.]+)%\s*\((.*?)\)',
                        r'[\s]*Point\s+([A-I]):\s*([\d.]+)%',
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
                                "fetched": False  # Initialize as not fetched
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
        self.log(f"FETCH REQUEST: {tool_name.upper()}", "system")
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
            import tkinter.messagebox as messagebox
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
                import tkinter.messagebox as messagebox
                warning_reply = messagebox.askyesno(
                    " Warning",
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
            
            # Return success - keep original return type for compatibility
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
            
            status = "+" if available_count > 0 else "-"
            
            if total_count > 1:
                self.log(f"  {status} {tool_name.upper():<15} → {available_count}/{total_count} available", "info")
            else:
                self.log(f"  {status} {tool_name.upper():<15} → {available_count}/{total_count} available", "info")