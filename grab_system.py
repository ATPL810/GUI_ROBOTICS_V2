import os
import json
import time
import re
import importlib.util

class GrabSystem:
    def __init__(self, arm_controller, snapshot_system, camera_system, log_callback=None):
        self.arm = arm_controller
        self.snapshot_system = snapshot_system
        self.camera_system = camera_system
        self.log_callback = log_callback
        
        # Tool mapping with duplicate handling
        self.tool_mapping = {}
        self.all_tool_locations = {}
        self.master_report_path = None
        
        # Fetched tools tracking
        self.fetched_tools_file = "data/fetched_tools.json"
        self.fetched_locations = self.load_fetched_locations()
        
        self.grab_scripts_dir = "movements"
        
    def log(self, message, level="info"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def load_fetched_locations(self):
        """Load previously fetched tool locations from file"""
        fetched_locations = {}
        
        if os.path.exists(self.fetched_tools_file):
            try:
                with open(self.fetched_tools_file, 'r') as f:
                    fetched_locations = json.load(f)
                self.log(f"Loaded {len(fetched_locations)} fetched locations from history", "info")
            except:
                self.log("Could not load fetched locations history", "warning")
                fetched_locations = {}
        
        return fetched_locations
    
    def load_mapping(self, master_report_path=None):
        """Load tool mapping from master report with duplicate handling"""
        if master_report_path:
            self.master_report_path = master_report_path
        else:
            self.master_report_path = self.find_latest_master_report()
        
        self.tool_mapping, self.all_tool_locations = self.parse_master_report()
        self.sync_fetched_status()
        
        self.log(f"Loaded {len(self.tool_mapping)} tool mappings with duplicate handling", "success")
        return self.tool_mapping
    
    def find_latest_master_report(self):
        """Find the latest master report"""
        import glob
        pattern = "data/mappings/mapping_*/master_report.txt"
        matches = glob.glob(pattern)
        
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
        
        raise FileNotFoundError("No master report found! Run scan first.")
    
    def parse_master_report(self):
        """Parse tool mapping from master report with duplicate handling"""
        tool_mapping = {}
        all_tool_locations = {}
        
        with open(self.master_report_path, 'r') as f:
            content = f.read()
        
        # Look for "ALL TOOL LOCATIONS (INCLUDING DUPLICATES):" section
        if "ALL TOOL LOCATIONS (INCLUDING DUPLICATES):" in content:
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
                
                # Parse location lines (e.g., "  Point A: 91.7% (initial position)")
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
                            confidence = float(match.group(2)) / 100
                            
                            # Try to extract position description
                            position_desc = "unknown"
                            if len(match.groups()) > 2 and match.group(3):
                                position_desc = match.group(3)
                            
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
                                continue
                            
                            location_data = {
                                "point": point,
                                "confidence": confidence,
                                "position": position,
                                "position_desc": position_desc,
                                "fetched": False
                            }
                            
                            tool_mapping[current_tool].append(location_data)
                            all_tool_locations[current_tool].append(location_data)
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
                else:
                    location["fetched"] = False
    
    def save_fetched_location(self, tool_name, point):
        """Save a fetched location to history"""
        location_key = f"{tool_name}_{point}"
        self.fetched_locations[location_key] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool_name,
            "point": point
        }
        
        # Update in-memory mapping
        if tool_name in self.tool_mapping:
            for location in self.tool_mapping[tool_name]:
                if location["point"] == point:
                    location["fetched"] = True
        
        # Save to file
        try:
            os.makedirs(os.path.dirname(self.fetched_tools_file), exist_ok=True)
            with open(self.fetched_tools_file, 'w') as f:
                json.dump(self.fetched_locations, f, indent=2)
        except Exception as e:
            self.log(f"Warning: Could not save fetched location: {e}", "warning")
    
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
                self.log(f"   Trying Pattern 1: with tool_type='{tool_name}'", "info")
                grab_func(self.arm.arm, tool_type=tool_name)
                self.log(f"   ✓ Success with tool_type parameter", "success")
                success = True
            except TypeError as e1:
                # Pattern 2: Try without parameter (for scripts like A)
                try:
                    self.log(f"   Trying Pattern 2: without parameters", "info")
                    grab_func(self.arm.arm)
                    self.log(f"   ✓ Success without parameters", "success")
                    success = True
                except TypeError as e2:
                    # Pattern 3: Try with tool_name (alternative parameter name)
                    try:
                        self.log(f"   Trying Pattern 3: with tool_name='{tool_name}'", "info")
                        grab_func(self.arm.arm, tool_name=tool_name)
                        self.log(f"   ✓ Success with tool_name parameter", "success")
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
            return False
        
        return success
    
    def fetch_tool(self, tool_name):
        """Fetch a specific tool by name with duplicate handling"""
        tool_name_lower = tool_name.lower().strip()
        
        self.log(f"\n{'='*60}", "system")
        self.log(f"🤖 FETCH REQUEST: {tool_name.upper()} (with duplicate handling)", "system")
        self.log(f"{'='*60}", "system")
        
        # Check if tool exists
        if tool_name_lower not in self.tool_mapping:
            self.log(f"❌ Tool '{tool_name}' not found in workshop!", "error")
            return False
        
        # Get next available location
        location = self.get_next_available_location(tool_name_lower)
        
        if location is None:
            self.log(f"⛔ All {tool_name_lower.upper()} instances have been fetched!", "error")
            self.log(f"   Total inventory: {len(self.tool_mapping[tool_name_lower])}", "info")
            return False
        
        grab_letter = location["point"]
        confidence = location["confidence"]
        position_desc = location["position_desc"]
        
        self.log(f"📍 Tool location: Point {grab_letter} ({position_desc})", "info")
        self.log(f"📊 Confidence: {confidence*100:.1f}%", "info")
        
        # Show inventory status
        total_count = len(self.tool_mapping[tool_name_lower])
        fetched_count = sum(1 for loc in self.tool_mapping[tool_name_lower] if loc.get("fetched", False))
        remaining = total_count - fetched_count - 1  # Minus the one we're about to fetch
        
        if total_count > 1:
            self.log(f"📦 Inventory: {remaining} more available after this fetch", "info")
        
        # Check if grab script exists
        if not self.check_grab_script_exists(grab_letter):
            self.log(f"❌ No grab script found for Point {grab_letter}", "error")
            self.log(f"   Please create movements/grab_point_{grab_letter}.py", "info")
            return False
        
        # Execute grab sequence
        self.log("\n🚀 Starting fetch sequence...", "success")
        success = self.execute_grab_script(grab_letter, tool_name_lower)
        
        if success:
            # Mark as fetched
            self.save_fetched_location(tool_name_lower, grab_letter)
            
            self.log(f"\n✅ Successfully fetched {tool_name_lower.upper()}!", "success")
            
            # Show remaining inventory
            if total_count > 1:
                new_remaining = total_count - (fetched_count + 1)
                if new_remaining > 0:
                    self.log(f"📦 {new_remaining} more {tool_name_lower.upper()} available in workshop", "info")
                else:
                    self.log(f"📦 Last {tool_name_lower.upper()} fetched!", "info")
            
            return True
        else:
            self.log(f"\n❌ Failed to fetch {tool_name_lower.upper()}", "error")
            return False