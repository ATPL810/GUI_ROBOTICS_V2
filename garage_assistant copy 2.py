"""
GARAGE ASSISTANT TOOL FETCHER - UPDATED
Handles tool-specific grip forces in grab scripts
"""

import os
import re
import time
import sys
from pathlib import Path
from Arm_Lib import Arm_Device
import json

class GarageAssistant:
    def __init__(self, master_report_path=None):
        """Initialize the garage assistant"""
        print("=" * 70)
        print("           GARAGE ASSISTANT - TOOL FETCHER v2.0")
        print("=" * 70)
        print("✨ NEW: Duplicate tool tracking enabled")
        print("=" * 70)
        
        # Initialize robot arm
        print("\nInitializing robot arm...")
        self.arm = self.initialize_arm()
        
        # Find master report
        self.master_report_path = self.find_master_report(master_report_path)
        print(f"Using tool mapping from: {self.master_report_path}")
        
        # Parse COMPLETE tool mapping from master report
        self.tool_mapping, self.tool_locations = self.parse_master_report()
        
        # NEW: Parse ALL tool locations including duplicates
        self.all_tool_locations = self.parse_all_tool_locations()
        
        # NEW: Tool status tracking for duplicates
        self.tool_status = self.load_tool_status()
        
        # Available grab scripts
        self.grab_scripts_dir = "movements"
        
        print("\n✅ System initialized successfully!")
        print(f"✅ Found {len(self.tool_mapping)} available tools")
        
        # NEW: Show duplicates summary
        self.show_duplicates_summary()
    
    def initialize_arm(self):
        """Initialize the robot arm"""
        try:
            arm = Arm_Device()
            time.sleep(2)  # Wait for arm to initialize
            
            # Move to safe starting position
            print("   Moving to safe starting position...")
            arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
            time.sleep(2)
            
            print("   Robot arm ready")
            return arm
            
        except Exception as e:
            print(f"   Failed to initialize robot arm: {e}")
            raise
    
    def find_master_report(self, path=None):
        """Find the latest master report"""
        if path and os.path.exists(path):
            return path
        
        # Look for master report in data directory
        possible_paths = [
            "data/mappings/mapping_*/master_report.txt",
            "data/snapshots/robot_snapshots_*/grab_point_assignments.txt",
            "master_report.txt"
        ]
        
        for pattern in possible_paths:
            import glob
            matches = glob.glob(pattern)
            if matches:
                # Get the latest one
                matches.sort(key=os.path.getmtime, reverse=True)
                return matches[0]
        
        raise FileNotFoundError("No master report found! Please run analyze_grab_points.py first.")
    
    def parse_master_report(self):
        """Parse tool mapping from master report - returns tool->point and full location data"""
        print("\nParsing tool mapping...")
        
        tool_mapping = {}  # tool_name -> grab_point
        tool_locations = {}  # grab_point -> tool_data
        
        with open(self.master_report_path, 'r') as f:
            content = f.read()
            
            # Parse the GRAB POINT TOOL ASSIGNMENTS section first
            if "GRAB POINT TOOL ASSIGNMENTS:" in content:
                sections = content.split("GRAB POINT TOOL ASSIGNMENTS:")[1].split("=")[0]
                
                # Extract tool assignments for each point
                for point_letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
                    # Look for pattern like "Point A (75,260): BOLT - 96.2% confidence"
                    pattern = rf'Point {point_letter}.*?:\s*([A-Z\s]+)\s*-'
                    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                    if match:
                        tool_name = match.group(1).strip().lower()
                        
                        # Store both mappings
                        tool_mapping[tool_name] = point_letter
                        
                        # Also store full data
                        tool_locations[point_letter] = {
                            "tool_name": tool_name,
                            "point": point_letter
                        }
                        print(f"   Point {point_letter}: {tool_name.upper()}")
            
            # Also parse TOOL TO GRAB POINT MAPPING section for verification
            if "TOOL TO GRAB POINT MAPPING:" in content and not tool_mapping:
                mapping_section = content.split("TOOL TO GRAB POINT MAPPING:")[1]
                mapping_section = mapping_section.split("=")[0]
                
                lines = mapping_section.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if "→" in line:
                        parts = line.split("→")
                        if len(parts) == 2:
                            tool_name = parts[0].strip().lower()
                            grab_point = parts[1].strip()
                            
                            match = re.search(r'Point\s+([A-I])', grab_point)
                            if match:
                                grab_letter = match.group(1)
                                tool_mapping[tool_name] = grab_letter
                                
                                # Store full data
                                tool_locations[grab_letter] = {
                                    "tool_name": tool_name,
                                    "point": grab_letter
                                }
                                print(f"   {tool_name.upper():<15} → Point {grab_letter}")
        
        return tool_mapping, tool_locations
    
    def parse_all_tool_locations(self):
        """NEW: Parse ALL tool locations including duplicates from master report"""
        print("\n🔍 Parsing ALL tool locations (including duplicates)...")
        
        all_locations = {}
        
        with open(self.master_report_path, 'r') as f:
            content = f.read()
            
            # Parse GRAB POINT TOOL ASSIGNMENTS section for ALL assignments
            if "GRAB POINT TOOL ASSIGNMENTS:" in content:
                print("✓ Found GRAB POINT TOOL ASSIGNMENTS section")
                assignments_section = content.split("GRAB POINT TOOL ASSIGNMENTS:")[1]
                assignments_section = assignments_section.split("=")[0] if "=" in assignments_section else assignments_section
                
                # Parse each line
                lines = assignments_section.strip().split('\n')
                
                matches_found = 0
                for line in lines:
                    line = line.strip()
                    
                    # Skip empty lines and section headers
                    if not line or line.startswith("---") or line.startswith("INITIAL") or line.startswith("SECOND") or line.startswith("THIRD"):
                        continue
                    
                    # Check if it's a valid tool line (contains "Point" and a letter)
                    if "Point" in line and any(letter in line for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']):
                        
                        # Skip "No tool assigned" lines
                        if "No tool assigned" in line:
                            print(f"  Skipping: {line[:50]}...")
                            continue
                        
                        print(f"  Parsing line: {line}")
                        
                        # Try to extract point letter
                        point_match = re.search(r'Point\s+([A-I])', line)
                        if not point_match:
                            continue
                        
                        point_letter = point_match.group(1)
                        
                        # Extract tool name and confidence - handle your exact format
                        # Pattern for: "Point A (75,260): BOLT - 90.6% confidence, 12px from tool center"
                        pattern = r'Point\s+[A-I]\s+\([^)]+\):\s*([A-Z\s]+)\s*-\s*([\d.]+)%'
                        match = re.search(pattern, line)
                        
                        if match:
                            tool_name = match.group(1).strip().lower()
                            confidence = float(match.group(2))
                            
                            print(f"    ✓ Found: Point {point_letter} = {tool_name} ({confidence}%)")
                            
                            if tool_name not in all_locations:
                                all_locations[tool_name] = []
                            
                            all_locations[tool_name].append({
                                "point": point_letter,
                                "confidence": confidence / 100,
                                "position": self.get_position_from_point(point_letter)
                            })
                            
                            matches_found += 1
                        else:
                            print(f"    ✗ Could not parse: {line[:50]}...")
                
                print(f"\nTotal matches found: {matches_found}")
                
                # If no matches found with GRAB POINT section, try ALL TOOL LOCATIONS section
                if matches_found == 0 and "ALL TOOL LOCATIONS (INCLUDING DUPLICATES):" in content:
                    print("\nTrying ALL TOOL LOCATIONS section instead...")
                    all_locations_section = content.split("ALL TOOL LOCATIONS (INCLUDING DUPLICATES):")[1]
                    all_locations_section = all_locations_section.split("=")[0] if "=" in all_locations_section else all_locations_section
                    
                    lines = all_locations_section.strip().split('\n')
                    
                    for line in lines:
                        line = line.strip()
                        
                        # Look for lines with tool names and points
                        if "⭐" in line or "Point" in line:
                            # Pattern for: "⭐ Point F: 91.4% (second position)"
                            pattern = r'[⭐•]\s*Point\s+([A-I]):\s*([\d.]+)%'
                            match = re.search(pattern, line)
                            
                            if match:
                                point_letter = match.group(1)
                                confidence = float(match.group(2))
                                
                                # Get tool name from previous lines
                                tool_name = None
                                # Look backward for tool name (like "BOLT (3 locations):")
                                for i in range(len(lines)):
                                    if lines[i].strip() == line:
                                        # Look at previous non-empty lines for tool name
                                        for j in range(i-1, max(-1, i-5), -1):
                                            prev_line = lines[j].strip()
                                            if prev_line and '(' in prev_line and ')' in prev_line:
                                                # Extract tool name from "BOLT (3 locations):"
                                                tool_match = re.search(r'([A-Z\s]+)\s*\(', prev_line)
                                                if tool_match:
                                                    tool_name = tool_match.group(1).strip().lower()
                                                    break
                                        break
                                
                                if tool_name:
                                    print(f"    ✓ Found in ALL LOCATIONS: {tool_name} at Point {point_letter} ({confidence}%)")
                                    
                                    if tool_name not in all_locations:
                                        all_locations[tool_name] = []
                                    
                                    all_locations[tool_name].append({
                                        "point": point_letter,
                                        "confidence": confidence / 100,
                                        "position": self.get_position_from_point(point_letter)
                                    })
                                    matches_found += 1
            
            else:
                print("✗ ERROR: GRAB POINT TOOL ASSIGNMENTS section not found!")
                return {}
        
        # Sort each tool's locations by confidence (highest first)
        for tool_name in all_locations:
            all_locations[tool_name].sort(key=lambda x: x["confidence"], reverse=True)
        
        print(f"\n✅ Successfully parsed {len(all_locations)} unique tool types:")
        for tool_name, locations in all_locations.items():
            print(f"  {tool_name.upper()}: {len(locations)} location(s)")
            for loc in locations:
                print(f"    • Point {loc['point']}: {loc['confidence']*100:.1f}% ({loc['position']})")
        
        return all_locations

    def get_position_from_point(self, point_letter):
        """NEW: Helper to get position from point letter"""
        if point_letter in ["A", "B", "C", "D"]:
            return "initial_position"
        elif point_letter in ["E", "F", "G"]:
            return "second_position"
        else:  # H, I
            return "third_position"

    def load_tool_status(self):
        """NEW: Load tool status from file"""
        self.tool_status_file = "data/tool_status.json"
        os.makedirs("data", exist_ok=True)
        
        if os.path.exists(self.tool_status_file):
            try:
                with open(self.tool_status_file, 'r') as f:
                    status = json.load(f)
                print("   Loaded existing tool status")
                return status
            except:
                print("   Creating new tool status")
        
        # Initialize new status
        status = {}
        for tool_name, locations in self.all_tool_locations.items():
            status[tool_name] = {
                "available": [loc["point"] for loc in locations],
                "fetched": [],
                "next_available": locations[0]["point"] if locations else None
            }
        
        self.save_tool_status(status)
        return status

    def save_tool_status(self, status=None):
        """NEW: Save tool status to file"""
        if status is None:
            status = self.tool_status
        
        with open(self.tool_status_file, 'w') as f:
            json.dump(status, f, indent=2)

    def update_tool_status(self, tool_name, fetched_point):
        """NEW: Update status after fetching"""
        if tool_name in self.tool_status:
            # Remove from available, add to fetched
            if fetched_point in self.tool_status[tool_name]["available"]:
                self.tool_status[tool_name]["available"].remove(fetched_point)
                self.tool_status[tool_name]["fetched"].append(fetched_point)
                
                # Update next available (highest confidence remaining)
                if self.tool_status[tool_name]["available"]:
                    remaining_points = self.tool_status[tool_name]["available"]
                    remaining_locations = [loc for loc in self.all_tool_locations[tool_name] 
                                        if loc["point"] in remaining_points]
                    if remaining_locations:
                        remaining_locations.sort(key=lambda x: x["confidence"], reverse=True)
                        self.tool_status[tool_name]["next_available"] = remaining_locations[0]["point"]
                    else:
                        self.tool_status[tool_name]["next_available"] = None
                else:
                    self.tool_status[tool_name]["next_available"] = None
                
                self.save_tool_status()
                return True
        
        return False

    def show_duplicates_summary(self):
        """NEW: Show summary of duplicate tools"""
        print("\n📊 DUPLICATE TOOLS INVENTORY:")
        print("-" * 40)
        
        total_tools = 0
        for tool_name, locations in self.all_tool_locations.items():
            count = len(locations)
            total_tools += count
            
            if count > 1:
                print(f"  {tool_name.upper():<15}: {count} locations")
                for loc in locations[:3]:  # Show top 3
                    print(f"    • Point {loc['point']}: {loc['confidence']*100:.1f}%")
                if count > 3:
                    print(f"    • ... and {count-3} more")
            else:
                print(f"  {tool_name.upper():<15}: 1 location (Point {locations[0]['point']})")
        
        print("-" * 40)
        print(f"Total tools (including duplicates): {total_tools}")
    
    def list_available_tools(self):
        """List all available tools - ENHANCED to show duplicates"""
        print("\n" + "=" * 70)
        print("AVAILABLE TOOLS INVENTORY:")
        print("-" * 40)
        
        if not self.tool_status:
            print("No tools found!")
            return
        
        # List all tools with availability
        for tool_name in sorted(self.tool_status.keys()):
            status = self.tool_status[tool_name]
            available_count = len(status["available"])
            fetched_count = len(status["fetched"])
            total_count = available_count + fetched_count
            
            if total_count > 1:
                # Tool has duplicates
                print(f"\n{tool_name.upper():<15} [Total: {total_count}, Available: {available_count}]")
                if available_count > 0:
                    print(f"   Available at: {', '.join(status['available'])}")
                if fetched_count > 0:
                    print(f"   Already taken: {', '.join(status['fetched'])}")
            else:
                # Single tool
                if available_count > 0:
                    print(f"  • {tool_name.upper():<15} [Available at: Point {status['available'][0]}]")
                else:
                    print(f"  • {tool_name.upper():<15} [OUT OF STOCK]")
        
        print("-" * 40)
        
        # Show quick summary
        total_available = sum(len(s["available"]) for s in self.tool_status.values())
        total_fetched = sum(len(s["fetched"]) for s in self.tool_status.values())
        print(f"Quick Stats: {total_available} tools available, {total_fetched} fetched")
        print("=" * 70)
    
    def check_grab_script_exists(self, grab_letter):
        """Check if grab script exists for a point"""
        script_path = os.path.join(self.grab_scripts_dir, f"grab_point_{grab_letter}.py")
        return os.path.exists(script_path)
    
    def execute_grab_script(self, grab_letter, tool_name):
        """Execute the grab script for a specific point - UNIVERSAL VERSION"""
        script_path = os.path.join(self.grab_scripts_dir, f"grab_point_{grab_letter}.py")
        
        if not os.path.exists(script_path):
            print(f"ERROR: Grab script not found: {script_path}")
            return False
        
        print(f"\nExecuting grab script for Point {grab_letter} - {tool_name.upper()}...")
        
        try:
            # Import and execute the script
            script_name = f"grab_point_{grab_letter}"
            
            # Dynamically import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(script_name, script_path)
            module = importlib.util.module_from_spec(spec)
            
            # Add arm object to module
            module.arm = self.arm
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Get the grab function
            grab_func_name = f"grab_point_{grab_letter}"
            if not hasattr(module, grab_func_name):
                print(f"❌ ERROR: Function {grab_func_name} not found in script")
                return False
            
            grab_func = getattr(module, grab_func_name)
            
            # UNIVERSAL APPROACH: Try both calling patterns
            success = False
            
            # Pattern 1: Try with tool_type parameter (for scripts like B)
            try:
                print(f"   Trying Pattern 1: with tool_type='{tool_name}'")
                grab_func(self.arm, tool_type=tool_name)
                print(f"   ✓ Success with tool_type parameter")
                success = True
            except TypeError as e1:
                # Pattern 2: Try without parameter (for scripts like A)
                try:
                    print(f"   Trying Pattern 2: without parameters")
                    grab_func(self.arm)
                    print(f"   ✓ Success without parameters")
                    success = True
                except TypeError as e2:
                    # Pattern 3: Try with tool_name (alternative parameter name)
                    try:
                        print(f"   Trying Pattern 3: with tool_name='{tool_name}'")
                        grab_func(self.arm, tool_name=tool_name)
                        print(f"   ✓ Success with tool_name parameter")
                        success = True
                    except TypeError as e3:
                        print(f"❌ ERROR: Function doesn't accept expected parameters")
                        print(f"   Error 1 (tool_type): {e1}")
                        print(f"   Error 2 (no params): {e2}")
                        print(f"   Error 3 (tool_name): {e3}")
                        success = False
            
            # Check if script actually grabbed the right tool
            if success:
                # Read the script to see what tool it's configured for
                with open(script_path, 'r') as f:
                    script_content = f.read()
                
                # Check if script mentions a specific tool
                if "BOLT" in script_content.upper() and "bolt" not in tool_name.lower():
                    print(f"⚠️  WARNING: Script appears to be for BOLTS, but fetching {tool_name.upper()}")
                elif "WRENCH" in script_content.upper() and "wrench" not in tool_name.lower():
                    print(f"⚠️  WARNING: Script appears to be for WRENCH, but fetching {tool_name.upper()}")
                    
        except Exception as e:
            print(f"❌ ERROR executing grab script: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return success
    
    def fetch_tool(self, tool_name):
        """Fetch a specific tool by name - ENHANCED for duplicates"""
        tool_name_lower = tool_name.lower().strip()
        
        print(f"\n" + "=" * 70)
        print(f"REQUEST: Fetch '{tool_name_lower.upper()}'")
        print("=" * 70)
        
        # Check if tool exists
        if tool_name_lower not in self.tool_status:
            print(f"❌ Tool '{tool_name}' not found in workshop!")
            print("\nAvailable tools are:")
            for available_tool in sorted(self.tool_status.keys()):
                count = len(self.tool_status[available_tool]["available"])
                status = f"({count} available)" if count > 0 else "(OUT OF STOCK)"
                print(f"  • {available_tool.upper():<15} {status}")
            return False
        
        # Check if tool is available
        status = self.tool_status[tool_name_lower]
        if not status["available"]:
            print(f"❌ No more '{tool_name}' available!")
            if status["fetched"]:
                print(f"   Previously fetched from: {', '.join(status['fetched'])}")
            return False
        
        # Get next available point (highest confidence)
        grab_letter = status["next_available"]
        
        # Show detailed tool status
        print(f"\n📋 TOOL STATUS: {tool_name_lower.upper()}")
        print(f"   Available at {len(status['available'])} location(s): {', '.join(status['available'])}")
        if status["fetched"]:
            print(f"   Already fetched from: {', '.join(status['fetched'])}")
        print(f"   Fetching from: Point {grab_letter} (highest confidence)")
        
        # Show all locations with confidence
        if tool_name_lower in self.all_tool_locations:
            print(f"\n📊 All locations for {tool_name_lower.upper()}:")
            for loc in self.all_tool_locations[tool_name_lower]:
                status_symbol = "✓" if loc["point"] in status["available"] else "✗"
                current_indicator = " ← CURRENT" if loc["point"] == grab_letter else ""
                print(f"   {status_symbol} Point {loc['point']}: {loc['confidence']*100:.1f}%{current_indicator}")
        
        # Check if grab script exists
        if not self.check_grab_script_exists(grab_letter):
            print(f"❌ No grab script found for Point {grab_letter}")
            print(f"   Please create movements/grab_point_{grab_letter}.py")
            return False
        
        # Show grab plan
        print(f"\n📋 GRAB PLAN:")
        print(f"   Tool: {tool_name_lower.upper()}")
        print(f"   Location: Point {grab_letter}")
        
        # Get which position this is in
        if grab_letter in ["A", "B", "C", "D"]:
            position = "Initial Position (Base: 90°)"
        elif grab_letter in ["E", "F", "G"]:
            position = "Second Position (Base: 40°)"
        else:  # H, I
            position = "Third Position (Base: 1°)"
        
        print(f"   Arm Position: {position}")
        
        # Confirm with user
        print(f"\nReady to fetch {tool_name_lower.upper()} from Point {grab_letter}")
        print("Starting in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        # Execute grab sequence
        print("\n🚀 Starting fetch sequence...")
        success = self.execute_grab_script(grab_letter, tool_name_lower)
        
        if success:
            print(f"\n✅ Successfully fetched {tool_name_lower.upper()} from Point {grab_letter}!")
            
            # Update tool status
            self.update_tool_status(tool_name_lower, grab_letter)
            
            # Show remaining inventory
            remaining = len(self.tool_status[tool_name_lower]["available"])
            if remaining > 0:
                print(f"\n🔄 {tool_name_lower.upper()} INVENTORY UPDATE:")
                print(f"   Remaining: {remaining} available")
                print(f"   Next available at: Point {self.tool_status[tool_name_lower]['next_available']}")
            else:
                print(f"\n⚠️  INVENTORY ALERT: No more {tool_name_lower.upper()} left!")
            
            # Ask if user wants to return tool
            if self.ask_return_tool(tool_name_lower, grab_letter):
                self.return_tool_to_position(tool_name_lower, grab_letter)
            
            return True
        else:
            print(f"\n❌ Failed to fetch {tool_name_lower.upper()}")
            return False
    
    def ask_return_tool(self, tool_name, grab_letter):
        """Ask user if they want to return the tool"""
        print("\n" + "-" * 40)
        response = input(f"Return {tool_name.upper()} to Point {grab_letter}? (yes/no): ").strip().lower()
        return response in ["yes", "y"]
    
    def return_tool_to_position(self, tool_name, grab_letter):
        """Return tool to its original position"""
        print(f"\nReturning {tool_name.upper()} to Point {grab_letter}...")
        
        # This would be the reverse sequence of your grab script
        # For now, just move to home position
        print("   Moving to home position...")
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
        time.sleep(2)
        
        print(f"✅ Tool returned (simulated)")
    
    def interactive_mode(self):
        """Run in interactive mode with user input"""
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE - WITH DUPLICATE TRACKING")
        print("=" * 70)
        
        # Show available tools
        self.list_available_tools()
        
        while True:
            print("\n" + "-" * 40)
            print("GARAGE ASSISTANT MENU:")
            print("  1. 🔧 Fetch a tool")
            print("  2. 📋 List available tools")
            print("  3. 🔄 Reset all tools (restock)")
            print("  4. 🧪 Test grab point (dry run)")
            print("  5. 🏠 Return to home position")
            print("  6. ❌ Exit")
            print("-" * 40)
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                # Fetch tool - UPDATED to use tool_status
                print("\nAvailable tools:")
                for tool_name in sorted(self.tool_status.keys()):
                    status = self.tool_status[tool_name]
                    available_count = len(status["available"])
                    if available_count > 0:
                        count_text = f"({available_count} available)" if available_count > 1 else ""
                        print(f"  • {tool_name.upper():<15} {count_text}")
                
                tool_name = input("\nWhich tool would you like? ").strip()
                if tool_name:
                    self.fetch_tool(tool_name)
            
            elif choice == "2":
                # List tools
                self.list_available_tools()
            
            elif choice == "3":
                # Reset tools
                confirm = input("\nAre you sure you want to reset all tools? (yes/no): ").strip().lower()
                if confirm in ["yes", "y"]:
                    self.reset_tool_status()
            
            elif choice == "4":
                # Test grab point (dry run)
                self.test_grab_point_dry_run()
            
            elif choice == "5":
                # Return to home
                print("\n🏠 Returning to home position...")
                self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
                time.sleep(2)
                print("✅ Arm at home position")
            
            elif choice == "6":
                # Exit
                print("\n🏠 Returning arm to home position...")
                self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
                time.sleep(2)
                print("\n👋 Goodbye! Have a productive day in the garage!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")
        
    def test_grab_point_dry_run(self):
        """Test a grab point without actually grabbing"""
        print("\n" + "=" * 70)
        print("TEST GRAB POINT (DRY RUN)")
        print("=" * 70)
        
        print("\nAvailable grab points:")
        for grab_letter in sorted(set(self.tool_mapping.values())):
            if grab_letter in self.tool_locations:
                tool_name = self.tool_locations[grab_letter]["tool_name"].upper()
                print(f"  Point {grab_letter}: {tool_name}")
        
        grab_letter = input("\nWhich point to test? (A-I): ").strip().upper()
        
        if grab_letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
            print(f"\n🧪 Testing Point {grab_letter} (Dry Run)...")
            print("   Moving to approach position...")
            print("   (Simulating grab sequence)")
            print("   ✓ Dry run completed successfully")
        else:
            print("❌ Invalid grab point")
    
    def voice_command_mode(self, command):
        """Handle voice command (for integration with voice assistant)"""
        # Normalize command
        command = command.lower().strip()
        
        # Check for fetch commands
        fetch_keywords = ["get", "fetch", "bring", "hand me", "give me", "pass", "i need"]
        
        for keyword in fetch_keywords:
            if keyword in command:
                # Extract tool name
                tool_name = command.replace(keyword, "").strip()
                tool_name = tool_name.replace("a ", "").replace("an ", "").replace("the ", "").strip()
                if tool_name:
                    return self.fetch_tool(tool_name)
        
        # Handle other commands
        if "list tools" in command or "what tools" in command or "available tools" in command:
            self.list_available_tools()
            return True
        
        elif "home" in command or "rest" in command or "park" in command:
            print("Returning to home position...")
            self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
            time.sleep(2)
            print("✅ Arm at home position")
            return True
        
        elif "test" in command:
            self.test_grab_point_dry_run()
            return True
        
        else:
            print(f"Sorry, I don't understand: '{command}'")
            print("Try: 'get hammer', 'list tools', 'test point A', or 'go home'")
            return False
        
    def reset_tool_status(self):
        """NEW: Reset all tools to available (simulate restocking)"""
        print("\n🔄 Resetting all tools to available...")
        
        for tool_name, locations in self.all_tool_locations.items():
            self.tool_status[tool_name] = {
                "available": [loc["point"] for loc in locations],
                "fetched": [],
                "next_available": locations[0]["point"] if locations else None
            }
        
        self.save_tool_status()
        print("✅ All tools reset to available!")
        self.list_available_tools()

def main():
    """Main execution function"""
    try:
        # Create assistant
        assistant = GarageAssistant()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            # Command line mode
            tool_name = " ".join(sys.argv[1:])
            assistant.fetch_tool(tool_name)
        else:
            # Interactive mode
            assistant.interactive_mode()
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease make sure you have:")
        print("1. Run take_snapshots.py to take photos of your workspace")
        print("2. Run analyze_grab_points.py to create the tool mapping")
        print("3. The master_report.txt file exists")
        print("\nExpected file: data/mappings/mapping_*/master_report.txt")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
        print("🏠 Returning arm to home position...")
        # Try to return to home if possible
        try:
            assistant.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
        except:
            pass
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()