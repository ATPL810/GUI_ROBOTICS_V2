"""
GARAGE ASSISTANT TOOL FETCHER - WITH DUPLICATE HANDLING
Uses master_report.txt to locate tools and runs corresponding grab scripts
Tracks duplicate tools and automatically selects next available instance
"""

import os
import re
import time
import sys
import json
from pathlib import Path
from Arm_Lib import Arm_Device

class GarageAssistant:
    def __init__(self, master_report_path=None):
        """Initialize the garage assistant with duplicate tracking"""
        print("=" * 70)
        print("           GARAGE ASSISTANT - TOOL FETCHER")
        print("           (WITH DUPLICATE HANDLING)")
        print("=" * 70)
        
        # Initialize robot arm
        print("\nInitializing robot arm...")
        self.arm = self.initialize_arm()
        
        # Find master report
        self.master_report_path = self.find_master_report(master_report_path)
        print(f"Using tool mapping from: {self.master_report_path}")
        
        # Track fetched tools (for duplicate handling) - INITIALIZE THIS FIRST!
        self.fetched_tools_file = "data/fetched_tools.json"
        self.fetched_locations = self.load_fetched_locations()
        
        # Parse tool mapping from master report - NOW THIS CAN ACCESS fetched_locations
        self.tool_mapping, self.all_tool_locations = self.parse_master_report()
        
        # Sync fetched status after parsing
        self.sync_fetched_status()
        
        # Available grab scripts
        self.grab_scripts_dir = "movements"
        
        print("\n✅ System initialized successfully!")
        print(f"✅ Found {len(self.tool_mapping)} unique tool types")
        print(f"✅ Tracking {len(self.fetched_locations)} previously fetched items")
    
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
            "data/snapshots/robot_snapshots_*/master_report.txt",
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
    
    def load_fetched_locations(self):
        """Load previously fetched tool locations from file"""
        fetched_locations = {}
        
        if os.path.exists(self.fetched_tools_file):
            try:
                with open(self.fetched_tools_file, 'r') as f:
                    fetched_locations = json.load(f)
                print(f"   Loaded {len(fetched_locations)} fetched locations from history")
            except:
                print("   Could not load fetched locations history")
                fetched_locations = {}
        
        return fetched_locations
    
    def parse_master_report(self):
        """
        Parse tool mapping from master report with duplicate handling
        Returns:
            - tool_mapping: Dict of tool_name -> list of available locations (sorted by confidence)
            - all_tool_locations: Complete mapping of all tool instances
        """
        print("\nParsing tool mapping (with duplicate handling)...")
        
        tool_mapping = {}
        all_tool_locations = {}
        
        with open(self.master_report_path, 'r') as f:
            content = f.read()
        
        # Look for "ALL TOOL LOCATIONS (INCLUDING DUPLICATES):" section
        if "ALL TOOL LOCATIONS (INCLUDING DUPLICATES):" in content:
            print("Found duplicate locations section")
            
            # Extract the entire section
            start_idx = content.find("ALL TOOL LOCATIONS (INCLUDING DUPLICATES):")
            # Find the end of this section (look for next major section)
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
                    # Extract tool name - match pattern like "BOLT (3 locations):"
                    match = re.search(r'^([A-Z\s]+)\s*\(', line)
                    if match:
                        current_tool = match.group(1).strip().lower()
                        tool_mapping[current_tool] = []
                        all_tool_locations[current_tool] = []
                        print(f"   Found tool: {current_tool}")
                
                # Parse location lines (e.g., "  Point A: 91.7% (initial position)")
                elif "Point" in line and current_tool and ":" in line and not line.startswith("="):
                    # Skip lines that are just dashes or empty
                    if line.startswith("-") or not line:
                        continue
                    
                    # Try multiple patterns
                    patterns = [
                        r'[⭐\s]*Point\s+([A-I]):\s*([\d.]+)%\s*\((.*?)\)',  # "Point A: 91.7% (initial position)"
                        r'[⭐\s]*Point\s+([A-I]):\s*([\d.]+)%',              # "Point A: 91.7%"
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, line)
                        if match:
                            point = match.group(1)
                            confidence = float(match.group(2))
                            
                            # Try to extract position description
                            position_desc = "unknown"
                            if len(match.groups()) > 2 and match.group(3):
                                position_desc = match.group(3)
                            else:
                                # Try to extract from the line
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
                                print(f"   Skipping Point E for bolt (E is hammer)")
                                continue
                            
                            location_data = {
                                "point": point,
                                "confidence": confidence / 100,
                                "position": position,
                                "position_desc": position_desc,
                                "fetched": False
                            }
                            
                            tool_mapping[current_tool].append(location_data)
                            all_tool_locations[current_tool].append(location_data)
                            
                            print(f"      Point {point}: {confidence}% confidence")
                            break
        
        # Clean up: Remove any tool entries with no locations
        tools_to_remove = [tool for tool, locations in tool_mapping.items() if not locations]
        for tool in tools_to_remove:
            del tool_mapping[tool]
            del all_tool_locations[tool]
        
        # Sort each tool's locations by confidence (highest first)
        for tool_name in tool_mapping:
            tool_mapping[tool_name].sort(key=lambda x: x["confidence"], reverse=True)
        
        # Print summary
        print("\n" + "-" * 40)
        print("TOOL MAPPING SUMMARY:")
        print("-" * 40)
        
        for tool_name, locations in sorted(tool_mapping.items()):
            count = len(locations)
            if count > 1:
                print(f"   {tool_name.upper():<15} → {count} locations (duplicate)")
                # List all locations
                for loc in locations:
                    print(f"      Point {loc['point']}: {loc['confidence']*100:.1f}%")
            else:
                print(f"   {tool_name.upper():<15} → 1 location")
        
        return tool_mapping, all_tool_locations
    
    def sync_fetched_status(self):
        """Sync fetched status from fetched_locations to tool_mapping"""
        print("\nSyncing fetched status...")
        
        for tool_name, locations in self.tool_mapping.items():
            for location in locations:
                location_key = f"{tool_name}_{location['point']}"
                if location_key in self.fetched_locations:
                    location["fetched"] = True
                    print(f"   Marked {tool_name.upper()} at Point {location['point']} as fetched")
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
            print(f"Warning: Could not save fetched location: {e}")
    
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
    
    def list_available_tools(self, show_details=False):
        """List all available tools with inventory status"""
        print("\n" + "=" * 70)
        print("WORKSHOP INVENTORY:")
        print("=" * 70)
        
        if not self.tool_mapping:
            print("No tools found in mapping!")
            return
        
        total_tools = 0
        available_tools = 0
        
        for tool_name in sorted(self.tool_mapping.keys()):
            locations = self.tool_mapping[tool_name]
            total_count = len(locations)
            fetched_count = sum(1 for loc in locations if loc.get("fetched", False))
            available_count = total_count - fetched_count
            
            total_tools += total_count
            available_tools += available_count
            
            status = "✅" if available_count > 0 else "⛔"
            
            if show_details:
                print(f"\n{status} {tool_name.upper():<15}")
                print(f"   Total: {total_count}, Available: {available_count}, Fetched: {fetched_count}")
                
                for i, loc in enumerate(locations, 1):
                    fetched_flag = "✓" if loc.get("fetched", False) else "○"
                    print(f"   {fetched_flag} Location {i}: Point {loc['point']} ({loc['confidence']*100:.1f}%)")
            else:
                if available_count > 1:
                    print(f"  {status} {tool_name.upper():<15} → {available_count} available ({total_count} total)")
                elif available_count == 1:
                    print(f"  {status} {tool_name.upper():<15} → 1 available ({total_count} total)")
                else:
                    print(f"  {status} {tool_name.upper():<15} → OUT OF STOCK ({total_count} total)")
        
        print("-" * 40)
        print(f"Total unique tools: {len(self.tool_mapping)}")
        print(f"Total tool instances: {total_tools}")
        print(f"Available for fetching: {available_tools}")
        print("=" * 70)
    
    def check_grab_script_exists(self, grab_letter):
        """Check if grab script exists for a point"""
        script_path = os.path.join(self.grab_scripts_dir, f"grab_point_{grab_letter}.py")
        return os.path.exists(script_path)
    
    def execute_grab_script(self, grab_letter):
        """Execute the grab script for a specific point"""
        script_path = os.path.join(self.grab_scripts_dir, f"grab_point_{grab_letter}.py")
        
        if not os.path.exists(script_path):
            print(f"ERROR: Grab script not found: {script_path}")
            return False
        
        print(f"\nExecuting grab script for Point {grab_letter}...")
        
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
            
            # Call the main function if it exists
            if hasattr(module, f"grab_point_{grab_letter}"):
                grab_func = getattr(module, f"grab_point_{grab_letter}")
                grab_func(self.arm)
            elif hasattr(module, "main"):
                module.main()
            else:
                print(f"Warning: No grab function found in {script_path}")
                # Try to execute the script directly
                exec(open(script_path).read(), {"arm": self.arm})
                
        except Exception as e:
            print(f"ERROR executing grab script: {e}")
            return False
        
        return True
    
    def fetch_tool(self, tool_name):
        """Fetch a specific tool by name with duplicate handling"""
        tool_name_lower = tool_name.lower().strip()
        
        print(f"\n" + "=" * 70)
        print(f"REQUEST: Fetch '{tool_name_lower.upper()}'")
        print("=" * 70)
        
        # Check if tool exists
        if tool_name_lower not in self.tool_mapping:
            print(f"❌ Tool '{tool_name_lower}' not found in workshop!")
            self.list_available_tools()
            return False
        
        # Get next available location
        location = self.get_next_available_location(tool_name_lower)
        
        if location is None:
            print(f"⛔ All {tool_name_lower.upper()} instances have been fetched!")
            print(f"   Total inventory: {len(self.tool_mapping[tool_name_lower])}")
            print(f"   Fetched: {len(self.tool_mapping[tool_name_lower])}")
            
            # Show all locations
            print("\nAll locations (all fetched):")
            for i, loc in enumerate(self.tool_mapping[tool_name_lower], 1):
                print(f"   {i}. Point {loc['point']} - {loc['confidence']*100:.1f}% confidence")
            
            return False
        
        grab_letter = location["point"]
        confidence = location["confidence"]
        position_desc = location["position_desc"]
        
        print(f"📍 Tool location: Point {grab_letter} ({position_desc})")
        print(f"📊 Confidence: {confidence*100:.1f}%")
        
        # Show inventory status
        total_count = len(self.tool_mapping[tool_name_lower])
        fetched_count = sum(1 for loc in self.tool_mapping[tool_name_lower] if loc.get("fetched", False))
        remaining = total_count - fetched_count - 1  # Minus the one we're about to fetch
        
        if total_count > 1:
            print(f"📦 Inventory: {remaining} more available after this fetch")
        
        # Check if grab script exists
        if not self.check_grab_script_exists(grab_letter):
            print(f"❌ No grab script found for Point {grab_letter}")
            print(f"   Please create movements/grab_point_{grab_letter}.py")
            return False
        
        # Confirm with user
        print(f"\nReady to fetch {tool_name_lower.upper()} from Point {grab_letter}")
        print("Starting in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        # Execute grab sequence
        print("\n🚀 Starting fetch sequence...")
        success = self.execute_grab_script(grab_letter)
        
        if success:
            # Mark as fetched
            self.save_fetched_location(tool_name_lower, grab_letter)
            
            print(f"\n✅ Successfully fetched {tool_name_lower.upper()}!")
            
            # Show remaining inventory
            if total_count > 1:
                new_remaining = total_count - (fetched_count + 1)
                if new_remaining > 0:
                    print(f"📦 {new_remaining} more {tool_name_lower.upper()} available in workshop")
                else:
                    print(f"📦 Last {tool_name_lower.upper()} fetched!")
            
            return True
        else:
            print(f"\n❌ Failed to fetch {tool_name_lower.upper()}")
            return False
    
    def reset_fetched_tools(self, tool_name=None):
        """Reset fetched status for tools (or specific tool)"""
        print("\n" + "=" * 70)
        print("RESET FETCHED TOOLS")
        print("=" * 70)
        
        if tool_name:
            # Reset specific tool
            if tool_name in self.tool_mapping:
                for location in self.tool_mapping[tool_name]:
                    location["fetched"] = False
                
                # Remove from fetched locations
                keys_to_remove = [k for k in self.fetched_locations.keys() if k.startswith(f"{tool_name}_")]
                for key in keys_to_remove:
                    del self.fetched_locations[key]
                
                print(f"Reset fetched status for {tool_name.upper()}")
                print(f"Removed {len(keys_to_remove)} fetched records")
            else:
                print(f"Tool {tool_name} not found")
        else:
            # Reset all tools
            for tool_name in self.tool_mapping:
                for location in self.tool_mapping[tool_name]:
                    location["fetched"] = False
            
            self.fetched_locations = {}
            print("Reset all fetched tools")
            print(f"Cleared {len(self.tool_mapping)} tools")
        
        # Save changes
        try:
            with open(self.fetched_tools_file, 'w') as f:
                json.dump(self.fetched_locations, f, indent=2)
            print("Saved reset to file")
        except Exception as e:
            print(f"Warning: Could not save reset: {e}")
    
    def interactive_mode(self):
        """Run in interactive mode with user input"""
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        
        # Show available tools
        self.list_available_tools()
        
        while True:
            print("\n" + "-" * 50)
            print("MAIN MENU:")
            print("  1. Fetch a tool")
            print("  2. List available tools (detailed)")
            print("  3. Reset fetched tools")
            print("  4. Test grab point")
            print("  5. Return to home position")
            print("  6. Exit")
            print("-" * 50)
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                # Fetch tool
                print("\nAvailable tools:")
                for tool_name in sorted(self.tool_mapping.keys()):
                    locations = self.tool_mapping[tool_name]
                    available_count = sum(1 for loc in locations if not loc.get("fetched", False))
                    total_count = len(locations)
                    
                    if available_count > 0:
                        print(f"  • {tool_name.upper():<15} ({available_count}/{total_count} available)")
                
                tool_name = input("\nWhich tool would you like? ").strip()
                if tool_name:
                    self.fetch_tool(tool_name)
            
            elif choice == "2":
                # List tools with details
                self.list_available_tools(show_details=True)
            
            elif choice == "3":
                # Reset fetched tools
                print("\nReset options:")
                print("  1. Reset all tools")
                print("  2. Reset specific tool")
                
                reset_choice = input("\nEnter choice (1-2): ").strip()
                
                if reset_choice == "1":
                    confirm = input("Are you sure you want to reset ALL fetched tools? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        self.reset_fetched_tools()
                elif reset_choice == "2":
                    print("\nAvailable tools to reset:")
                    for tool_name in sorted(self.tool_mapping.keys()):
                        print(f"  • {tool_name.upper()}")
                    
                    tool_name = input("\nWhich tool to reset? ").strip().lower()
                    if tool_name:
                        self.reset_fetched_tools(tool_name)
            
            elif choice == "4":
                # Test specific grab point
                print("\nTest grab point:")
                grab_letter = input("Enter grab point letter (A-I): ").strip().upper()
                if grab_letter and grab_letter in "ABCDEFGHI":
                    if self.check_grab_script_exists(grab_letter):
                        print(f"Testing Point {grab_letter}...")
                        self.execute_grab_script(grab_letter)
                    else:
                        print(f"No script for Point {grab_letter}")
                else:
                    print("Invalid grab point")
            
            elif choice == "5":
                # Return to home
                print("\nReturning to home position...")
                self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
                time.sleep(2)
                print("✅ Arm at home position")
            
            elif choice == "6":
                # Exit
                print("\nReturning arm to home position...")
                self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
                time.sleep(2)
                print("\n👋 Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
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
                if tool_name:
                    return self.fetch_tool(tool_name)
        
        # Handle other commands
        if "list" in command and "tool" in command:
            self.list_available_tools()
            return True
        
        elif "reset" in command and "tool" in command:
            parts = command.split()
            if len(parts) > 2:
                # Try to extract tool name
                tool_name = parts[-1]  # Last word might be tool name
                self.reset_fetched_tools(tool_name)
            else:
                self.reset_fetched_tools()
            return True
        
        elif "home" in command or "rest" in command:
            print("Returning to home position...")
            self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
            time.sleep(2)
            return True
        
        elif "inventory" in command or "status" in command:
            self.list_available_tools(show_details=True)
            return True
        
        else:
            print(f"Sorry, I don't understand: '{command}'")
            print("Try: 'get hammer', 'list tools', 'reset tools', or 'go home'")
            return False

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
        print("Returning arm to home position...")
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