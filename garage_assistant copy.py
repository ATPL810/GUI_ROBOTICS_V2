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

class GarageAssistant:
    def __init__(self, master_report_path=None):
        """Initialize the garage assistant"""
        print("=" * 70)
        print("           GARAGE ASSISTANT - TOOL FETCHER v2.0")
        print("=" * 70)
        
        # Initialize robot arm
        print("\nInitializing robot arm...")
        self.arm = self.initialize_arm()
        
        # Find master report
        self.master_report_path = self.find_master_report(master_report_path)
        print(f"Using tool mapping from: {self.master_report_path}")
        
        # Parse COMPLETE tool mapping from master report
        self.tool_mapping, self.tool_locations = self.parse_master_report()
        
        # Available grab scripts
        self.grab_scripts_dir = "movements"
        
        print("\n✅ System initialized successfully!")
        print(f"✅ Found {len(self.tool_mapping)} available tools")
    
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
    
    def list_available_tools(self):
        """List all available tools"""
        print("\n" + "=" * 70)
        print("AVAILABLE TOOLS IN WORKSHOP:")
        print("-" * 40)
        
        if not self.tool_mapping:
            print("No tools found in mapping!")
            return
        
        # List all tools alphabetically with their grab points
        for tool_name in sorted(self.tool_mapping.keys()):
            grab_letter = self.tool_mapping[tool_name]
            print(f"  • {tool_name.upper():<15} (at Point {grab_letter})")
        
        print("-" * 40)
        print(f"Total tools: {len(self.tool_mapping)}")
        
        # Show grab point layout
        print("\nWORKSHOP LAYOUT:")
        print("-" * 40)
        layout = {
            "Initial Position (Base 90°)": ["A", "B", "C", "D"],
            "Second Position (Base 40°)": ["E", "F", "G"],
            "Third Position (Base 1°)": ["H", "I"]
        }
        
        for position_name, points in layout.items():
            print(f"\n{position_name}:")
            for point in points:
                if point in self.tool_locations:
                    tool_name = self.tool_locations[point]["tool_name"].upper()
                    print(f"  Point {point}: {tool_name}")
                else:
                    print(f"  Point {point}: Empty")
        
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
        """Fetch a specific tool by name"""
        tool_name_lower = tool_name.lower().strip()
        
        print(f"\n" + "=" * 70)
        print(f"REQUEST: Fetch '{tool_name_lower.upper()}'")
        print("=" * 70)
        
        # Check if tool exists
        if tool_name_lower not in self.tool_mapping:
            print(f"❌ Tool '{tool_name}' not found in workshop!")
            print("\nAvailable tools are:")
            for available_tool in sorted(self.tool_mapping.keys()):
                print(f"  • {available_tool.upper()}")
            return False
        
        # Get grab point for this tool
        grab_letter = self.tool_mapping[tool_name_lower]
        print(f"📍 Tool location: Point {grab_letter}")
        
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
            print(f"\n✅ Successfully fetched {tool_name_lower.upper()}!")
            
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
        print("INTERACTIVE MODE")
        print("=" * 70)
        
        # Show available tools
        self.list_available_tools()
        
        while True:
            print("\n" + "-" * 40)
            print("GARAGE ASSISTANT MENU:")
            print("  1. 🔧 Fetch a tool")
            print("  2. 📋 List available tools")
            print("  3. 🧪 Test grab point (dry run)")
            print("  4. 🏠 Return to home position")
            print("  5. ❌ Exit")
            print("-" * 40)
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                # Fetch tool
                print("\nAvailable tools:")
                for tool_name in sorted(self.tool_mapping.keys()):
                    grab_letter = self.tool_mapping[tool_name]
                    print(f"  • {tool_name.upper():<15} (Point {grab_letter})")
                
                tool_name = input("\nWhich tool would you like? ").strip()
                if tool_name:
                    self.fetch_tool(tool_name)
            
            elif choice == "2":
                # List tools
                self.list_available_tools()
            
            elif choice == "3":
                # Test grab point (dry run)
                self.test_grab_point_dry_run()
            
            elif choice == "4":
                # Return to home
                print("\n🏠 Returning to home position...")
                self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
                time.sleep(2)
                print("✅ Arm at home position")
            
            elif choice == "5":
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