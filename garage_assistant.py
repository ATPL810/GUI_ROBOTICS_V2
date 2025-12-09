"""
GARAGE ASSISTANT TOOL FETCHER
Uses master_report.txt to locate tools and runs corresponding grab scripts
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
        print("           GARAGE ASSISTANT - TOOL FETCHER")
        print("=" * 70)
        
        # Initialize robot arm
        print("\nInitializing robot arm...")
        self.arm = self.initialize_arm()
        
        # Find master report
        self.master_report_path = self.find_master_report(master_report_path)
        print(f"Using tool mapping from: {self.master_report_path}")
        
        # Parse tool mapping from master report
        self.tool_mapping = self.parse_master_report()
        
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
        """Parse tool mapping from master report"""
        print("\nParsing tool mapping...")
        
        tool_mapping = {}
        
        with open(self.master_report_path, 'r') as f:
            content = f.read()
            
            # Look for "TOOL TO GRAB POINT MAPPING" section
            if "TOOL TO GRAB POINT MAPPING:" in content:
                # Extract the mapping section
                mapping_section = content.split("TOOL TO GRAB POINT MAPPING:")[1]
                mapping_section = mapping_section.split("=")[0]  # Get up to next section
                
                # Parse each tool line
                lines = mapping_section.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if "→" in line:
                        # Extract tool name and grab point
                        parts = line.split("→")
                        if len(parts) == 2:
                            tool_name = parts[0].strip().lower()  # Lowercase for matching
                            grab_point = parts[1].strip()
                            
                            # Extract just the letter (e.g., "Point A" -> "A")
                            match = re.search(r'Point\s+([A-I])', grab_point)
                            if match:
                                grab_letter = match.group(1)
                                tool_mapping[tool_name] = grab_letter
                                print(f"   {tool_name.upper():<15} → Point {grab_letter}")
        
        if not tool_mapping:
            # Try alternative parsing from ROBOT ACTION PLAN section
            if "ROBOT ACTION PLAN:" in content:
                action_section = content.split("ROBOT ACTION PLAN:")[1]
                lines = action_section.strip().split('\n')
                
                for line in lines:
                    if "When user requests" in line:
                        # Extract tool name and point from pattern
                        match = re.search(r"requests '([^']+)'.*Point ([A-I])", line)
                        if match:
                            tool_name = match.group(1).lower()
                            grab_letter = match.group(2)
                            tool_mapping[tool_name] = grab_letter
                            print(f"   {tool_name.upper():<15} → Point {grab_letter}")
        
        return tool_mapping
    
    def list_available_tools(self):
        """List all available tools"""
        print("\n" + "=" * 70)
        print("AVAILABLE TOOLS IN WORKSHOP:")
        print("-" * 40)
        
        if not self.tool_mapping:
            print("No tools found in mapping!")
            return
        
        # Group by tool type
        tools_by_letter = {}
        for tool_name, grab_letter in self.tool_mapping.items():
            tools_by_letter[grab_letter] = tool_name
        
        # List all tools alphabetically
        for tool_name in sorted(self.tool_mapping.keys()):
            grab_letter = self.tool_mapping[tool_name]
            print(f"  • {tool_name.upper():<15} (at Point {grab_letter})")
        
        print("-" * 40)
        print(f"Total tools: {len(self.tool_mapping)}")
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
            else:
                print(f"Warning: No function grab_point_{grab_letter} found in script")
                
        except Exception as e:
            print(f"ERROR executing grab script: {e}")
            return False
        
        return True
    
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
            print(f"\n✅ Successfully fetched {tool_name_lower.upper()}!")
            return True
        else:
            print(f"\n❌ Failed to fetch {tool_name_lower.upper()}")
            return False
    
    def interactive_mode(self):
        """Run in interactive mode with user input"""
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        
        # Show available tools
        self.list_available_tools()
        
        while True:
            print("\n" + "-" * 40)
            print("OPTIONS:")
            print("  1. Fetch a tool")
            print("  2. List available tools")
            print("  3. Test all grab points")
            print("  4. Return to home position")
            print("  5. Exit")
            print("-" * 40)
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                # Fetch tool
                print("\nAvailable tools:")
                for tool_name in sorted(self.tool_mapping.keys()):
                    print(f"  • {tool_name.upper()}")
                
                tool_name = input("\nWhich tool would you like? ").strip()
                if tool_name:
                    self.fetch_tool(tool_name)
            
            elif choice == "2":
                # List tools
                self.list_available_tools()
            
            elif choice == "3":
                # Test all grab points
                self.test_all_grab_points()
            
            elif choice == "4":
                # Return to home
                print("\nReturning to home position...")
                self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
                time.sleep(2)
                print("✅ Arm at home position")
            
            elif choice == "5":
                # Exit
                print("\nReturning arm to home position...")
                self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
                time.sleep(2)
                print("\n👋 Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    def test_all_grab_points(self):
        """Test all grab points (optional)"""
        print("\n" + "=" * 70)
        print("TESTING ALL GRAB POINTS")
        print("=" * 70)
        
        # Get all unique grab points from mapping
        grab_points = set(self.tool_mapping.values())
        
        print(f"Found {len(grab_points)} grab points to test:")
        
        for grab_letter in sorted(grab_points):
            print(f"\nTesting Point {grab_letter}...")
            
            if self.check_grab_script_exists(grab_letter):
                # Execute without actually gripping (simulated)
                print(f"  ✓ Script exists: movements/grab_point_{grab_letter}.py")
                # You could do a dry run here if needed
            else:
                print(f"  ✗ Script missing: movements/grab_point_{grab_letter}.py")
        
        print("\n✅ Test complete")
    
    def voice_command_mode(self, command):
        """Handle voice command (for integration with voice assistant)"""
        # Normalize command
        command = command.lower().strip()
        
        # Check for fetch commands
        fetch_keywords = ["get", "fetch", "bring", "hand me", "give me", "pass"]
        
        for keyword in fetch_keywords:
            if keyword in command:
                # Extract tool name
                tool_name = command.replace(keyword, "").strip()
                if tool_name:
                    return self.fetch_tool(tool_name)
        
        # Handle other commands
        if "list tools" in command or "what tools" in command:
            self.list_available_tools()
            return True
        
        elif "home" in command or "rest" in command:
            print("Returning to home position...")
            self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
            time.sleep(2)
            return True
        
        else:
            print(f"Sorry, I don't understand: '{command}'")
            print("Try: 'get hammer', 'list tools', or 'go home'")
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