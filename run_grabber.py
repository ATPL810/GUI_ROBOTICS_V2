#!/usr/bin/env python3
"""
Simple CLI to grab tools based on master report
Usage: python run_grabber.py [--list] [--grab TOOL_NAME]
"""

import argparse
import sys
from tool_inventory import ToolInventory
from grab_executor import GrabExecutor

# Import YOUR existing arm controller
try:
    # Adjust this import based on your actual arm module
    from arm_controller import ArmController
    arm = ArmController()
    print("✓ Arm controller initialized")
except ImportError:
    print("⚠️  Using mock arm controller for testing")
    class MockArm:
        def Arm_serial_servo_write6(self, *args):
            print(f"   [MOCK] Arm move to: {args[:6]} in {args[6]}ms")
        def Arm_serial_servo_write(self, servo, angle, time_ms):
            print(f"   [MOCK] Servo {servo} → {angle}° in {time_ms}ms")
    arm = MockArm()

def main():
    parser = argparse.ArgumentParser(
        description="Grab tools based on master report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_grabber.py --list          # Show available tools
  python run_grabber.py --grab hammer   # Grab a hammer
  python run_grabber.py --grab bolt     # Grab a bolt
  python run_grabber.py --scan          # Update inventory (calls your camera code)
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List all available tools from master report')
    parser.add_argument('--grab', type=str, metavar='TOOL_NAME',
                       help='Grab a specific tool (e.g., hammer, bolt)')
    parser.add_argument('--scan', action='store_true',
                       help='Scan all points and update master report')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Initialize inventory
    inventory = ToolInventory()
    
    # Handle --scan (call your existing camera code)
    if args.scan:
        print("\n🔍 Scanning all points...")
        print("NOTE: This should call your existing camera analysis code")
        
        # TODO: Call your existing scan_all_points() function here
        # For now, we'll show what would happen
        print("Would call: camera.take_snapshots(['A','B','C','D','E','F','G','H','I'])")
        print("Would call: camera.analyze_tools()")
        
        # Simulated update (replace with actual results from your camera)
        simulated_results = {
            'A': ['bolt'],
            'B': ['wrench_red', 'hammer'],
            'C': ['bolt'],
            'D': ['wrench_yellow'],
            'E': ['hammer'],
            'F': ['bolt'],
            'G': ['pliers'],
            'H': ['measuring_tape'],
            'I': ['hammer']
        }
        
        for point, tools in simulated_results.items():
            if tools:
                inventory.update_point(point, tools)
        
        print("\n✅ Inventory updated from scan")
        inventory.print_inventory()
    
    # Handle --list
    if args.list:
        inventory.print_inventory()
    
    # Handle --grab TOOL_NAME
    if args.grab:
        tool_name = args.grab.lower().strip()
        
        # Check if tool exists
        if not inventory.is_tool_available(tool_name):
            print(f"\n❌ Tool '{tool_name}' not found in inventory!")
            print("Available tools:")
            all_tools = inventory.get_all_tools()
            for tool, point in all_tools.items():
                print(f"  • {tool} (Point {point})")
            return
        
        # Get tool location
        point = inventory.get_tool_location(tool_name)
        print(f"\n🎯 Found '{tool_name}' at Point {point}")
        
        # Confirm with user
        confirm = input(f"Grab {tool_name} from Point {point}? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return
        
        # Execute grab sequence
        success = GrabExecutor.execute_grab(arm, point, tool_name)
        
        if success:
            # Remove from inventory
            inventory.remove_tool(tool_name)
            print(f"\n✅ Success! '{tool_name}' has been grabbed and delivered to drop zone")
        else:
            print(f"\n❌ Failed to grab '{tool_name}'")

if __name__ == "__main__":
    main()