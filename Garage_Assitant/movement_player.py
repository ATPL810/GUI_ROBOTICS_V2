#!/usr/bin/env python3
"""
Movement Player - Executes stored movement sequences from JSON
Complete test sequence: home -> pick up -> return -> drop
"""

import json
import time
from Arm_Lib import Arm_Device

class MovementPlayer:
    def __init__(self, movements_file="config/movements.json"):
        """Initialize arm and load movements"""
        print("Initializing Movement Player...")
        
        try:
            self.arm = Arm_Device()
            time.sleep(2)  # Wait for arm initialization
            print("Arm initialized")
            
            self.load_movements(movements_file)
            print(f"Loaded movements for {len(self.movements['grab_points'])} grab points")
            
        except Exception as e:
            print(f"Failed to initialize: {e}")
            raise
    
    def load_movements(self, filename):
        """Load movement sequences from JSON file"""
        with open(filename, 'r') as f:
            self.movements = json.load(f)
        print(f"Loaded movements from {filename}")
    
    def execute_step(self, step):
        """Execute a single movement step"""
        print(f"    Step {step['step']}: {step['name']}")
        
        # Handle gripper movement separately if specified
        if "gripper_angle" in step:
            gripper_angle = step["gripper_angle"]
            self.arm.Arm_serial_servo_write(6, gripper_angle, step["move_time"])
            print(f"      Gripper: {gripper_angle} degrees")
        
        # Handle full arm movement
        elif "servo_angles" in step:
            angles = step["servo_angles"]
            self.arm.Arm_serial_servo_write6(
                angles[0], angles[1], angles[2],
                angles[3], angles[4], angles[5],
                step["move_time"]
            )
            print(f"      Servos: {angles}")
        
        # Wait for stabilization
        stabilization = step.get("stabilization_time", 0.5)
        if stabilization > 0:
            time.sleep(stabilization)
    
    def execute_sequence(self, point_id, sequence_type="pickup"):
        """
        Execute a movement sequence for a specific grab point
        
        Args:
            point_id: Grab point ID (A, B, C, etc.)
            sequence_type: "pickup", "return", or "drop"
        """
        print(f"\nExecuting {sequence_type} sequence for Point {point_id}")
        
        # Check if point exists
        if point_id not in self.movements["grab_points"]:
            print(f"Point {point_id} not found in movements file")
            return False
        
        point_data = self.movements["grab_points"][point_id]
        
        # Select sequence type
        if sequence_type == "pickup":
            sequence = point_data["movement_sequence"]
        elif sequence_type == "return":
            sequence = point_data["return_to_home_sequence"]
        elif sequence_type == "drop":
            sequence = point_data.get("drop_sequence", [])
        else:
            print(f"Unknown sequence type: {sequence_type}")
            return False
        
        # Execute each step
        for step in sequence:
            self.execute_step(step)
        
        print(f"{sequence_type.capitalize()} sequence complete for Point {point_id}")
        return True
    
    def pick_tool_from_point(self, point_id):
        """Complete pickup operation: home -> pick up -> return home"""
        print(f"\nStarting complete pickup from Point {point_id}")
        print("=" * 50)
        
        try:
            # 1. Go to home position first
            print("\n1. Moving to home position...")
            self.go_home()
            
            # 2. Execute pickup sequence
            if not self.execute_sequence(point_id, "pickup"):
                return False
            
            # 3. Return to home with tool
            print("\n3. Returning to home with tool...")
            self.execute_sequence(point_id, "return")
            
            print(f"\nSuccessfully picked tool from Point {point_id}")
            return True
            
        except Exception as e:
            print(f"Pickup failed: {e}")
            return False
    
    def deliver_tool(self, point_id=None):
        """Deliver tool to drop zone"""
        print(f"\nDelivering tool to drop zone")
        print("=" * 50)
        
        try:
            # Execute drop sequence (if point-specific)
            if point_id and point_id in self.movements["grab_points"]:
                self.execute_sequence(point_id, "drop")
            else:
                # Default drop sequence
                print("\nUsing default drop sequence...")
                self.arm.Arm_serial_servo_write6(180, 90, 90, 90, 90, 135, 2000)
                time.sleep(2)
                self.arm.Arm_serial_servo_write(6, 90, 1000)  # Open gripper
                time.sleep(1)
            
            print("Tool delivered to drop zone")
            return True
            
        except Exception as e:
            print(f"Delivery failed: {e}")
            return False
    
    def go_home(self):
        """Return to initial/home position"""
        print("Returning to home position...")
        
        home_angles = self.movements["metadata"]["initial_position"]
        self.arm.Arm_serial_servo_write6(
            home_angles[0], home_angles[1], home_angles[2],
            home_angles[3], home_angles[4], home_angles[5],
            2000
        )
        time.sleep(2)
        print("At home position")
    
    def test_complete_sequence(self, point_id):
        """
        Test complete sequence: home -> pick up -> return -> drop
        This simulates the entire process from start to finish
        """
        print(f"\nTESTING COMPLETE SEQUENCE FOR POINT {point_id}")
        print("=" * 60)
        print("Sequence: HOME -> PICK UP -> RETURN HOME -> DROP")
        print("=" * 60)
        
        try:
            # 1. Start at home position
            print("\n[PHASE 1] Starting from home position")
            self.go_home()
            time.sleep(1)
            
            # 2. Pick up tool from specified point
            print("\n[PHASE 2] Picking up tool from Point", point_id)
            print("-" * 40)
            self.execute_sequence(point_id, "pickup")
            time.sleep(1)
            
            # 3. Return to home with tool
            print("\n[PHASE 3] Returning to home with tool")
            print("-" * 40)
            self.execute_sequence(point_id, "return")
            time.sleep(1)
            
            # 4. Deliver tool to drop zone
            print("\n[PHASE 4] Delivering tool to drop zone")
            print("-" * 40)
            self.deliver_tool(point_id)
            time.sleep(1)
            
            # 5. Return to home (empty)
            print("\n[PHASE 5] Returning to home position (empty)")
            print("-" * 40)
            self.go_home()
            
            print(f"\nCOMPLETE SEQUENCE TEST SUCCESSFUL FOR POINT {point_id}")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\nSequence test failed: {e}")
            print("Returning to home position for safety...")
            self.go_home()
            return False
    
    def test_point(self, point_id):
        """Test all sequences for a point"""
        print(f"\nTesting Point {point_id}")
        print("=" * 50)
        
        self.go_home()
        time.sleep(1)
        
        # Test pickup
        self.execute_sequence(point_id, "pickup")
        time.sleep(1)
        
        # Test return
        self.execute_sequence(point_id, "return")
        time.sleep(1)
        
        print(f"\nPoint {point_id} test complete")
    
    def list_available_points(self):
        """List all calibrated grab points"""
        print("\nAvailable Grab Points:")
        print("-" * 40)
        
        for point_id, data in self.movements["grab_points"].items():
            print(f"  Point {point_id}: {data['description']}")
            print(f"      Position: {data['position_name']}")
            print(f"      Image Coords: {data['image_coords']}")
            print(f"      Steps in pickup sequence: {len(data['movement_sequence'])}")
            print()


# ============================================
# MAIN TEST FUNCTION
# ============================================

def test_complete_workflow():
    """
    Test the complete workflow from start to finish
    You can run this with different point IDs as arguments
    """
    print("MOVEMENT PLAYER - COMPLETE WORKFLOW TEST")
    print("=" * 70)
    print("This will test: HOME -> PICK UP -> RETURN HOME -> DROP")
    print("=" * 70)
    
    try:
        # Create player
        player = MovementPlayer()
        
        # List available points
        player.list_available_points()
        
        # Get point to test from user or use default
        import sys
        if len(sys.argv) > 1:
            point_to_test = sys.argv[1].upper()
        else:
            point_to_test = "A"  # Default to Point A
        
        print(f"\nTesting complete workflow for Point {point_to_test}")
        print("Press Ctrl+C to stop at any time")
        print("-" * 50)
        
        # Ask for confirmation
        response = input(f"Test Point {point_to_test}? (y/n): ").lower()
        if response != 'y':
            print("Test cancelled")
            return
        
        # Run the complete sequence
        success = player.test_complete_sequence(point_to_test)
        
        if success:
            print("\nALL TESTS COMPLETED SUCCESSFULLY!")
        else:
            print("\nTEST FAILED - Check movements.json and arm position")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        print("Returning arm to home position...")
        player.go_home()
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # You can run this in multiple ways:
    # 1. Test complete workflow for Point A: python movement_player.py
    # 2. Test complete workflow for Point B: python movement_player.py B
    # 3. Test complete workflow for Point C: python movement_player.py C
    # etc.
    
    test_complete_workflow()