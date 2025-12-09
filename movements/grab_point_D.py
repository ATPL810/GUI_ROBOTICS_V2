import time

def grab_point_D(arm, tool_type="yellow_wrench"):
    """Grabbing point D - Multiple tools"""
    print(f"Starting Grab Point D - {tool_type}")
    
    # Define grip force for each tool
    grip_forces = {
        "yellow_wrench": 153,
        "screwdriver": 165,
        "hammer": 169,
        "spanner": 176
    }
    
    grip_force = grip_forces.get(tool_type, 153)  # default to yellow_wrench
    
    # move to pos Before grabbing
    time.sleep(3)
    arm.Arm_serial_servo_write6(105, 40, 25, 50, 89, 111, 1000)
    time.sleep(2)   
    arm.Arm_serial_servo_write6(105, 36, 27, 53, 90, 123, 1000)
    
    # grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write(6, grip_force, 1000)

    # move to pos After grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write6(103, 54, 40, 34, 90, grip_force, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, grip_force, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, grip_force, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
    print(f"Grab Point D completed for {tool_type}")