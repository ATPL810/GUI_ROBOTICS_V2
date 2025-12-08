import time

def grab_point_E(arm, tool_type="hammer"):
    """Grabbing point E - Multiple tools"""
    print(f"Starting Grab Point E - {tool_type}")
    
    # Define grip force for each tool
    grip_forces = {
        "hammer": 169,
        "screwdriver": 169,
        "yellow_wrench": 154,
        "red_spanner": 176
    }
    
    grip_force = grip_forces.get(tool_type, 169)  # default to hammer
    
    # move to pos Before grabbing
    time.sleep(3)
    arm.Arm_serial_servo_write6(31, 39, 48, 22, 101, 133, 1000)
    time.sleep(2)   
    arm.Arm_serial_servo_write6(31, 33, 48, 22, 102, 140, 1000)
    
    # grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write(6, grip_force, 1000)

    # move to pos After grabbing object while holding it
    time.sleep(2)
    arm.Arm_serial_servo_write6(31, 65, 71, 22, 102, 176, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, 176, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, 176, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 176, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
    print(f"Grab Point E completed for {tool_type}")