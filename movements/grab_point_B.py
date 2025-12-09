import time

def grab_point_B(arm, tool_type="wrench"):
    """Grabbing point B - Multiple tools"""
    print(f"Starting Grab Point B - {tool_type}")
    
    # Define grip force for each tool
    grip_forces = {
        "wrench": 176,
        "hammer": 170,
        "screwdriver": 169
        
    }
    
    grip_force = grip_forces.get(tool_type, 176)  # default to wrench
    
    # move to pos Before grabbing
    time.sleep(3)
    arm.Arm_serial_servo_write6(81, 33, 54, 17, 76, 139, 1000)
    time.sleep(2)   
    arm.Arm_serial_servo_write6(81, 29, 55, 18, 76, 150, 1000)
    
    # grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write(6, grip_force, 1000)

    # move to pos After grabbing object while holding it
    time.sleep(2)
    arm.Arm_serial_servo_write6(81, 29, 54, 18, 76, 177, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, 178, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, 177, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
    print(f"Grab Point B completed for {tool_type}")