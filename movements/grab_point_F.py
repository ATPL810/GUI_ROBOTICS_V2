import time

def grab_point_F(arm):
    """Grabbing point F - Bolt"""
    print("Starting Grab Point F - Bolt")
    
    # move to pos Before grabbing
    time.sleep(3)
    arm.Arm_serial_servo_write6(38, 33, 53, 35, 90, 142, 1000)
    time.sleep(2)   
    arm.Arm_serial_servo_write6(38, 25, 54, 44, 90, 143, 1000)
    
    # grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 177, 1000)

    # move to pos After grabbing object while holding it
    time.sleep(2)
    arm.Arm_serial_servo_write6(37, 36, 71, 44, 80, 177, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, 177, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, 177, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 130, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
    print("Grab Point F completed")