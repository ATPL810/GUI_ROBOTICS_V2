import time

def grab_point_C(arm):
    """Grabbing point C - Bolt"""
    print("Starting Grab Point C - Bolt")
    
    # move to pos Before grabbing
    time.sleep(3)
    arm.Arm_serial_servo_write6(91, 47, 6, 85, 89, 91, 1000)
    time.sleep(2)   
    arm.Arm_serial_servo_write6(91, 45, 6, 85, 89, 119, 1000)
    
    # grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 176, 1000)

    # move to pos After grabbing object while holding it
    time.sleep(2)
    arm.Arm_serial_servo_write6(103, 54, 40, 34, 90, 177, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, 178, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, 178, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
    print("Grab Point C completed")