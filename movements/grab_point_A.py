import time

def grab_point_A(arm):
    """Grabbing point A - BOLTS"""
    print("Starting Grab Point A - BOLTS")
    
    # move to pos Before grabbing
    time.sleep(3)
    arm.Arm_serial_servo_write6(72, 33, 31, 59, 110, 89, 1000)
    time.sleep(2)   
    arm.Arm_serial_servo_write6(72, 35, 31, 59, 85, 112, 1000)
    
    # grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 178, 1000)

    # move to pos After grabbing object while holding it
    time.sleep(2)
    arm.Arm_serial_servo_write6(69, 58, 39, 59, 118, 178, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, 178, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, 179, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 125, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
    print("Grab Point A completed")