import time

def grab_point_H(arm):
    """Grabbing point H - Measuring Tape"""
    print("Starting Grab Point H - Measuring Tape")
    
    # move to pos Before grabbing
    time.sleep(3)
    arm.Arm_serial_servo_write6(0, 23, 90, 2, 40, 62, 1000)
    time.sleep(2)   
    arm.Arm_serial_servo_write6(-15, 19, 90, 6, 40, 45, 1000)
    
    # grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 163, 1000)

    # move to pos After grabbing object while holding it
    time.sleep(2)
    arm.Arm_serial_servo_write6(50, 30, 107, 18, 89, 163, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, 163, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, 163, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 100, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 80, 1000)
    print("Grab Point H completed")