import time

def grab_point_H(arm):
    """Grabbing point H - Measuring Tape"""
    print("Starting Grab Point H - Measuring Tape")
    
    # move to pos Before grabbing
    time.sleep(3)
    arm.Arm_serial_servo_write6(1, 105, 45, -2, 89, 20, 1000)
    time.sleep(2)   
    arm.Arm_serial_servo_write6(-10, 71, 39, 2, 88, 20, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(-10, 67, 39, 2, 10, 20, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(-10, 38, 49, 16, 10, 20, 1000)
    # grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 159, 1000)

    # move to pos After grabbing object while holding it
    time.sleep(2)
    arm.Arm_serial_servo_write6(5, 72, 59, 18, 10, 159, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(40, 90, 55, 60, 90, 160, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, 160, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 160, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(130,30,55,45,90,36, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 105, 45, -35, 90, 90, 1000)
    print("Grab Point H completed")