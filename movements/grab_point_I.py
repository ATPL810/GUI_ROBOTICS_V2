import time

def grab_point_I(arm):
    """Grabbing point I - Hammer"""
    print("Starting Grab Point I - Hammer")
    
    # move to pos Before grabbing
    time.sleep(3)
    arm.Arm_serial_servo_write6(13, 50, 37, 37, 93, 87, 1000)
    time.sleep(2)   
    arm.Arm_serial_servo_write6(8, 3, 86, 34, 123, 86, 1000)
    
    # grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 167, 1000)

    # move to pos After grabbing object while holding it
    time.sleep(2)
    arm.Arm_serial_servo_write6(9, 26, 98, 35, 123, 167, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, 163, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(130, 40, 55, 45, 90, 163, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(130, 30, 55, 45, 90, 100, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 80, 1000)
    print("Grab Point I completed")