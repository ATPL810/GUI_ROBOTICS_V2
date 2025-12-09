import time

def grab_point_G(arm):
    """Grabbing point G - Pliers"""
    print("Starting Grab Point G - Pliers")
    
    # move to pos Before grabbing
    time.sleep(3)
    arm.Arm_serial_servo_write6(50, 17, 93, 1, 90, 59, 1000)
    time.sleep(2)   
    arm.Arm_serial_servo_write6(49, 2, 93, 16, 89, 59, 1000)
    
    # grabbing
    time.sleep(2)
    arm.Arm_serial_servo_write(6, 125, 1000)

    # move to pos After grabbing object while holding it
    time.sleep(2)
    arm.Arm_serial_servo_write6(50, 30, 107, 18, 89, 125, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(120, 90, 55, 60, 90, 125, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(150, 40, 55, 45, 90, 126, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(150, 30, 55, 45, 90, 125, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(150,30,55,45,90,36, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 20, 1000)
    print("Grab Point G completed")