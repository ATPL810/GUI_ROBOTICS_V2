#!/usr/bin/env python3
"""
Quick test script for Yahboom arm
"""

import time

try:
    # import Arm_Lib
    from Arm_Lib import Arm_Device
    print("✅ Arm_Lib imported successfully")
    
    # Initialize arm
    arm = Arm_Device()
    time.sleep(2)
    
    print("Testing servo movements...")
    
    # Test each servo
    # for servo in range(2, 3):
    #     print(f"Moving servo {servo} to 90 degrees...")
    #     arm.Arm_serial_servo_write(servo, 135, 1000)
    #     time.sleep(1)
    
    print("✅ Arm test complete!")
#  INiitial positions
    arm.Arm_serial_servo_write(1,1, 1000)
    arm.Arm_serial_servo_write(2, 105, 1000)
    arm.Arm_serial_servo_write(3, 45, 1000)
    arm.Arm_serial_servo_write(4, -35, 1000)
    arm.Arm_serial_servo_write(5, 90, 1000)

    # # move to pos Before grabbing
    # time.sleep(2)   
    # arm.Arm_serial_servo_write6(52,35,49,45,89,125, 1000)
    
    # #  grabbing

    # time.sleep(2)
    # arm.Arm_serial_servo_write(6, 135, 1000)

    # # arm.Arm_serial_servo_write(6, 145, 1000)

    # # move to pos After grabbing
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(60,45,50,90,90,135, 1000)
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(100,90,55,60,90,135, 1000)
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(130,40,55,45,90,135, 1000)
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(130,30,55,45,90,125, 1000)

    # time.sleep(2)
    # arm.Arm_serial_servo_write6(90,90,90,90,90,90, 1000)


# position for angle 0

    # # move to pos Before grabbing
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(0,67,17,41,117,57, 1000)
    # time.sleep(2)   
    # arm.Arm_serial_servo_write6(0,60,18,47,117,138, 1000)
    
    # #  grabbing

    # time.sleep(2)
    # arm.Arm_serial_servo_write(6, 169, 1000)

    # # arm.Arm_serial_servo_write(6, 145, 1000)

    # # move to pos After grabbing
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(0,45,50,90,90,169, 1000)
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(60,90,55,60,90,169, 1000)
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(130,40,55,45,90,169,1000)
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(130,30,55,45,90,125, 1000)

    # time.sleep(2)
    # arm.Arm_serial_servo_write6(90,90,90,90,90,90, 1000)

    # Grabbing point I - 

    # move to pos Before grabbing - hammer(167), 
    # time.sleep(3)
    # arm.Arm_serial_servo_write6(13,50,37,37,93,87, 1000)
    # time.sleep(2)   
    # arm.Arm_serial_servo_write6(8,3,86,34,123,86, 1000)
    
    # #  grabbing

    # time.sleep(2)
    # arm.Arm_serial_servo_write(6, 167, 1000)

    # # arm.Arm_serial_servo_write(6, 145, 1000)

    # # move to pos After grabbing object while hlding it
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(9,26,98,35,123,167, 1000)
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(120,90,55,60,90,163, 1000)
    # time.sleep(2)
    # # tightening
    # arm.Arm_serial_servo_write6(130,40,55,45,90,163,1000)
    # time.sleep(2)
    # arm.Arm_serial_servo_write6(130,30,55,45,90,100, 1000)

    # time.sleep(2)
    # arm.Arm_serial_servo_write6(90,90,90,90,90,80, 1000)
# ------------------------------------------------------- Measuring tape
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
    arm.Arm_serial_servo_write(6, 156, 1000)

    # move to pos After grabbing object while holding it
    time.sleep(2)
    arm.Arm_serial_servo_write6(5, 72, 59, 18, 10, 157, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(40, 90, 55, 60, 90, 157, 1000)
    time.sleep(2)
    
    # tightening
    arm.Arm_serial_servo_write6(150, 40, 55, 45, 90, 157, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(150, 30, 55, 45, 90, 155, 1000)
    time.sleep(2)
    arm.Arm_serial_servo_write6(150,30,55,45,90,36, 1000)

    time.sleep(2)
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 20, 1000)
    print("Grab Point G completed")









    # LOgging all the current servo positions
    positions = []
    for servo in range(1, 7):
        pos = arm.Arm_serial_servo_read(servo)
        positions.append((servo, pos))
    print("Current servo positions:", positions)
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("Check:")
    print("1. Arm is powered ON")
    print("2. USB cable is connected")
    print("3. Arm_Lib is installed")