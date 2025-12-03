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

    arm.Arm_serial_servo_write(1, 90, 1000)
    arm.Arm_serial_servo_write(2, 115, 1000)
    arm.Arm_serial_servo_write(3, 5, 1000)
    arm.Arm_serial_servo_write(4, 15, 1000)
    arm.Arm_serial_servo_write(5, 90, 1000)



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