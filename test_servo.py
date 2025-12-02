#!/usr/bin/env python3
"""
Quick test script for Yahboom arm
"""

import time

try:
    import Arm_Lib
    from Arm_Lib import Arm_Device
    print("✅ Arm_Lib imported successfully")
    
    # Initialize arm
    arm = Arm_Device()
    time.sleep(2)
    
    print("Testing servo movements...")
    
    # Test each servo
    for servo in range(3, 5):
        print(f"Moving servo {servo} to 90 degrees...")
        arm.Arm_serial_servo_write(servo, 35, 1000)
        time.sleep(1)
    
    print("✅ Arm test complete!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("Check:")
    print("1. Arm is powered ON")
    print("2. USB cable is connected")
    print("3. Arm_Lib is installed")