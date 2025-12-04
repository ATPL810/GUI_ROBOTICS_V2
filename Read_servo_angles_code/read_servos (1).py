import time
from Arm_Lib import Arm_Device

# Engage torque on all servos
# Arm.Arm_serial_set_torque(1)

# Disengage torque on all servos
Arm = Arm_Device()
Arm.Arm_serial_set_torque(0)

# Read and print the current position of all servos
# While key press isnt q
while True:

    for i in range(6):
        aa = Arm.Arm_serial_servo_read(i+1)
        # print(i + "" + aa)
        print(f"Servo {i+1} Angle: {aa}")
        time.sleep(1)

    key = input("Press 'q' to quit or Enter to read servo angles: ")
    if key.lower() == 'q':
        break
    