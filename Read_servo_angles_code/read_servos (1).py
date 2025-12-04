import time
from Arm_Lib import Arm

# Engage torque on all servos
Arm.Arm_serial_set_torque(1)

# Disengage torque on all servos
Arm.Arm_serial_set_torque(0)

# Read and print the current position of all servos
for i in range(6):
    aa = Arm.Arm_serial_servo_read(i+1)
    print(aa)
    time.sleep(1)