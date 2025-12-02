# Quick I2C test script
import smbus
import time

# Initialize I2C
bus = smbus.SMBus(1)  # Bus 1
address = 0x15        # Yahboom address

# Open gripper (servo 5 to 180°)
data = [0x55, 0x55, 5, 1000 & 0xFF, (1000 >> 8) & 0xFF, 2500 & 0xFF, (2500 >> 8) & 0xFF]
bus.write_i2c_block_data(address, 0, data)
time.sleep(1)

# Close gripper (servo 5 to 30°)
data = [0x55, 0x55, 5, 1000 & 0xFF, (1000 >> 8) & 0xFF, 500 & 0xFF, (500 >> 8) & 0xFF]
bus.write_i2c_block_data(address, 0, data)
time.sleep(1)