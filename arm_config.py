"""
Yahboom Arm Configuration File
Edit this based on your specific Yahboom kit
"""

# ============================================
# ARM TYPE CONFIGURATION
# ============================================
# Uncomment ONE of these based on your Yahboom kit:

# 1. For DOFBOT (6-DOF Robotic Arm Kit)
# ARM_TYPE = "DOFBOT"
# SERVO_COUNT = 6
# LIBRARY_MODULE = "Arm_Lib"

# 2. For Alphabot (with 6-DOF arm)
# ARM_TYPE = "ALPHABOT"
# SERVO_COUNT = 6
# LIBRARY_MODULE = "Arm_Lib"

# 3. For Robotic Arm Pi (Raspberry Pi version)
ARM_TYPE = "ROBOTIC_ARM_PI"
SERVO_COUNT = 6
LIBRARY_MODULE = "Arm_Lib.Arm_Device"

# 4. For AI Robotic Arm Kit
# ARM_TYPE = "AI_ROBOTIC_ARM"
# SERVO_COUNT = 6
# LIBRARY_MODULE = "yahboom"

# ============================================
# SERVO CONFIGURATION
# ============================================
# Servo IDs (1-6 for 6-DOF arm)
SERVO_BASE = 1        # Servo 1: Base rotation
SERVO_SHOULDER = 2    # Servo 2: Shoulder
SERVO_ELBOW = 3       # Servo 3: Elbow
SERVO_WRIST = 4       # Servo 4: Wrist pitch
SERVO_GRIPPER = 5     # Servo 5: Gripper
SERVO_WRIST_ROT = 6   # Servo 6: Wrist rotation

# Servo angle limits (degrees)
SERVO_LIMITS = {
    SERVO_BASE: (0, 180),
    SERVO_SHOULDER: (20, 160),
    SERVO_ELBOW: (30, 150),
    SERVO_WRIST: (30, 150),
    SERVO_GRIPPER: (10, 80),   # Adjust based on gripper
    SERVO_WRIST_ROT: (0, 180)
}

# Gripper positions
GRIPPER_OPEN = 70     # Angle for open gripper
GRIPPER_CLOSED = 30   # Angle for closed gripper

# Movement speed (ms)
MOVE_SPEED = 1000
FAST_MOVE_SPEED = 500
SLOW_MOVE_SPEED = 1500

# ============================================
# CAMERA CALIBRATION (Adjust these!)
# ============================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_INDEX = 0  # Usually 0, try 1, 2 if camera not found

# Workspace boundaries in meters (CALIBRATE THESE!)
WORKSPACE_X = (-0.25, 0.25)   # Left/Right
WORKSPACE_Y = (0.15, 0.45)    # Forward/Back
WORKSPACE_Z = (-0.05, 0.10)   # Up/Down

# ============================================
# DISPLACEMENT SETTINGS
# ============================================
DROP_ZONE_X = 0.20     # X position for drop zone
DROP_ZONE_Y = 0.30     # Y position for drop zone
DROP_ZONE_Z = 0.03     # Z position for drop zone
DROP_SPACING = 0.05    # Spacing between tools in drop zone

SAFE_HEIGHT = 0.15     # Safe Z for moving above objects
APPROACH_HEIGHT_OFFSET = 0.08  # Height above object for approach