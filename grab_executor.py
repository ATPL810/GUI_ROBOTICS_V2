import time
import importlib.util
import sys

class GrabExecutor:
    # Grip forces from your movement.txt notes
    GRIP_FORCES = {
        'bolt': 176,
        'wrench_red': 176,
        'hammer': 169,
        'screwdriver': 169,
        'wrench': 153,  # yellow wrench
        'wrench_yellow': 153,
        'pliers': 125,
        'measuring_tape': 163,
        'spanner': 176,
        'hammer_i': 167  # hammer at point I
    }
    
    # Map tool names to script parameters
    TOOL_TO_SCRIPT_PARAM = {
        'wrench_red': 'wrench_red',
        'wrench_yellow': 'yellow_wrench',
        'hammer': 'hammer',
        'screwdriver': 'screwdriver',
        'spanner': 'spanner',
        'pliers': 'pliers',
        'measuring_tape': 'measuring_tape',
        'bolt': 'bolt'
    }
    
    @staticmethod
    def execute_grab(arm, point, tool_name):
        """Execute the correct grab script for point and tool"""
        print(f"\n🤖 EXECUTING GRAB SEQUENCE")
        print(f"   Tool: {tool_name}")
        print(f"   Point: {point}")
        
        # Get appropriate grip force
        grip_force = GrabExecutor.GRIP_FORCES.get(tool_name, 150)
        print(f"   Grip force: {grip_force}")
        
        # Construct script filename
        script_name = f"grab_point_{point}.py"
        
        try:
            # Import the grab script dynamically
            print(f"   Loading script: {script_name}")
            
            # Check if script exists
            import importlib.util
            spec = importlib.util.spec_from_file_location(script_name, script_name)
            if spec is None:
                raise FileNotFoundError(f"Script {script_name} not found")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[script_name] = module
            spec.loader.exec_module(module)
            
            # Determine which function to call
            if point in ['B', 'D', 'E']:
                # Points that can grab multiple tool types
                tool_param = GrabExecutor.TOOL_TO_SCRIPT_PARAM.get(tool_name, tool_name)
                print(f"   Using tool type: {tool_param}")
                
                if hasattr(module, f'grab_point_{point}'):
                    func = getattr(module, f'grab_point_{point}')
                    func(arm, tool_type=tool_param, grip_force=grip_force)
                elif hasattr(module, 'main'):
                    module.main(arm, tool_type=tool_param, grip_force=grip_force)
                else:
                    # Try any function that starts with 'grab'
                    for attr_name in dir(module):
                        if attr_name.startswith('grab'):
                            func = getattr(module, attr_name)
                            func(arm, tool_type=tool_param, grip_force=grip_force)
                            break
            else:
                # Points for single tool type
                if hasattr(module, f'grab_point_{point}'):
                    func = getattr(module, f'grab_point_{point}')
                    func(arm, grip_force=grip_force)
                elif hasattr(module, 'main'):
                    module.main(arm, grip_force=grip_force)
                else:
                    # Try default function
                    module.grab(arm, grip_force=grip_force)
            
            print(f"✅ Grab sequence completed for {tool_name}")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ ERROR: {e}")
            print("   Please create the script file first")
            return False
        except Exception as e:
            print(f"❌ ERROR executing {script_name}: {e}")
            print("   Using emergency default movements...")
            return GrabExecutor.execute_default(arm, grip_force)
    
    @staticmethod
    def execute_default(arm, grip_force):
        """Emergency default grab sequence"""
        print("   Running default grab sequence...")
        try:
            # Home position
            arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
            time.sleep(1)
            
            # Close gripper
            arm.Arm_serial_servo_write(6, grip_force, 1000)
            time.sleep(1)
            
            # Return to home/drop zone
            arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
            time.sleep(1)
            
            # Open gripper
            arm.Arm_serial_servo_write(6, 90, 1000)
            
            print("✅ Default grab completed")
            return True
        except:
            print("❌ Failed even with default sequence")
            return False