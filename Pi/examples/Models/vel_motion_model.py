import numpy as np
import sys
import os
import time
import threading
import termios
import tty
import select
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from kinematics import IK, FK, velocities_to_pwm , pot_readings_to_velocities
from nano_interface import initialize_serial, get_encoder_counts, send_vel_cmd, prime_encoders
from ProbabilisticMotionModel import sample_motion_velocity_model, command_correction   

# Serial connection
ser = initialize_serial()

# Robot parameters
L = 0.081  # Distance between wheels (m)
D = 0.037  # Wheel diameter (m)

# Keyboard control parameters
v_accel_rate = 2      # m/s^2 - acceleration rate when key pressed
w_accel_rate = 2       # rad/s^2 - angular acceleration rate when key pressed
v_max = 0.03              # m/s - maximum linear velocity
w_max = 0.41111                  # rad/s - maximum angular velocity

# Command state
v_cmd = 0.0              # Current commanded linear velocity
w_cmd = 0.0              # Current commanded angular velocity
key_states = {
    'w': False,  # Forward
    's': False,  # Backward
    'a': False,  # Turn left
    'd': False   # Turn right
}

running = True
stdin_settings = None
old_settings = None
last_input_time = time.time()
key_timeout = 0.15  # If no input for 150ms, consider keys released

def input_thread():
    """Thread to handle keyboard input with proper key press/release detection"""
    global key_states, running, old_settings, last_input_time
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        
        while running:
            try:
                # Use select with very short timeout for responsiveness
                ready, _, _ = select.select([sys.stdin], [], [], 0.001)
                
                if ready:
                    char = sys.stdin.read(1)
                    if char:
                        char = char.lower()
                        last_input_time = time.time()  # Update last input time
                        
                        if char == '\x03':  # Ctrl+C
                            running = False
                            break
                        # Ignore newlines and carriage returns
                        if char in ['\n', '\r', '\x00']:
                            continue
                        # Key press detected - set to True
                        if char == 'w':
                            key_states['w'] = True
                            key_states['s'] = False  # Can't go forward and backward
                        elif char == 's':
                            key_states['s'] = True
                            key_states['w'] = False
                        elif char == 'a':
                            key_states['a'] = True
                            key_states['d'] = False  # Can't turn both directions
                        elif char == 'd':
                            key_states['d'] = True
                            key_states['a'] = False
                        elif char == ' ':
                            # Space to stop
                            key_states['w'] = False
                            key_states['s'] = False
                            key_states['a'] = False
                            key_states['d'] = False
                else:
                    # Check if keys have timed out (no input received for key_timeout seconds)
                    current_time = time.time()
                    if current_time - last_input_time > key_timeout:
                        # Release all keys due to timeout
                        if key_states['w'] or key_states['s'] or key_states['a'] or key_states['d']:
                            key_states['w'] = False
                            key_states['s'] = False
                            key_states['a'] = False
                            key_states['d'] = False
                            last_input_time = current_time  # Reset timeout
                    
            except KeyboardInterrupt:
                running = False
                break
                
    finally:
        if old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass

# Start input thread
input_thread_obj = threading.Thread(target=input_thread, daemon=True)
input_thread_obj.start()

# States
x, y, theta = 0.0, 0.0, 0.0
prev_left, prev_right = prime_encoders(ser)
prev_time = time.time()

print("Velocity Motion Model with Keyboard Control")
print("Controls: W=forward, S=backward, A=turn left, D=turn right, SPACE=stop")
print("Press Ctrl+C to stop")
print("-" * 80)
print(f"{'Time (s)':>10} | {'v (m/s)':>10} | {'w (rad/s)':>10} | {'X (m)':>10} | {'Y (m)':>10} | {'Theta':>10}")
print("-" * 80)

try:
    start_time = time.time()
    last_update = time.time()
    fixed_dt = 0.01  # Target 100 Hz
    
    # Data collection for plotting
    time_data = []
    u_cmd_list = []
    u_fb_list = []
    
    while running:
        if not running:
            break
        
        # Use fixed dt for predictable behavior
        dt = fixed_dt
        
        # Update v_cmd based on keyboard input
        if key_states['w']:
            v_cmd = min(v_cmd + v_accel_rate * dt, v_max)
        elif key_states['s']:
            v_cmd = max(v_cmd - v_accel_rate * dt, -v_max)
        else:
            # Key released - set to zero immediately
            v_cmd = 0.0
        
        # Update w_cmd based on keyboard input
        if key_states['a']:
            w_cmd = min(w_cmd + w_accel_rate * dt, w_max)
        elif key_states['d']:
            w_cmd = max(w_cmd - w_accel_rate * dt, -w_max)
        else:
            # Key released - set to zero immediately
            w_cmd = 0.0
        
        current_time = time.time()
        actual_dt = current_time - last_update
        elapsed = current_time - start_time

        #feedback
        pot_l, pot_r = get_encoder_counts(ser)        
        wl_fb, wr_fb = pot_readings_to_velocities(pot_l, pot_r, prev_left, prev_right, actual_dt,
                                            pot_diameter_mm=10, wheel_diameter_mm=37)
        prev_left, prev_right = pot_l, pot_r

        v_result, w_result = FK(wl_fb, wr_fb, L=L, D=D)

        # Command input (v, w)
        ut_hat = (v_cmd, w_cmd)

        #command correction based on feedback
        ut = command_correction(ut_hat, (v_result, w_result))
        v_corr , w_corr = ut
        # Send velocity command to robot ASAP (before motion model calculation)
        wl, wr = IK(v_corr , w_corr, L=L, D=D)
        pwm_left, pwm_right = velocities_to_pwm(wl, wr)
        send_vel_cmd(ser, pwm_left, pwm_right)
        
        #sample from motion model
        prev_Xt = np.array([x, y, theta])
        Xt = sample_motion_velocity_model(prev_Xt, ut, dt=dt)
        x, y, theta = Xt

        # Use \r to overwrite the same line, and flush to ensure immediate output
        sys.stdout.write(f"\rTime: {elapsed:6.2f}s | dt: {actual_dt*1000:5.2f}ms | v_cmd: {v_cmd:7.4f} m/s | w_cmd: {w_corr:7.4f} rad/s | v_fb: {v_result:7.4f} m/s | w_fb: {w_result:7.4f} rad/s\r")
        sys.stdout.flush()
        
        # Collect data for plotting
        time_data.append(elapsed)
        u_cmd_list.append([v_cmd, w_cmd])
        u_fb_list.append([v_result, w_result])
        
        last_update = current_time
        time.sleep(0.01)  # 100 Hz update rate


except KeyboardInterrupt:
    print("\n\nStopped.")
    running = False
finally:
    send_vel_cmd(ser, 0, 0)
    ser.close()
    # Save plot if data was collected
    try:
        if len(time_data) > 0:
            # Convert lists to arrays
            time_data_array = np.array(time_data)
            u_cmd_array = np.array(u_cmd_list)
            u_fb_array = np.array(u_fb_list)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Linear velocity v over time
            ax1.plot(time_data_array, u_cmd_array[:, 0], label='v_cmd', linewidth=2, color='blue')
            ax1.plot(time_data_array, u_fb_array[:, 0], label='v_fb', linewidth=2, color='red', linestyle='--')
            ax1.set_xlabel('Time (s)', fontsize=11)
            ax1.set_ylabel('Linear Velocity (m/s)', fontsize=11)
            ax1.set_title('v_cmd(t) vs v_fb(t)', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Angular velocity w over time
            ax2.plot(time_data_array, u_cmd_array[:, 1], label='w_cmd', linewidth=2, color='green')
            ax2.plot(time_data_array, u_fb_array[:, 1], label='w_fb', linewidth=2, color='orange', linestyle='--')
            ax2.set_xlabel('Time (s)', fontsize=11)
            ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
            ax2.set_title('w_cmd(t) vs w_fb(t)', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out', 'examples')
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, 'command_feedback_comparison.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"\n✓ Plot saved to {output_path}")
    except Exception as e:
        print(f"\nError saving plot: {e}")

