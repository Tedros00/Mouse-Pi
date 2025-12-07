#!/usr/bin/env python3
import sys
import time
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/mouse/AMR/src')

from kinematics import pot_readings_to_velocities, FK
from nano_interface import initialize_serial, get_encoder_counts, send_vel_cmd

# Initialize serial connection
ser = initialize_serial()

# Robot parameters
L = 0.081  # Distance between wheels (m)
D = 0.037  # Wheel diameter (m)

print("Reading encoder counts and motor velocities. Press Ctrl+C to stop.")

# Data collection lists
time_data = []
left_counts = []
right_counts = []
wl_velocities = []
wr_velocities = []

prev_left, prev_right = None, None
prev_time = time.time()
start_time = prev_time
cmd_sent = False
try:
    while True:
        current_time = time.time()
        actual_dt = current_time - prev_time
        left, right = get_encoder_counts(ser)
        
        # Send motor command once on first successful read
        if not cmd_sent and left is not None and right is not None:
            send_vel_cmd(ser, 0, 0)
            cmd_sent = True
        
        if left is not None and right is not None:
            # Calculate motor velocities if we have previous readings
            if prev_left is not None and prev_right is not None:
                wl, wr = pot_readings_to_velocities(
                    left, right, prev_left, prev_right, actual_dt,
                    pot_diameter_mm=10, wheel_diameter_mm=37
                )
                v, w = FK(wl, wr, L=L, D=D)
                sys.stdout.write(f"\rdt:{actual_dt*1000:5.1f}ms | L:{left:6d} | R:{right:6d} | wl:{wl:8.4f} rad/s | wr:{wr:8.4f} rad/s | v:{v:8.4f} m/s | w:{w:8.4f} rad/s")
                
                # Collect data for plots
                time_data.append(current_time - start_time)
                left_counts.append(left)
                right_counts.append(right)
                wl_velocities.append(wl)
                wr_velocities.append(wr)
            else:
                sys.stdout.write(f"\rdt:{actual_dt*1000:5.1f}ms | L:{left:6d} | R:{right:6d} | wl:     -- | wr:     -- | v:     -- | w:     --")
            
            sys.stdout.flush()
            prev_left, prev_right = left, right
            prev_time = current_time
        
        time.sleep(0.01)  # Small sleep to prevent CPU spinning
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    send_vel_cmd(ser, 0, 0)

    ser.close()
    

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Encoder counts
    ax1.plot(time_data, left_counts, label='Left Encoder Count', linewidth=2)
    ax1.plot(time_data, right_counts, label='Right Encoder Count', linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Encoder Count', fontsize=12)
    ax1.set_title('Filtered Encoder Counts Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Angular velocities
    ax2.plot(time_data, wl_velocities, label='Left Wheel Angular Velocity (wl)', linewidth=2)
    ax2.plot(time_data, wr_velocities, label='Right Wheel Angular Velocity (wr)', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax2.set_title('Motor Angular Velocities Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/mouse/AMR/examples/encoder_analysis.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to /home/mouse/AMR/examples/encoder_analysis.png")
