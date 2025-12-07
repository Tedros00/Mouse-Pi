#!/usr/bin/env python3
import sys
import time
sys.path.insert(0, '/home/mouse/AMR/src')

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
from collections import deque

from nano_interface import initialize_serial, get_encoder_counts
from kinematics import pot_readings_to_velocities, FK, integrate_pose

# Initialize serial connection
ser = initialize_serial()

# Robot parameters
L = 0.081  # Distance between wheels (m)
D = 0.037  # Wheel diameter (m)

# Setup matplotlib for animation (non-interactive)
plt.ioff()  # Turn off interactive mode
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Robot Pose Tracker')

# Robot ellipse parameters
robot_length = 0.1  # Length of robot (m)
robot_width = 0.06  # Width of robot (m)
robot_ellipse = Ellipse((0, 0), robot_width, robot_length, 
                        angle=0, facecolor='lightblue', 
                        edgecolor='darkblue', linewidth=2)
ax.add_patch(robot_ellipse)

# Add origin marker
ax.plot(0, 0, 'r+', markersize=15, markeredgewidth=2, label='Origin')

# Store animation frames
frames_data = deque()

# Pose state
x, y, theta = 0.0, 0.0, 0.0
prev_left, prev_right = 0, 0
prev_time = time.time()

print("Tracking robot pose. Press Ctrl+C to stop.")
print("-" * 60)
print(f"{'Time (s)':>10} | {'X (m)':>10} | {'Y (m)':>10} | {'Theta (rad)':>12}")
print("-" * 60)

try:
    start_time = time.time()
    while True:
        current_time = time.time()
        dt = current_time - prev_time
        
        left_count, right_count = get_encoder_counts(ser)
        
        if left_count is None or right_count is None:
            continue
        
        # Get wheel angular velocities from encoder readings
        wl, wr = pot_readings_to_velocities(left_count, right_count,
                                            prev_left, prev_right, dt,
                                            pot_diameter_mm=10, wheel_diameter_mm=37)
        
        # Get robot velocities
        v, w = FK(wl, wr, L=L, D=D)
        
        # Integrate pose
        x, y, theta = integrate_pose(v, w, dt, x, y, theta)
        
        elapsed = current_time - start_time
        print(f"{elapsed:10.2f} | {x:10.4f} | {y:10.4f} | {theta:12.4f}")
        
        # Store pose data for animation
        frames_data.append((x, y, theta))
        
        prev_left = left_count
        prev_right = right_count
        prev_time = current_time
        
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    ser.close()
    
    # Create animation from collected frames
    print("\nCreating animation from recorded frames...")
    
    def animate(frame_idx):
        if frame_idx < len(frames_data):
            x_pos, y_pos, theta_pos = frames_data[frame_idx]
            robot_ellipse.set_center((x_pos, y_pos))
            robot_ellipse.set_angle(np.degrees(theta_pos))
        return robot_ellipse,
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data),
                                   interval=50, blit=True, repeat=True)
    
    # Save animation as GIF using Pillow
    output_file = '/home/mouse/AMR/examples/pose_animation.gif'
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer='pillow', fps=20)
    print(f"Animation saved successfully!")
