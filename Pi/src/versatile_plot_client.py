import socket
import time
import json
from nano_interface import initialize_serial, get_encoder_counts
from kinematics import pot_readings_to_velocities, FK

# Configuration
PC_IP = '192.168.1.101'
PC_PORT = 5005

# Initialize serial
ser = initialize_serial()
prev_left, prev_right = 0, 0
prev_time = time.time()

# Connect to PC
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Connecting to PC at {PC_IP}:{PC_PORT}...")
try:
    sock.connect((PC_IP, PC_PORT))
    print("Connected to PC!")
except Exception as e:
    print(f"Failed to connect: {e}")
    exit(1)

print("Sending data to PC...")

frame_count = 0
try:
    while True:
        current_time = time.time()
        dt = current_time - prev_time if prev_time else 0.1
        
        left_reading, right_reading = get_encoder_counts(ser)
        
        # Skip invalid readings
        if left_reading is None or right_reading is None:
            continue
        
        # Get wheel angular velocities from encoder readings
        wl, wr = pot_readings_to_velocities(left_reading, right_reading, 
                                            prev_left, prev_right, dt)
        
        # Apply forward kinematics
        v, w = FK(wl, wr, L=0.081, D=0.037)
        
        # Build data dictionary - add/remove parameters as needed
        data = {
            'wheel_velocities': (wl, wr),           # Tuple = single plot with both
            'robot_velocities': (v, w),             # Tuple = single plot with both
            'linear_velocity': v,                    # Scalar = individual plot
            'angular_velocity': w,                   # Scalar = individual plot
            'encoder_readings': (left_reading, right_reading),  # Tuple = single plot
            # Add more parameters here as needed:
            # 'temperature': sensor_temp,
            # 'battery': battery_voltage,
            # 'imu': (accel_x, accel_y, accel_z),
        }
        
        try:
            sock.sendall((json.dumps(data) + '\n').encode('utf-8'))
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Sent {frame_count} frames - v:{v:.3f} m/s, w:{w:.3f} rad/s")
        except Exception as e:
            print(f"Connection lost: {e}")
            break
        
        # Update prev values
        prev_left, prev_right = left_reading, right_reading
        prev_time = current_time
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    sock.close()
    print("Connection closed")
