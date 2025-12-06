import socket
import time
import json
from nano_interface import initialize_serial, get_encoder_counts
from kinematics import pot_readings_to_velocities, FK

# Configuration
PC_IP = '192.168.1.101'  # Replace with your PC's IP address
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
        
        # Send data as JSON
        data = {
            'wl': wl,
            'wr': wr,
            'v': v,
            'w': w,
            'timestamp': current_time
        }
        
        try:
            sock.sendall((json.dumps(data) + '\n').encode('utf-8'))
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Sent {frame_count} frames - wl:{wl:.3f}, wr:{wr:.3f}, v:{v:.3f}, w:{w:.3f}")
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
