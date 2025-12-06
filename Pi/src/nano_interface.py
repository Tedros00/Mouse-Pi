import serial
import time
from kinematics import pot_readings_to_velocities

def initialize_serial(port='/dev/ttyUSB0', baudrate=115200, timeout=1):
    """Initialize the serial connection to the Arduino Nano."""
    ser = serial.Serial(port, baudrate, timeout=timeout)
    time.sleep(2)  # Wait for the connection to establish
    return ser

def get_encoder_counts(ser):
    #arduino sends Pot1: INT | Pot2: INT
    """Request and read encoder counts from the Arduino Nano."""
    try:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            return None, None
        
        counts = line.split('|')
        if len(counts) != 2:
            return None, None
        
        left_count = int(counts[0].split(':')[1].strip())
        right_count = int(counts[1].split(':')[1].strip())
        
        # Validate counts are in expected range (0-1023 for 10-bit)
        if 0 <= left_count <= 1023 and 0 <= right_count <= 1023:
            return left_count, right_count
        return None, None
    except (ValueError, IndexError):
        return None, None

def send_vel_cmd(ser, pwm_left, pwm_right):
    """Send velocity command to the Arduino Nano."""
    #cmd format: pwm_left pwm_right\n
    #pwm are 2 ints ranging from -255 to 255
    
    """Send velocity command to the Arduino Nano."""
    command = f"{pwm_right:.2f} {pwm_left:.2f}\n"
    ser.write(command.encode('utf-8'))
    ser.flush()