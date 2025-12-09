import serial
import time
from collections import deque
from kinematics import pot_readings_to_velocities

# Global variables to track integrated encoder counts
_integrated_left_count = 0
_integrated_right_count = 0
_prev_left_raw = None
_prev_right_raw = None
_max_count = 1024  # 2^10 for 10-bit encoder (counts from 0-1023)

# Moving average filter buffers for integrated encoder counts
_MA_WINDOW = 5
_left_count_buffer = deque(maxlen=_MA_WINDOW)
_right_count_buffer = deque(maxlen=_MA_WINDOW)

def _apply_count_filter(left_integrated, right_integrated):
    """Apply moving average filter to integrated encoder counts."""
    _left_count_buffer.append(left_integrated)
    _right_count_buffer.append(right_integrated)
    
    # Return mean of buffers
    left_filtered = sum(_left_count_buffer) / len(_left_count_buffer)
    right_filtered = sum(_right_count_buffer) / len(_right_count_buffer)
    
    return int(left_filtered), int(right_filtered)

def initialize_serial(port='/dev/ttyUSB0', baudrate=115200, timeout=1):
    """Initialize the serial connection to the Arduino Nano."""
    ser = serial.Serial(port, baudrate, timeout=timeout)
    # Clear any buffered data from reset
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    time.sleep(3)  # Wait for the connection to establish
    return ser

def init_encoders(ser, max_attempts=20, verbose=True):
    """Prime the encoder readings to ensure valid data is available.
    
    Call this after initialize_serial() and before the main control loop.
    Waits for the first successful encoder read.
    
    Parameters:
    ser : serial.Serial
        Serial connection from initialize_serial()
    max_attempts : int
        Maximum number of attempts to read encoders (default 20 = ~1 second)
    verbose : bool
        Print status messages (default True)
    
    Returns:
    tuple
        (left_count, right_count) from first successful read, or (None, None) if failed
    """
    if verbose:
        print("Priming encoder readings...")
    
    for attempt in range(max_attempts):
        left, right = get_encoder_counts(ser)
        if left is not None and right is not None:
            if verbose:
                print(f"  Encoders ready (attempt {attempt + 1}/{max_attempts})")
            return left, right
        time.sleep(0.05)
    
    if verbose:
        print("  WARNING: Encoders did not respond after {} attempts".format(max_attempts))
    return None, None

def get_encoder_counts(ser):
    #arduino sends: LEFT_COUNT RIGHT_COUNT
    """Request and read encoder counts from the Arduino Nano.
    
    Sends 'R' command to request encoder data, then reads the response.
    Response format: "LEFT_COUNT RIGHT_COUNT\n"
    Returns integrated counts that continue to increment even when the raw
    encoder count wraps from 1023 to 0. Applies moving average filter.
    """
    global _integrated_left_count, _integrated_right_count, _prev_left_raw, _prev_right_raw
    
    try:
        # Request encoder data from Arduino
        ser.write(b'R\n')
        ser.flush()
        
        line = ser.readline().decode('utf-8').strip()
        if not line:
            return None, None
        
        # Parse space-separated values: "LEFT RIGHT"
        parts = line.split()
        if len(parts) != 2:
            return None, None
        
        left_raw = int(parts[0])
        right_raw = int(parts[1])
        
        # On first call, initialize previous values
        if _prev_left_raw is None:
            _prev_left_raw = left_raw
            _integrated_left_count = left_raw
        if _prev_right_raw is None:
            _prev_right_raw = right_raw
            _integrated_right_count = right_raw
        
        # Detect wraparound and integrate counts
        # Left wheel
        delta_left = left_raw - _prev_left_raw
        if delta_left < -_max_count / 2:  # Wrapped forward (e.g., 1023 -> 0)
            delta_left += _max_count
        elif delta_left > _max_count / 2:  # Wrapped backward (shouldn't happen in normal operation)
            delta_left -= _max_count
        _integrated_left_count += delta_left
        _prev_left_raw = left_raw
        
        # Right wheel
        delta_right = right_raw - _prev_right_raw
        if delta_right < -_max_count / 2:  # Wrapped forward (e.g., 1023 -> 0)
            delta_right += _max_count
        elif delta_right > _max_count / 2:  # Wrapped backward (shouldn't happen in normal operation)
            delta_right -= _max_count
        _integrated_right_count += delta_right
        _prev_right_raw = right_raw
        
        # Apply moving average filter to integrated counts
        left_filtered, right_filtered = _apply_count_filter(_integrated_left_count, _integrated_right_count)
        
        return left_filtered, right_filtered
    except (ValueError, IndexError):
        return None, None

def send_vel_cmd(ser, pwm_left, pwm_right):
    """Send velocity command to the Arduino Nano."""
    #cmd format: pwm_left pwm_right\n
    #pwm are 2 ints ranging from -255 to 255
    
    """Send velocity command to the Arduino Nano."""
    #pwm_left is mirrored to account for motor orientation
    command = f"{pwm_right:.2f} {-pwm_left:.2f}\n"
    ser.write(command.encode('utf-8'))
    ser.flush()