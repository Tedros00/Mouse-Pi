import numpy as np 

def IK(v, w, L=0.081, D = 0.037):
    """
    Inverse Kinematics for a differential drive robot.
    
    Parameters:
    v : float
        Linear velocity (m/s)
    w : float
        Angular velocity (rad/s)
    L : float
        Distance between the wheels (m)
    D : float
        Diameter of the wheels (m)
        
    Returns:
    tuple
        Left and right wheel angular velocities (rad/s)
    """
    r = D / 2  # Radius of the wheel
    wl = (2 * v - w * L) / (2 * r)  # Left wheel angular velocity
    wr = (2 * v + w * L) / (2 * r)  # Right wheel angular velocity
    return wl, wr

def FK(wl, wr, L=0.081, D = 0.037):
    """
    Forward Kinematics for a differential drive robot.
    
    Parameters:
    wl : float
        Left wheel angular velocity (rad/s)
    wr : float
        Right wheel angular velocity (rad/s)
    L : float
        Distance between the wheels (m)
    D : float
        Diameter of the wheels (m)
        
    Returns:
    tuple
        Linear and angular velocities (m/s, rad/s)
    """
    r = D / 2  # Radius of the wheel
    v = r * (wl + wr) / 2  # Linear velocity
    w = r * (wr - wl) / (2*L)  # Angular velocity
    return v, w

def pot_readings_to_velocities(left_reading, right_reading, prev_left_reading, prev_right_reading, dt, bits=10, pot_range_rad=2*np.pi, pot_diameter_mm=10, wheel_diameter_mm=37):
    """
    Convert potentiometer readings to wheel angular velocities, handling wraparound.
    
    Parameters:
    left_reading : int
        Current left potentiometer reading (0-1023)
    right_reading : int
        Current right potentiometer reading (0-1023)
    prev_left_reading : int
        Previous left potentiometer reading
    prev_right_reading : int
        Previous right potentiometer reading
    dt : float
        Time interval (s)
    bits : int
        Number of bits for potentiometer (10 bits = 0-1023)
    pot_range_rad : float
        Range of potentiometer in radians (typically 2π for full rotation)
    pot_diameter_mm : float
        Diameter of potentiometer wheel (mm)
    wheel_diameter_mm : float
        Diameter of the actual wheel (mm)
        
    Returns:
    tuple
        (left_angular_velocity, right_angular_velocity) in rad/s (actual wheel velocities)
    """
    max_count = 2 ** bits  # 1024 for 10-bit
    
    # Gear ratio: potentiometer wheel to actual wheel
    gear_ratio = pot_diameter_mm / wheel_diameter_mm 
    
    # Calculate delta for left wheel
    delta_left = left_reading - prev_left_reading
    if delta_left > max_count / 2:
        delta_left -= max_count  # Wrapped backwards
    elif delta_left < -max_count / 2:
        delta_left += max_count  # Wrapped forwards
    
    # Calculate delta for right wheel
    delta_right = right_reading - prev_right_reading
    if delta_right > max_count / 2:
        delta_right -= max_count  # Wrapped backwards
    elif delta_right < -max_count / 2:
        delta_right += max_count  # Wrapped forwards
    
    # Convert deltas to radians and then to angular velocities
    angle_delta_left = delta_left * (pot_range_rad / max_count)
    angle_delta_right = delta_right * (pot_range_rad / max_count)
    
    # Potentiometer angular velocities
    wl_pot = angle_delta_left / dt
    wr_pot = angle_delta_right / dt
    
    # Convert to actual wheel angular velocities using gear ratio
    # right wheel gets a mirror to account for orientation 
    wl = wl_pot * gear_ratio
    wr = -wr_pot * gear_ratio
    
    return wl, wr

def velocities_to_pwm(wl, wr, max_pwm=255, max_wheel_rad_s=3.14):
    """
    Convert wheel angular velocities to PWM commands.
    
    Parameters:
    wl : float
        Left wheel angular velocity (rad/s)
    wr : float
        Right wheel angular velocity (rad/s)
    max_pwm : int
        Maximum PWM value
    max_wheel_rad_s : float
        Maximum wheel angular velocity (rad/s) corresponding to max PWM
        
    Returns:
    tuple
        (pwm_left, pwm_right) as integers
    """
    pwm_left = int(np.clip((wl / max_wheel_rad_s) * max_pwm, -max_pwm, max_pwm))
    pwm_right = int(np.clip((wr / max_wheel_rad_s) * max_pwm, -max_pwm, max_pwm))
    
    return pwm_left, pwm_right