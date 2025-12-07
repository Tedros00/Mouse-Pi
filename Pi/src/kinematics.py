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
    Convert potentiometer readings to wheel angular velocities based on rate of change.
    
    Parameters:
    left_reading : int
        Current left integrated encoder count
    right_reading : int
        Current right integrated encoder count
    prev_left_reading : int
        Previous left integrated encoder count
    prev_right_reading : int
        Previous right integrated encoder count
    dt : float
        Time interval (s)
    bits : int
        Number of bits for potentiometer (10 bits = 0-1023, 1024 counts per full turn)
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
    max_count = 2 ** bits  # 1024 for 10-bit (1024 counts = 1 full turn)
    
    # Gear ratio: potentiometer wheel to actual wheel
    gear_ratio = pot_diameter_mm / wheel_diameter_mm 
    
    # Calculate change in encoder counts (integrated, so no wraparound handling needed)
    delta_left = left_reading - prev_left_reading
    delta_right = right_reading - prev_right_reading
    
    # Convert count deltas to radians (1024 counts = 2π radians)
    angle_delta_left = delta_left * (pot_range_rad / max_count)
    angle_delta_right = delta_right * (pot_range_rad / max_count)
    
    # Calculate potentiometer angular velocities (rad/s)
    wl_pot = angle_delta_left / dt
    wr_pot = angle_delta_right / dt
    
    # Convert to actual wheel angular velocities using gear ratio
    # right wheel gets a mirror to account for orientation 
    wl = np.clip(wl_pot * gear_ratio, -1.8, 1.8)
    wr = np.clip(-wr_pot * gear_ratio, -1.8, 1.8)
    
    return wl, wr

def integrate_pose(v, w, dt, x=0, y=0, theta=0):
    """
    Integrate robot velocities to update pose.
    
    Parameters:
    v : float
        Linear velocity (m/s)
    w : float
        Angular velocity (rad/s)
    dt : float
        Time step (s)
    x : float
        Current x position (m)
    y : float
        Current y position (m)
    theta : float
        Current orientation (rad)
        
    Returns:
    tuple
        (x, y, theta) updated pose
    """
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + w * dt
    
    return x_new, y_new, theta_new

def velocities_to_pwm(wl, wr, min_pwm=50, max_pwm=255, min_wheel_rad_s=0, max_wheel_rad_s=1.8):

    """
    Convert wheel angular velocities to PWM commands.
    
    Parameters:
    wl : float
        Left wheel angular velocity (rad/s)
    wr : float
        Right wheel angular velocity (rad/s)
    min_pwm : int
        Minimum PWM value (threshold for motor movement)
    max_pwm : int
        Maximum PWM value
    min_wheel_rad_s : float
        Minimum wheel angular velocity (rad/s) threshold
    max_wheel_rad_s : float
        Maximum wheel angular velocity (rad/s) corresponding to max PWM
        
    Returns:
    tuple
        (pwm_left, pwm_right) as integers
    """
    # Apply minimum velocity threshold
    wl = wl if abs(wl) >= min_wheel_rad_s else 0
    wr = wr if abs(wr) >= min_wheel_rad_s else 0
    
    # Normalize velocities to [-1, 1]
    norm_left = np.clip(wl / max_wheel_rad_s, -1, 1)
    norm_right = np.clip(wr / max_wheel_rad_s, -1, 1)
    
    # Convert to PWM with dead-band handling
    def norm_to_pwm(norm_vel):
        if norm_vel == 0:
            return 0
        elif norm_vel > 0:
            # Map [0, 1] to [min_pwm, max_pwm]
            return int(min_pwm + norm_vel * (max_pwm - min_pwm))
        else:
            # Map [-1, 0] to [-max_pwm, -min_pwm]
            return int(-min_pwm + norm_vel * (max_pwm - min_pwm))
    
    pwm_left = norm_to_pwm(norm_left)
    pwm_right = norm_to_pwm(norm_right)
    
    return pwm_left, pwm_right
