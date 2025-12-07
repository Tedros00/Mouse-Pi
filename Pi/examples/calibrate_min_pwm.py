#!/usr/bin/env python3
"""
Calibration script to find minimum PWM required to move motors.
Tests increasing PWM values and monitors encoder readings to detect movement.
"""

import sys
import time
sys.path.insert(0, '/home/mouse/AMR/src')

from nano_interface import initialize_serial, get_encoder_counts, send_vel_cmd

# Initialize serial connection
ser = initialize_serial()

# Test parameters
pwm_start = 0
pwm_end = 255
pwm_step = 5
settle_time = 0.5  # Time to let motor settle (seconds)
sample_time = 1.0  # Time to sample readings (seconds)
samples_per_test = 10  # Number of readings to take

print("=" * 80)
print("Minimum PWM Calibration Script")
print("=" * 80)
print(f"Testing PWM from {pwm_start} to {pwm_end} in steps of {pwm_step}")
print(f"Settle time: {settle_time}s, Sample time: {sample_time}s")
print("=" * 80)

left_min_pwm = None
right_min_pwm = None

try:
    for pwm in range(pwm_start, pwm_end + 1, pwm_step):
        print(f"\n--- Testing PWM: {pwm} ---")
        
        # Stop motors
        send_vel_cmd(ser, 0, 0)
        time.sleep(0.2)
        
        # Get baseline readings
        baseline_left_readings = []
        baseline_right_readings = []
        for _ in range(samples_per_test):
            left, right = get_encoder_counts(ser)
            if left is not None and right is not None:
                baseline_left_readings.append(left)
                baseline_right_readings.append(right)
            time.sleep(0.05)
        
        baseline_left = sum(baseline_left_readings) / len(baseline_left_readings)
        baseline_right = sum(baseline_right_readings) / len(baseline_right_readings)
        print(f"Baseline: Left={baseline_left:.0f}, Right={baseline_right:.0f}")
        
        # Let motor settle
        time.sleep(settle_time)
        
        # Send PWM command (same to both wheels for forward motion)
        send_vel_cmd(ser, pwm, pwm)
        
        # Let motor accelerate
        time.sleep(settle_time)
        
        # Take readings after movement
        final_left_readings = []
        final_right_readings = []
        for _ in range(samples_per_test):
            left, right = get_encoder_counts(ser)
            if left is not None and right is not None:
                final_left_readings.append(left)
                final_right_readings.append(right)
            time.sleep(0.05)
        
        final_left = sum(final_left_readings) / len(final_left_readings)
        final_right = sum(final_right_readings) / len(final_right_readings)
        
        delta_left = final_left - baseline_left
        delta_right = final_right - baseline_right
        
        print(f"Final:    Left={final_left:.0f}, Right={final_right:.0f}")
        print(f"Delta:    Left={delta_left:.0f}, Right={delta_right:.0f}")
        
        # Check for movement (threshold = 5 counts)
        movement_threshold = 5
        left_moved = abs(delta_left) > movement_threshold
        right_moved = abs(delta_right) > movement_threshold
        
        if left_moved and left_min_pwm is None:
            left_min_pwm = pwm
            print(f"✓ LEFT motor started moving at PWM {pwm}")
        
        if right_moved and right_min_pwm is None:
            right_min_pwm = pwm
            print(f"✓ RIGHT motor started moving at PWM {pwm}")
        
        if left_min_pwm is not None and right_min_pwm is not None:
            print("\n✓ Both motors calibrated - stopping test")
            break
    
    # Stop motors
    send_vel_cmd(ser, 0, 0)
    time.sleep(0.2)
    
    # Print results
    print("\n" + "=" * 80)
    print("CALIBRATION RESULTS")
    print("=" * 80)
    if left_min_pwm is not None:
        print(f"Left motor  minimum PWM:  {left_min_pwm}")
    else:
        print(f"Left motor  minimum PWM:  NOT FOUND (tested up to {pwm_end})")
    
    if right_min_pwm is not None:
        print(f"Right motor minimum PWM:  {right_min_pwm}")
    else:
        print(f"Right motor minimum PWM:  NOT FOUND (tested up to {pwm_end})")
    
    if left_min_pwm is not None and right_min_pwm is not None:
        avg_min_pwm = (left_min_pwm + right_min_pwm) / 2
        print(f"\nAverage minimum PWM: {avg_min_pwm:.0f}")
        print(f"Recommended min_pwm setting: {int(avg_min_pwm)}")
    
    print("=" * 80)

except KeyboardInterrupt:
    print("\n\nCalibration interrupted by user")
    send_vel_cmd(ser, 0, 0)
finally:
    ser.close()
    print("Serial connection closed")
