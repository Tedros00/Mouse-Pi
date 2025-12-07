#!/usr/bin/env python3
"""
Calibration script to estimate sigma_cmd and sigma_fb for motion model.
Sends controlled movements and logs command vs actual velocities to estimate error distributions.
"""

import sys
import time
import numpy as np
sys.path.insert(0, '/home/mouse/AMR/src')

from kinematics import IK, FK, velocities_to_pwm, pot_readings_to_velocities, integrate_pose
from nano_interface import initialize_serial, get_encoder_counts, send_vel_cmd

# Initialize serial connection
ser = initialize_serial()

# Robot parameters
L = 0.081  # Distance between wheels (m)
D = 0.037  # Wheel diameter (m)

# Test parameters
fixed_dt = 0.01  # 100 Hz
settle_time = 0.5  # Time to let robot reach steady state (seconds)
sample_duration = 2.0  # Duration to collect samples (seconds)
samples_per_test = int(sample_duration / fixed_dt)

# Test cases: (v_cmd, w_cmd, description)
test_cases = [
    (0.002, 0.0, "Forward 0.005 m/s"),
    (-0.002, 0.0, "Backward 0.005 m/s"),
    (0.0, 0.05, "Rotate 0.05 rad/s"),
    (0.0, -0.05, "Rotate -0.05 rad/s"),
    (0.0, 0.0, "Stationary (zero command)"),
]

print("=" * 80)
print("Sigma Calibration Script")
print("=" * 80)
print(f"Fixed dt: {fixed_dt}s")
print(f"Settle time: {settle_time}s")
print(f"Sample duration: {sample_duration}s per test")
print(f"Samples per test: {samples_per_test}")
print("=" * 80)

# Storage for results
results = {}
stationary_result = None

try:
    # Initial encoder read to sync
    print("Syncing encoders...")
    for attempt in range(10):
        baseline_left, baseline_right = get_encoder_counts(ser)
        if baseline_left is not None and baseline_right is not None:
            print("  Encoders synced")
            break
        time.sleep(0.1)
    
    for v_cmd, w_cmd, description in test_cases:
        print(f"\n>>> Test: {description} (v={v_cmd}, w={w_cmd})")
        print("  Settling...")
        
        # Stop motors
        send_vel_cmd(ser, 0, 0)
        time.sleep(0.5)
        
        # Get baseline encoder readings
        baseline_left, baseline_right = get_encoder_counts(ser)
        if baseline_left is None or baseline_right is None:
            print("  ERROR: Could not read encoders")
            continue
        
        prev_left, prev_right = baseline_left, baseline_right
        
        # Send command and let it settle
        wl, wr = IK(v_cmd, w_cmd, L=L, D=D)
        pwm_left, pwm_right = velocities_to_pwm(wl, wr)
        send_vel_cmd(ser, pwm_left, pwm_right)
        time.sleep(settle_time)
        
        print("  Collecting samples...")
        
        # Collect samples
        v_actual_list = []
        w_actual_list = []
        v_cmd_list = []
        w_cmd_list = []
        
        sample_start = time.time()
        
        for i in range(samples_per_test):
            current_time = time.time()
            dt = current_time - sample_start - i * fixed_dt
            
            # Get encoder readings
            left, right = get_encoder_counts(ser)
            if left is None or right is None:
                continue
            
            # Calculate actual velocities from encoder deltas
            wl_actual, wr_actual = pot_readings_to_velocities(
                left, right, prev_left, prev_right, fixed_dt,
                pot_diameter_mm=10, wheel_diameter_mm=37
            )
            
            # Get actual v, w from FK
            v_actual, w_actual = FK(wl_actual, wr_actual, L=L, D=D)
            
            v_actual_list.append(v_actual)
            w_actual_list.append(w_actual)
            v_cmd_list.append(v_cmd)
            w_cmd_list.append(w_cmd)
            
            prev_left = left
            prev_right = right
            
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{samples_per_test} samples")
            
            # Maintain timing
            elapsed = time.time() - sample_start
            expected_time = (i + 1) * fixed_dt
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)
        
        # Stop motors
        send_vel_cmd(ser, 0, 0)
        time.sleep(0.2)
        
        # Calculate statistics
        v_actual_arr = np.array(v_actual_list)
        w_actual_arr = np.array(w_actual_list)
        
        # Error = actual - commanded
        v_error = v_actual_arr - v_cmd
        w_error = w_actual_arr - w_cmd
        
        # Standard deviations
        sigma_v = np.std(v_error)
        sigma_w = np.std(w_error)
        
        # Mean and max error
        v_mean_error = np.mean(v_error)
        w_mean_error = np.mean(w_error)
        v_max_error = np.max(np.abs(v_error))
        w_max_error = np.max(np.abs(w_error))
        
        results[description] = {
            'v_cmd': v_cmd,
            'w_cmd': w_cmd,
            'sigma_v': sigma_v,
            'sigma_w': sigma_w,
            'v_mean_error': v_mean_error,
            'w_mean_error': w_mean_error,
            'v_max_error': v_max_error,
            'w_max_error': w_max_error,
            'v_actual_mean': np.mean(v_actual_arr),
            'w_actual_mean': np.mean(w_actual_arr),
        }
        
        # Store stationary result for sigma_fb calculation
        if v_cmd == 0.0 and w_cmd == 0.0:
            stationary_result = {
                'v_actual_arr': v_actual_arr,
                'w_actual_arr': w_actual_arr,
                'description': description,
            }
        
        print(f"  Results:")
        print(f"    v: mean={v_actual_arr.mean():7.4f} m/s, σ_error={sigma_v:7.4f}, max_error={v_max_error:7.4f}")
        print(f"    w: mean={w_actual_arr.mean():7.4f} rad/s, σ_error={sigma_w:7.4f}, max_error={w_max_error:7.4f}")

except KeyboardInterrupt:
    print("\n\nCalibration interrupted")
    send_vel_cmd(ser, 0, 0)
finally:
    ser.close()

# Print summary
print("\n" + "=" * 80)
print("CALIBRATION SUMMARY")
print("=" * 80)

if results:
    # Calculate overall sigmas for sigma_cmd (from moving tests)
    all_sigma_v = []
    all_sigma_w = []
    
    for test_name, data in results.items():
        print(f"\n{test_name}")
        print(f"  Commanded: v={data['v_cmd']:7.4f} m/s, w={data['w_cmd']:7.4f} rad/s")
        print(f"  Actual:    v={data['v_actual_mean']:7.4f} m/s, w={data['w_actual_mean']:7.4f} rad/s")
        print(f"  Error Std: σ_v={data['sigma_v']:7.4f}, σ_w={data['sigma_w']:7.4f}")
        print(f"  Max Error: v={data['v_max_error']:7.4f} m/s, w={data['w_max_error']:7.4f} rad/s")
        
        # Only include moving tests for sigma_cmd (exclude stationary)
        if data['v_cmd'] != 0.0 or data['w_cmd'] != 0.0:
            if data['sigma_v'] > 0:
                all_sigma_v.append(data['sigma_v'])
            if data['sigma_w'] > 0:
                all_sigma_w.append(data['sigma_w'])
    
    # Calculate sigma_fb from stationary test (encoder noise only)
    sigma_fb_v = 0.01
    sigma_fb_w = 0.01
    if stationary_result is not None:
        # When stationary, all measured velocities are noise
        sigma_fb_v = np.std(stationary_result['v_actual_arr'])
        sigma_fb_w = np.std(stationary_result['w_actual_arr'])
        print(f"\n{stationary_result['description']}")
        print(f"  Encoder feedback noise (σ_fb):")
        print(f"    σ_fb_v={sigma_fb_v:7.4f} m/s (measurement noise)")
        print(f"    σ_fb_w={sigma_fb_w:7.4f} rad/s (measurement noise)")
    
    if all_sigma_v and all_sigma_w:
        # Use RMS of all sigmas for command execution error
        sigma_v_final = np.sqrt(np.mean(np.array(all_sigma_v)**2))
        sigma_w_final = np.sqrt(np.mean(np.array(all_sigma_w)**2))
        
        print("\n" + "-" * 80)
        print("RECOMMENDED SIGMAS FOR KALMAN FUSION")
        print("-" * 80)
        print(f"sigma_cmd = ({sigma_v_final:.4f}, {sigma_w_final:.4f})  # Command execution error")
        print(f"sigma_fb  = ({sigma_fb_v:.4f}, {sigma_fb_w:.4f})  # Encoder measurement noise")
        print(f"\nUse in command_correction():")
        print(f"  u_fused = command_correction(u_cmd=ut, u_fb=(v_result, w_result),")
        print(f"                               sigma_cmd=({sigma_v_final:.4f}, {sigma_w_final:.4f}),")
        print(f"                               sigma_fb=({sigma_fb_v:.4f}, {sigma_fb_w:.4f}))")
        
        print("\n" + "-" * 80)
        print("Python code to add to vel_motion_model.py:")
        print("-" * 80)
        print(f"""
# After motion model update:
from ProbabilisticMotionModel import command_correction

# Calibrated sigmas from calibrate_sigmas.py
SIGMA_CMD = ({sigma_v_final:.4f}, {sigma_w_final:.4f})  # Command execution error
SIGMA_FB = ({sigma_fb_v:.4f}, {sigma_fb_w:.4f})    # Encoder measurement noise

# In main loop after: Xt = sample_motion_velocity_model(...)
u_fused = command_correction(u_cmd=ut, u_fb=(v_result, w_result),
                             sigma_cmd=SIGMA_CMD, sigma_fb=SIGMA_FB)
# u_fused is the corrected (v, w) to use for pose integration
""")

print("=" * 80)
