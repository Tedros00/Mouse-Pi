"""
EKF-SLAM with Monte Carlo Localization (MCL) - Minimal Visualization
Random Initial Position → Localization → Path Planning → Following
Simplified version with only: Robot (GT), Robot (Est), Map, Planned Path
"""
import sys
import os
import argparse
import time
import numpy as np
from PIL import Image
import threading

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--headless', action='store_true', help='Run without displaying plots')
parser.add_argument('--save-gif', type=str, default=None, help='Path to save animation GIF (GIF will be written even in headless mode)')
args = parser.parse_args()

import matplotlib
if args.headless:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from EKF_SLAM import EKF_SLAM
from ProbabilisticMotionModel import sample_motion_velocity_model
from ProbabilisticSensorModel import scan_likelihood
from GridLocalization import monte_carlo_localization

print("Loading ground truth maze...")
maze_path = os.path.join(os.path.dirname(__file__), '..', 'SimulationEnv', 'TestMaze.JPG')
maze_image = np.array(Image.open(maze_path).convert('L'))
maze_binary = (maze_image > 127).astype(np.uint8)
maze_height_px, maze_width_px = maze_binary.shape

# Real world dimensions (in METERS)
REAL_WIDTH = 1.8
REAL_HEIGHT = 2.0
LIDAR_MAX_RANGE = 15.0
LIDAR_EFFECTIVE_RANGE = np.sqrt(REAL_WIDTH**2 + REAL_HEIGHT**2) * 1.5

# Calculate scale factors
scale_x = REAL_WIDTH / maze_width_px
scale_y = REAL_HEIGHT / maze_height_px

print(f"Maze dimensions: {maze_width_px}x{maze_height_px} pixels")
print(f"Real world dimensions: {REAL_WIDTH}x{REAL_HEIGHT} meters")
print(f"Scale: {scale_x:.6f} m/pixel (x), {scale_y:.6f} m/pixel (y)")

def compute_expected_measurements(x, y, angle, map_data, max_range, num_beams=360):
    """Compute expected LiDAR measurements by raycasting against map."""
    measurements = np.zeros(num_beams)
    step_size_m = raycast_step_m  # Use global raycast step size
    angles = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)
    
    for i in range(num_beams):
        beam_angle = angle + angles[i]
        cos_angle = np.cos(beam_angle)
        sin_angle = np.sin(beam_angle)
        
        for dist_m in np.arange(step_size_m, max_range, step_size_m):
            check_x_m = x + dist_m * cos_angle
            check_y_m = y + dist_m * sin_angle
            
            check_x_px = int(check_x_m / scale_x)
            check_y_px = int(check_y_m / scale_y)
            
            if check_x_px < 0 or check_x_px >= map_data.shape[1] or check_y_px < 0 or check_y_px >= map_data.shape[0]:
                measurements[i] = dist_m
                break
            
            if map_data[check_y_px, check_x_px] == 0:
                measurements[i] = dist_m
                break
        else:
            measurements[i] = max_range
    
    return measurements

def compute_from_map_wrapper(x, y, angle, map_data, max_range):
    """Compute expected range for a single beam."""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    step_size_m = 0.01
    
    for dist_m in np.arange(step_size_m, max_range, step_size_m):
        check_x_m = x + dist_m * cos_angle
        check_y_m = y + dist_m * sin_angle
        
        check_x_px = int(check_x_m / scale_x)
        check_y_px = int(check_y_m / scale_y)
        
        if check_x_px < 0 or check_x_px >= map_data.shape[1] or check_y_px < 0 or check_y_px >= map_data.shape[0]:
            return dist_m
        
        if map_data[check_y_px, check_x_px] == 0:
            return dist_m
    
    return max_range

print("Initializing Monte Carlo Localization...")
num_particles = 1000  # Reduced from 1000 for speed
num_beams = 360  # Number of LiDAR beams
raycast_step_m = 0.01  # Raycasting step size in meters
print(f"Using {num_particles} particles for MCL")
print(f"LiDAR: {num_beams} beams, {raycast_step_m}m raycast steps")

particles = np.zeros((num_particles, 3))
particles[:, 0] = np.random.uniform(0, REAL_WIDTH, num_particles)
particles[:, 1] = np.random.uniform(0, REAL_HEIGHT, num_particles)
particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)

# Ground truth position
gt_x_px = 150
gt_y_px = 1200
gt_theta = -np.pi/2
gt_pose = np.array([gt_x_px * scale_x, gt_y_px * scale_y, gt_theta])

print(f"Ground truth robot position: ({gt_pose[0]:.3f}, {gt_pose[1]:.3f}) m")

# Goal position
end_x_px = 1000
end_y_px = 350
goal_pos = np.array([end_x_px * scale_x, end_y_px * scale_y])
print(f"Goal position: ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}) m")

print("Initializing EKF-SLAM...")
cell_size_m = 0.02
grid_width = int(REAL_WIDTH / cell_size_m)
grid_height = int(REAL_HEIGHT / cell_size_m)
grid_shape = (grid_height, grid_width)

ekf_slam = EKF_SLAM(grid_shape=grid_shape, 
                    grid_bounds=((0, REAL_WIDTH), (0, REAL_HEIGHT)),
                    compute_from_map=compute_expected_measurements, 
                    dt=0.1,
                    motion_noise_std=(0.01, 0.005), 
                    measurement_noise_std=0.01)

ekf_slam.load_ground_truth_maze(maze_path)

# Simulation states
PHASE_LOCALIZATION = 0
PHASE_PATH_PLANNING = 1
PHASE_FOLLOWING = 2

current_phase = PHASE_LOCALIZATION
localization_complete = False
trajectory = None
current_waypoint_idx = 0
waypoint_tolerance = 5.0  # 5cm tolerance for advancing to next waypoint (from ekf_slam_II)

print("\n" + "="*70)
print("SIMULATION SETUP")
print("="*70)
print(f"Phase 1: Monte Carlo Localization (target uncertainty < 0.03 m)")
print(f"Phase 2: Path planning from localized position to goal")
print(f"Phase 3: Follow trajectory with EKF-SLAM")
print("="*70)

# Minimal visualization setup (only if not headless)
if not args.headless:
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_xlim(0, REAL_WIDTH)
    ax.set_ylim(0, REAL_HEIGHT)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('MCL Localization → Path Planning → EKF-SLAM Following')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    # Display maze
    ax.imshow(maze_binary, extent=[0, REAL_WIDTH, REAL_HEIGHT, 0], cmap='gray', origin='upper', alpha=0.6)
    
    # Plot elements
    gt_pos_plot, = ax.plot([], [], 'bo', markersize=12, label='GT Position')
    est_pos_plot, = ax.plot([], [], 'ro', markersize=12, label='Est Position')
    goal_marker, = ax.plot([goal_pos[0]], [goal_pos[1]], 'g^', markersize=14, label='Goal')
    trajectory_line, = ax.plot([], [], 'b--', alpha=0.5, linewidth=2, label='Planned Path')
    gt_heading, = ax.plot([], [], 'b-', linewidth=2)
    est_heading, = ax.plot([], [], 'r-', linewidth=2)
    
    # Status text
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

# History tracking
gt_trajectory_history = []
est_trajectory_history = []
errors = []
uncertainties = []
max_frames = 5000

# Path planning automatically triggered at frame 5 (no threading needed)

# Define motion and sensor models (used in all phases with monte_carlo_localization)
def sample_motion_model(prev_pose, u, dt=0.1):
    return sample_motion_velocity_model(prev_pose, u, dt=dt, 
                                       alphas=(0.001, 0.001, 0.001, 0.001, 0.001, 0.001))

def sensor_model(pose, z, m, sigma_hit=0.1):
    x, y, theta = pose
    z_expected = compute_expected_measurements(x, y, theta, m, LIDAR_EFFECTIVE_RANGE, num_beams=num_beams)
    z_expected = np.clip(z_expected, 0.001, LIDAR_EFFECTIVE_RANGE)
    return scan_likelihood(z, z_expected, sigma_hit=sigma_hit)

# Initialize estimated poses
est_pose_mcl = np.array([particles[0, 0], particles[0, 1], 0.0])
est_pose = est_pose_mcl.copy()  # Initialize for use in FOLLOWING phase

def update(frame):
    """Main simulation update loop"""
    global gt_pose, particles, current_phase, localization_complete, trajectory, current_waypoint_idx
    global gt_trajectory_history, est_trajectory_history
    global est_pose_mcl, est_pose, weights
    
    frame_start_time = time.time()
    
    if frame >= max_frames:
        return []
    
    # ===== PHASE 1: LOCALIZATION (MCL) =====
    if current_phase == PHASE_LOCALIZATION:
        u = np.array([0.0, 0.0])
        gt_pose = sample_motion_velocity_model(gt_pose, u, dt=0.1, alphas=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        
        while gt_pose[2] > np.pi:
            gt_pose[2] -= 2*np.pi
        while gt_pose[2] < -np.pi:
            gt_pose[2] += 2*np.pi
        
        # Get ground truth measurement
        z_gt = compute_expected_measurements(gt_pose[0], gt_pose[1], gt_pose[2], 
                                            maze_binary, LIDAR_EFFECTIVE_RANGE, num_beams=num_beams)
        z_gt = np.clip(z_gt, 0.001, LIDAR_EFFECTIVE_RANGE)
        
        # MCL update using GridLocalization
        particles, weights = monte_carlo_localization(particles, u, z_gt, 
                                                       sample_motion_model, sensor_model,
                                                       maze_binary)
        
        # Estimate pose as best particle
        best_particle_idx = np.argmax(weights)
        est_pose_mcl = particles[best_particle_idx]
        
        # Estimate uncertainty from top 5% particles
        top_5_percent = max(1, int(0.05 * num_particles))
        top_indices = np.argsort(weights)[-top_5_percent:]
        top_particles = particles[top_indices]
        
        distances = np.sqrt((top_particles[:, 0] - est_pose_mcl[0])**2 + (top_particles[:, 1] - est_pose_mcl[1])**2)
        uncertainty = np.std(distances)
        uncertainties.append(uncertainty)
        
        # Calculate error
        error = np.linalg.norm(est_pose_mcl[:2] - gt_pose[:2])
        errors.append(error)
        
        gt_trajectory_history.append(gt_pose[:2].copy())
        est_trajectory_history.append(est_pose_mcl[:2].copy())
        
        # Auto-trigger path planning at frame 5
        if frame == 5:
                if not localization_complete:
                    ekf_slam.set_initial_pose(est_pose_mcl, uncertainty=uncertainty/100.0)
                
                # Execute path planning immediately
                print(f"\n✓ Localization converged at frame {frame}!")
                print(f"  Uncertainty: {uncertainty:.4f} m")
                print(f"  Position error: {error:.4f} m")
                print(f"\n  Planning path from ({est_pose_mcl[0]:.3f}, {est_pose_mcl[1]:.3f}) to ({goal_pos[0]:.3f}, {goal_pos[1]:.3f})")
                start_x_px = int(est_pose_mcl[0] / scale_x)
                start_y_px = int(est_pose_mcl[1] / scale_y)
                end_x_px_plan = int(goal_pos[0] / scale_x)
                end_y_px_plan = int(goal_pos[1] / scale_y)
                
                trajectory_px = ekf_slam.generate_trajectory(start_x_px, start_y_px, end_x_px_plan, end_y_px_plan, 
                                                            smooth=True, smoothing_factor=0.5)
                
                if trajectory_px is not None:
                    trajectory = [(wp[0] * scale_x, wp[1] * scale_y, wp[2]) for wp in trajectory_px]
                    
                    # Downsample trajectory like ekf_slam_II does
                    downsample_factor = 10  # Take every 10th waypoint
                    trajectory = trajectory[::downsample_factor]
                    
                    print(f"  ✓ Path planned with {len(trajectory)} waypoints")
                    print(f"  Transition to FOLLOWING phase")
                    
                    current_phase = PHASE_FOLLOWING
                    current_waypoint_idx = 0
                    localization_complete = True
                else:
                    print(f"  ✗ Path planning failed")
    
    # ===== PHASE 2: PATH PLANNING =====
    elif current_phase == PHASE_PATH_PLANNING:
        # This phase should not be reached since path planning is done immediately
        pass
    
    # ===== PHASE 3: FOLLOWING (using ekf_slam_II logic) =====
    elif current_phase == PHASE_FOLLOWING:
        if trajectory is None or len(trajectory) == 0:
            return []
        
        # Get current target waypoint (ekf_slam_II approach)
        if current_waypoint_idx >= len(trajectory) - 1:
            # Reached goal, hold position
            current_wp = trajectory[-1]
            next_wp = trajectory[-1]
        else:
            current_wp = trajectory[current_waypoint_idx]
            next_wp = trajectory[current_waypoint_idx + 1]
        
        # Calculate velocity toward next waypoint (ekf_slam_II approach)
        dx = next_wp[0] - current_wp[0]
        dy = next_wp[1] - current_wp[1]
        dtheta = next_wp[2] - current_wp[2]
        
        # Normalize angle difference
        while dtheta > np.pi:
            dtheta -= 2*np.pi
        while dtheta < -np.pi:
            dtheta += 2*np.pi
        
        # Convert to velocity commands (ekf_slam_II approach)
        dt = 0.1
        v = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0  # in cm/s
        w = dtheta / dt if dt > 0 else 0  # in rad/s
        
        # Get estimated robot position for weak heading correction (ekf_slam_II approach)
        # Use MCL for first 10 frames, then switch to EKF-SLAM
        if frame < 10:
            est_pose_current = est_pose_mcl.copy()  # Use MCL estimate
        else:
            est_pose_current = est_pose.copy()  # Use EKF-SLAM estimate
        
        # Calculate heading error to current waypoint
        waypoint_direction = np.arctan2(current_wp[1] - est_pose_current[1], 
                                         current_wp[0] - est_pose_current[0])
        heading_error = waypoint_direction - est_pose_current[2]
        
        # Normalize heading error to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2*np.pi
        while heading_error < -np.pi:
            heading_error += 2*np.pi
        
        # Very weak PI-controller on heading only (ekf_slam_II approach)
        Kp_heading = 0.1  # Weak proportional gain for heading
        Ki_heading = 0.01  # Weak integral gain for heading
        
        # Proportional term: small correction based on heading error
        w_correction = Kp_heading * heading_error
        
        # Integral term: accumulate heading error (use frame as proxy for time)
        integral_heading = frame * heading_error * Ki_heading if frame > 0 else 0
        w_integral = Ki_heading * integral_heading * 0.0001  # Scale down significantly
        
        # Apply very weak heading corrections (only 5% of calculated correction)
        w = w + 0.05 * (w_correction + w_integral)
        
        dist_to_waypoint = np.linalg.norm(est_pose_current[:2] - np.array([current_wp[0], current_wp[1]]))
        
        u = np.array([v, w])
        
        if dist_to_waypoint < waypoint_tolerance and current_waypoint_idx < len(trajectory) - 1:
            current_waypoint_idx += 1
        
        # Debug: print every 10 frames
        if frame % 10 == 0:
            print(f"\rFrame {frame:4d}: WP {current_waypoint_idx}/{len(trajectory)}, dist_to_wp={dist_to_waypoint:.2f}cm", end='', flush=True)
        
        # Get LiDAR measurement from GT pose
        z_gt = compute_expected_measurements(gt_pose[0], gt_pose[1], gt_pose[2],
                                            maze_binary, LIDAR_EFFECTIVE_RANGE, num_beams=num_beams)
        z_gt = z_gt + np.random.normal(0, 0.01, len(z_gt))
        z_gt = np.clip(z_gt, 0.01, LIDAR_EFFECTIVE_RANGE)
        
        # ===== MCL ONLY FOR FIRST 10 FRAMES =====
        if frame < 10:
            particles, weights = monte_carlo_localization(particles, u, z_gt,
                                                           sample_motion_model, sensor_model,
                                                           maze_binary)
            
            # Estimate pose as best particle
            best_particle_idx = np.argmax(weights)
            est_pose_mcl = particles[best_particle_idx]
            
            # Calculate uncertainty from top 5% particles
            top_5_percent = max(1, int(0.05 * num_particles))
            top_indices = np.argsort(weights)[-top_5_percent:]
            top_particles = particles[top_indices]
            distances = np.sqrt((top_particles[:, 0] - est_pose_mcl[0])**2 + (top_particles[:, 1] - est_pose_mcl[1])**2)
            uncertainty = np.std(distances)
            uncertainties.append(uncertainty)
        
        # ===== EKF-SLAM UPDATE (ALWAYS AFTER FRAME 5) =====
        ekf_slam.update(u, z_gt)
        
        est_pose = ekf_slam.get_estimated_pose()
        pose_uncertainty = ekf_slam.get_pose_uncertainty()
        
        # Ground truth moves
        gt_pose = sample_motion_velocity_model(gt_pose, u, dt=0.1)
        while gt_pose[2] > np.pi:
            gt_pose[2] -= 2*np.pi
        while gt_pose[2] < -np.pi:
            gt_pose[2] += 2*np.pi
        
        gt_trajectory_history.append(gt_pose[:2].copy())
        # Use EKF-SLAM pose for history after frame 5
        est_trajectory_history.append(est_pose[:2].copy())
        
        error = np.linalg.norm(est_pose[:2] - gt_pose[:2])
        errors.append(error)

    
    # ===== VISUALIZATION =====
    if not args.headless:
        # GT position
        gt_pos_plot.set_data([gt_pose[0]], [gt_pose[1]])
        
        # Estimated position (update based on phase)
        if current_phase == PHASE_LOCALIZATION or (current_phase == PHASE_FOLLOWING and frame < 10):
            est_pos_plot.set_data([est_pose_mcl[0]], [est_pose_mcl[1]])
            est_heading_x = est_pose_mcl[0] + 0.15 * np.cos(est_pose_mcl[2])
            est_heading_y = est_pose_mcl[1] + 0.15 * np.sin(est_pose_mcl[2])
        else:
            est_pos_plot.set_data([est_pose[0]], [est_pose[1]])
            est_heading_x = est_pose[0] + 0.15 * np.cos(est_pose[2])
            est_heading_y = est_pose[1] + 0.15 * np.sin(est_pose[2])
        
        # GT heading arrow
        gt_heading_x = gt_pose[0] + 0.15 * np.cos(gt_pose[2])
        gt_heading_y = gt_pose[1] + 0.15 * np.sin(gt_pose[2])
        gt_heading.set_data([gt_pose[0], gt_heading_x], [gt_pose[1], gt_heading_y])
        
        # Est heading arrow
        if current_phase == PHASE_LOCALIZATION or (current_phase == PHASE_FOLLOWING and frame < 10):
            est_heading.set_data([est_pose_mcl[0], est_heading_x], [est_pose_mcl[1], est_heading_y])
        else:
            est_heading.set_data([est_pose[0], est_heading_x], [est_pose[1], est_heading_y])
        
        # Planned trajectory
        if trajectory is not None and len(trajectory) > 0:
            traj_array = np.array([(wp[0], wp[1]) for wp in trajectory])
            trajectory_line.set_data(traj_array[:, 0], traj_array[:, 1])
        
        # Status text
        phase_names = {
            PHASE_LOCALIZATION: "Localization (MCL)",
            PHASE_PATH_PLANNING: "Path Planning",
            PHASE_FOLLOWING: "Following (EKF-SLAM)"
        }
        phase_name = phase_names.get(current_phase, "Unknown")
        
        if len(uncertainties) > 0:
            status = (f"Frame: {frame} | Phase: {phase_name}\n"
                     f"Uncertainty: {uncertainties[-1]:.4f} m\n"
                     f"Error: {errors[-1]:.4f} m\n"
                     f"GT Pos: ({gt_pose[0]:.3f}, {gt_pose[1]:.3f})\n"
                     f"Est Pos: ({est_pose_mcl[0]:.3f}, {est_pose_mcl[1]:.3f})" if current_phase == PHASE_LOCALIZATION
                     else f"Est Pos: ({est_pose[0]:.3f}, {est_pose[1]:.3f})")
            
            status_text.set_text(status)
    
    return [gt_pos_plot, est_pos_plot, trajectory_line, status_text, gt_heading, est_heading]

print("Creating animation...")

def frame_generator():
    frame = 0
    while frame < max_frames:
        yield frame
        frame += 1
        if current_phase == PHASE_FOLLOWING and len(est_trajectory_history) > 0:
            dist_to_goal = np.linalg.norm(np.array(est_trajectory_history[-1]) - goal_pos)
            if dist_to_goal < 0.05:
                break

try:
    anim = FuncAnimation(fig, update, frames=frame_generator(),
                         interval=50, blit=True, repeat=False, cache_frame_data=False)

    # If requested, save the animation to GIF (works in headless and non-headless)
    if args.save_gif:
        try:
            writer = PillowWriter(fps=20)
            anim.save(args.save_gif, writer=writer)
            print(f"Saved animation to {args.save_gif}")
        except Exception as e:
            print(f"Error saving GIF: {e}")

    if not args.headless:
        print("Starting animation... (close window when done)")
        plt.show()
    else:
        # Headless: run through frames to collect stats/prints (animation already saved if requested)
        print("Running in headless mode (no visualization)")
        print("="*70)
        start_total = time.time()
        for frame_num in frame_generator():
            frame_start = time.time()
            update(frame_num)
            frame_time = (time.time() - frame_start) * 1000
            if len(errors) > 0:
                print(f"Frame {frame_num:4d} | Time: {frame_time:6.2f} ms | Error: {errors[-1]:.4f} m | Uncertainty: {uncertainties[-1]:.4f} m")

        total_time = (time.time() - start_total) * 1000
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)
        print(f"Total time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
        print(f"Frames completed: {len(errors)}")
        if len(errors) > 0:
            avg_frame_time = total_time / len(errors)
            print(f"Average frame time: {avg_frame_time:.2f} ms")
            print(f"Final uncertainty: {uncertainties[-1]:.4f} m")
            print(f"Final error: {errors[-1]:.4f} m")
        print("="*70)
except Exception as e:
    print(f"Error in animation: {e}")

# Final statistics
print("\n" + "="*70)
print("SIMULATION COMPLETE - STATISTICAL ANALYSIS")
print("="*70)

if len(errors) > 0:
    print(f"\nPosition Error Statistics (meters):")
    print(f"  Mean: {np.mean(errors):.4f} m")
    print(f"  Std: {np.std(errors):.4f} m")
    print(f"  Min: {np.min(errors):.4f} m")
    print(f"  Max: {np.max(errors):.4f} m")
    print(f"  Final: {errors[-1]:.4f} m")

if len(uncertainties) > 0:
    print(f"\nUncertainty Statistics:")
    print(f"  Initial: {uncertainties[0]:.4f} m")
    print(f"  Final: {uncertainties[-1]:.6f}")
    print(f"  Min: {np.min(uncertainties):.4f} m")
    if np.any(np.array(uncertainties) < 0.03):
        convergence_frame = np.argmax(np.array(uncertainties) < 0.03)
        print(f"  Localization time: {convergence_frame} frames")
    else:
        print(f"  Localization time: Not achieved")

if localization_complete:
    print(f"\n✓ Localization successful!")
    print(f"✓ Path planning and following initiated")
else:
    print(f"\n✗ Localization not completed within {len(uncertainties)} frames")

print("="*70)