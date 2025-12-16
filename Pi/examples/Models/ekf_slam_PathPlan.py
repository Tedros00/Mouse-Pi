"""
EKF-SLAM Animation - Uses Maze from TestMaze.JPG
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from EKF_SLAM import EKF_SLAM

print("Loading ground truth maze...")
# Use relative path for TestMaze.JPG
maze_path = os.path.join(os.path.dirname(__file__), '..', 'SimulationEnv', 'TestMaze.JPG')
maze_image = np.array(Image.open(maze_path).convert('L'))
maze_binary = (maze_image > 127).astype(np.uint8)
maze_height_px, maze_width_px = maze_binary.shape

# Real world dimensions (in CENTIMETERS)
REAL_WIDTH = 180.0  # centimeters (1.8m)
REAL_HEIGHT = 200.0  # centimeters (2.0m)
LIDAR_MAX_RANGE = 1500.0  # centimeters (15m)

# Calculate scale factors
scale_x = REAL_WIDTH / maze_width_px
scale_y = REAL_HEIGHT / maze_height_px

print(f"Maze dimensions: {maze_width_px}x{maze_height_px} pixels")
print(f"Real world dimensions: {REAL_WIDTH}x{REAL_HEIGHT} centimeters ({REAL_WIDTH/100}m x {REAL_HEIGHT/100}m)")
print(f"LiDAR max range: {LIDAR_MAX_RANGE} cm")
print(f"Scale: {scale_x:.4f} cm/pixel (x), {scale_y:.4f} cm/pixel (y)")

def compute_expected_measurements(x, y, angle, map_data, max_range, num_beams=360):
    """
    Compute expected LiDAR measurements by raycasting against map.
    Works in pixel coordinates of the maze image.
    
    Parameters:
    - x, y: robot position (in real world cm)
    - angle: robot heading (radians)
    - map_data: occupancy grid (1=free, 0=occupied, in pixels)
    - max_range: maximum LiDAR range (cm)
    - num_beams: number of beams
    
    Returns:
    - array of range measurements for each beam (in cm)
    """
    # Convert real world position to pixel coordinates
    x_px = x / scale_x
    y_px = y / scale_y
    
    measurements = np.zeros(num_beams)
    max_range_px = max_range / min(scale_x, scale_y)  # Convert max range to pixels
    
    for i in range(num_beams):
        # Beam angle relative to robot heading
        beam_angle = angle + (i - num_beams/2) * (2 * np.pi / num_beams)
        
        cos_angle = np.cos(beam_angle)
        sin_angle = np.sin(beam_angle)
        
        # Step along ray in pixel space
        for dist_px in np.arange(0.1, max_range_px, 0.5):
            check_x_px = x_px + dist_px * cos_angle
            check_y_px = y_px + dist_px * sin_angle
            
            col = int(check_x_px)
            row = int(check_y_px)
            
            # Out of bounds = hit boundary
            if col < 0 or col >= map_data.shape[1] or row < 0 or row >= map_data.shape[0]:
                measurements[i] = dist_px * min(scale_x, scale_y)
                break
            
            # Hit obstacle (0 = black = occupied)
            if map_data[row, col] == 0:
                measurements[i] = dist_px * min(scale_x, scale_y)
                break
        else:
            # Never hit anything, use max range
            measurements[i] = max_range
    
    return measurements

print("Initializing EKF-SLAM with real world coordinates...")
# Use coarser grid for speed (2cm per cell = 90x100 cells)
cell_size_cm = 2.0  # 2cm per cell
grid_width = int(REAL_WIDTH / cell_size_cm)
grid_height = int(REAL_HEIGHT / cell_size_cm)
grid_shape = (grid_height, grid_width)
print(f"Using grid resolution: {grid_shape} ({cell_size_cm}cm per cell, {grid_width}x{grid_height} cells)")

# DO NOT upsample maze - keep original for correct raycasting against original trajectory
print(f"Using original maze for raycasting: {maze_binary.shape}")

ekf_slam = EKF_SLAM(grid_shape=grid_shape, 
                    grid_bounds=((0, REAL_WIDTH), (0, REAL_HEIGHT)),
                    compute_from_map=compute_expected_measurements, 
                    dt=0.1,
                    motion_noise_std=(0.01, 0.005), 
                    measurement_noise_std=0.3)

# Load ground truth maze
ekf_slam.load_ground_truth_maze(maze_path)

# Generate trajectory in maze (using pixel coordinates from the image)
print("Generating trajectory from maze (in pixel coordinates)...")
start_x_px = 150 
start_y_px = 1200 
end_x_px = 1000 
end_y_px = 350 

# Ensure start and end are within bounds
print(f"Using start: ({start_x_px}, {start_y_px}), end: ({end_x_px}, {end_y_px})")

trajectory_px = ekf_slam.generate_trajectory(start_x_px, start_y_px, end_x_px, end_y_px, smooth=True, smoothing_factor=0.5)

if trajectory_px is None:
    print("Failed to generate trajectory!")
    sys.exit(1)

# Convert trajectory from pixel coordinates to real world coordinates
trajectory = [(wp[0] * scale_x, wp[1] * scale_y, wp[2]) for wp in trajectory_px]

# Downsample trajectory: take every Nth waypoint to reduce steps
downsample_factor = 10  # Take every 10th waypoint (less aggressive to preserve smooth interpolation)
trajectory = trajectory[::downsample_factor]

# Convert trajectory to format compatible with visualization
trajectory_array = np.array([(wp[0], wp[1]) for wp in trajectory])
initial_pose = np.array([trajectory[0][0], trajectory[0][1], trajectory[0][2]])
ekf_slam.set_initial_pose(initial_pose, uncertainty=0.05)

# Cell size tolerance for waypoint switching
waypoint_tolerance = 5.0  # 5cm tolerance for advancing to next waypoint

print(f"Generated trajectory with {len(trajectory)} waypoints")
print(f"Trajectory start: ({trajectory[0][0]:.2f}, {trajectory[0][1]:.2f}) centimeters")
print(f"Trajectory end: ({trajectory[-1][0]:.2f}, {trajectory[-1][1]:.2f}) centimeters")
if len(trajectory) > 1:
    avg_dist = np.mean([np.linalg.norm(np.array(trajectory[i][:2]) - np.array(trajectory[i-1][:2])) for i in range(1, min(10, len(trajectory)))])
    print(f"Average distance between first 10 waypoints: {avg_dist:.3f} cm")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Ground Truth subplot
ax_gt = fig.add_subplot(gs[0, 0])
ax_gt.set_xlim(0, REAL_WIDTH)
ax_gt.set_ylim(0, REAL_HEIGHT)
ax_gt.set_aspect('equal')
ax_gt.invert_yaxis()
ax_gt.set_title('Ground Truth - Maze + Trajectory')
ax_gt.set_xlabel('X (centimeters)')
ax_gt.set_ylabel('Y (centimeters)')
# Display ground truth maze (scaled to real world)
ax_gt.imshow(maze_binary, extent=[0, REAL_WIDTH, REAL_HEIGHT, 0], cmap='gray', origin='upper', alpha=0.7)
# Display ground truth trajectory
ax_gt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b--', alpha=0.6, linewidth=2, label='GT Trajectory')
gt_pos, = ax_gt.plot([], [], 'bo', markersize=10, label='GT Position')
gt_heading, = ax_gt.plot([], [], 'b-', linewidth=2)
gt_path, = ax_gt.plot([], [], 'b-', alpha=0.6, linewidth=1)
gt_uncertainty = ax_gt.scatter([], [], c='blue', s=200, alpha=0.1, marker='o')
gt_rays, = ax_gt.plot([], [], 'r--', linewidth=0.5, alpha=0.4, label='GT LiDAR Rays')
step_text = ax_gt.text(0.02, 0.05, '', transform=ax_gt.transAxes, fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        verticalalignment='bottom')
ax_gt.legend(loc='upper right', fontsize=8)

# Estimated pose subplot
ax_est = fig.add_subplot(gs[0, 1])
ax_est.set_xlim(0, REAL_WIDTH)
ax_est.set_ylim(0, REAL_HEIGHT)
ax_est.set_aspect('equal')
ax_est.invert_yaxis()
ax_est.set_title('Estimated Pose - Maze + Trajectory')
ax_est.set_xlabel('X (centimeters)')
ax_est.set_ylabel('Y (centimeters)')
# Display maze
ax_est.imshow(maze_binary, extent=[0, REAL_WIDTH, REAL_HEIGHT, 0], cmap='gray', origin='upper', alpha=0.7)
# Display trajectory
ax_est.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b--', alpha=0.3, linewidth=1, label='GT Traj')
est_pos, = ax_est.plot([], [], 'ro', markersize=10, label='Est Position')
est_heading, = ax_est.plot([], [], 'r-', linewidth=2)
est_path, = ax_est.plot([], [], 'r-', alpha=0.6, linewidth=1)
est_uncertainty = ax_est.scatter([], [], c='red', s=200, alpha=0.1, marker='o')
est_rays, = ax_est.plot([], [], 'r--', linewidth=0.5, alpha=0.4, label='Est LiDAR Rays')
ax_est.legend(loc='upper right', fontsize=8)

ax_error = fig.add_subplot(gs[0, 2])
ax_error.set_title('Position Error (centimeters)')
ax_error.set_xlabel('Step')
ax_error.set_ylabel('Error (cm)')
ax_error.grid(True, alpha=0.3)
error_line, = ax_error.plot([], [], 'r-', linewidth=2)

ax_gt_map = fig.add_subplot(gs[1, 0])
ax_gt_map.set_title('Ground Truth Map (from Maze)')
ax_gt_map.set_aspect('equal')
ax_gt_map.invert_yaxis()
ax_gt_map.set_xlabel('X (centimeters)')
ax_gt_map.set_ylabel('Y (centimeters)')
gt_map_im = ax_gt_map.imshow(maze_binary, extent=[0, REAL_WIDTH, REAL_HEIGHT, 0], cmap='gray', origin='upper', vmin=0, vmax=1, alpha=0.8)

ax_est_map = fig.add_subplot(gs[1, 1])
ax_est_map.set_title('EKF-SLAM Estimated Map')
ax_est_map.set_aspect('equal')
ax_est_map.invert_yaxis()
ax_est_map.set_xlabel('X (centimeters)')
ax_est_map.set_ylabel('Y (centimeters)')
est_map_im = ax_est_map.imshow(np.zeros(maze_binary.shape), extent=[0, REAL_WIDTH, REAL_HEIGHT, 0], cmap='hot', origin='upper', vmin=0, vmax=1, alpha=0.8)
plt.colorbar(est_map_im, ax=ax_est_map, label='Occupancy')

ax_uncertainty = fig.add_subplot(gs[1, 2])
ax_uncertainty.set_title('Pose Uncertainty Over Time')
ax_uncertainty.set_xlabel('Step')
ax_uncertainty.set_ylabel('Uncertainty (σ)')
ax_uncertainty.grid(True, alpha=0.3)
uncertainty_line, = ax_uncertainty.plot([], [], 'g-', linewidth=2)

gt_trajectory_history = []
est_trajectory_history = []
errors = []
uncertainties = []
max_frames = 10000  # Safety limit, but animation should end when est reaches goal
current_waypoint_idx = 0  # Track current target waypoint
accumulated_lidar_points = {}  # Dict to store merged LiDAR points by cell: (cell_x, cell_y) -> [x, y]

def update(frame):
    """
    EKF-SLAM update visualization using maze trajectory.
    Robot follows waypoints, switching to next when within cell_size tolerance.
    Animation continues until estimated robot reaches the goal.
    """
    global gt_trajectory_history, est_trajectory_history, current_waypoint_idx, accumulated_lidar_points
    
    # Stop if we've hit the safety limit
    if frame >= max_frames:
        return [gt_pos, gt_heading, gt_path, est_pos, est_heading, est_path, 
                error_line, est_map_im, uncertainty_line, step_text]
    
    # Get current target waypoint
    if current_waypoint_idx >= len(trajectory) - 1:
        # Reached goal, hold position
        current_wp = trajectory[-1]
        next_wp = trajectory[-1]
    else:
        current_wp = trajectory[current_waypoint_idx]
        next_wp = trajectory[current_waypoint_idx + 1]
    
    # Ground truth is at current waypoint
    gt_pose = np.array([current_wp[0], current_wp[1], current_wp[2]])
    
    # Calculate velocity toward next waypoint
    dx = next_wp[0] - current_wp[0]
    dy = next_wp[1] - current_wp[1]
    dtheta = next_wp[2] - current_wp[2]
    
    # Normalize angle difference
    while dtheta > np.pi:
        dtheta -= 2*np.pi
    while dtheta < -np.pi:
        dtheta += 2*np.pi
    
    # Convert to velocity commands (same as before)
    dt = 0.1
    v = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0  # in cm/s
    w = dtheta / dt if dt > 0 else 0  # in rad/s
    
    # Get estimated robot position for weak heading correction
    est_pose_current = ekf_slam.get_estimated_pose()
    
    # Calculate heading error to current waypoint
    waypoint_direction = np.arctan2(current_wp[1] - est_pose_current[1], 
                                     current_wp[0] - est_pose_current[0])
    heading_error = waypoint_direction - est_pose_current[2]
    
    # Normalize heading error to [-pi, pi]
    while heading_error > np.pi:
        heading_error -= 2*np.pi
    while heading_error < -np.pi:
        heading_error += 2*np.pi
    
    # Very weak PI-controller on heading only
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
    
    import time
    
    # Simulate LiDAR measurement from GT pose (for EKF-SLAM)
    # Generate 360 beams, raycast against GT maze
    t0 = time.time()
    z = compute_expected_measurements(gt_pose[0], gt_pose[1], gt_pose[2], 
                                            maze_binary, LIDAR_MAX_RANGE, num_beams=360)
    z = z + np.random.normal(0, 1.0, len(z))
    z = np.clip(z, 0.1, LIDAR_MAX_RANGE)
    t_sensor = time.time() - t0
    
    # ===== EKF-SLAM UPDATE =====
    t1 = time.time()
    result = ekf_slam.update(u, z)
    t_ekf = time.time() - t1
    
    # ===== GET RESULTS =====
    # Get actual estimated pose from EKF (has uncertainty from motion + measurement noise)
    est_pose = ekf_slam.get_estimated_pose()
    est_map = ekf_slam.get_estimated_map_probability()
    pose_uncertainty = ekf_slam.get_pose_uncertainty()
    
    # Generate LiDAR from estimated pose for visualization
    z_est = compute_expected_measurements(est_pose[0], est_pose[1], est_pose[2], 
                                          maze_binary, LIDAR_MAX_RANGE, num_beams=360)
    
    # Print timing every 20 frames
    if frame % 20 == 0:
        print(f" | Sensor: {t_sensor*1000:.1f}ms, EKF: {t_ekf*1000:.1f}ms", flush=True)
    
    # Use GT pose from trajectory for visualization
    gt_trajectory_history.append(gt_pose[:2])
    est_trajectory_history.append(est_pose[:2])
    error = np.linalg.norm(est_pose[:2] - gt_pose[:2])
    errors.append(error)
    uncertainties.append(pose_uncertainty)
    
    # ===== VISUALIZATION ===== 
    # Ground truth (from trajectory waypoint)
    gt_pos.set_data([gt_pose[0]], [gt_pose[1]])
    arrow_len = 20.0
    gt_heading.set_data([gt_pose[0], gt_pose[0] + arrow_len * np.cos(gt_pose[2])],
                        [gt_pose[1], gt_pose[1] + arrow_len * np.sin(gt_pose[2])])
    if len(gt_trajectory_history) > 1:
        gt_traj = np.array(gt_trajectory_history)
        gt_path.set_data(gt_traj[:, 0], gt_traj[:, 1])
    
    try:
        gt_uncertainty.set_offsets([[gt_pose[0], gt_pose[1]]])
        gt_uncertainty.set_sizes([pose_uncertainty * 500])
    except:
        pass
    
    # Estimated pose
    est_pos.set_data([est_pose[0]], [est_pose[1]])
    est_heading.set_data([est_pose[0], est_pose[0] + arrow_len * np.cos(est_pose[2])],
                         [est_pose[1], est_pose[1] + arrow_len * np.sin(est_pose[2])])
    if len(est_trajectory_history) > 1:
        est_traj = np.array(est_trajectory_history)
        est_path.set_data(est_traj[:, 0], est_traj[:, 1])
    
    try:
        est_uncertainty.set_offsets([[est_pose[0], est_pose[1]]])
        est_uncertainty.set_sizes([pose_uncertainty * 500])
    except:
        pass
    
    # Draw GT LiDAR rays (dashed red)
    gt_rays_x = []
    gt_rays_y = []
    num_beams = len(z)
    for i in range(num_beams):
        beam_angle = gt_pose[2] + (i - num_beams/2) * (2 * np.pi / num_beams)
        end_x = gt_pose[0] + z[i] * np.cos(beam_angle)
        end_y = gt_pose[1] + z[i] * np.sin(beam_angle)
        gt_rays_x.extend([gt_pose[0], end_x, None])
        gt_rays_y.extend([gt_pose[1], end_y, None])
    if gt_rays_x:
        gt_rays.set_data(gt_rays_x, gt_rays_y)
    
    # Draw Estimated LiDAR rays (dashed red)
    est_rays_x = []
    est_rays_y = []
    for i in range(len(z_est)):
        beam_angle = est_pose[2] + (i - len(z_est)/2) * (2 * np.pi / len(z_est))
        end_x = est_pose[0] + z_est[i] * np.cos(beam_angle)
        end_y = est_pose[1] + z_est[i] * np.sin(beam_angle)
        est_rays_x.extend([est_pose[0], end_x, None])
        est_rays_y.extend([est_pose[1], end_y, None])
    if est_rays_x:
        est_rays.set_data(est_rays_x, est_rays_y)
    
    # Error over time
    if errors:
        error_line.set_data(np.arange(len(errors)), errors)
        ax_error.set_xlim(0, max(len(errors), 10))
        ax_error.set_ylim(0, max(np.max(errors) * 1.1, 1.0))
    
    # Estimated map
    est_map_im.set_data(est_map)
    
    # Display step number
    step_text.set_text(f'Step: {frame}/{len(trajectory)}')
    
    # Uncertainty over time
    # Check if estimated robot has reached the goal
    goal_pos = np.array(trajectory[-1][:2])
    dist_to_goal = np.linalg.norm(est_pose[:2] - goal_pos)
    
    if uncertainties:
        uncertainty_line.set_data(np.arange(len(uncertainties)), uncertainties)
        ax_uncertainty.set_xlim(0, max(len(uncertainties), 10))
        ax_uncertainty.set_ylim(0, max(np.max(uncertainties) * 1.1, 1.0))
    
    # Update step text to show distance to goal for estimated robot
    step_text.set_text(f'Step: {frame}\nEst dist to goal: {dist_to_goal:.2f} cm')
    
    return [gt_pos, gt_heading, gt_path, est_pos, est_heading, est_path, 
            error_line, est_map_im, uncertainty_line, step_text, gt_rays, est_rays]

print("Creating animation...")
# Create dynamic frame generator that stops when estimated robot reaches goal
def frame_generator():
    frame = 0
    goal_pos = np.array(trajectory[-1][:2])
    while frame < max_frames:
        yield frame
        # Check if estimated robot reached goal (within 5cm) and GT already finished
        if frame > len(trajectory) and len(est_trajectory_history) > 0:
            est_pos_current = est_trajectory_history[-1]
            dist_to_goal = np.linalg.norm(np.array(est_pos_current) - goal_pos)
            if dist_to_goal < 5.0:  # Within 5cm of goal
                break
        frame += 1

anim = FuncAnimation(fig, update, frames=frame_generator(), 
                     interval=50, blit=True, repeat=False)

plt.suptitle('EKF-SLAM: Real-Time Localization and Mapping on Maze', fontsize=14, fontweight='bold')
print("Starting animation... (close window when done)")
plt.show()

print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)
if len(errors) > 0:
    print(f"\nPosition Error Statistics (centimeters):")
    print(f"  Mean: {np.mean(errors) * scale_x:.4f} cm")
    print(f"  Std: {np.std(errors) * scale_x:.4f} cm")
    print(f"  Min: {np.min(errors) * scale_x:.4f} cm")
    print(f"  Max: {np.max(errors) * scale_x:.4f} cm")
    print(f"  Final: {errors[-1] * scale_x:.4f} cm")
    
if len(uncertainties) > 0:
    print(f"\nPose Uncertainty Statistics:")
    print(f"  Mean: {np.mean(uncertainties):.4f}")
    print(f"  Std: {np.std(uncertainties):.4f}")
    print(f"  Min: {np.min(uncertainties):.4f}")
    print(f"  Max: {np.max(uncertainties):.4f}")
    print(f"  Final: {uncertainties[-1]:.4f}")

if len(errors) > 0:
    gt_traj = np.array(gt_trajectory_history)
    trajectory_dist = np.sum(np.linalg.norm(np.diff(gt_traj, axis=0), axis=1))
    trajectory_dist_cm = trajectory_dist * scale_x
    print(f"\nTrajectory: {trajectory_dist_cm:.2f} centimeters total ({trajectory_dist_cm/100:.2f} m)")
    error_percent = (np.mean(errors) * scale_x / trajectory_dist_cm) * 100 if trajectory_dist_cm > 0 else 0
    print(f"Error as % of distance: {error_percent:.2f}%")
print("="*70)

# Save final OGM and trajectory to out/examples folder
print("\nSaving results...")
out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out', 'examples')
os.makedirs(out_dir, exist_ok=True)

# Save the final estimated map (OGM)
from OGM import log_odds_to_occupancy_probability, save_ogm_to_pgm
estimated_map = log_odds_to_occupancy_probability(ekf_slam.map_grid)
estimated_map_image = (estimated_map * 255).astype(np.uint8)
ogm_output_path = os.path.join(out_dir, 'ekf_slam_final_ogm.pgm')
save_ogm_to_pgm(ekf_slam.map_grid, ogm_output_path)
print(f"✓ Final OGM saved to: {ogm_output_path}")

# Save the robot's actual trajectory (estimated poses, NOT the planned path)
trajectory_output_path = os.path.join(out_dir, 'ekf_slam_robot_trajectory.txt')
with open(trajectory_output_path, 'w') as f:
    f.write("Timestep X Y theta\n")
    for timestep, (x, y, theta) in enumerate(est_trajectory_history):
        f.write(f"{timestep} {x:.4f} {y:.4f} {theta:.4f}\n")
print(f"✓ Robot trajectory saved to: {trajectory_output_path}")
print(f"  Total timesteps: {len(est_trajectory_history)}")
if len(est_trajectory_history) > 0:
    print(f"  Start position: ({est_trajectory_history[0][0]:.2f}, {est_trajectory_history[0][1]:.2f})")
    print(f"  End position: ({est_trajectory_history[-1][0]:.2f}, {est_trajectory_history[-1][1]:.2f})")

