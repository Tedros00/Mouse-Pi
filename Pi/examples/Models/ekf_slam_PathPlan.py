"""
EKF-SLAM with Monte Carlo Localization (MCL) - Random Initial Position
Key difference from ekf_slam_II: Robot starts at random position and must localize itself.
Once localized (uncertainty < 3cm), performs path planning and follows trajectory to goal.
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
args = parser.parse_args()

# Only import matplotlib if not headless
if not args.headless:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from EKF_SLAM import EKF_SLAM
from GridLocalization import monte_carlo_localization
from ProbabilisticMotionModel import sample_motion_velocity_model
from ProbabilisticSensorModel import beam_range_finder_model, scan_likelihood

print("Loading ground truth maze...")
# Use relative path for TestMaze.JPG
maze_path = os.path.join(os.path.dirname(__file__), '..', 'SimulationEnv', 'TestMaze.JPG')
maze_image = np.array(Image.open(maze_path).convert('L'))
maze_binary = (maze_image > 127).astype(np.uint8)
maze_height_px, maze_width_px = maze_binary.shape

# Real world dimensions (in METERS)
REAL_WIDTH = 1.8  # meters
REAL_HEIGHT = 2.0  # meters
LIDAR_MAX_RANGE = 15.0  # meters
# Effective LiDAR range clipped to maze diagonal
LIDAR_EFFECTIVE_RANGE = np.sqrt(REAL_WIDTH**2 + REAL_HEIGHT**2) * 1.5  # 1.5x diagonal

# Calculate scale factors (convert from pixels to meters)
scale_x = REAL_WIDTH / maze_width_px
scale_y = REAL_HEIGHT / maze_height_px

print(f"Maze dimensions: {maze_width_px}x{maze_height_px} pixels")
print(f"Real world dimensions: {REAL_WIDTH}x{REAL_HEIGHT} meters")
print(f"LiDAR max range: {LIDAR_MAX_RANGE} m")
print(f"LiDAR effective range: {LIDAR_EFFECTIVE_RANGE:.2f} m")
print(f"Scale: {scale_x:.6f} m/pixel (x), {scale_y:.6f} m/pixel (y)")

def compute_expected_measurements(x, y, angle, map_data, max_range, num_beams=360):
    """
    Compute expected LiDAR measurements by raycasting against map.
    Raycasting works in meter coordinates, map lookup uses pixel coordinates.
    
    Parameters:
    - x, y: robot position (in meters)
    - angle: robot heading (radians)
    - map_data: occupancy grid (1=free, 0=occupied, in pixels)
    - max_range: maximum LiDAR range (meters)
    - num_beams: number of beams
    
    Returns:
    - array of range measurements for each beam (in meters)
    """
    measurements = np.zeros(num_beams)
    step_size_m = 0.01  # Step in meters for raycasting precision
    angles = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)
    
    for i in range(num_beams):
        # Beam angle relative to robot heading
        beam_angle = angle + angles[i]
        
        cos_angle = np.cos(beam_angle)
        sin_angle = np.sin(beam_angle)
        
        # Step along ray in meter space
        for dist_m in np.arange(step_size_m, max_range, step_size_m):
            # Current point in meters
            check_x_m = x + dist_m * cos_angle
            check_y_m = y + dist_m * sin_angle
            
            # Convert to pixel coordinates for map lookup
            check_x_px = int(check_x_m / scale_x)
            check_y_px = int(check_y_m / scale_y)
            
            # Out of bounds = hit boundary
            if check_x_px < 0 or check_x_px >= map_data.shape[1] or check_y_px < 0 or check_y_px >= map_data.shape[0]:
                measurements[i] = dist_m
                break
            
            # Hit obstacle (0 = black = occupied)
            if map_data[check_y_px, check_x_px] == 0:
                measurements[i] = dist_m
                break
        else:
            # Never hit anything, use max range
            measurements[i] = max_range
    
    return measurements

def compute_from_map_wrapper(x, y, angle, map_data, max_range):
    """
    Compute expected range for a single beam at the given absolute angle.
    
    Used by the sensor model. The angle parameter is the absolute beam angle
    (not a relative offset), so we need to raycast directly without the 
    num_beams adjustment that compute_expected_measurements applies.
    """
    # Raycast in meter space
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    step_size_m = 0.01
    
    for dist_m in np.arange(step_size_m, max_range, step_size_m):
        check_x_m = x + dist_m * cos_angle
        check_y_m = y + dist_m * sin_angle
        
        # Convert to pixel coordinates for map lookup
        check_x_px = int(check_x_m / scale_x)
        check_y_px = int(check_y_m / scale_y)
        
        # Out of bounds = hit boundary
        if check_x_px < 0 or check_x_px >= map_data.shape[1] or check_y_px < 0 or check_y_px >= map_data.shape[0]:
            return dist_m
        
        # Hit obstacle
        if map_data[check_y_px, check_x_px] == 0:
            return dist_m
    
    # Never hit anything
    return max_range

print("Initializing Monte Carlo Localization...")
# Number of particles for MCL
num_particles = 500
print(f"Using {num_particles} particles for MCL")

# Initialize particles uniformly across the environment (x, y, theta)
particles = np.zeros((num_particles, 3))
particles[:, 0] = np.random.uniform(0, REAL_WIDTH, num_particles)   # x positions
particles[:, 1] = np.random.uniform(0, REAL_HEIGHT, num_particles)  # y positions
particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)   # theta orientations

# Ground truth position (random, but not known by robot)
gt_x_px = 150
gt_y_px = 1200
gt_theta = -np.pi/2

gt_pose = np.array([gt_x_px * scale_x, gt_y_px * scale_y, gt_theta])

print(f"Ground truth robot position: ({gt_pose[0]:.3f}, {gt_pose[1]:.3f}) m, heading: {gt_pose[2]:.2f} rad")
print(f"Localization task: Find position using MCL")

# End position (same as ekf_slam_II)
end_x_px = 1000
end_y_px = 350
goal_pos = np.array([end_x_px * scale_x, end_y_px * scale_y])
print(f"Goal position: ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}) m")

print("Initializing EKF-SLAM...")
cell_size_m = 0.02  # 2 cm grid cells = 0.02 m
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
waypoint_tolerance = 0.05  # 5 cm in meters

print("\n" + "="*70)
print("SIMULATION SETUP")
print("="*70)
print(f"Phase 1: Monte Carlo Localization (target uncertainty < 0.03 m)")
if not args.headless:
    print(f"Phase 2: Path planning from localized position to goal")
    print(f"Phase 3: Follow trajectory with EKF-SLAM")
print("="*70)

# Visualization setup (only if not headless)
if not args.headless:
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax_mcl = fig.add_subplot(gs[0, 0])
    ax_mcl.set_xlim(0, REAL_WIDTH)
    ax_mcl.set_ylim(0, REAL_HEIGHT)
    ax_mcl.set_aspect('equal')
    ax_mcl.invert_yaxis()
    ax_mcl.set_title('MCL Particle Filter')
    ax_mcl.set_xlabel('X (meters)')
    ax_mcl.set_ylabel('Y (meters)')
    ax_mcl.imshow(maze_binary, extent=[0, REAL_WIDTH, REAL_HEIGHT, 0], cmap='gray', origin='upper', alpha=0.7)
    particles_scatter = ax_mcl.scatter([], [], c='green', s=15, alpha=0.3, label='Particles')
    gt_pos_mcl, = ax_mcl.plot([], [], 'bo', markersize=10, label='GT Position')
    best_particle, = ax_mcl.plot([], [], 'o', color='orange', markersize=12, label='Best Particle', markerfacecolor='none', markeredgewidth=2)
    gt_lidar_rays = ax_mcl.plot([], [], 'b--', alpha=0.1, linewidth=0.5, label='GT LiDAR')[0]
    ax_mcl.legend(loc='upper right', fontsize=8)

    ax_est = fig.add_subplot(gs[0, 1])
    ax_est.set_xlim(0, REAL_WIDTH)
    ax_est.set_ylim(0, REAL_HEIGHT)
    ax_est.set_aspect('equal')
    ax_est.invert_yaxis()
    ax_est.set_title('EKF-SLAM: Estimated Pose + Goal')
    ax_est.set_xlabel('X (meters)')
    ax_est.set_ylabel('Y (meters)')
    ax_est.imshow(maze_binary, extent=[0, REAL_WIDTH, REAL_HEIGHT, 0], cmap='gray', origin='upper', alpha=0.7)
    est_pos_ekf, = ax_est.plot([], [], 'ro', markersize=10, label='Est Position')
    est_heading, = ax_est.plot([], [], 'r-', linewidth=2)
    est_path, = ax_est.plot([], [], 'r-', alpha=0.6, linewidth=1, label='Est Path')
    est_lidar_rays = ax_est.plot([], [], 'r--', alpha=0.1, linewidth=0.5, label='Est LiDAR')[0]
    goal_marker, = ax_est.plot([goal_pos[0]], [goal_pos[1]], 'b^', markersize=12, label='Goal')
    trajectory_line, = ax_est.plot([], [], 'b--', alpha=0.3, linewidth=1, label='Planned Path')
    ax_est.legend(loc='upper right', fontsize=8)

    ax_beams = fig.add_subplot(gs[0, 2])
    ax_beams.set_title('LiDAR Scan Comparison')
    ax_beams.set_xlabel('X (meters)')
    ax_beams.set_ylabel('Y (meters)')
    ax_beams.set_aspect('equal')
    ax_beams.grid(True, alpha=0.3)
    # Origin marker
    ax_beams.plot(0, 0, 'ko', markersize=8, label='Robot (Origin)', zorder=10)
    # Pre-allocate scatter containers for efficiency
    beam_scatter_gt = ax_beams.scatter([], [], c='red', s=10, alpha=0.6, label='GT Measurement')
    beam_scatter_exp = ax_beams.scatter([], [], c='blue', s=10, alpha=0.6, label='Expected (Est Particle)')
    ax_beams.legend(loc='upper right', fontsize=8)

    ax_weights = fig.add_subplot(gs[1, 0])
    ax_weights.set_title('Particle Weight Distribution')
    ax_weights.set_xlabel('Particle Index')
    ax_weights.set_ylabel('Weight')
    ax_weights.grid(True, alpha=0.3)
    ax_weights.set_ylim(0, 0.1)
    # Create bar containers once for efficient updating
    weight_bars = ax_weights.bar(np.arange(num_particles), np.zeros(num_particles), color='blue', alpha=0.7)

    ax_est_map = fig.add_subplot(gs[1, 1])
    ax_est_map.set_title('EKF-SLAM Estimated Map')
    ax_est_map.set_aspect('equal')
    ax_est_map.invert_yaxis()
    ax_est_map.set_xlabel('X (meters)')
    ax_est_map.set_ylabel('Y (meters)')
    est_map_im = ax_est_map.imshow(np.zeros(maze_binary.shape), extent=[0, REAL_WIDTH, REAL_HEIGHT, 0], 
                                   cmap='hot', origin='upper', vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(est_map_im, ax=ax_est_map, label='Occupancy')

    ax_status = fig.add_subplot(gs[1, 2])
    ax_status.axis('off')
    status_text = ax_status.text(0.1, 0.9, '', transform=ax_status.transAxes, fontsize=11,
                                verticalalignment='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# History tracking
gt_trajectory_history = []
est_trajectory_history = []
errors = []
uncertainties = []
phases_log = []
z_gt = np.zeros(360)  # Initialize for beam comparison plot
z_expected = np.zeros(360)  # Initialize for beam comparison plot
max_frames = 5000

# Headless mode - no plotting
headless = args.headless

# User control for phase transitions
ready_for_phase_2 = False
phase_2_approved = False
localization_converged = False

def listen_for_user_input():
    """Thread function to listen for user pressing ENTER"""
    global phase_2_approved
    print("\n[WAITING FOR USER INPUT]")
    print("When localization converges, press ENTER in this terminal to proceed to Phase 2")
    print("="*70)
    while not localization_converged:
        try:
            input()  # Wait for user to press ENTER
            phase_2_approved = True
            print("✓ User approved Phase 2 transition")
            break
        except EOFError:
            break

# Start input listener thread
input_thread = threading.Thread(target=listen_for_user_input, daemon=True)
input_thread.start()

# Initialize MCL estimated pose
est_pose_mcl = np.array([particles[0, 0], particles[0, 1], 0.0])

def estimate_particle_uncertainty(particles):
    """Estimate position uncertainty from particle distribution"""
    if len(particles) == 0:
        return 999.0
    mean_x = np.mean(particles[:, 0])
    mean_y = np.mean(particles[:, 1])
    distances = np.sqrt((particles[:, 0] - mean_x)**2 + (particles[:, 1] - mean_y)**2)
    # Use standard deviation as uncertainty metric
    return np.std(distances)

def update(frame):
    """
    Main simulation update loop with three phases:
    1. Localization (MCL until uncertainty < 3cm)
    2. Path planning (after localization, plan path to goal)
    3. Following (EKF-SLAM trajectory following)
    """
    global gt_pose, particles, current_phase, localization_complete, trajectory, current_waypoint_idx
    global gt_trajectory_history, est_trajectory_history, z_gt, z_expected
    global ready_for_phase_2, phase_2_approved, localization_converged, est_pose_mcl, weights
    
    if frame >= max_frames:
        return [particles_scatter, gt_pos_mcl, est_pos_ekf, est_heading, 
                est_path, est_map_im, status_text]
    
    # ===== PHASE 1: LOCALIZATION (MCL) =====
    if current_phase == PHASE_LOCALIZATION:
        # Only move if we haven't converged yet. Once converged, wait for user input
        # Ground truth moves in random direction for exploration
        u = np.array([0.0, 0.0])  # 0.05 m/s forward, random rotation
        gt_pose = sample_motion_velocity_model(gt_pose, u, dt=0.1, alphas=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        
        # Normalize angle
        while gt_pose[2] > np.pi:
            gt_pose[2] -= 2*np.pi
        while gt_pose[2] < -np.pi:
            gt_pose[2] += 2*np.pi
        
        # Get ground truth measurement (EVERY FRAME, even when stopped)
        z_gt = compute_expected_measurements(gt_pose[0], gt_pose[1], gt_pose[2], 
                                            maze_binary, LIDAR_EFFECTIVE_RANGE, num_beams=360)
        z_gt = np.clip(z_gt, 0.001, LIDAR_EFFECTIVE_RANGE)
        
        # MCL update
        def sample_motion_model(prev_pose, u, PDF=np.random.normal, dt=0.1, alphas=(0.001, 0.001, 0.001, 0.001)):
            return sample_motion_velocity_model(prev_pose, u, PDF=PDF, dt=dt, alphas=(*alphas, 0.001, 0.001))
        
        def sensor_model(pose, z, m, compute_from_map, PDF=np.random.normal, 
                        min_theta=-np.pi, max_theta=np.pi, 
                        sigma_hit=0.1, lambda_short=0.2, z_hit=1, z_short=0.0, 
                        z_max=15.0, z_rand=0.02):
                        
            # Compute expected scan for this pose using the SAME function as plotting
            x, y, theta = pose
            z_expected = compute_expected_measurements(x, y, theta, m, LIDAR_EFFECTIVE_RANGE, num_beams=360)
            z_expected = np.clip(z_expected, 0.001, LIDAR_EFFECTIVE_RANGE)
            
            # Compare with measured scan using scan_likelihood
            return scan_likelihood(z, z_expected, sigma_hit=sigma_hit)
        
        # MCL update: Motion model + Sensor model
        weights = np.zeros(num_particles)
        
        # Motion model: update particle poses
        for i in range(num_particles):
            particles[i] = sample_motion_model(particles[i], u, PDF=np.random.normal, dt=0.1)
        
        # Sensor model: weight particles based on scan likelihood
        best_weight = 0.0
        best_particle_idx = 0
        
        for i in range(num_particles):
            pose = particles[i]
            weights[i] = sensor_model(pose, z_gt, maze_binary, compute_from_map_wrapper)
            if weights[i] > best_weight:
                best_weight = weights[i]
                best_particle_idx = i
        
        # Normalize weights
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights /= weights_sum
        else:
            weights = np.ones(num_particles) / num_particles
        
        # Debug: check if weights are zero
        if best_weight == 0:
            print(f"Frame {frame}: WARNING - All weights are 0! Z measurements: {z_gt[:5]} (first 5)")
            # Test sensor model on GT pose
            test_weight = sensor_model(gt_pose, z_gt, maze_binary, compute_from_map_wrapper)
            print(f"  Sensor model output for GT pose: {test_weight}")
        
        # Estimate pose as best particle (highest weight)
        best_particle_idx = np.argmax(weights)
        est_pose_mcl = particles[best_particle_idx]
        
        # Skip visualization computations in headless mode
        if not args.headless:
            # Compute expected measurements for estimated pose (visualization only)
            z_expected = compute_expected_measurements(est_pose_mcl[0], est_pose_mcl[1], est_pose_mcl[2],
                                                      maze_binary, LIDAR_EFFECTIVE_RANGE, num_beams=360)
            z_expected = np.clip(z_expected, 0.01, LIDAR_EFFECTIVE_RANGE)
        
        # Estimate position uncertainty from top 5% particles only
        top_5_percent = max(1, int(0.002 * num_particles))
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
        
        # Check if localized
        if uncertainty < 0.03 and frame > 50:  # Wait at least 50 frames before declaring localization (0.03 m = 3 cm)
            if not localization_converged:
                localization_converged = True
                ready_for_phase_2 = True
                if args.headless:
                    print(f"\n✓ Localization converged at frame {frame}!", flush=True)
                    print(f"  Uncertainty: {uncertainty:.4f} m (target: < 0.03 m)", flush=True)
                    print(f"  Position error: {error:.4f} m", flush=True)
                else:
                    print(f"\n✓ Localization converged at frame {frame}!")
                    print(f"  Uncertainty: {uncertainty:.4f} m (target: < 0.03 m)")
                    print(f"  Position error: {error:.4f} m")
                    print(f"  Estimated position: ({est_pose_mcl[0]:.3f}, {est_pose_mcl[1]:.3f}) m")
                    print(f"\n  [WAITING FOR USER INPUT]")
                    print(f"  Press ENTER in terminal to proceed to Phase 2 (Path Planning & Following)")
            
            # Only transition if user has approved
            if phase_2_approved:
                # Initialize EKF-SLAM with best estimate (do this once when transitioning)
                if not localization_complete:
                    ekf_slam.set_initial_pose(est_pose_mcl, uncertainty=uncertainty/100.0)
                
                print(f"\n  Transition to PATH PLANNING phase")
                current_phase = PHASE_PATH_PLANNING
                localization_complete = True
    
    # ===== PHASE 2: PATH PLANNING =====
    elif current_phase == PHASE_PATH_PLANNING:
        # Execute path planning once
        print(f"\n  Planning path from ({est_pose_mcl[0]:.3f}, {est_pose_mcl[1]:.3f}) to ({goal_pos[0]:.3f}, {goal_pos[1]:.3f})")
        start_x_px = int(est_pose_mcl[0] / scale_x)
        start_y_px = int(est_pose_mcl[1] / scale_y)
        end_x_px_plan = int(goal_pos[0] / scale_x)
        end_y_px_plan = int(goal_pos[1] / scale_y)
        
        trajectory_px = ekf_slam.generate_trajectory(start_x_px, start_y_px, end_x_px_plan, end_y_px_plan, 
                                                    smooth=True, smoothing_factor=0.5)
        
        if trajectory_px is not None:
            # Convert to real world coordinates
            trajectory = [(wp[0] * scale_x, wp[1] * scale_y, wp[2]) for wp in trajectory_px]
            
            # Downsample
            downsample_factor = 1
            trajectory = trajectory[::downsample_factor]
            
            print(f"  ✓ Path planned with {len(trajectory)} waypoints")
            print(f"  Transition to FOLLOWING phase")
            
            current_phase = PHASE_FOLLOWING
            current_waypoint_idx = 0
        else:
            print(f"  ✗ Path planning failed, staying in PATH PLANNING phase")
    
    # ===== PHASE 3: FOLLOWING =====
    elif current_phase == PHASE_FOLLOWING:
        if trajectory is None or len(trajectory) == 0:
            status_text.set_text("Error: No trajectory generated")
            return [particles_scatter, gt_pos_mcl, est_pos_ekf, est_heading,
                    est_path, est_map_im, status_text]
        
        # Get current waypoint
        if current_waypoint_idx >= len(trajectory) - 1:
            current_wp = trajectory[-1]
            next_wp = trajectory[-1]
        else:
            current_wp = trajectory[current_waypoint_idx]
            next_wp = trajectory[current_waypoint_idx + 1]
        
        # Ground truth moves to follow trajectory (with realistic noise)
        dx = next_wp[0] - current_wp[0]
        dy = next_wp[1] - current_wp[1]
        dtheta = next_wp[2] - current_wp[2]
        
        while dtheta > np.pi:
            dtheta -= 2*np.pi
        while dtheta < -np.pi:
            dtheta += 2*np.pi
        
        dt = 0.1
        v = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
        w = dtheta / dt if dt > 0 else 0
        u = np.array([v, w])
        
        # Get LiDAR measurement from GT pose
        z_gt = compute_expected_measurements(gt_pose[0], gt_pose[1], gt_pose[2],
                                            maze_binary, LIDAR_EFFECTIVE_RANGE, num_beams=360)
        z_gt = z_gt + np.random.normal(0, 0.01, len(z_gt))
        z_gt = np.clip(z_gt, 0.01, LIDAR_EFFECTIVE_RANGE)
        
        # EKF-SLAM update
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
        est_trajectory_history.append(est_pose[:2].copy())
        
        error = np.linalg.norm(est_pose[:2] - gt_pose[:2])
        errors.append(error)
        uncertainties.append(pose_uncertainty)
        
        # Check if reached current waypoint
        dist_to_waypoint = np.linalg.norm(est_pose[:2] - np.array([current_wp[0], current_wp[1]]))
        if dist_to_waypoint < waypoint_tolerance and current_waypoint_idx < len(trajectory) - 1:
            current_waypoint_idx += 1
        
        # Update estimated map
        est_map = ekf_slam.get_estimated_map_probability()
    
    # ===== VISUALIZATION (skip if headless) =====
    if not args.headless:
        # Get indices of top 1% particles by weight
        num_to_plot = max(1, int(0.01 * num_particles))
        top_indices = np.argsort(weights)[-num_to_plot:]
        particles_scatter.set_offsets(particles[top_indices, :2])
        gt_pos_mcl.set_data([gt_pose[0]], [gt_pose[1]])
        
        # Highlight best particle (highest weight) in orange
        if current_phase == PHASE_LOCALIZATION and len(weights) > 0:
            best_idx = np.argmax(weights)
            best_particle.set_data([particles[best_idx, 0]], [particles[best_idx, 1]])
        else:
            best_particle.set_data([], [])
        
        # Visualize GT LiDAR measurements (every 10th beam to reduce clutter)
        if current_phase == PHASE_LOCALIZATION:
            gt_lidar_rays_x = []
            gt_lidar_rays_y = []
            z_gt_display = compute_expected_measurements(gt_pose[0], gt_pose[1], gt_pose[2], 
                                                         maze_binary, LIDAR_EFFECTIVE_RANGE, num_beams=360)
            for i in range(0, 360, 10):  #Every 10th beam
                beam_angle = gt_pose[2] + (i - 180) * (2 * np.pi / 360)
                end_x = gt_pose[0] + z_gt_display[i] * np.cos(beam_angle)
                end_y = gt_pose[1] + z_gt_display[i] * np.sin(beam_angle)
                gt_lidar_rays_x.extend([gt_pose[0], end_x, None])
                gt_lidar_rays_y.extend([gt_pose[1], end_y, None])
            gt_lidar_rays.set_data(gt_lidar_rays_x, gt_lidar_rays_y)
        
        # EKF-SLAM visualization - estimated position only
        if current_phase == PHASE_LOCALIZATION:
            # Update estimated position marker (MCL estimate)
            est_pos_ekf.set_data([est_pose_mcl[0]], [est_pose_mcl[1]])
        
        # Update estimated heading arrow
        heading_length = 4.0
        heading_end_x = est_pose_mcl[0] + heading_length * np.cos(est_pose_mcl[2])
        heading_end_y = est_pose_mcl[1] + heading_length * np.sin(est_pose_mcl[2])
        est_heading.set_data([est_pose_mcl[0], heading_end_x], [est_pose_mcl[1], heading_end_y])
        
        # Update estimated path traveled
        if len(est_trajectory_history) > 1:
            traj_array = np.array(est_trajectory_history)
            est_path.set_data(traj_array[:, 0], traj_array[:, 1])
        
        # Visualize estimated LiDAR measurements from estimated particle pose (every 10th beam to reduce clutter)
        est_lidar_rays_x = []
        est_lidar_rays_y = []
        z_est_display = compute_expected_measurements(est_pose_mcl[0], est_pose_mcl[1], est_pose_mcl[2],
                                                      maze_binary, LIDAR_EFFECTIVE_RANGE, num_beams=360)
        for i in range(0, 360, 10):  # Every 10th beam
            beam_angle = est_pose_mcl[2] + (i - 180) * (2 * np.pi / 360)
            end_x = est_pose_mcl[0] + z_est_display[i] * np.cos(beam_angle)
            end_y = est_pose_mcl[1] + z_est_display[i] * np.sin(beam_angle)
            est_lidar_rays_x.extend([est_pose_mcl[0], end_x, None])
            est_lidar_rays_y.extend([est_pose_mcl[1], end_y, None])
        est_lidar_rays.set_data(est_lidar_rays_x, est_lidar_rays_y)
    
    elif current_phase == PHASE_FOLLOWING:
        # Update estimated position marker
        est_pos_ekf.set_data([est_pose[0]], [est_pose[1]])

        print(f"Frame {frame}: EKF Pose: ({est_pose[0]:.3f}, {est_pose[1]:.3f}), Uncertainty: {pose_uncertainty:.4f} m, Error: {error:.4f} m")
        
        # Update estimated heading arrow
        heading_length = 15.0
        heading_end_x = est_pose[0] + heading_length * np.cos(est_pose[2])
        heading_end_y = est_pose[1] + heading_length * np.sin(est_pose[2])
        est_heading.set_data([est_pose[0], heading_end_x], [est_pose[1], heading_end_y])
        
        # Update estimated path traveled
        if len(est_trajectory_history) > 1:
            traj_array = np.array(est_trajectory_history)
            est_path.set_data(traj_array[:, 0], traj_array[:, 1])
        
        if trajectory is not None:
            traj_array = np.array([(wp[0], wp[1]) for wp in trajectory])
            trajectory_line.set_data(traj_array[:, 0], traj_array[:, 1])
        
        est_map = ekf_slam.get_estimated_map_probability()
        est_map_im.set_data(est_map)
        
        # Visualize estimated LiDAR measurements (every 10th beam to reduce clutter)
        est_lidar_rays_x = []
        est_lidar_rays_y = []
        z_est_display = compute_expected_measurements(est_pose[0], est_pose[1], est_pose[2], 
                                                      maze_binary, LIDAR_EFFECTIVE_RANGE, num_beams=360)
        for i in range(0, 360, 10):  # Every 10th beam
            beam_angle = est_pose[2] + (i - 180) * (2 * np.pi / 360)
            end_x = est_pose[0] + z_est_display[i] * np.cos(beam_angle)
            end_y = est_pose[1] + z_est_display[i] * np.sin(beam_angle)
            est_lidar_rays_x.extend([est_pose[0], end_x, None])
            est_lidar_rays_y.extend([est_pose[1], end_y, None])
        est_lidar_rays.set_data(est_lidar_rays_x, est_lidar_rays_y)
    
    # Beam comparison scatter plot (visualization only)
    if not args.headless and current_phase == PHASE_LOCALIZATION:
        # Convert range measurements to Cartesian coordinates
        # Beams are indexed from 0 to 359, offset by theta
        beam_angles = est_pose_mcl[2] + (np.arange(360) - 180) * (2 * np.pi / 360)
        
        # GT scan in Cartesian
        gt_x = z_gt * np.cos(beam_angles)
        gt_y = z_gt * np.sin(beam_angles)
        beam_scatter_gt.set_offsets(np.c_[gt_x, gt_y])
        
        # Expected scan in Cartesian
        exp_x = z_expected * np.cos(beam_angles)
        exp_y = z_expected * np.sin(beam_angles)
        beam_scatter_exp.set_offsets(np.c_[exp_x, exp_y])
        
        # Fixed 2x2 meter view centered on robot
        ax_beams.set_xlim(-1, 1)
        ax_beams.set_ylim(-1, 1)
    
    # ===== UPDATE LIDAR DIAGNOSTIC WINDOW (DISABLED) =====
    # if current_phase == PHASE_LOCALIZATION:
    #     ax_lidar.clear()
    #     ax_lidar.set_aspect('equal')
    #     ax_lidar.set_xlim(-LIDAR_EFFECTIVE_RANGE, LIDAR_EFFECTIVE_RANGE)
    #     ax_lidar.set_ylim(-LIDAR_EFFECTIVE_RANGE, LIDAR_EFFECTIVE_RANGE)
    #     ax_lidar.grid(True, alpha=0.3)
    #     ax_lidar.axhline(y=0, color='k', linewidth=0.5)
    #     ax_lidar.axvline(x=0, color='k', linewidth=0.5)
    #     ax_lidar.plot(0, 0, 'ko', markersize=10, label='Robot at Origin', zorder=10)
    #     ax_lidar.set_xlabel('X (m)')
    #     ax_lidar.set_ylabel('Y (m)')
    #     
    #     # Convert LiDAR measurements to cartesian rays
    #     theta_array = np.linspace(0, 2*np.pi, len(z_gt), endpoint=False)
    #     
    #     # GT rays (blue)
    #     for angle, distance in zip(theta_array, z_gt):
    #         x = distance * np.cos(angle)
    #         y = distance * np.sin(angle)
    #         ax_lidar.plot([0, x], [0, y], 'b-', alpha=0.3, linewidth=0.5)
    #     
    #     # Expected rays (red) - overlay
    #     z_expected = compute_expected_measurements(gt_pose[0], gt_pose[1], gt_pose[2], 
    #                                                maze_binary, LIDAR_EFFECTIVE_RANGE, num_beams=360)
    #     for angle, distance in zip(theta_array, z_expected):
    #         x = distance * np.cos(angle)
    #         y = distance * np.sin(angle)
    #         ax_lidar.plot([0, x], [0, y], 'r-', alpha=0.3, linewidth=0.5)
    #     
    #     # Add legend with GT and expected endpoints
    #     from matplotlib.patches import Patch
    #     legend_elements = [
    #         Patch(facecolor='blue', alpha=0.3, label='GT Measurement'),
    #         Patch(facecolor='red', alpha=0.3, label='Expected from Map')
    #     ]
    #     ax_lidar.legend(handles=legend_elements, loc='upper right')
    #     
    #     # Calculate difference stats
    #     diff = np.abs(z_gt - z_expected)
    #     rmse = np.sqrt(np.mean(diff**2))
    #     ax_lidar.set_title(f'LiDAR Rays (Frame {frame}) - RMSE: {rmse:.4f}m, Mean Diff: {np.mean(diff):.4f}m')
    #     fig_lidar.canvas.draw_idle()
    
    # Weight distribution plot (MCL phase only, visualization)
    if not args.headless and current_phase == PHASE_LOCALIZATION:
        if len(weights) > 0:
            # Update bar heights and colors
            max_weight = np.max(weights) if np.max(weights) > 0 else 0.1
            best_idx = np.argmax(weights)
            for i, (bar, weight) in enumerate(zip(weight_bars, weights)):
                bar.set_height(weight)
                # Highlight best particle in orange
                if i == best_idx:
                    bar.set_color('orange')
                else:
                    bar.set_color('blue')
            ax_weights.set_ylim(0, max_weight * 1.2)
            ax_weights.set_title(f'Particle Weight Distribution (max={max_weight:.2e})')
    
    # Status text (visualization only)
    if not args.headless:
        phase_names = {
            PHASE_LOCALIZATION: "Localization (MCL)",
            PHASE_PATH_PLANNING: "Path Planning",
            PHASE_FOLLOWING: "Following (EKF-SLAM)"
        }
        phase_name = phase_names.get(current_phase, "Unknown")
        if current_phase == PHASE_LOCALIZATION:
            w_min = np.min(weights) if len(weights) > 0 else 0
            w_max = np.max(weights) if len(weights) > 0 else 0
            w_mean = np.mean(weights) if len(weights) > 0 else 0
            w_nonzero = np.sum(weights > 0) if len(weights) > 0 else 0
            
            # Add message if localization converged
            if ready_for_phase_2 and not phase_2_approved:
                convergence_msg = "\n[WAITING FOR INPUT]\nPress ENTER to proceed"
            else:
                convergence_msg = ""
            
            status_text.set_text(
                f"Frame: {frame} | {phase_name}\n"
                f"\nUncertainty: {uncertainties[-1]:.4f} m\n"
                f"Error: {errors[-1]:.4f} m\n"
                f"\n--- WEIGHTS ---\n"
                f"Min: {w_min:.2e}\n"
                f"Max: {w_max:.2e}\n"
                f"Mean: {w_mean:.2e}\n"
                f"Non-zero: {w_nonzero}/{num_particles}\n"
                f"\n--- POSITIONS ---\n"
                f"GT: ({gt_pose[0]:.2f}, {gt_pose[1]:.2f})\n"
                f"Est: ({est_pose_mcl[0]:.2f}, {est_pose_mcl[1]:.2f})"
                f"{convergence_msg}"
            )
        else:
            dist_to_goal = np.linalg.norm(est_pose[:2] - goal_pos)
            status_text.set_text(
                f"Frame: {frame}\n"
                f"Phase: {phase_name}\n"
                f"Uncertainty: {uncertainties[-1]:.6f}\n"
                f"Error: {errors[-1]:.4f} m\n"
                f"GT Pos: ({gt_pose[0]:.2f}, {gt_pose[1]:.2f})\n"
                f"Est Pos: ({est_pose[0]:.2f}, {est_pose[1]:.2f})\n"
                f"Dist to Goal: {dist_to_goal:.4f} m"
            )
    
    # Only return visualization objects if not headless
    if not args.headless:
        return [particles_scatter, gt_pos_mcl, est_pos_ekf, est_heading,
                est_path, est_map_im, status_text, gt_lidar_rays, est_lidar_rays, beam_scatter_gt, beam_scatter_exp, best_particle]
    else:
        return []

print("Creating animation...")

# LiDAR diagnostic window disabled
# fig_lidar = plt.figure(figsize=(12, 12))
# ax_lidar = fig_lidar.add_subplot(111)
# ax_lidar.set_aspect('equal')
# ax_lidar.set_xlim(-LIDAR_EFFECTIVE_RANGE, LIDAR_EFFECTIVE_RANGE)
# ax_lidar.set_ylim(-LIDAR_EFFECTIVE_RANGE, LIDAR_EFFECTIVE_RANGE)
# ax_lidar.grid(True, alpha=0.3)
# ax_lidar.axhline(y=0, color='k', linewidth=0.5)
# ax_lidar.axvline(x=0, color='k', linewidth=0.5)
# ax_lidar.plot(0, 0, 'ko', markersize=8, label='Robot at Origin')
# ax_lidar.set_xlabel('X (m)')
# ax_lidar.set_ylabel('Y (m)')
# ax_lidar.set_title('LiDAR Scans - GT (blue) vs Expected (red)')
# ax_lidar.legend(loc='upper right')
# 
# fig_lidar.tight_layout()

def frame_generator():
    frame = 0
    while frame < max_frames:
        yield frame
        frame += 1
        # Early stopping: if following phase and reached goal
        if current_phase == PHASE_FOLLOWING and len(est_trajectory_history) > 0:
            dist_to_goal = np.linalg.norm(np.array(est_trajectory_history[-1]) - goal_pos)
            if dist_to_goal < 5.0:  # Within 0.05 m of goal
                break

if not args.headless:
    print("Creating animation...")
    
    try:
        anim = FuncAnimation(fig, update, frames=frame_generator(),
                             interval=50, blit=True, repeat=False, cache_frame_data=False)
        
        plt.suptitle('EKF-SLAM with MCL: Random Initial Position → Localization → Path Planning → Following', 
                     fontsize=12, fontweight='bold')
        print("Starting animation... (close window when done)")
        plt.show()
    except Exception as e:
        print(f"Error in animation: {e}")
else:
    # Headless mode: run simulation without animation
    print("Running in headless mode (no visualization)")
    print("="*70)
    start_total = time.time()
    for frame_num in frame_generator():
        frame_start = time.time()
        update(frame_num)
        frame_time = (time.time() - frame_start) * 1000  # Convert to ms
        
        # Print timing every frame (buffered output)
        print(f"Frame {frame_num:4d} | Time: {frame_time:6.2f} ms | Error: {errors[-1]:.4f} m | Uncertainty: {uncertainties[-1]:.4f} m")
    
    total_time = (time.time() - start_total) * 1000
    # Print final results
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
    print(f"Frames completed: {len(errors)}")
    avg_frame_time = total_time / len(errors) if len(errors) > 0 else 0
    print(f"Average frame time: {avg_frame_time:.2f} ms")
    print(f"Final uncertainty: {uncertainties[-1]:.4f} m")
    print(f"Final error: {errors[-1]:.4f} m")
    if np.array(uncertainties).min() < 0.03:
        convergence_frame = np.where(np.array(uncertainties) < 0.03)[0][0]
        print(f"Convergence achieved at frame {convergence_frame}")
    print("="*70)
    import traceback
    traceback.print_exc()

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
    print(f"  Localization time: {np.argmax(np.array(uncertainties) < 0.03) if np.any(np.array(uncertainties) < 0.03) else 'Not achieved'} frames")

if localization_complete:
    print(f"\n✓ Localization successful!")
    print(f"✓ Path planning and following initiated")
else:
    print(f"\n✗ Localization not completed within {len(uncertainties)} frames")

print("="*70)

# Save results
print("\nSaving results...")
out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out', 'examples')
os.makedirs(out_dir, exist_ok=True)

if localization_complete:
    from OGM import log_odds_to_occupancy_probability, save_ogm_to_pgm
    ogm_output_path = os.path.join(out_dir, 'ekf_slam_I_final_ogm.pgm')
    save_ogm_to_pgm(ekf_slam.map_grid, ogm_output_path)
    print(f"✓ Final OGM saved to: {ogm_output_path}")
    
    trajectory_output_path = os.path.join(out_dir, 'ekf_slam_I_robot_trajectory.txt')
    with open(trajectory_output_path, 'w') as f:
        f.write("Timestep X Y\n")
        for timestep, (x, y) in enumerate(est_trajectory_history):
            f.write(f"{timestep} {x:.4f} {y:.4f}\n")
    print(f"✓ Robot trajectory saved to: {trajectory_output_path}")
