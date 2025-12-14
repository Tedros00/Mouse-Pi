"""
Real Maze Simulation - Uses ground truth path from OGM maze and path finding results.
Simulates a robot following a path through a real maze with LiDAR measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image

# Add src to path for path finding
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from path_finding import find_path


class MazeSimulation:
    """
    Simulation of a robot moving through a real maze from an OGM image.
    Loads ground truth path from path finding results.
    """
    
    def __init__(self,
                 ogm_image_path: str,
                 path_output_file: str,
                 real_width_cm: float = 180.0,
                 real_height_cm: float = 200.0,
                 num_beams: int = 181,
                 lidar_fov: float = 2*np.pi,
                 max_range: float = 15.0,
                 measurement_noise_std: float = 0.001,
                 motion_noise_std: tuple = (0.1, 0.05),
                 ogm_downsample_factor: int = 10,
                 obstacle_inflation_radius_px: int = 0):
        """
        Initialize the maze simulation.
        
        Parameters:
        -----------
        ogm_image_path : str
            Path to the OGM image file (TestMaze.jpg)
        path_output_file : str
            Path to the file containing ground truth path (X Y grid cells)
        real_width_cm : float
            Real-world width of the maze in centimeters
        real_height_cm : float
            Real-world height of the maze in centimeters
        num_beams : int
            Number of LiDAR beams
        lidar_fov : float
            Field of view of LiDAR in radians
        max_range : float
            Maximum range of LiDAR
        measurement_noise_std : float
            Standard deviation of measurement noise
        motion_noise_std : tuple
            Standard deviation of motion noise (linear, angular)
        ogm_downsample_factor : int
            Downsample OGM by this factor (e.g., 2 = half resolution). Default: 1 (no downsampling)
            Maximum range of LiDAR in meters
        obstacle_inflation_radius_px : int
            Obstacle inflation radius in pixels (based on original OGM). Default: 0 (no inflation)
        """
        self.ogm_image_path = ogm_image_path
        self.path_output_file = path_output_file
        self.real_width_cm = real_width_cm
        self.real_height_cm = real_height_cm
        self.real_width_m = real_width_cm / 100.0
        self.real_height_m = real_height_cm / 100.0
        self.measurement_noise_std = measurement_noise_std
        self.motion_noise_std = motion_noise_std
        self.obstacle_inflation_radius_px = obstacle_inflation_radius_px
        
        # Load OGM image
        try:
            pil_image = Image.open(ogm_image_path)
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            self.ogm_image = np.array(pil_image)
        except Exception as e:
            raise ValueError(f"Failed to load OGM image: {ogm_image_path} - {str(e)}")
        
        self.image_height, self.image_width = self.ogm_image.shape
        self.ogm_downsample_factor = ogm_downsample_factor  # Store for later use
        
        # Downsample OGM if requested (makes cells bigger = sparser grid)
        if ogm_downsample_factor > 1:
            self.ogm_image = self.ogm_image[::ogm_downsample_factor, ::ogm_downsample_factor]
            self.image_height, self.image_width = self.ogm_image.shape
            print(f"[MazeSimulation] Downsampled OGM by factor {ogm_downsample_factor}")
        
        print(f"[MazeSimulation] Loaded OGM image: {self.image_width} x {self.image_height} pixels")
        print(f"[MazeSimulation] Real-world dimensions: {self.real_width_cm}cm x {self.real_height_cm}cm")
        
        # Calculate conversion factors: grid cells to meters
        self.cell_size_x = self.real_width_m / self.image_width
        self.cell_size_y = self.real_height_m / self.image_height
        print(f"[MazeSimulation] Cell size: {self.cell_size_x*100:.2f}cm x {self.cell_size_y*100:.2f}cm")
        
        # Create occupancy map for LiDAR raycasting
        # OGM image: white (255) = free, black (0) = obstacle
        # occupancy_map: 1 = obstacle, 0 = free space
        self.occupancy_map = (self.ogm_image < 127).astype(np.uint8)
        
        # Debug: print occupancy map stats
        print(f"[MazeSimulation] Occupancy map shape: {self.occupancy_map.shape}")
        print(f"[MazeSimulation] Occupancy map - Min: {self.occupancy_map.min()}, Max: {self.occupancy_map.max()}, Mean: {self.occupancy_map.mean():.3f}")
        print(f"[MazeSimulation] Obstacles (1s): {np.sum(self.occupancy_map)}, Free (0s): {np.sum(1 - self.occupancy_map)}")
        
        # LiDAR parameters
        self.num_beams = num_beams
        self.lidar_fov = lidar_fov
        self.max_range = max_range
        self.beam_angles = np.linspace(-lidar_fov/2, lidar_fov/2, num_beams)
        
        # Storage
        self.trajectory = None      # (N, 3) - poses [x, y, theta] in meters
        self.controls = None        # (N-1, 2) - [v, w]
        self.measurements = None    # (N, num_beams) - range measurements
        self.time_steps = None      # (N,) - time values
        
    def generate_path_from_grid(self, trajectory_downsample_factor: int = 1):
        """
        Generate ground truth path based on the current downsampled OGM grid using BFS pathfinding.
        Converts grid cells to real-world meters and computes heading angles.
        Smooths sharp corners with tangent radius and optionally repeats points for speed control.
        
        Parameters:
        -----------
        trajectory_downsample_factor : int
            Downsample trajectory by keeping every Nth waypoint (e.g., 5 = fewer steps)
            Default: 1 (no downsampling)
        
        Returns:
        --------
        np.ndarray
            Trajectory as (N, 3) array of [x, y, theta] in meters
        """
        print(f"[MazeSimulation] Generating path for grid {self.image_width}x{self.image_height} using BFS...")
        
        # Scale start/end coordinates from original OGM to current (downsampled) grid
        # Original coordinates: start (150, 1200), end (950, 400)
        start_x_original = 150
        start_y_original = 1200
        end_x_original = 950
        end_y_original = 400
        
        # Scale to current grid
        start_x = start_x_original // self.ogm_downsample_factor
        start_y = start_y_original // self.ogm_downsample_factor
        end_x = end_x_original // self.ogm_downsample_factor
        end_y = end_y_original // self.ogm_downsample_factor
        
        print(f"[MazeSimulation] Start: ({start_x}, {start_y}) | End: ({end_x}, {end_y})")
        
        # Scale obstacle inflation radius to downsampled grid
        # R is specified in pixels of the original OGM
        R_scaled = max(1, self.obstacle_inflation_radius_px // self.ogm_downsample_factor)
        if self.obstacle_inflation_radius_px > 0:
            print(f"[MazeSimulation] Obstacle inflation: {self.obstacle_inflation_radius_px}px (original) -> {R_scaled}px (downsampled)")
        
        # Use BFS path finding on the current (downsampled) grid
        path_grid_cells = find_path(
            ogm_image=self.ogm_image,
            start_x=start_x, start_y=start_y,
            end_x=end_x, end_y=end_y,
            diagonal=True,
            R=R_scaled
        )
        
        if path_grid_cells is None or len(path_grid_cells) == 0:
            raise RuntimeError("Path finding failed - no valid path found!")
        
        path_grid_cells = np.array(path_grid_cells)
        print(f"[MazeSimulation] Found path with {len(path_grid_cells)} waypoints")
        
        # Downsample trajectory by skipping waypoints
        if trajectory_downsample_factor > 1:
            path_grid_cells = path_grid_cells[::trajectory_downsample_factor]
            print(f"[MazeSimulation] Downsampled trajectory (keep every {trajectory_downsample_factor}th waypoint)")
        
        print(f"[MazeSimulation] Final trajectory: {len(path_grid_cells)} waypoints")
        
        # Convert grid cells to real-world meters
        x_meters = path_grid_cells[:, 0] * self.cell_size_x
        y_meters = path_grid_cells[:, 1] * self.cell_size_y
        
        # Compute heading angles from consecutive points
        theta = np.zeros(len(path_grid_cells))
        for i in range(len(path_grid_cells) - 1):
            dx = x_meters[i + 1] - x_meters[i]
            dy = y_meters[i + 1] - y_meters[i]
            theta[i] = np.arctan2(dy, dx)
        
        # For the last point, use the same heading as the previous point
        if len(path_grid_cells) > 1:
            theta[-1] = theta[-2]
        
        trajectory = np.column_stack([x_meters, y_meters, theta])
        
        # Smooth corners and interpolate trajectory
        trajectory = self._smooth_trajectory(trajectory, tangent_radius=0.2, point_repeat_factor=2)
        
        return trajectory
    
    def _smooth_trajectory(self, trajectory: np.ndarray, tangent_radius: float = 0.3, point_repeat_factor: int = 2):
        """
        Smooth sharp corners in trajectory using Bezier curves and optionally repeat points for slower travel.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Original trajectory as (N, 3) array of [x, y, theta]
        tangent_radius : float
            Radius of curvature for corner smoothing (meters)
        point_repeat_factor : int
            How many times to repeat each point (>1 slows down robot, 1 = no repeat)
        
        Returns:
        --------
        np.ndarray
            Smoothed trajectory with interpolated corners
        """
        if len(trajectory) < 3:
            return trajectory
        
        smoothed_points = [trajectory[0]]
        
        # Process each corner (point i is corner between point i-1 and i+1)
        for i in range(1, len(trajectory) - 1):
            prev_point = trajectory[i - 1, :2]
            curr_point = trajectory[i, :2]
            next_point = trajectory[i + 1, :2]
            
            # Distance to neighbors
            dist_to_prev = np.linalg.norm(curr_point - prev_point)
            dist_to_next = np.linalg.norm(next_point - curr_point)
            
            if dist_to_prev > 0.01 and dist_to_next > 0.01:
                # Direction vectors
                dir_in = (curr_point - prev_point) / dist_to_prev  # Direction coming into corner
                dir_out = (next_point - curr_point) / dist_to_next  # Direction leaving corner
                
                # Calculate angle between directions
                cos_angle = np.dot(dir_in, dir_out)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                # Only smooth if there's a noticeable corner (angle > 15 degrees)
                if angle > np.deg2rad(15):
                    # Use a larger radius for better smoothing
                    r = max(tangent_radius, min(dist_to_prev, dist_to_next) * 0.4)
                    
                    # Calculate how far along each segment to place tangent points
                    # Using: distance = r / tan(angle/2)
                    tan_dist = r / np.tan(angle / 2)
                    
                    # Clamp to reasonable distances (don't go past halfway)
                    tan_dist = min(tan_dist, dist_to_prev * 0.4, dist_to_next * 0.4)
                    
                    # Tangent points
                    tan_in = curr_point - dir_in * tan_dist
                    tan_out = curr_point - dir_out * tan_dist
                    
                    # Add incoming tangent point with interpolated heading
                    theta_in = np.arctan2(curr_point[1] - prev_point[1], curr_point[0] - prev_point[0])
                    smoothed_points.append(np.array([tan_in[0], tan_in[1], theta_in]))
                    
                    # Generate smooth arc between tangent points using Bezier-like interpolation
                    arc_length = np.linalg.norm(tan_out - tan_in)
                    num_interp = max(3, int(arc_length / 0.02))  # One point every 2cm
                    
                    for t in np.linspace(0, 1, num_interp + 2)[1:-1]:  # Exclude endpoints
                        # Quadratic Bezier: B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
                        # Control point at the corner
                        one_minus_t = 1 - t
                        bezier_point = (one_minus_t**2 * tan_in + 
                                       2 * one_minus_t * t * curr_point + 
                                       t**2 * tan_out)
                        
                        # Interpolate heading smoothly through the corner
                        theta_interp = theta_in + t * angle
                        smoothed_points.append(np.array([bezier_point[0], bezier_point[1], theta_interp]))
                    
                    # Add outgoing tangent point
                    theta_out = np.arctan2(next_point[1] - curr_point[1], next_point[0] - curr_point[0])
                    smoothed_points.append(np.array([tan_out[0], tan_out[1], theta_out]))
                else:
                    # No corner smoothing needed, just add the point
                    smoothed_points.append(trajectory[i])
            else:
                smoothed_points.append(trajectory[i])
        
        # Add last point
        smoothed_points.append(trajectory[-1])
        
        smoothed_array = np.array(smoothed_points)
        
        # Repeat points for speed control
        if point_repeat_factor > 1:
            repeated_points = []
            for point in smoothed_array:
                for _ in range(point_repeat_factor):
                    repeated_points.append(point)
            smoothed_array = np.array(repeated_points)
        
        print(f"[MazeSimulation] Trajectory smoothing: {len(trajectory)} -> {len(smoothed_array)} waypoints")
        
        return smoothed_array
    
    def generate_trajectory(self, dt: float = 0.01, trajectory_downsample_factor: int = 1):
        """
        Generate the ground truth trajectory from the current downsampled grid.
        
        Parameters:
        -----------
        dt : float
            Time step between trajectory points
        trajectory_downsample_factor : int
            Downsample trajectory by keeping every Nth waypoint (e.g., 5 = fewer steps)
        """
        self.trajectory = self.generate_path_from_grid(trajectory_downsample_factor=trajectory_downsample_factor)
        num_points = len(self.trajectory)
        self.time_steps = np.arange(num_points) * dt
        
        # Compute controls (velocity and angular velocity)
        self.controls = np.zeros((num_points - 1, 2))
        
        for i in range(len(self.trajectory) - 1):
            curr_pose = self.trajectory[i]
            next_pose = self.trajectory[i + 1]
            
            # Linear distance
            dist = np.linalg.norm(next_pose[:2] - curr_pose[:2])
            v = dist / dt  # Linear velocity
            
            # Angular difference
            dtheta = next_pose[2] - curr_pose[2]
            # Normalize to [-pi, pi]
            while dtheta > np.pi:
                dtheta -= 2 * np.pi
            while dtheta < -np.pi:
                dtheta += 2 * np.pi
            w = dtheta / dt  # Angular velocity
            
            # Add Gaussian noise to controls
            v_noisy = v + np.random.normal(0, self.motion_noise_std[0])
            w_noisy = w + np.random.normal(0, self.motion_noise_std[1])
            
            self.controls[i] = [v_noisy, w_noisy]
    
    def _grid_to_world(self, grid_x: float, grid_y: float) -> tuple:
        """
        Convert grid cell coordinates to world coordinates in meters.
        """
        world_x = grid_x * self.cell_size_x
        world_y = grid_y * self.cell_size_y
        return world_x, world_y
    
    def _world_to_grid(self, world_x: float, world_y: float) -> tuple:
        """
        Convert world coordinates in meters to grid cell coordinates.
        """
        grid_x = int(world_x / self.cell_size_x)
        grid_y = int(world_y / self.cell_size_y)
        # Clamp to valid range
        grid_x = max(0, min(grid_x, self.image_width - 1))
        grid_y = max(0, min(grid_y, self.image_height - 1))
        return grid_x, grid_y
    
    def _raycast_lidar(self, x: float, y: float, theta: float) -> np.ndarray:
        """
        Simulate LiDAR measurements using raycasting on the occupancy grid.
        Uses proper line-of-sight tracing to detect first wall hit.
        
        Parameters:
        -----------
        x : float
            Robot X position in meters
        y : float
            Robot Y position in meters
        theta : float
            Robot heading in radians
        
        Returns:
        --------
        np.ndarray
            Range measurements for each beam
        """
        ranges = np.full(self.num_beams, self.max_range)
        
        # Convert robot position to grid coordinates
        start_grid_x, start_grid_y = self._world_to_grid(x, y)
        
        # For each beam
        for beam_idx, beam_angle in enumerate(self.beam_angles):
            # Global angle of this beam
            global_angle = theta + beam_angle
            cos_angle = np.cos(global_angle)
            sin_angle = np.sin(global_angle)
            
            # Find the range at which this ray hits an obstacle
            # Use small step size to not skip over thin walls
            step_size = 0.005  # 0.5 cm steps for accuracy
            current_range = step_size
            
            while current_range < self.max_range:
                # Endpoint of ray
                end_x = x + current_range * cos_angle
                end_y = y + current_range * sin_angle
                
                # Convert to grid coordinates
                end_grid_x, end_grid_y = self._world_to_grid(end_x, end_y)
                
                # Check if out of bounds
                if (end_grid_x < 0 or end_grid_x >= self.image_width or
                    end_grid_y < 0 or end_grid_y >= self.image_height):
                    ranges[beam_idx] = current_range
                    break
                
                # Check if hit obstacle (occupancy == 1, meaning obstacle; 0 = free space)
                if self.occupancy_map[int(end_grid_y), int(end_grid_x)] == 1:
                    ranges[beam_idx] = current_range
                    break
                
                # Step forward
                current_range += step_size
        
        # Add measurement noise
        noisy_ranges = ranges + np.random.normal(0, self.measurement_noise_std, self.num_beams)
        noisy_ranges = np.clip(noisy_ranges, 0, self.max_range)
        
        return noisy_ranges
    
    def generate_measurements(self):
        """
        Generate LiDAR measurements for the entire trajectory using raycasting.
        """
        print("[MazeSimulation] Generating LiDAR measurements...")
        num_points = len(self.trajectory)
        self.measurements = np.zeros((num_points, self.num_beams))
        
        for i, pose in enumerate(self.trajectory):
            if i % max(1, num_points // 10) == 0:
                print(f"  Progress: {i}/{num_points}")
            
            x, y, theta = pose
            self.measurements[i] = self._raycast_lidar(x, y, theta)
        
        # Debug: check measurement stats
        print(f"[MazeSimulation] Measurement generation complete")
        print(f"  Mean range: {np.mean(self.measurements):.3f} m")
        print(f"  Min range: {np.min(self.measurements):.3f} m")
        print(f"  Max range: {np.max(self.measurements):.3f} m")
        print(f"  % at max range: {100 * np.sum(self.measurements >= self.max_range - 0.01) / self.measurements.size:.1f}%")
        
        # Debug sample measurement
        if len(self.trajectory) > 0:
            sample_pose = self.trajectory[0]
            print(f"\n[DEBUG] Sample at start pose {sample_pose}:")
            sample_ranges = self._raycast_lidar(sample_pose[0], sample_pose[1], sample_pose[2])
            occupied_beams = np.sum(sample_ranges < self.max_range - 0.01)
            print(f"  Occupied beams: {occupied_beams} / {self.num_beams}")
            print(f"  Range stats: min={np.min(sample_ranges):.3f}, max={np.max(sample_ranges):.3f}, mean={np.mean(sample_ranges):.3f}")
        print(f"[MazeSimulation] Measurement stats:")
        print(f"  Min range: {np.min(self.measurements):.3f} m")
        print(f"  Max range: {np.max(self.measurements):.3f} m")
        print(f"  Mean range: {np.mean(self.measurements):.3f} m")
        print(f"  Median range: {np.median(self.measurements):.3f} m")
        print(f"  % at max range: {np.sum(self.measurements >= 14.9) / self.measurements.size * 100:.1f}%")
        
        print("[MazeSimulation] Measurement generation complete")
    
    def visualize_maze_and_path(self):
        """
        Visualize the OGM maze with the ground truth trajectory overlaid.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Display OGM
        ax.imshow(self.occupancy_map, cmap='gray', origin='upper', extent=[0, self.real_width_m, 0, self.real_height_m])
        
        # Plot trajectory
        if self.trajectory is not None:
            traj = self.trajectory
            ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Ground Truth Path')
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
            ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=20, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Maze with Ground Truth Path')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    ogm_path = os.path.join(os.path.dirname(__file__), 'TestMaze.JPG')
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out', 'examples')
    os.makedirs(out_dir, exist_ok=True)
    path_file = os.path.join(out_dir, 'path_output.txt')
    
    try:
        print("Creating maze simulation...")
        sim = MazeSimulation(ogm_path, path_file, real_width_cm=180.0, real_height_cm=200.0)
        
        print("Generating trajectory from ground truth path...")
        sim.generate_trajectory(dt=0.01)
        
        print("Generating LiDAR measurements...")
        sim.generate_measurements()
        
        print("Visualizing maze and path...")
        sim.visualize_maze_and_path()
        
        print(f"\nSimulation Summary:")
        print(f"  Trajectory points: {len(sim.trajectory)}")
        print(f"  Total path length: {np.sum(np.linalg.norm(np.diff(sim.trajectory[:, :2], axis=0), axis=1)):.2f} m")
        
    except Exception as e:
        print(f"Error: {e}")
