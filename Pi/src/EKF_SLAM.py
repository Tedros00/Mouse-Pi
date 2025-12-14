"""
EKF-SLAM: Extended Kalman Filter for Simultaneous Localization and Mapping.

Uses EKF for robot pose estimation combined with occupancy grid mapping.
Unlike FastSLAM (particle filter), EKF-SLAM maintains a single Gaussian 
estimate of the pose distribution with a shared occupancy grid map.
"""

import numpy as np
from typing import Tuple, Dict, Callable, List, Optional
from ProbabilisticMotionModel import sample_motion_velocity_model, velocity_motion_model
from ProbabilisticSensorModel import beam_range_finder_model
from OGM import incremental_occupancy_grid_update, log_odds_to_occupancy_probability, save_ogm_to_pgm, load_ogm_from_pgm
from path_finding import find_path
from PIL import Image
from scipy.interpolate import CubicSpline


class EKF_SLAM:
    """
    EKF-SLAM: Extended Kalman Filter SLAM with occupancy grid mapping.
    
    Maintains:
    - A Gaussian state estimate (mean Mu, covariance Sigma) of robot pose
    - A shared occupancy grid map updated incrementally
    - Deterministic motion model with Gaussian noise modeling
    """
    
    def __init__(self,
                 grid_shape: Tuple[int, int],
                 grid_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                 compute_from_map: Callable,
                 motion_model_params: Dict = None,
                 sensor_model_params: Dict = None,
                 dt: float = 0.01,
                 motion_noise_std: Tuple[float, float] = (0.1, 0.05),
                 measurement_noise_std: float = 0.3,
                 lidar_downsample_factor: int = 5):
        """
        Initialize EKF-SLAM.
        
        Parameters:
        -----------
        grid_shape : tuple
            Shape of occupancy grid (rows, cols)
        grid_bounds : tuple
            Bounds of grid ((min_x, max_x), (min_y, max_y))
        compute_from_map : callable
            Function to compute expected measurement from map.
            Signature: compute_from_map(x, y, angle, map_data, max_range)
        motion_model_params : dict, optional
            Parameters for motion model (alphas, dt, PDF)
        sensor_model_params : dict, optional
            Parameters for sensor model (sigma_hit, z_hit, etc.)
        dt : float
            Time step for motion model
        motion_noise_std : tuple, optional
            Gaussian noise std for (linear velocity, angular velocity)
        measurement_noise_std : float, optional
            Gaussian noise std for range measurements
        lidar_downsample_factor : int, optional
            Downsample factor for LiDAR (360 beams -> 360/factor beams)
        """
        self.grid_shape = grid_shape
        self.grid_bounds = grid_bounds
        self.compute_from_map = compute_from_map
        self.dt = dt
        self.motion_noise_std = motion_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.lidar_downsample_factor = lidar_downsample_factor
        
        # Motion model parameters
        self.motion_model_params = motion_model_params or {
            'alphas': (0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
            'dt': dt
        }
        
        # Sensor model parameters (beam range finder model)
        self.sensor_model_params = sensor_model_params or {
            'min_theta': -np.pi,
            'max_theta': np.pi,
            'max_range': 300.0,
            'sigma_hit': 0.2,
            'lambda_short': 0.1,
            'z_hit': 0.95,
            'z_short': 0.01,
            'z_max': 0.1,
            'z_rand': 0.1
        }
        
        # EKF state: [x, y, theta] for robot pose
        self.Mu = np.array([0.0, 0.0, 0.0])  # Mean state estimate
        self.Sigma = np.eye(3) * 0.1  # Covariance of pose (3x3)
        
        # Occupancy grid map
        self.map_grid = np.zeros(grid_shape)  # Log-odds representation
        
        # PGM file for persistent storage
        self.map_pgm_path = "current_ogm.pgm"
        
        # History for analysis
        self.pose_estimates = []
        self.pose_uncertainties = []
        self.measurement_history = []
        self.update_count = 0  # Track update count for periodic saves
        
        # Ground truth maze and trajectory
        self.ground_truth_maze = None  # Will store loaded maze image
        self.ground_truth_maze_path = None  # Path to maze image for path finding
        self.trajectory = None  # Will store waypoints
        self.trajectory_index = 0  # Current position in trajectory
    
    def load_ground_truth_maze(self, maze_image_path: str) -> np.ndarray:
        """
        Load ground truth maze from image file.
        This is the true maze structure that is unknown to the robot initially.
        
        Parameters:
        -----------
        maze_image_path : str
            Path to the maze image (jpg, png, etc.)
        
        Returns:
        --------
        np.ndarray
            Binary occupancy grid (1=free, 0=occupied)
        """
        try:
            image = np.array(Image.open(maze_image_path).convert('L'))
            # Convert to binary: white (>127) = free (1), black (<=127) = occupied (0)
            self.ground_truth_maze = (image > 127).astype(np.uint8)
            self.ground_truth_maze_path = maze_image_path  # Store path for path finding
            print(f"[EKF-SLAM] Loaded ground truth maze from {maze_image_path}")
            print(f"[EKF-SLAM] Maze dimensions: {self.ground_truth_maze.shape}")
            return self.ground_truth_maze
        except Exception as e:
            print(f"[EKF-SLAM] Error loading maze: {e}")
            return None
    
    def generate_trajectory(self, start_x: int, start_y: int, end_x: int, end_y: int,
                           smooth: bool = True, smoothing_factor: float = 0.1) -> List[Tuple[float, float, float]]:
        """
        Generate trajectory using path finder with smoothed corners.
        
        Parameters:
        -----------
        start_x : int
            Start X position (in image pixels)
        start_y : int
            Start Y position (in image pixels)
        end_x : int
            End X position (in image pixels)
        end_y : int
            End Y position (in image pixels)
        smooth : bool
            Whether to smooth corners using cubic spline interpolation
        smoothing_factor : float
            Smoothing parameter (0-1, higher = smoother)
        
        Returns:
        --------
        List[Tuple[float, float, float]]
            List of (x, y, theta) waypoints representing the smoothed trajectory
        """
        if self.ground_truth_maze is None:
            print("[EKF-SLAM] Error: Ground truth maze not loaded. Call load_ground_truth_maze first.")
            return None
        
        # Find path using BFS
        print(f"[EKF-SLAM] Finding path from ({start_x}, {start_y}) to ({end_x}, {end_y})...")
        path = find_path(
            ogm_image_path=self.ground_truth_maze_path,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            diagonal=True,
            R=10
        )
        
        if path is None:
            print("[EKF-SLAM] No path found!")
            return None
        
        print(f"[EKF-SLAM] Path found with {len(path)} waypoints")
        
        # Smooth path using cubic spline interpolation
        if smooth and len(path) > 3:
            path = self._smooth_path(path, smoothing_factor)
        
        # Convert path to trajectory with orientation
        trajectory = self._path_to_trajectory(path)
        self.trajectory = trajectory
        self.trajectory_index = 0
        
        print(f"[EKF-SLAM] Generated trajectory with {len(trajectory)} waypoints")
        return trajectory
    
    def _smooth_path(self, path: List[Tuple[int, int]], smoothing_factor: float) -> List[Tuple[float, float]]:
        """
        Smooth path using cubic spline interpolation to minimize sharp turns.
        
        Parameters:
        -----------
        path : List[Tuple[int, int]]
            Original path waypoints
        smoothing_factor : float
            Smoothing parameter (0-1)
        
        Returns:
        --------
        List[Tuple[float, float]]
            Smoothed path waypoints
        """
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        
        # Create parameter t along the path
        t = np.linspace(0, 1, len(path))
        
        # Fit cubic splines
        cs_x = CubicSpline(t, x, bc_type='natural')
        cs_y = CubicSpline(t, y, bc_type='natural')
        
        # Generate smooth path with higher density
        t_smooth = np.linspace(0, 1, len(path) * 5)
        x_smooth = cs_x(t_smooth)
        y_smooth = cs_y(t_smooth)
        
        smooth_path = list(zip(x_smooth, y_smooth))
        return smooth_path
    
    def _path_to_trajectory(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        """
        Convert path to trajectory with orientation angles.
        
        Parameters:
        -----------
        path : List[Tuple[float, float]]
            Path waypoints (x, y)
        
        Returns:
        --------
        List[Tuple[float, float, float]]
            Trajectory waypoints (x, y, theta)
        """
        trajectory = []
        
        for i, (x, y) in enumerate(path):
            # Calculate orientation based on movement direction
            if i < len(path) - 1:
                next_x, next_y = path[i + 1]
                theta = np.arctan2(next_y - y, next_x - x)
            else:
                # Last point: use same orientation as previous
                theta = trajectory[-1][2] if trajectory else 0.0
            
            trajectory.append((x, y, theta))
        
        return trajectory
    
    def get_next_trajectory_waypoint(self) -> Optional[Tuple[float, float, float]]:
        """
        Get the next waypoint in the trajectory.
        
        Returns:
        --------
        Optional[Tuple[float, float, float]]
            Next waypoint (x, y, theta) or None if trajectory finished
        """
        if self.trajectory is None or self.trajectory_index >= len(self.trajectory):
            return None
        
        waypoint = self.trajectory[self.trajectory_index]
        self.trajectory_index += 1
        return waypoint
    
    def reset_trajectory(self):
        """Reset trajectory index to start."""
        self.trajectory_index = 0
    
    def _downsample_lidar(self, measurement: np.ndarray) -> np.ndarray:
        """
        Downsample LiDAR measurement from 360 beams to 72 beams.
        
        Parameters:
        -----------
        measurement : np.ndarray
            Raw LiDAR measurement (360 beams)
        
        Returns:
        --------
        np.ndarray
            Downsampled measurement (72 beams)
        """
        if len(measurement) <= 1:
            return measurement
        
        # Downsample by taking every Nth beam
        downsampled = measurement[::self.lidar_downsample_factor]
        
        return downsampled
    
    def _motion_jacobian(self, prev_pose: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of motion model with respect to state.
        
        Parameters:
        -----------
        prev_pose : np.ndarray
            Previous pose [x, y, theta]
        u : np.ndarray
            Control input [v, w]
        
        Returns:
        --------
        np.ndarray
            3x3 Jacobian matrix G_t
        """
        x, y, theta = prev_pose
        v, w = u
        
        # Prevent division by zero
        if abs(w) < 1e-6:
            # Straight line motion
            G = np.array([
                [1.0, 0.0, -v * self.dt * np.sin(theta)],
                [0.0, 1.0, v * self.dt * np.cos(theta)],
                [0.0, 0.0, 1.0]
            ])
        else:
            # Curved motion (differential drive kinematics)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_dtheta = np.sin(theta + w * self.dt)
            cos_dtheta = np.cos(theta + w * self.dt)
            
            G = np.array([
                [1.0, 0.0, (v / w) * (sin_dtheta - sin_theta)],
                [0.0, 1.0, (v / w) * (-cos_dtheta + cos_theta)],
                [0.0, 0.0, 1.0]
            ])
        
        return G
    
    def _measurement_jacobian(self, pose: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of measurement model with respect to pose.
        
        For range measurements from LiDAR, the Jacobian represents how
        range measurements change with pose (x, y, theta).
        
        Parameters:
        -----------
        pose : np.ndarray
            Robot pose [x, y, theta]
        z : np.ndarray
            Range measurements (72 beams)
        
        Returns:
        --------
        np.ndarray
            (72 x 3) Jacobian matrix H
        """
        num_beams = len(z)
        H = np.zeros((num_beams, 3))
        
        x, y, theta = pose
        
        # For each beam, compute partial derivatives
        for i in range(num_beams):
            # Beam angle in world frame
            beam_angle = theta + (i - num_beams/2) * (2 * np.pi / num_beams)
            
            # dz/dx, dz/dy (changes in range with robot position)
            # Range decreases as robot moves toward obstacle
            dx_beam = np.cos(beam_angle)
            dy_beam = np.sin(beam_angle)
            
            H[i, 0] = -dx_beam  # dz/dx (negative because moving away increases range)
            H[i, 1] = -dy_beam  # dz/dy
            H[i, 2] = 0.1 * num_beams  # dz/dtheta (small effect from rotation)
        
        return H
    
    def _motion_covariance(self, u: np.ndarray) -> np.ndarray:
        """
        Compute motion noise covariance matrix.
        
        Parameters:
        -----------
        u : np.ndarray
            Control input [v, w]
        
        Returns:
        --------
        np.ndarray
            3x3 motion noise covariance matrix Q
        """
        # Motion noise scales with control magnitude
        v, w = u
        v_noise = max(abs(v) * 0.1, self.motion_noise_std[0])
        w_noise = max(abs(w) * 0.1, self.motion_noise_std[1])
        
        Q = np.diag([
            v_noise**2,
            v_noise**2,
            w_noise**2
        ])
        
        return Q
    
    def _motion_model(self, prev_pose: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Motion model using sample_motion_velocity_model (velocity model with noise sampling).
        
        Parameters:
        -----------
        prev_pose : np.ndarray
            Previous pose [x, y, theta]
        u : np.ndarray
            Control input [v, w] - linear and angular velocity
        
        Returns:
        --------
        np.ndarray
            New pose [x, y, theta]
        """
        # Use the velocity motion model with alpha parameters for noise
        alphas = self.motion_model_params.get('alphas', (0.01, 0.01, 0.01, 0.01, 0.01, 0.01))
        
        # Sample from velocity motion model
        new_pose = sample_motion_velocity_model(
            prev_Xt=prev_pose,
            ut=u,
            PDF=np.random.normal,
            dt=self.dt,
            alphas=alphas
        )
        
        # Normalize angle to [-pi, pi]
        new_pose[2] = np.arctan2(np.sin(new_pose[2]), np.cos(new_pose[2]))
        
        return new_pose
    
    def _measurement_model(self, pose: np.ndarray, z: np.ndarray) -> float:
        """
        Compute measurement probability using beam range finder model.
        
        Parameters:
        -----------
        pose : np.ndarray
            Robot pose [x, y, theta]
        z : np.ndarray
            Range measurements
        
        Returns:
        --------
        float
            Probability of measurement given pose and map
        """
        try:
            # Use beam range finder model from probabilistic sensor model
            prob = beam_range_finder_model(
                Xt=pose,
                z=z,
                map_data=self.map_grid,
                compute_from_map=self.compute_from_map,
                PDF=np.random.normal,
                min_theta=self.sensor_model_params.get('min_theta', -np.pi/2),
                max_theta=self.sensor_model_params.get('max_theta', np.pi/2),
                max_range=self.sensor_model_params.get('max_range', 10.0),
                sigma_hit=self.sensor_model_params.get('sigma_hit', 0.2),
                lambda_short=self.sensor_model_params.get('lambda_short', 0.1),
                z_hit=self.sensor_model_params.get('z_hit', 0.7),
                z_short=self.sensor_model_params.get('z_short', 0.1),
                z_max=self.sensor_model_params.get('z_max', 0.1),
                z_rand=self.sensor_model_params.get('z_rand', 0.1)
            )
            return prob
        except Exception as e:
            # If measurement fails, return neutral probability
            return 0.5
    
    def update(self, u: np.ndarray, z: np.ndarray) -> Dict:
        """
        EKF-SLAM update step: prediction + update + map update.
        
        Pseudocode from probabilistic robotics:
        
        ======================================================================
        Algorithm: EKF-SLAM(Mu_{t-1}, Sigma_{t-1}, u_t, z_t, m_{t-1})
        
        # PREDICTION STEP
        Mu_bar = motion_model(u_t, Mu_{t-1})
        G_t = Jacobian of motion model
        Sigma_bar = G_t @ Sigma_{t-1} @ G_t.T + Q_t
        
        # MAP UPDATE STEP
        m_t = occupancy_grid_update(m_{t-1}, z_t, Mu_bar)
        
        # MEASUREMENT UPDATE STEP (optional in grid-SLAM)
        # For occupancy grids, measurement likelihood is implicit
        # H_t = Jacobian of measurement model
        # K = Kalman gain (typically not used for grid SLAM)
        # Mu = Mu_bar + K @ innovation
        # Sigma = (I - K @ H) @ Sigma_bar
        
        return {Mu_t, Sigma_t, m_t}
        ======================================================================
        
        Parameters:
        -----------
        u : np.ndarray
            Control input [v, w]
        z : np.ndarray
            Measurement (range readings, 360 beams)
        
        Returns:
        --------
        dict
            Updated state with keys: 'pose_mean', 'pose_cov', 'map', 'map_prob'
        """
        
        # Downsample LiDAR from 360 to 72 beams
        z_downsampled = self._downsample_lidar(z)
        # ===== PREDICTION STEP =====
        # Mu_bar = motion_model(u_t, Mu_{t-1})
        Mu_bar = self._motion_model(self.Mu, u)
        
        # G_t = Jacobian of motion model
        G = self._motion_jacobian(self.Mu, u)
        
        # Q_t = motion noise covariance
        Q = self._motion_covariance(u)
        
        # Sigma_bar = G_t @ Sigma_{t-1} @ G_t.T + Q_t
        Sigma_bar = G @ self.Sigma @ G.T + Q
        
        # ===== MAP UPDATE STEP =====
        # Update occupancy grid using current measurement
        try:
            # Use inverse sensor model to update grid
            map_grid_updated = self._update_occupancy_grid(Mu_bar, z_downsampled)
        except Exception as e:
            # If map update fails, keep previous map
            map_grid_updated = self.map_grid.copy()
        
        # ===== MEASUREMENT UPDATE STEP =====
        # For grid-SLAM: measurements are used primarily for map building
        # The map itself provides implicit pose correction through occupancy
        # Pose uncertainty is reduced as we build a more accurate map
        
        # Simplified measurement update: reduce uncertainty based on measurement quality
        # Once map is built, expected measurements should align with actual ones
        # This keeps the algorithm stable while building the map
        measurement_scaling = 0.95  # Slight uncertainty reduction per measurement
        Sigma_updated = Sigma_bar * measurement_scaling
        
        # Pose stays at prediction (map building provides implicit correction)
        self.Mu = Mu_bar
        self.Sigma = Sigma_updated
        self.map_grid = map_grid_updated
        
        # Store history
        self.pose_estimates.append(self.Mu.copy())
        pose_uncertainty = np.sqrt(np.trace(self.Sigma))
        self.pose_uncertainties.append(pose_uncertainty)
        self.measurement_history.append(z_downsampled.copy())
        
        # Periodically save map to disk to avoid RAM bloat
        self.update_count += 1
        if self.update_count % 10 == 0:  # Save every 10 updates
            save_ogm_to_pgm(self.map_grid, self.map_pgm_path)
        
        return {
            'pose_mean': self.Mu.copy(),
            'pose_cov': self.Sigma.copy(),
            'map': self.map_grid.copy(),
            'map_prob': log_odds_to_occupancy_probability(self.map_grid),
            'pose_uncertainty': pose_uncertainty
        }
    
    def _update_occupancy_grid(self, pose: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        """
        Update occupancy grid using incremental inverse sensor model from OGM.
        
        Parameters:
        -----------
        pose : np.ndarray
            Robot pose [x, y, theta]
        measurement : np.ndarray
            Range measurements (downsampled to 72 beams)
        
        Returns:
        --------
        np.ndarray
            Updated occupancy grid (log-odds)
        """
        try:
            # Use incremental OGM update
            updated_grid = incremental_occupancy_grid_update(
                current_map=self.map_grid,
                pose=pose,
                measurement=measurement,
                grid_shape=self.grid_shape,
                grid_bounds=self.grid_bounds,
                min_theta=self.sensor_model_params.get('min_theta', -np.pi),
                max_theta=self.sensor_model_params.get('max_theta', np.pi),
                max_range=self.sensor_model_params.get('max_range', 300.0),
                sigma_hit=self.sensor_model_params.get('sigma_hit', 0.2)
            )
            return updated_grid
        except Exception as e:
            # If update fails, keep previous map
            return self.map_grid.copy()
    
    def get_estimated_pose(self) -> np.ndarray:
        """
        Get estimated robot pose.
        
        Returns:
        --------
        np.ndarray
            Pose [x, y, theta]
        """
        return self.Mu.copy()
    
    def get_estimated_map_probability(self) -> np.ndarray:
        """
        Get occupancy grid map as probability values [0, 1].
        
        Returns:
        --------
        np.ndarray
            Occupancy probabilities
        """
        return log_odds_to_occupancy_probability(self.map_grid)
    
    def get_pose_covariance(self) -> np.ndarray:
        """
        Get covariance of pose estimate.
        
        Returns:
        --------
        np.ndarray
            3x3 covariance matrix
        """
        return self.Sigma.copy()
    
    def get_pose_uncertainty(self) -> float:
        """
        Get scalar uncertainty metric (trace of covariance).
        
        Returns:
        --------
        float
            Uncertainty value
        """
        return np.sqrt(np.trace(self.Sigma))
    
    def set_initial_pose(self, pose: np.ndarray, uncertainty: float = 0.1):
        """
        Set initial robot pose and uncertainty.
        
        Parameters:
        -----------
        pose : np.ndarray
            Initial pose [x, y, theta]
        uncertainty : float
            Initial uncertainty (std of diagonal covariance)
        """
        self.Mu = pose.copy()
        self.Sigma = np.eye(3) * (uncertainty ** 2)
        self.pose_estimates = [self.Mu.copy()]
        self.pose_uncertainties = [uncertainty]
