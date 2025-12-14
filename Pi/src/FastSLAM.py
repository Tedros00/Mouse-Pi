import numpy as np
import sys
import os
from typing import List, Tuple, Dict, Callable
import copy

# Add exam2 directory to path for motion and sensor models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ProbabilisticMotionModel import sample_motion_velocity_model, velocity_motion_model
from ProbabilisticSensorModel import beam_range_finder_model


class Particle:
    """
    Represents a single particle in FastSLAM.
    
    Each particle maintains:
    - A pose estimate (x, y, theta)
    - Its own occupancy grid map
    - A weight (importance sampling weight)
    """
    
    def __init__(self, pose: np.ndarray, map_grid: np.ndarray = None, weight: float = 1.0):
        """
        Initialize a particle.
        
        Parameters:
        -----------
        pose : np.ndarray
            Pose estimate [x, y, theta]
        map_grid : np.ndarray, optional
            Occupancy grid. If None, initialized as zeros.
        weight : float
            Importance sampling weight
        """
        self.pose = pose.copy()
        self.map_grid = map_grid.copy() if map_grid is not None else None
        self.weight = weight
    
    def copy(self):
        """Create a deep copy of the particle."""
        return Particle(
            pose=self.pose.copy(),
            map_grid=self.map_grid.copy() if self.map_grid is not None else None,
            weight=self.weight
        )


class FastSLAM:
    """
    FastSLAM: Fast Simultaneous Localization and Mapping using Particle Filters.
    
    Combines particle filtering with per-particle occupancy grid mapping.
    Each particle represents a hypothesis of the robot's trajectory and maintains
    its own map estimate.
    """
    
    def __init__(self,
                 num_particles: int,
                 grid_shape: Tuple[int, int],
                 grid_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                 compute_from_map: Callable,
                 motion_model_params: Dict = None,
                 sensor_model_params: Dict = None,
                 dt: float = 0.01,
                 motion_noise_std: Tuple[float, float] = (0.1, 0.05),
                 measurement_noise_std: float = 0.3):
        """
        Initialize FastSLAM algorithm.
        
        Parameters:
        -----------
        num_particles : int
            Number of particles (M in the algorithm)
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
        """
        self.num_particles = num_particles
        self.grid_shape = grid_shape
        self.grid_bounds = grid_bounds
        self.compute_from_map = compute_from_map
        self.dt = dt
        self.motion_noise_std = motion_noise_std
        self.measurement_noise_std = measurement_noise_std
        
        # Motion model parameters (velocity model with noise)
        self.motion_model_params = motion_model_params or {
            'alphas': (0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
            'dt': dt
        }
        
        # Sensor model parameters (beam range finder model)
        self.sensor_model_params = sensor_model_params or {
            'min_theta': -np.pi/2,
            'max_theta': np.pi/2,
            'max_range': 10.0,
            'sigma_hit': 0.2,
            'lambda_short': 0.1,
            'z_hit': 0.7,
            'z_short': 0.1,
            'z_max': 0.1,
            'z_rand': 0.1
        }
        
        # Initialize particles with uniform poses and empty maps
        self.particles: List[Particle] = []
        
        # Track ESS before resampling (for diagnostics)
        self.ess_before_resample = self.num_particles
        
        # Burn-in period: skip measurement weighting for first N steps
        # to let particles build diverse maps
        self.burn_in_steps = 50
        self.step_count = 0
        
        self._initialize_particles()
    
    def _initialize_particles(self):
        """Initialize particles with uniform prior."""
        (min_x, max_x), (min_y, max_y) = self.grid_bounds
        
        for _ in range(self.num_particles):
            # Random initial pose
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            theta = np.random.uniform(-np.pi, np.pi)
            pose = np.array([x, y, theta])
            
            # Equal weights initially
            weight = 1.0 / self.num_particles
            
            # Initialize empty map (0 = free, 1 = occupied)
            map_grid = np.zeros(self.grid_shape)
            
            particle = Particle(pose, map_grid, weight)
            self.particles.append(particle)
    
    def sample_motion_model(self, particle: Particle, control_input: np.ndarray) -> Particle:
        """
        Sample new pose from motion model.
        
        Parameters:
        -----------
        particle : Particle
            Current particle
        control_input : np.ndarray
            Control [v, w] - linear and angular velocity
        
        Returns:
        --------
        Particle
            New particle with sampled pose
        """
        prev_pose = particle.pose.copy()
        v, w = control_input
        
        # Add Gaussian noise to control inputs
        v_noisy = v + np.random.normal(0, self.motion_noise_std[0])
        w_noisy = w + np.random.normal(0, self.motion_noise_std[1])
        
        # Sample new pose using velocity motion model with noise
        new_pose = sample_motion_velocity_model(
            prev_pose,
            (v_noisy, w_noisy),
            PDF=np.random.normal,
            dt=self.dt,
            alphas=self.motion_model_params['alphas']
        )
        
        # Create new particle with sampled pose, keep map from previous
        return Particle(new_pose, particle.map_grid.copy(), particle.weight)
    
    def measurement_model_map(self, particle: Particle, measurement: np.ndarray) -> float:
        """
        Compute measurement likelihood given particle's pose and map.
        Simple Gaussian comparison between actual and expected measurements.
        
        Parameters:
        -----------
        particle : Particle
            Current particle
        measurement : np.ndarray
            Measurement vector z (range readings)
        
        Returns:
        --------
        float
            Weight w_t = p(z_t | x_t, m_t)
        """
        pose = particle.pose
        map_grid = particle.map_grid
        x, y, theta = pose
        
        # Downsample to 72 beams for faster computation
        beam_indices = np.linspace(0, len(measurement)-1, 72, dtype=int)
        measurement = measurement[beam_indices]
        num_beams = len(measurement)
        (min_x, max_x), (min_y, max_y) = self.grid_bounds
        rows, cols = self.grid_shape
        
        expected_ranges = np.full(num_beams, self.sensor_model_params.get('max_range', 15.0))
        
        # Raycast on particle's map to get expected measurements
        cell_size_x = (max_x - min_x) / cols
        cell_size_y = (max_y - min_y) / rows
        max_range = self.sensor_model_params.get('max_range', 15.0)
        step_size = 0.02  # Increased from 0.005 for speed
        
        for i in range(num_beams):
            ray_angle = theta + (i / num_beams) * 2 * np.pi
            cos_angle = np.cos(ray_angle)
            sin_angle = np.sin(ray_angle)
            
            for range_m in np.arange(0, max_range, step_size):
                check_x = x + range_m * cos_angle
                check_y = y + range_m * sin_angle
                
                grid_col = int((check_x - min_x) / cell_size_x)
                grid_row = int((check_y - min_y) / cell_size_y)
                
                if grid_col < 0 or grid_col >= cols or grid_row < 0 or grid_row >= rows:
                    expected_ranges[i] = range_m
                    break
                
                if map_grid[grid_row, grid_col] > 0.5:
                    expected_ranges[i] = range_m
                    break
        
        # Gaussian likelihood in log-space
        sigma = 0.5  # 50cm std
        diff = np.abs(measurement - expected_ranges)
        
        # Log-space computation to avoid underflow
        log_likelihoods = -0.5 * (diff / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        log_weight = np.sum(log_likelihoods)
        
        # Convert back with clipping
        log_weight = np.clip(log_weight, -700, 700)
        weight = np.exp(log_weight)
        weight = np.clip(weight, 1e-100, 1e100)
        
        return weight

    
    def motion_model_likelihood(self, prev_pose: np.ndarray, curr_pose: np.ndarray, control: np.ndarray) -> float:
        """
        Compute likelihood of transition from prev_pose to curr_pose given control input.
        How well did the motion model predict the actual motion?
        
        Parameters:
        -----------
        prev_pose : np.ndarray
            Previous pose [x, y, theta]
        curr_pose : np.ndarray
            Current pose [x, y, theta]
        control : np.ndarray
            Control input [v, w]
        
        Returns:
        --------
        float
            Likelihood of this motion
        """
        v, w = control
        
        # Predict where we should be using ideal motion model
        predicted_pose = velocity_motion_model(prev_pose, (v, w), self.dt, self.motion_model_params['alphas'])
        
        # Compute distance between predicted and actual
        pose_diff = curr_pose - predicted_pose
        pos_diff = np.linalg.norm(pose_diff[:2])
        angle_diff = abs(pose_diff[2])
        
        # Gaussian likelihood on position and angle
        sigma_pos = 0.2  # 20cm std
        sigma_angle = 0.1  # 0.1 rad std
        
        likelihood_pos = np.exp(-0.5 * (pos_diff / sigma_pos) ** 2)
        likelihood_angle = np.exp(-0.5 * (angle_diff / sigma_angle) ** 2)
        
        return likelihood_pos * likelihood_angle
    
    def normalize_weights(self):
        """Normalize particle weights to sum to 1."""
        total_weight = sum(p.weight for p in self.particles)
        
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # Uniform weights if all weights were zero
            uniform_weight = 1.0 / self.num_particles
            for particle in self.particles:
                particle.weight = uniform_weight
    
        
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # Uniform weights if all weights were zero
            uniform_weight = 1.0 / self.num_particles
            for particle in self.particles:
                particle.weight = uniform_weight
    
    def resample_particles(self) -> List[Particle]:
        """
        Resample particles proportional to their weights.
        Uses low-variance resampling to reduce variance.
        
        Returns:
        --------
        List[Particle]
            Resampled particles
        """
        weights = np.array([p.weight for p in self.particles])
        
        # Low-variance resampling
        n = len(self.particles)
        indices = np.zeros(n, dtype=int)
        cumsum = np.cumsum(weights)
        
        i = 0
        u = np.random.uniform(0, 1.0 / n)
        
        for j in range(n):
            while u > cumsum[i]:
                i += 1
            indices[j] = min(i, n - 1)
        
        # Create resampled particles with equal weights
        resampled = []
        for idx in indices:
            particle = self.particles[idx].copy()
            particle.weight = 1.0 / n
            resampled.append(particle)
        
        return resampled
    
    def update_occupancy_grid(self, particle: Particle, measurement: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Update particle's occupancy grid with LiDAR measurements using inverse sensor model.
        
        For each ray:
        - If z >= z_max (no hit): mark cells along ray as FREE
        - If z < z_max (hit detected): mark hit cell as OCCUPIED
        
        Parameters:
        -----------
        particle : Particle
            Current particle
        measurement : np.ndarray
            Range readings from all LiDAR rays
        pose : np.ndarray
            Current pose [x, y, theta]
        
        Returns:
        --------
        np.ndarray
            Updated occupancy grid
        """
        map_grid = particle.map_grid.copy()
        x, y, theta = pose
        
        # Grid bounds and shape
        (min_x, max_x), (min_y, max_y) = self.grid_bounds
        rows, cols = self.grid_shape
        max_range = 14.9  # Threshold for "hit vs no-hit"
        
        # For each ray (360 rays total)
        num_rays = len(measurement)
        
        for i, ray_range in enumerate(measurement):
            # Skip invalid measurements
            if ray_range <= 0:
                continue
            
            # Angle of this ray in world frame
            ray_angle = theta + (i / num_rays) * 2 * np.pi
            cos_angle = np.cos(ray_angle)
            sin_angle = np.sin(ray_angle)
            
            if ray_range >= max_range:
                # Ray reached max range without hitting: mark path as FREE
                # Sample along the ray to mark cells as free
                num_samples = 8
                for sample_dist in np.linspace(0.1, ray_range, num_samples):
                    sample_x = x + sample_dist * cos_angle
                    sample_y = y + sample_dist * sin_angle
                    
                    # Convert to grid cell coordinates
                    grid_col = int((sample_x - min_x) / (max_x - min_x) * cols)
                    grid_row = int((sample_y - min_y) / (max_y - min_y) * rows)
                    
                    # Clamp to bounds
                    grid_col = int(np.clip(grid_col, 0, cols - 1))
                    grid_row = int(np.clip(grid_row, 0, rows - 1))
                    
                    # Mark as free (0.0)
                    map_grid[grid_row, grid_col] = 0.0
            else:
                # Ray hit something: mark path to hit point as FREE, then hit point as OCCUPIED
                # Sample along the ray UP TO (but not including) the hit point
                num_samples = 8
                for sample_dist in np.linspace(0.1, ray_range * 0.95, num_samples):
                    sample_x = x + sample_dist * cos_angle
                    sample_y = y + sample_dist * sin_angle
                    
                    # Convert to grid cell coordinates
                    grid_col = int((sample_x - min_x) / (max_x - min_x) * cols)
                    grid_row = int((sample_y - min_y) / (max_y - min_y) * rows)
                    
                    # Clamp to bounds
                    grid_col = int(np.clip(grid_col, 0, cols - 1))
                    grid_row = int(np.clip(grid_row, 0, rows - 1))
                    
                    # Mark as free (0.0)
                    map_grid[grid_row, grid_col] = 0.0
                
                # Mark hit point as OCCUPIED
                hit_x = x + ray_range * cos_angle
                hit_y = y + ray_range * sin_angle
                
                # Convert to grid cell coordinates
                grid_col = int((hit_x - min_x) / (max_x - min_x) * cols)
                grid_row = int((hit_y - min_y) / (max_y - min_y) * rows)
                
                # Clamp to bounds
                grid_col = int(np.clip(grid_col, 0, cols - 1))
                grid_row = int(np.clip(grid_row, 0, rows - 1))
                
                # Mark as occupied (1.0)
                map_grid[grid_row, grid_col] = 1.0
        
        return map_grid
    
    def update(self, control_input: np.ndarray, measurement: np.ndarray):
        """
        Perform one FastSLAM update step.
        
        Algorithm:
        1. For each particle:
           a. Sample new pose from motion model
           b. Compute measurement likelihood
           c. Update occupancy grid
           d. Store in predicted set with weight
        2. Resample particles based on weights
        
        Parameters:
        -----------
        control_input : np.ndarray
            Control input [v, w] - linear and angular velocity
        measurement : np.ndarray
            Measurement vector z (range readings)
        """
        predicted_particles = []
        
        # ======== Particle Sampling Phase ========
        for k in range(self.num_particles):
            particle = self.particles[k]
            
            # Step 1: Sample motion model
            particle_bar = self.sample_motion_model(particle, control_input)
            
            # Step 2: Weight particles
            # During burn-in, use motion model likelihood. After burn-in, use measurement model.
            if self.step_count < self.burn_in_steps:
                # Motion model only: particles that move as expected get higher weight
                alphas_4 = self.motion_model_params['alphas'][:4]
                weight = velocity_motion_model(particle_bar.pose, control_input, particle.pose, 
                                              np.random.normal, self.dt, alphas_4)
                particle_bar.weight = max(weight, 1e-20)
            else:
                # Measurement model: particles with maps get weighted by measurement fit
                weight = self.measurement_model_map(particle_bar, measurement)
                particle_bar.weight = weight
            
            # Step 3: Update occupancy grid
            particle_bar.map_grid = self.update_occupancy_grid(particle_bar, measurement, particle_bar.pose)
            
            predicted_particles.append(particle_bar)
        
        # ======== Normalize Weights ========
        self.particles = predicted_particles
        
        # Store ESS before resampling
        self.ess_before_resample = self.get_effective_sample_size()
        
        self.normalize_weights()
        
        # ======== Resampling Phase ========
        self.particles = self.resample_particles()
        
        # Increment step counter
        self.step_count += 1
    
    def get_estimated_pose(self) -> np.ndarray:
        """
        Get estimated robot pose as weighted average of particles.
        
        Returns:
        --------
        np.ndarray
            Estimated pose [x, y, theta]
        """
        if len(self.particles) == 0:
            return np.array([0, 0, 0])
        
        weighted_pose = np.zeros(3)
        total_weight = 0
        
        for particle in self.particles:
            weighted_pose += particle.weight * particle.pose
            total_weight += particle.weight
        
        if total_weight > 0:
            weighted_pose /= total_weight
        
        return weighted_pose
    

    
    def get_particle_poses(self) -> np.ndarray:
        """
        Get all particle poses.
        
        Returns:
        --------
        np.ndarray
            Array of shape (num_particles, 3) with poses [x, y, theta]
        """
        return np.array([p.pose for p in self.particles])
    
    def get_particle_weights(self) -> np.ndarray:
        """
        Get all particle weights.
        
        Returns:
        --------
        np.ndarray
            Array of shape (num_particles,) with weights
        """
        return np.array([p.weight for p in self.particles])
    
    def get_effective_sample_size(self) -> float:
        """
        Compute effective sample size (ESS) to assess particle degeneracy.
        
        Returns:
        --------
        float
            ESS in range [0, num_particles]. Higher is better.
        """
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(weights ** 2)



if __name__ == "__main__":
    print("FastSLAM Algorithm Implementation")
    print("=" * 50)
    print("\nFastSLAM combines particle filtering with per-particle occupancy")
    print("grid mapping for simultaneous localization and mapping.")
    print("\nKey features:")
    print("- M particles, each with pose estimate and map")
    print("- Motion model: Sampled from velocity motion model")
    print("- Sensor model: Beam range finder model")
    print("- Map updates: Inverse sensor model for occupancy grid")
    print("- Resampling: Low-variance resampling based on weights")
