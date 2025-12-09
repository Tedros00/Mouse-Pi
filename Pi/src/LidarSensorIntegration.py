"""
Integration module for LIDAR scans with beam range finder model.

Converts LIDAR scans to range measurements compatible with the probabilistic sensor model.
"""

import numpy as np
from ProbabilisticSensorModel import beam_range_finder_model, likelyhood_field_model


def lidar_scan_to_ranges(scan_data: np.ndarray, 
                         robot_pose: tuple,
                         num_beams: int = 360,
                         max_range: float = 5000) -> np.ndarray:
    """
    Convert raw LIDAR scan data to a normalized range array for sensor model.
    
    Interpolates scan data to a uniform angular distribution.
    
    Args:
        scan_data (np.ndarray): Nx3 array [angle, distance, quality]
        robot_pose (tuple): Robot pose (x, y, theta) - used for coordinate transformation
        num_beams (int): Number of beams to resample to (default: 360)
        max_range (float): Maximum range in mm (default: 5000)
    
    Returns:
        np.ndarray: Normalized range measurements (0-1 scale) for num_beams angles
    """
    if len(scan_data) == 0:
        return np.full(num_beams, 1.0)  # Return max range if no data
    
    # Extract angles and distances from scan
    angles = scan_data[:, 0]
    distances = scan_data[:, 1]
    
    # Normalize distances to 0-1 range
    normalized_distances = np.clip(distances / max_range, 0, 1)
    
    # Create uniform angular grid for resampling
    uniform_angles = np.linspace(0, 360, num_beams, endpoint=False)
    
    # Interpolate distances for uniform angles
    resampled_distances = np.interp(uniform_angles, angles, normalized_distances, 
                                    left=1.0, right=1.0)
    
    return resampled_distances


def lidar_scan_to_cartesian(scan_data: np.ndarray, 
                            robot_pose: tuple) -> np.ndarray:
    """
    Convert LIDAR scan to cartesian coordinates relative to world frame.
    
    Args:
        scan_data (np.ndarray): Nx3 array [angle, distance, quality]
        robot_pose (tuple): Robot pose (x, y, theta) in world frame
    
    Returns:
        np.ndarray: Nx2 array of [x, y] points in world frame
    """
    x_robot, y_robot, theta_robot = robot_pose
    
    angles_deg = scan_data[:, 0]
    distances_mm = scan_data[:, 1]
    
    # Convert angles to radians and add robot orientation
    angles_rad = np.radians(angles_deg) + theta_robot
    
    # Convert to cartesian in robot frame
    x_robot_frame = distances_mm * np.cos(angles_rad)
    y_robot_frame = distances_mm * np.sin(angles_rad)
    
    # Transform to world frame
    x_world = x_robot + x_robot_frame * np.cos(theta_robot) - y_robot_frame * np.sin(theta_robot)
    y_world = y_robot + x_robot_frame * np.sin(theta_robot) + y_robot_frame * np.cos(theta_robot)
    
    return np.column_stack([x_world, y_world])


def filter_scan_by_quality(scan_data: np.ndarray, 
                          min_quality: int = 5) -> np.ndarray:
    """
    Filter LIDAR scan points by signal quality.
    
    Args:
        scan_data (np.ndarray): Nx3 array [angle, distance, quality]
        min_quality (int): Minimum quality threshold (0-255)
    
    Returns:
        np.ndarray: Filtered scan data
    """
    if scan_data.shape[1] < 3:
        return scan_data
    
    quality = scan_data[:, 2]
    mask = quality >= min_quality
    return scan_data[mask]


def filter_scan_by_distance(scan_data: np.ndarray,
                           min_distance: float = 0,
                           max_distance: float = 5000) -> np.ndarray:
    """
    Filter LIDAR scan points by distance range.
    
    Args:
        scan_data (np.ndarray): Nx3 array [angle, distance, quality]
        min_distance (float): Minimum distance in mm
        max_distance (float): Maximum distance in mm
    
    Returns:
        np.ndarray: Filtered scan data
    """
    distances = scan_data[:, 1]
    mask = (distances >= min_distance) & (distances <= max_distance)
    return scan_data[mask]


def evaluate_scan_with_sensor_model(scan_data: np.ndarray,
                                    robot_pose: tuple,
                                    map_data,
                                    compute_from_map,
                                    num_beams: int = 360,
                                    max_range: float = 5000,
                                    sensor_params: dict = None) -> float:
    """
    Evaluate a LIDAR scan using the beam range finder model.
    
    Computes probability of observation given robot pose and map.
    
    Args:
        scan_data (np.ndarray): Nx3 array [angle, distance, quality]
        robot_pose (tuple): Robot pose (x, y, theta)
        map_data: Map data for sensor model evaluation
        compute_from_map: Function to compute expected range from map
        num_beams (int): Number of beams for evaluation
        max_range (float): Maximum range in mm
        sensor_params (dict): Sensor model parameters
                            - 'min_theta': minimum beam angle (default: -π/2)
                            - 'max_theta': maximum beam angle (default: π/2)
                            - 'sigma_hit': hit component std dev (default: 0.2)
                            - 'lambda_short': short component lambda (default: 0.1)
                            - 'z_hit': hit weight (default: 0.7)
                            - 'z_short': short weight (default: 0.1)
                            - 'z_max': max range weight (default: 0.1)
                            - 'z_rand': random weight (default: 0.1)
    
    Returns:
        float: Probability of observation given pose
    """
    if sensor_params is None:
        sensor_params = {}
    
    # Set defaults
    params = {
        'min_theta': sensor_params.get('min_theta', -np.pi/2),
        'max_theta': sensor_params.get('max_theta', np.pi/2),
        'sigma_hit': sensor_params.get('sigma_hit', 0.2),
        'lambda_short': sensor_params.get('lambda_short', 0.1),
        'z_hit': sensor_params.get('z_hit', 0.7),
        'z_short': sensor_params.get('z_short', 0.1),
        'z_max': sensor_params.get('z_max', 0.1),
        'z_rand': sensor_params.get('z_rand', 0.1),
    }
    
    # Convert scan to normalized ranges
    z_normalized = lidar_scan_to_ranges(scan_data, robot_pose, num_beams, max_range)
    
    # Denormalize for sensor model (convert back to actual mm)
    z_measured = z_normalized * max_range
    
    # Evaluate with beam range finder model
    prob = beam_range_finder_model(
        robot_pose,
        z_measured,
        map_data,
        compute_from_map,
        min_theta=params['min_theta'],
        max_theta=params['max_theta'],
        max_range=max_range,
        sigma_hit=params['sigma_hit'],
        lambda_short=params['lambda_short'],
        z_hit=params['z_hit'],
        z_short=params['z_short'],
        z_max=params['z_max'],
        z_rand=params['z_rand']
    )
    
    return prob


def evaluate_scan_with_likelihood_field(scan_data: np.ndarray,
                                       robot_pose: tuple,
                                       get_distance_to_closest_obstacle,
                                       map_data,
                                       sigma: float = 0.5,
                                       z_max: float = 5000) -> float:
    """
    Evaluate a LIDAR scan using the likelihood field model.
    
    Args:
        scan_data (np.ndarray): Nx3 array [angle, distance, quality]
        robot_pose (tuple): Robot pose (x, y, theta)
        get_distance_to_closest_obstacle: Function to compute distance to obstacle
        map_data: Map data
        sigma (float): Standard deviation for likelihood field
        z_max (float): Maximum range in mm
    
    Returns:
        float: Likelihood of observation
    """
    distances = scan_data[:, 1]
    prob = likelyhood_field_model(
        robot_pose,
        distances,
        get_distance_to_closest_obstacle,
        map_data,
        sigma=sigma,
        z_max=z_max
    )
    
    return prob


def accumulate_scans(scan_list: list,
                     robot_poses: list,
                     coordinate_frame: str = 'world') -> np.ndarray:
    """
    Accumulate multiple LIDAR scans from different robot poses.
    
    Useful for creating maps from multiple measurements.
    
    Args:
        scan_list (list): List of scan_data arrays
        robot_poses (list): List of robot poses (x, y, theta)
        coordinate_frame (str): 'robot' or 'world' frame
    
    Returns:
        np.ndarray: Accumulated point cloud in specified frame
    """
    all_points = []
    
    for scan, pose in zip(scan_list, robot_poses):
        if coordinate_frame == 'world':
            points = lidar_scan_to_cartesian(scan, pose)
        else:
            # Robot frame - just convert angles/distances to cartesian
            angles_rad = np.radians(scan[:, 0])
            distances = scan[:, 1]
            x = distances * np.cos(angles_rad)
            y = distances * np.sin(angles_rad)
            points = np.column_stack([x, y])
        
        all_points.append(points)
    
    return np.vstack(all_points) if all_points else np.array([])


def compute_scan_statistics(scan_data: np.ndarray) -> dict:
    """
    Compute statistics from a LIDAR scan.
    
    Args:
        scan_data (np.ndarray): Nx3 array [angle, distance, quality]
    
    Returns:
        dict: Statistics including min/max/mean distances, quality info
    """
    if len(scan_data) == 0:
        return {
            'num_points': 0,
            'min_distance': None,
            'max_distance': None,
            'mean_distance': None,
            'median_distance': None,
            'quality_mean': None,
        }
    
    distances = scan_data[:, 1]
    quality = scan_data[:, 2] if scan_data.shape[1] > 2 else np.zeros_like(distances)
    
    return {
        'num_points': len(scan_data),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'std_distance': float(np.std(distances)),
        'quality_mean': float(np.mean(quality)) if len(quality) > 0 else None,
        'quality_min': float(np.min(quality)) if len(quality) > 0 else None,
        'quality_max': float(np.max(quality)) if len(quality) > 0 else None,
    }
