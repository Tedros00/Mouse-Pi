"""
Quick start guide for LIDAR Sensor Integration with Beam Range Finder Model.

This module provides functions to integrate raw LIDAR scans with probabilistic
sensor models for localization and mapping.
"""

# ============================================================================
# BASIC USAGE
# ============================================================================

"""
1. CAPTURE A SCAN AND CONVERT TO RANGES
"""

from lidar import connect_lidar, init_lidar, capture_map, disconnect_lidar
from LidarSensorIntegration import lidar_scan_to_ranges, compute_scan_statistics

# Connect to LIDAR
lidar = connect_lidar('/dev/ttyUSB1')
init_lidar(lidar)

# Capture a scan
scan_data = capture_map(lidar, max_distance=5000, max_points=None)

# Get statistics
stats = compute_scan_statistics(scan_data)
print(f"Captured {stats['num_points']} points")
print(f"Distance range: {stats['min_distance']} to {stats['max_distance']} mm")

# Disconnect
disconnect_lidar(lidar)


"""
2. PREPARE SCAN FOR SENSOR MODEL
"""

from LidarSensorIntegration import lidar_scan_to_ranges

robot_pose = (0, 0, 0)  # (x, y, theta)
normalized_ranges = lidar_scan_to_ranges(scan_data, robot_pose, num_beams=360)
# Now use normalized_ranges with beam_range_finder_model


"""
3. EVALUATE SCAN WITH BEAM RANGE FINDER MODEL
"""

from LidarSensorIntegration import evaluate_scan_with_sensor_model
from ProbabilisticSensorModel import beam_range_finder_model

# You need to provide:
# - map_data: Your occupancy grid or map
# - compute_from_map: Function that computes expected range for (x, y, angle)

def compute_expected_range(x, y, angle, map_data, max_range):
    # Implement ray casting through your map
    # Return expected range measurement
    pass

# Evaluate scan
robot_pose = (0, 0, 0)  # Robot position and orientation
prob = evaluate_scan_with_sensor_model(
    scan_data,
    robot_pose,
    map_data,
    compute_expected_range,
    num_beams=360,
    max_range=5000
)
print(f"Observation likelihood: {prob}")


"""
4. FILTER AND PROCESS SCANS
"""

from LidarSensorIntegration import (
    filter_scan_by_quality,
    filter_scan_by_distance
)

# Remove low-quality points
filtered = filter_scan_by_quality(scan_data, min_quality=5)

# Keep only points within 200-3000mm
filtered = filter_scan_by_distance(filtered, min_distance=200, max_distance=3000)


"""
5. CONVERT TO CARTESIAN COORDINATES
"""

from LidarSensorIntegration import lidar_scan_to_cartesian

robot_pose = (100, 200, 0)
cartesian_points = lidar_scan_to_cartesian(scan_data, robot_pose)
# Each point: [x, y] in world frame


"""
6. ACCUMULATE SCANS FOR MAPPING
"""

from LidarSensorIntegration import accumulate_scans

# Multiple scans from different poses
scans = [scan1, scan2, scan3]
poses = [(0, 0, 0), (100, 0, 0), (200, 0, 0)]

# Accumulate in world frame
map_points = accumulate_scans(scans, poses, coordinate_frame='world')


# ============================================================================
# INTEGRATION WITH PARTICLE FILTER LOCALIZATION
# ============================================================================

"""
Example: Using with particle filter
"""

import numpy as np
from lidar import connect_lidar, init_lidar, stream_map_to_pc
from LidarSensorIntegration import evaluate_scan_with_sensor_model

def particle_filter_step(particles, scan_data, map_data, compute_fn):
    """
    Update particle weights based on LIDAR observation.
    
    Args:
        particles: Nx3 array of particle poses [x, y, theta]
        scan_data: Raw LIDAR scan
        map_data: Occupancy grid
        compute_fn: Function to compute expected ranges
    
    Returns:
        weights: Updated particle weights
    """
    weights = np.zeros(len(particles))
    
    for i, pose in enumerate(particles):
        # Evaluate how well this scan matches each particle's pose
        likelihood = evaluate_scan_with_sensor_model(
            scan_data,
            pose,
            map_data,
            compute_fn,
            num_beams=360
        )
        weights[i] = likelihood
    
    # Normalize weights
    weights /= np.sum(weights) + 1e-10
    return weights


# ============================================================================
# KEY FUNCTIONS SUMMARY
# ============================================================================

"""
Core Functions:

1. lidar_scan_to_ranges(scan_data, robot_pose, num_beams=360, max_range=5000)
   - Converts raw scan to normalized range array for sensor model
   - Output: array of normalized distances (0-1)

2. lidar_scan_to_cartesian(scan_data, robot_pose)
   - Converts scan to world-frame cartesian coordinates
   - Output: Nx2 array of [x, y] points

3. filter_scan_by_quality(scan_data, min_quality=5)
   - Removes low signal-quality points
   - Output: filtered scan array

4. filter_scan_by_distance(scan_data, min_distance=0, max_distance=5000)
   - Keeps only points within distance range
   - Output: filtered scan array

5. compute_scan_statistics(scan_data)
   - Computes min, max, mean, median distances, quality info
   - Output: dict with statistics

6. accumulate_scans(scan_list, robot_poses, coordinate_frame='world')
   - Combines multiple scans from different poses
   - Output: Nx2 array of accumulated cartesian points

7. evaluate_scan_with_sensor_model(scan_data, robot_pose, map_data, ...)
   - Computes likelihood using beam range finder model
   - Output: float probability

8. evaluate_scan_with_likelihood_field(scan_data, robot_pose, ...)
   - Computes likelihood using likelihood field model
   - Output: float likelihood
"""

# ============================================================================
# COMMON WORKFLOWS
# ============================================================================

# Workflow 1: Real-time localization
"""
lidar = connect_lidar()
init_lidar(lidar)

while True:
    scan = capture_map(lidar, max_distance=5000, max_points=1000)
    
    # Update particle filter weights
    particles = ...  # your particle set
    weights = particle_filter_step(particles, scan, map_data, compute_fn)
    
    # Resample particles based on weights
    particles = resample_particles(particles, weights)
"""

# Workflow 2: Offline mapping
"""
scans = [scan1, scan2, scan3]  # Captured scans
poses = [pose1, pose2, pose3]  # Known/estimated poses

# Accumulate into map
map_points = accumulate_scans(scans, poses, coordinate_frame='world')

# Save map
np.save('map.npy', map_points)
"""

# Workflow 3: Real-time visualization streaming
"""
lidar = connect_lidar()
init_lidar(lidar)

stream_map_to_pc(lidar, pc_ip='192.168.1.101', pc_port=5006)
# Run visualize_lidar_stream.py on your PC
"""
