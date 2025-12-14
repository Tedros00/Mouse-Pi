"""
Example script demonstrating LIDAR scan integration with beam range finder model.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from lidar import connect_lidar, init_lidar, capture_map, disconnect_lidar
from LidarSensorIntegration import (
    lidar_scan_to_ranges,
    lidar_scan_to_cartesian,
    filter_scan_by_quality,
    filter_scan_by_distance,
    compute_scan_statistics,
    accumulate_scans
)


def main():
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 5 + "LIDAR Sensor Integration Example" + " " * 10 + "║")
    print("╚" + "=" * 48 + "╝\n")
    
    lidar = None
    
    try:
        # Connect and initialize LIDAR
        print("Connecting to LIDAR...")
        lidar = connect_lidar(port='/dev/ttyUSB1')
        init_lidar(lidar)
        
        # Capture a scan
        print("\nCapturing LIDAR scan...")
        scan_data = capture_map(lidar, max_distance=5000, max_points=1000)
        
        # 1. Compute scan statistics
        print("\n" + "=" * 50)
        print("Scan Statistics")
        print("=" * 50)
        stats = compute_scan_statistics(scan_data)
        for key, value in stats.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"{key:20s}: {value:.2f}")
                else:
                    print(f"{key:20s}: {value}")
        
        # 2. Filter by quality
        print("\n" + "=" * 50)
        print("Filtering by Quality")
        print("=" * 50)
        filtered_quality = filter_scan_by_quality(scan_data, min_quality=5)
        print(f"Original points: {len(scan_data)}")
        print(f"Filtered points (quality >= 5): {len(filtered_quality)}")
        print(f"Retained: {len(filtered_quality) / len(scan_data) * 100:.1f}%")
        
        # 3. Filter by distance
        print("\n" + "=" * 50)
        print("Filtering by Distance")
        print("=" * 50)
        filtered_distance = filter_scan_by_distance(scan_data, min_distance=200, max_distance=3000)
        print(f"Original points: {len(scan_data)}")
        print(f"Filtered points (200-3000mm): {len(filtered_distance)}")
        print(f"Retained: {len(filtered_distance) / len(scan_data) * 100:.1f}%")
        
        # 4. Convert to ranges (for sensor model)
        print("\n" + "=" * 50)
        print("Converting to Normalized Ranges")
        print("=" * 50)
        robot_pose = (0, 0, 0)  # At origin, facing forward
        normalized_ranges = lidar_scan_to_ranges(scan_data, robot_pose, num_beams=360)
        print(f"Number of beams: {len(normalized_ranges)}")
        print(f"Min normalized range: {np.min(normalized_ranges):.3f}")
        print(f"Max normalized range: {np.max(normalized_ranges):.3f}")
        print(f"Mean normalized range: {np.mean(normalized_ranges):.3f}")
        
        # 5. Convert to cartesian
        print("\n" + "=" * 50)
        print("Converting to Cartesian Coordinates")
        print("=" * 50)
        cartesian_points = lidar_scan_to_cartesian(scan_data, robot_pose)
        print(f"Number of cartesian points: {len(cartesian_points)}")
        print(f"X range: {np.min(cartesian_points[:, 0]):.0f} to {np.max(cartesian_points[:, 0]):.0f} mm")
        print(f"Y range: {np.min(cartesian_points[:, 1]):.0f} to {np.max(cartesian_points[:, 1]):.0f} mm")
        
        # 6. Demonstrate accumulation of multiple scans
        print("\n" + "=" * 50)
        print("Accumulating Multiple Scans")
        print("=" * 50)
        # Simulate moving the robot
        poses = [
            (0, 0, 0),
            (100, 0, 0),
            (200, 0, 0),
        ]
        scans = [scan_data] * len(poses)  # Use same scan for all poses (for demo)
        
        accumulated = accumulate_scans(scans, poses, coordinate_frame='world')
        print(f"Total points accumulated: {len(accumulated)}")
        print(f"Accumulated X range: {np.min(accumulated[:, 0]):.0f} to {np.max(accumulated[:, 0]):.0f} mm")
        print(f"Accumulated Y range: {np.min(accumulated[:, 1]):.0f} to {np.max(accumulated[:, 1]):.0f} mm")
        
        print("\n" + "=" * 50)
        print("Integration example complete!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Use compute_scan_statistics() to analyze scans")
        print("2. Use lidar_scan_to_ranges() to prepare for sensor model")
        print("3. Use evaluate_scan_with_sensor_model() with your map")
        print("4. Use accumulate_scans() to build maps from multiple poses")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if lidar:
            disconnect_lidar(lidar)


if __name__ == "__main__":
    main()
