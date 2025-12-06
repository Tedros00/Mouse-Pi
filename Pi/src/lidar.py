#!/usr/bin/env python3
"""
LIDAR Scan Utility - Simple parametric function
Captures LIDAR scans with customizable parameters
"""
from rplidar import RPLidar
import numpy as np
from typing import List, Tuple

def get_lidar_scan(port: str = '/dev/ttyUSB1',
                   min_distance: int = 0,
                   max_distance: int = 5000,
                   fov_start: float = 0,
                   fov_end: float = 360,
                   point_density: float = 1.0,
                   num_scans: int = 1) -> List[List[Tuple[float, float, int]]]:
    """
    Capture LIDAR scans with custom parameters
    
    Parameters:
    port : str
        Serial port for LIDAR (default: /dev/ttyUSB1)
    min_distance : int
        Minimum distance threshold in mm (default: 0)
    max_distance : int
        Maximum distance threshold in mm (default: 5000)
    fov_start : float
        Start angle of field of view in degrees (default: 0)
    fov_end : float
        End angle of field of view in degrees (default: 360)
    point_density : float
        Point density (0.0-1.0). 1.0 = all points, 0.5 = every other point (default: 1.0)
    num_scans : int
        Number of complete scans to capture (default: 1)
    
    Returns:
    List of scans, each scan is a list of (angle, distance, quality) tuples
    """
    
    # Input validation
    if not (0 <= point_density <= 1.0):
        raise ValueError("point_density must be between 0.0 and 1.0")
    
    if not (0 <= fov_start < 360 and 0 <= fov_end <= 360):
        raise ValueError("FOV angles must be between 0 and 360 degrees")
    
    # Convert point density to skip rate
    skip_rate = int(1.0 / point_density) if point_density > 0 else 1
    
    all_scans = []
    lidar = RPLidar(port)
    
    try:
        print(f"Connecting to LIDAR on {port}...")
        lidar.connect()
        print("Connected")
        
        scan_count = 0
        iterator = lidar.iter_scans()
        
        for scan in iterator:
            filtered_scan = []
            
            # Process each measurement in the scan
            for i, (quality, angle, distance) in enumerate(scan):
                # Apply point density filter (skip points based on density)
                if i % skip_rate != 0:
                    continue
                
                # Apply distance filter
                if not (min_distance <= distance <= max_distance):
                    continue
                
                # Apply FOV filter
                if fov_start <= fov_end:
                    # Normal case: FOV doesn't wrap around
                    if not (fov_start <= angle <= fov_end):
                        continue
                else:
                    # Wraparound case: e.g., 350-10 degrees
                    if not (angle >= fov_start or angle <= fov_end):
                        continue
                
                # Add filtered point (angle, distance, quality)
                filtered_scan.append((angle, distance, quality))
            
            if filtered_scan:
                all_scans.append(filtered_scan)
                scan_count += 1
                print(f"Scan {scan_count}: {len(filtered_scan)} points")
                
                if scan_count >= num_scans:
                    break
    
    except KeyboardInterrupt:
        print(f"Stopped by user. Captured {scan_count} scans.")
    except Exception as e:
        print(f"Error during scan: {e}")
    finally:
        lidar.stop()
        lidar.disconnect()
        print("LIDAR disconnected")
    
    return all_scans


def get_lidar_scan_numpy(port: str = '/dev/ttyUSB1',
                         min_distance: int = 0,
                         max_distance: int = 5000,
                         fov_start: float = 0,
                         fov_end: float = 360,
                         point_density: float = 1.0,
                         num_scans: int = 1) -> List[np.ndarray]:
    """
    Capture LIDAR scans and return as numpy arrays
    
    Returns:
    List of numpy arrays with shape (N, 3) containing (angle, distance, quality)
    """
    scans = get_lidar_scan(port, min_distance, max_distance, fov_start, fov_end, 
                           point_density, num_scans)
    
    return [np.array(scan) for scan in scans]


def get_lidar_scan_cartesian(port: str = '/dev/ttyUSB1',
                              min_distance: int = 0,
                              max_distance: int = 5000,
                              fov_start: float = 0,
                              fov_end: float = 360,
                              point_density: float = 1.0,
                              num_scans: int = 1) -> List[np.ndarray]:
    """
    Capture LIDAR scans and convert to Cartesian coordinates (x, y, quality)
    
    Returns:
    List of numpy arrays with shape (N, 3) containing (x, y, quality)
    """
    scans = get_lidar_scan(port, min_distance, max_distance, fov_start, fov_end, 
                           point_density, num_scans)
    
    cartesian_scans = []
    for scan in scans:
        points = []
        for angle, distance, quality in scan:
            # Convert polar to Cartesian
            rad = np.radians(angle)
            x = distance * np.cos(rad)
            y = distance * np.sin(rad)
            points.append([x, y, quality])
        
        cartesian_scans.append(np.array(points))
    
    return cartesian_scans


# Example usage
if __name__ == '__main__':
    # Example 1: Simple scan with distance filter
    print("=== Example 1: Distance filtered scan ===")
    scans = get_lidar_scan(
        min_distance=100,
        max_distance=3000,
        num_scans=1
    )
    print(f"Captured {len(scans)} scan(s)\n")
    
    # Example 2: Front-facing FOV (120 degrees)
    print("=== Example 2: Front-facing FOV ===")
    scans = get_lidar_scan(
        min_distance=200,
        max_distance=4000,
        fov_start=300,
        fov_end=60,
        num_scans=1
    )
    print(f"Captured {len(scans)} scan(s)\n")
    
    # Example 3: Low density scan
    print("=== Example 3: Low density scan (25% points) ===")
    scans = get_lidar_scan(
        min_distance=150,
        max_distance=2500,
        point_density=0.25,
        num_scans=1
    )
    print(f"Captured {len(scans)} scan(s)\n")
