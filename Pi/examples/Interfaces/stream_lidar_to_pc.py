"""
Stream LIDAR scans to PC in real-time with robot pose and statistics.

Run this script on the AMR to continuously stream LIDAR data to your PC.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from lidar import connect_lidar, init_lidar, disconnect_lidar
from LidarSensorIntegration import compute_scan_statistics, lidar_scan_to_ranges
import socket
import json
import time


def apply_beam_model_filter(scan_array, max_distance=5000, 
                           sigma_hit=0.2, lambda_short=0.1,
                           z_hit=0.7, z_short=0.1, z_max=0.1, z_rand=0.1):
    """
    Filter scan points using beam range finder model criteria.
    
    Applies probabilistic filtering based on beam model components:
    - Hit model: Points close to expected distance
    - Short model: Points closer than expected (occlusion)
    - Max range: Points at maximum range
    - Random: Uniform noise
    
    Args:
        scan_array: Nx3 array [angle, distance, quality]
        max_distance: Maximum valid range
        Other args: Beam model weights
    
    Returns:
        Filtered scan array with beam model confidence weights
    """
    if len(scan_array) == 0:
        return scan_array
    
    distances = scan_array[:, 1]
    quality = scan_array[:, 2]
    
    # Calculate beam model confidence for each point
    confidences = np.zeros(len(scan_array))
    
    for i, (distance, qual) in enumerate(zip(distances, quality)):
        # Normalize quality to probability (0-15 -> 0-1)
        q_prob = qual / 15.0
        
        # Hit component: Points with good signal quality
        p_hit = q_prob if 0 <= distance <= max_distance else 0
        
        # Short component: Penalize points much closer than neighbors
        # (indicates occlusion/noise)
        p_short = 0
        if distance < max_distance * 0.5:  # Points in close range
            p_short = lambda_short * np.exp(-lambda_short * distance / 1000.0)
        
        # Max range component: Points at max range (low confidence)
        p_max = 0.1 if distance >= max_distance * 0.95 else 0
        
        # Random component: Uniform noise (low confidence)
        p_rand = 0.05 if 0 <= distance <= max_distance else 0
        
        # Combined confidence (weighted mixture model)
        confidence = (z_hit * p_hit + 
                     z_short * p_short + 
                     z_max * p_max + 
                     z_rand * p_rand)
        
        confidences[i] = confidence
    
    # Filter: keep points with confidence above threshold
    # Threshold adaptively set based on quality distribution
    quality_threshold = np.mean(confidences) * 0.3  # Keep if above 30% of mean
    
    valid_mask = confidences > quality_threshold
    filtered_scan = scan_array[valid_mask].copy()
    
    # Update weights with beam model confidence
    if len(filtered_scan) > 0:
        filtered_scan[:, 2] = confidences[valid_mask]  # Replace with confidence weights
    
    return filtered_scan


def stream_lidar_with_pose(lidar, pc_ip: str, pc_port: int = 5006,
                          robot_pose_callback=None,
                          max_distance: float = 5000):
    """
    Stream LIDAR scans with robot pose and statistics to PC.
    
    Args:
        lidar: Connected RPLidar object
        pc_ip: Target PC IP address
        pc_port: Target PC port
        robot_pose_callback: Optional function that returns (x, y, theta) robot pose
        max_distance: Maximum range in mm
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Connecting to PC at {pc_ip}:{pc_port}...")
        sock.connect((pc_ip, pc_port))
        print("Connected to PC! Starting stream...")
        
        frame_count = 0
        
        try:
            for scan in lidar.iter_scans(max_buf_meas=3000, min_len=5):
                scan_data = []
                
                for (quality, angle, distance) in scan:
                    if distance <= max_distance:
                        scan_data.append([angle, distance, quality])
                
                # Convert to numpy array for analysis
                scan_array = np.array(scan_data)
                
                # Apply beam model filtering to remove low-confidence points
                # This filters based on hit/short/max/random model components
                filtered_scan = apply_beam_model_filter(scan_array, max_distance=max_distance)
                
                # filtered_scan now contains beam model confidence weights (0-1)
                # instead of raw quality values
                
                # Get robot pose (default to origin if no callback provided)
                if robot_pose_callback:
                    robot_pose = robot_pose_callback()
                else:
                    robot_pose = (0, 0, 0)
                
                # Compute scan statistics on filtered data
                stats = compute_scan_statistics(filtered_scan)
                
                # Convert to normalized ranges for beam model
                normalized_ranges = lidar_scan_to_ranges(filtered_scan, robot_pose, num_beams=360)
                
                # Prepare data packet
                data = {
                    'timestamp': time.time(),
                    'frame': frame_count,
                    'robot_pose': {
                        'x': robot_pose[0],
                        'y': robot_pose[1],
                        'theta': robot_pose[2]
                    },
                    'points': scan_data,  # Original points with quality (0-15)
                    'weighted_points': filtered_scan.tolist(),  # Points with beam model confidence (0-1)
                    'num_points': len(scan_data),
                    'statistics': {
                        'min_distance': stats['min_distance'],
                        'max_distance': stats['max_distance'],
                        'mean_distance': stats['mean_distance'],
                        'median_distance': stats['median_distance'],
                        'std_distance': stats['std_distance'],
                        'quality_mean': stats['quality_mean'],
                        'quality_min': stats['quality_min'],
                        'quality_max': stats['quality_max'],
                    },
                    'normalized_ranges': normalized_ranges.tolist()
                }
                
                try:
                    message = json.dumps(data) + '\n'
                    sock.sendall(message.encode('utf-8'))
                    frame_count += 1
                    
                    if frame_count % 5 == 0:
                        print(f"Frame {frame_count}: {len(scan_data)} points | "
                              f"Distance: {stats['mean_distance']:.0f}±{stats['std_distance']:.0f}mm | "
                              f"Pose: ({robot_pose[0]:.1f}, {robot_pose[1]:.1f}, {np.degrees(robot_pose[2]):.1f}°)")
                except Exception as e:
                    print(f"Connection lost: {e}")
                    break
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            sock.close()
            print(f"Stream ended. Total frames sent: {frame_count}")
    
    except Exception as e:
        print(f"Failed to establish connection to PC: {e}")
        raise


def main():
    # Configuration - CHANGE THIS TO YOUR PC'S IP ADDRESS
    PC_IP = '192.168.1.102'  # Replace with your PC's IP address
    PC_PORT = 5006
    
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 8 + "LIDAR Stream with Pose to PC" + " " * 10 + "║")
    print("╚" + "=" * 48 + "╝\n")
    
    print(f"Target PC: {PC_IP}:{PC_PORT}")
    print("Make sure the PC visualization script is running first!\n")
    
    lidar = None
    try:
        # Connect and initialize LIDAR
        lidar = connect_lidar(port='/dev/ttyUSB1')
        init_lidar(lidar)
        
        # Define a robot pose callback (modify as needed for your system)
        def get_robot_pose():
            # TODO: Connect to your odometry/localization system
            # For now, return default pose at origin
            return (0, 0, 0)
        
        # Stream data with pose and statistics
        stream_lidar_with_pose(lidar, pc_ip=PC_IP, pc_port=PC_PORT,
                              robot_pose_callback=get_robot_pose,
                              max_distance=5000)
    
    except KeyboardInterrupt:
        print("\n\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if lidar:
            disconnect_lidar(lidar)
        print("Stream ended")


if __name__ == "__main__":
    main()
