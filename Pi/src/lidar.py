"""
RPlidar interface module for capturing and processing LIDAR data.
"""

from rplidar import RPLidar
import numpy as np
import socket
import json
import time


def connect_lidar(port: str = '/dev/ttyUSB1', baudrate: int = 115200) -> RPLidar:
    """
    Connect to the RPlidar device.
    
    Args:
        port (str): Serial port path (default: /dev/ttyUSB1)
        baudrate (int): Serial communication baudrate (default: 115200)
    
    Returns:
        RPLidar: Connected RPlidar object
    
    Raises:
        Exception: If connection fails
    """
    try:
        lidar = RPLidar(port, baudrate)
        print(f"Successfully connected to RPlidar on {port}")
        return lidar
    except Exception as e:
        print(f"Failed to connect to RPlidar: {e}")
        raise


def init_lidar(lidar: RPLidar) -> None:
    """
    Initialize the RPlidar device.
    
    Performs startup sequence including getting device info and starting scanning.
    
    Args:
        lidar (RPLidar): Connected RPlidar object
    
    Raises:
        Exception: If initialization fails
    """
    try:
        # Get device info
        info = lidar.get_info()
        print(f"LIDAR Info: {info}")
        
        # Get device health
        health = lidar.get_health()
        print(f"LIDAR Health: {health}")
        
        # Start motor
        lidar.start_motor()
        print("LIDAR motor started")
        
        print("LIDAR initialization complete")
    except Exception as e:
        print(f"Failed to initialize LIDAR: {e}")
        raise


def capture_map(lidar: RPLidar, max_distance: float = 5000, max_points: int = None) -> np.ndarray:
    """
    Capture a complete scan/map from the RPlidar.
    
    Collects scan data until a full rotation is completed or max_points is reached.
    
    Args:
        lidar (RPLidar): Connected and initialized RPlidar object
        max_distance (float): Maximum distance in mm to include in scan (default: 5000)
        max_points (int): Maximum number of points to collect. If None, captures one full rotation
    
    Returns:
        np.ndarray: Nx3 array where each row is [angle, distance, quality]
                    - angle: in degrees (0-360)
                    - distance: in mm
                    - quality: signal quality (0-255)
    """
    try:
        print("Starting LIDAR scan...")
        scan_data = []
        
        for scan in lidar.iter_scans(max_buf_meas=3000, min_len=5):
            for (quality, angle, distance) in scan:
                # Filter by max distance
                if distance <= max_distance:
                    scan_data.append([angle, distance, quality])
            
            # Check if we've reached max points
            if max_points and len(scan_data) >= max_points:
                scan_data = scan_data[:max_points]
                break
        
        print(f"Scan complete. Captured {len(scan_data)} points")
        return np.array(scan_data)
    
    except Exception as e:
        print(f"Failed to capture map: {e}")
        raise


def disconnect_lidar(lidar: RPLidar) -> None:
    """
    Safely disconnect from the RPlidar device.
    
    Args:
        lidar (RPLidar): Connected RPLidar object
    """
    try:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        print("LIDAR disconnected safely")
    except Exception as e:
        print(f"Error during disconnection: {e}")


def send_map_to_pc(scan_data: np.ndarray, pc_ip: str, pc_port: int = 5006) -> None:
    """
    Send LIDAR scan data to PC via socket connection.
    
    Converts scan data to JSON format and sends to PC for visualization.
    
    Args:
        scan_data (np.ndarray): Scan data array with shape (N, 3) containing [angle, distance, quality]
        pc_ip (str): IP address of the PC
        pc_port (int): Port number on PC (default: 5006)
    
    Raises:
        Exception: If connection or transmission fails
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Connecting to PC at {pc_ip}:{pc_port}...")
        sock.connect((pc_ip, pc_port))
        print("Connected to PC!")
        
        # Convert numpy array to list for JSON serialization
        points = scan_data.tolist()
        
        # Prepare data in JSON format
        data = {
            'timestamp': time.time(),
            'num_points': len(points),
            'points': points  # Each point: [angle, distance, quality]
        }
        
        # Send data
        message = json.dumps(data) + '\n'
        sock.sendall(message.encode('utf-8'))
        print(f"Sent {len(points)} scan points to PC")
        
        sock.close()
    except Exception as e:
        print(f"Failed to send map to PC: {e}")
        raise


def stream_map_to_pc(lidar: RPLidar, pc_ip: str, pc_port: int = 5006, 
                     max_distance: float = 5000, max_points: int = None) -> None:
    """
    Continuously capture and stream LIDAR maps to PC for real-time visualization.
    
    Args:
        lidar (RPLidar): Connected and initialized RPLidar object
        pc_ip (str): IP address of the PC
        pc_port (int): Port number on PC (default: 5006)
        max_distance (float): Maximum distance in mm to include in scan (default: 5000)
        max_points (int): Maximum number of points per scan
    
    Raises:
        Exception: If connection fails
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
                
                if max_points and len(scan_data) > max_points:
                    scan_data = scan_data[:max_points]
                
                # Send scan to PC
                data = {
                    'timestamp': time.time(),
                    'frame': frame_count,
                    'num_points': len(scan_data),
                    'points': scan_data
                }
                
                try:
                    message = json.dumps(data) + '\n'
                    sock.sendall(message.encode('utf-8'))
                    frame_count += 1
                    
                    if frame_count % 5 == 0:
                        print(f"Streamed {frame_count} frames - {len(scan_data)} points in latest scan")
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
