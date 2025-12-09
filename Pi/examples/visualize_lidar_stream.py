"""
Real-time LIDAR visualization receiver with robot pose and scan statistics.

Run this script on your PC to receive and visualize LIDAR scans from the AMR.
Shows robot pose, scan points, and statistical analysis.
"""

import socket
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
from collections import deque
import threading
import time

# Configuration
SERVER_IP = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 5006

# Data storage
latest_scan = None
scan_lock = threading.Lock()
data_queue = deque(maxlen=10)  # Keep last 10 scans


def polar_to_cartesian(angles, distances):
    """Convert polar coordinates to cartesian."""
    angles_rad = np.radians(angles)
    x = distances * np.cos(angles_rad)
    y = distances * np.sin(angles_rad)
    return x, y


def receive_data():
    """Receive LIDAR data from the AMR."""
    global latest_scan
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen(1)
        print(f"Listening on {SERVER_IP}:{SERVER_PORT}...")
        
        while True:
            try:
                client_socket, client_address = server_socket.accept()
                print(f"Connected to {client_address[0]}:{client_address[1]}")
                
                buffer = ""
                while True:
                    try:
                        data = client_socket.recv(4096).decode('utf-8')
                        if not data:
                            break
                        
                        buffer += data
                        
                        # Process complete messages (separated by newline)
                        while '\n' in buffer:
                            message, buffer = buffer.split('\n', 1)
                            
                            if message.strip():
                                try:
                                    scan_data = json.loads(message)
                                    
                                    with scan_lock:
                                        latest_scan = scan_data
                                        data_queue.append(scan_data)
                                    
                                    frame = scan_data.get('frame', '?')
                                    points = scan_data.get('num_points', 0)
                                    pose = scan_data.get('robot_pose', {})
                                    stats = scan_data.get('statistics', {})
                                    
                                    x, y, theta = pose.get('x', 0), pose.get('y', 0), pose.get('theta', 0)
                                    mean_dist = stats.get('mean_distance', 0)
                                    
                                    print(f"Frame {frame}: {points} points | Pose: ({x:.1f}, {y:.1f}, {np.degrees(theta):.1f}°) | Mean dist: {mean_dist:.0f}mm")
                                
                                except json.JSONDecodeError as e:
                                    print(f"JSON decode error: {e}")
                    
                    except Exception as e:
                        print(f"Error receiving data: {e}")
                        break
                
                client_socket.close()
                print("Client disconnected")
            
            except Exception as e:
                print(f"Connection error: {e}")
                continue
    
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server_socket.close()


def draw_robot(ax, x, y, theta, size=150, color='red', alpha=0.8):
    """Draw robot position and orientation."""
    # Robot body
    circle = Circle((x, y), size, color=color, alpha=alpha, fill=True, label='Robot')
    ax.add_patch(circle)
    
    # Robot heading arrow
    arrow_length = size * 2
    dx = arrow_length * np.cos(theta)
    dy = arrow_length * np.sin(theta)
    arrow = FancyArrowPatch((x, y), (x + dx, y + dy),
                           arrowstyle='->', color=color, linewidth=2.5, 
                           mutation_scale=20, alpha=0.8)
    ax.add_patch(arrow)


def update_plot(frame_num):
    """Update the plot with new LIDAR data."""
    ax.clear()
    
    with scan_lock:
        if latest_scan is not None:
            # Use weighted points if available, otherwise fall back to original points
            points = latest_scan.get('weighted_points', latest_scan.get('points', []))
            robot_pose = latest_scan.get('robot_pose', {})
            statistics = latest_scan.get('statistics', {})
            frame_num = latest_scan.get('frame', 0)
            
            # Extract robot pose
            robot_x = robot_pose.get('x', 0)
            robot_y = robot_pose.get('y', 0)
            robot_theta = robot_pose.get('theta', 0)
            
            # Plot scan points
            if len(points) > 0:
                points_array = np.array(points)
                angles = points_array[:, 0]
                distances = points_array[:, 1]
                weights = points_array[:, 2] if points_array.shape[1] > 2 else np.ones_like(distances)
                
                # Convert to cartesian coordinates relative to robot
                x, y = polar_to_cartesian(angles, distances)
                
                # Transform to world frame
                x_world = robot_x + x * np.cos(robot_theta) - y * np.sin(robot_theta)
                y_world = robot_y + x * np.sin(robot_theta) + y * np.cos(robot_theta)
                
                # Color points by weight (0-1 normalized quality)
                scatter = ax.scatter(x_world, y_world, c=weights, cmap='viridis', s=10, alpha=0.6)
                
                # Add colorbar for weights
                if not hasattr(update_plot, 'cbar') or update_plot.cbar is None:
                    update_plot.cbar = plt.colorbar(scatter, ax=ax, label='Weight (0-1)')
                else:
                    try:
                        update_plot.cbar.update_normal(scatter)
                    except:
                        pass
            
            # Draw robot on plot
            draw_robot(ax, robot_x, robot_y, robot_theta, size=150, color='red')
            
            # Format title with statistics
            num_points = latest_scan.get('num_points', 0)
            mean_dist = statistics.get('mean_distance', 0)
            std_dist = statistics.get('std_distance', 0)
            quality_mean = statistics.get('quality_mean', 0)
            
            title = f'LIDAR Scan - Frame {frame_num}\n'
            title += f'Points: {num_points} | '
            title += f'Distance: {mean_dist:.0f}±{std_dist:.0f}mm | '
            title += f'Quality: {quality_mean:.1f}'
            
            ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.grid(True, alpha=0.3)
    
    # Dynamic axis limits centered on robot
    with scan_lock:
        if latest_scan is not None:
            robot_pose = latest_scan.get('robot_pose', {})
            robot_x = robot_pose.get('x', 0)
            robot_y = robot_pose.get('y', 0)
            
            # Center view on robot with 3000mm buffer
            ax.set_xlim(robot_x - 3000, robot_x + 3000)
            ax.set_ylim(robot_y - 3000, robot_y + 3000)


def main():
    """Main function."""
    global fig, ax
    
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 5 + "LIDAR Real-time Visualization" + " " * 12 + "║")
    print("║" + " " * 4 + "with Pose & Statistics" + " " * 20 + "║")
    print("╚" + "=" * 48 + "╝\n")
    
    # Start receiver thread
    receiver_thread = threading.Thread(target=receive_data, daemon=True)
    receiver_thread.start()
    
    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create animation
    ani = FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)
    
    plt.tight_layout()
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nVisualization closed")


if __name__ == "__main__":
    main()
