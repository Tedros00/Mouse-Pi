"""
Advanced LIDAR visualization with beam range finder model visualization.

Run this script on your PC for detailed scan analysis with beam model overlay.
Shows scan points, robot pose, statistics, and expected beam ranges.
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
SERVER_IP = '0.0.0.0'
SERVER_PORT = 5006

# Data storage
latest_scan = None
scan_lock = threading.Lock()


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
                        
                        while '\n' in buffer:
                            message, buffer = buffer.split('\n', 1)
                            
                            if message.strip():
                                try:
                                    scan_data = json.loads(message)
                                    with scan_lock:
                                        latest_scan = scan_data
                                    
                                    frame = scan_data.get('frame', '?')
                                    points = scan_data.get('num_points', 0)
                                    stats = scan_data.get('statistics', {})
                                    mean_dist = stats.get('mean_distance', 0)
                                    
                                    print(f"Frame {frame}: {points} points | Mean: {mean_dist:.0f}mm")
                                
                                except json.JSONDecodeError:
                                    pass
                    
                    except Exception as e:
                        print(f"Error receiving: {e}")
                        break
                
                client_socket.close()
                print("Client disconnected")
            
            except Exception as e:
                print(f"Connection error: {e}")
    
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server_socket.close()


def draw_robot(ax, x, y, theta, size=150, color='red', alpha=0.8):
    """Draw robot position and orientation."""
    circle = Circle((x, y), size, color=color, alpha=alpha, fill=True, label='Robot')
    ax.add_patch(circle)
    
    arrow_length = size * 2
    dx = arrow_length * np.cos(theta)
    dy = arrow_length * np.sin(theta)
    arrow = FancyArrowPatch((x, y), (x + dx, y + dy),
                           arrowstyle='->', color=color, linewidth=2.5, 
                           mutation_scale=20, alpha=0.8)
    ax.add_patch(arrow)


def draw_beam_coverage(ax, robot_x, robot_y, robot_theta, normalized_ranges, max_range=5000, alpha=0.1):
    """Draw beam coverage visualization."""
    num_beams = len(normalized_ranges)
    angles = np.linspace(0, 360, num_beams)
    
    for i, (angle, norm_range) in enumerate(zip(angles, normalized_ranges)):
        if norm_range > 0.01:  # Only draw if range > 0
            distance = norm_range * max_range
            beam_angle = np.radians(angle) + robot_theta
            
            x_end = robot_x + distance * np.cos(beam_angle)
            y_end = robot_y + distance * np.sin(beam_angle)
            
            ax.plot([robot_x, x_end], [robot_y, y_end], 
                   color='cyan', alpha=alpha, linewidth=0.5)


def update_plot(frame_num):
    """Update the plot with new LIDAR data."""
    fig.clear()
    
    # Create subplots
    ax_scan = fig.add_subplot(221)
    ax_dist = fig.add_subplot(222)
    ax_quality = fig.add_subplot(223)
    ax_ranges = fig.add_subplot(224)
    
    with scan_lock:
        if latest_scan is not None:
            # Use weighted points if available, otherwise fall back to original points
            points = latest_scan.get('weighted_points', latest_scan.get('points', []))
            robot_pose = latest_scan.get('robot_pose', {})
            statistics = latest_scan.get('statistics', {})
            normalized_ranges = latest_scan.get('normalized_ranges', [])
            frame_num = latest_scan.get('frame', 0)
            
            robot_x = robot_pose.get('x', 0)
            robot_y = robot_pose.get('y', 0)
            robot_theta = robot_pose.get('theta', 0)
            
            # ===== SUBPLOT 1: SCAN VISUALIZATION =====
            if len(points) > 0:
                points_array = np.array(points)
                angles = points_array[:, 0]
                distances = points_array[:, 1]
                weights = points_array[:, 2] if points_array.shape[1] > 2 else np.ones_like(distances)
                
                # Convert to cartesian
                x, y = polar_to_cartesian(angles, distances)
                
                # Transform to world frame
                x_world = robot_x + x * np.cos(robot_theta) - y * np.sin(robot_theta)
                y_world = robot_y + x * np.sin(robot_theta) + y * np.cos(robot_theta)
                
                # Plot points colored by weight
                scatter = ax_scan.scatter(x_world, y_world, c=weights, cmap='viridis', s=8, alpha=0.7)
                
                # Draw beam coverage
                if normalized_ranges:
                    draw_beam_coverage(ax_scan, robot_x, robot_y, robot_theta, normalized_ranges)
            
            # Draw robot
            draw_robot(ax_scan, robot_x, robot_y, robot_theta, size=120, color='red')
            
            ax_scan.set_aspect('equal')
            ax_scan.set_xlabel('X (mm)')
            ax_scan.set_ylabel('Y (mm)')
            ax_scan.grid(True, alpha=0.3)
            ax_scan.set_xlim(robot_x - 3000, robot_x + 3000)
            ax_scan.set_ylim(robot_y - 3000, robot_y + 3000)
            ax_scan.set_title('Scan Visualization\n(Cyan = Beam Coverage)', fontsize=10, fontweight='bold')
            
            # ===== SUBPLOT 2: DISTANCE HISTOGRAM =====
            if len(points) > 0:
                distances = points_array[:, 1]
                ax_dist.hist(distances, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                ax_dist.axvline(statistics.get('mean_distance', 0), color='red', 
                               linestyle='--', linewidth=2, label=f"Mean: {statistics.get('mean_distance', 0):.0f}mm")
                ax_dist.axvline(statistics.get('median_distance', 0), color='green', 
                               linestyle='--', linewidth=2, label=f"Median: {statistics.get('median_distance', 0):.0f}mm")
                ax_dist.set_xlabel('Distance (mm)')
                ax_dist.set_ylabel('Count')
                ax_dist.legend(fontsize=8)
            
            ax_dist.set_title('Distance Distribution', fontsize=10, fontweight='bold')
            
            # ===== SUBPLOT 3: QUALITY ANALYSIS =====
            if len(points) > 0:
                weights = points_array[:, 2] if points_array.shape[1] > 2 else np.zeros_like(distances)
                ax_quality.scatter(distances, weights, c=weights, cmap='viridis', s=10, alpha=0.6)
                ax_quality.set_xlabel('Distance (mm)')
                ax_quality.set_ylabel('Weight (0-1)')
                ax_quality.grid(True, alpha=0.3)
            
            ax_quality.set_title('Weight vs Distance', fontsize=10, fontweight='bold')
            
            # ===== SUBPLOT 4: STATISTICS TEXT =====
            ax_ranges.axis('off')
            
            stats_text = f"""
Frame: {frame_num}
Timestamp: {latest_scan.get('timestamp', 0):.2f}

POSE:
  X: {robot_x:.1f} mm
  Y: {robot_y:.1f} mm
  θ: {np.degrees(robot_theta):.1f}°

POINTS: {len(points)}

DISTANCE STATS:
  Min: {statistics.get('min_distance', 0):.0f} mm
  Max: {statistics.get('max_distance', 0):.0f} mm
  Mean: {statistics.get('mean_distance', 0):.0f} mm
  Median: {statistics.get('median_distance', 0):.0f} mm
  Std Dev: {statistics.get('std_distance', 0):.0f} mm

QUALITY STATS:
  Min: {statistics.get('quality_min', 0):.0f}
  Max: {statistics.get('quality_max', 0):.0f}
  Mean: {statistics.get('quality_mean', 0):.1f}

BEAM MODEL:
  Beams: {len(normalized_ranges) if normalized_ranges else 0}
  Coverage: {np.sum(np.array(normalized_ranges) > 0) if normalized_ranges else 0}%
            """
            
            ax_ranges.text(0.05, 0.95, stats_text, transform=ax_ranges.transAxes,
                          fontsize=8, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('LIDAR Sensor Analysis Dashboard', fontsize=12, fontweight='bold')
    fig.tight_layout()


def main():
    """Main function."""
    global fig
    
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 3 + "LIDAR Advanced Visualization Dashboard" + " " * 7 + "║")
    print("║" + " " * 5 + "with Beam Model & Statistics" + " " * 14 + "║")
    print("╚" + "=" * 48 + "╝\n")
    
    # Start receiver thread
    receiver_thread = threading.Thread(target=receive_data, daemon=True)
    receiver_thread.start()
    
    # Setup matplotlib figure
    fig = plt.figure(figsize=(16, 12))
    
    # Create animation
    ani = FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nVisualization closed")


if __name__ == "__main__":
    main()
