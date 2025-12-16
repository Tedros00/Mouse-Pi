from matplotlib.path import Path
# --- Simple OGM: fill inside LIDAR polygon as free (black) ---
def simple_polygon_ogm(px, py, ptheta, z_lidar, grid_shape, grid_bounds):
    """
    Create a binary OGM where everything inside the LIDAR scan polygon is free (black), rest is unknown (orange).
    """
    rows, cols = grid_shape
    (min_x, max_x), (min_y, max_y) = grid_bounds
    # Compute LIDAR endpoints in world coordinates
    num_beams = len(z_lidar)
    angles = np.linspace(-np.pi, np.pi, num_beams, endpoint=False) + ptheta + np.pi
    xs = px + z_lidar * np.cos(angles)
    ys = py + z_lidar * np.sin(angles)
    # Polygon: robot position + all LIDAR endpoints
    polygon = np.concatenate([[[px, py]], np.column_stack((xs, ys))], axis=0)
    # Create grid of cell centers (not edges)
    xg = np.linspace(min_x + (max_x - min_x) / (2 * cols), max_x - (max_x - min_x) / (2 * cols), cols)
    yg = np.linspace(min_y + (max_y - min_y) / (2 * rows), max_y - (max_y - min_y) / (2 * rows), rows)
    xv, yv = np.meshgrid(xg, yg)
    points = np.column_stack((xv.ravel(), yv.ravel()))
    # Check which points are inside the polygon
    path = Path(polygon)
    inside = path.contains_points(points)
    ogm = np.ones((rows, cols))  # 1 = unknown (orange)
    ogm.flat[inside] = 0.0      # 0 = free (black)
    return ogm


import socket
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch
import threading
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from OGM import occupancy_grid_mapping, log_odds_to_occupancy_probability
from EKF_SLAM import EKF_SLAM

PC_IP = '0.0.0.0'  # Listen on all interfaces
PC_PORT = 5010

# Visualization parameters
ROBOT_RADIUS = 0.08  # meters
LIDAR_RANGE = 15.0    # meters (for plot limits)





# Data storage
latest_msg = None
msg_lock = threading.Lock()
accumulated_points = []  # List of (x, y) tuples in world frame (all scans)
pose_history = []        # List of (x, y, theta)
scan_history = []        # List of lidar scan arrays

# EKF-SLAM setup
GRID_SHAPE = (90, 100)
GRID_BOUNDS = ((-0.9, 0.9), (-1.0, 1.0))
def dummy_compute_from_map(x, y, angle, map_data, max_range):
    return max_range
ekf_slam = EKF_SLAM(
    grid_shape=GRID_SHAPE,
    grid_bounds=GRID_BOUNDS,
    compute_from_map=dummy_compute_from_map,
    dt=0.1,
    motion_noise_std=(0.05, 0.02),
    measurement_noise_std=0.3
)
ekf_initialized = False

def plot_robot(ax, x, y, theta, radius=ROBOT_RADIUS):
    circle = Circle((x, y), radius, color='blue', fill=False, linewidth=2)
    ax.add_patch(circle)
    dx = radius * 1.5 * np.cos(theta)
    dy = radius * 1.5 * np.sin(theta)
    arrow = FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle='->', color='blue', linewidth=2, mutation_scale=15)
    ax.add_patch(arrow)

def update_plot(frame):
    ax = getattr(main, 'ax', None)
    if ax is None:
        return
    ax.clear()
    ax.set_title('LIDAR Scan')
    ax.set_xlabel('Y (meters, right)')
    ax.set_ylabel('X (meters, up)')
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    with msg_lock:
        msg = latest_msg.copy() if latest_msg else None
    if msg and 'pose' in msg and 'lidar' in msg and 'ranges' in msg['lidar']:
        px = msg['pose']['x']
        py = msg['pose']['y']
        ptheta = msg['pose']['theta'] + np.pi/2
        z_lidar = np.array(msg['lidar']['ranges'])
        if z_lidar.size > 0:
            num_beams = len(z_lidar)
            angles = np.linspace(-np.pi, np.pi, num_beams, endpoint=False) + ptheta
            # Swap axes: X (up) is vertical, Y (right) is horizontal
            xs = px + z_lidar * np.cos(angles)  # X (up)
            ys = py + z_lidar * np.sin(angles)  # Y (right)
            # --- OGM computation and display (overlay on scatter) ---
            from OGM import incremental_occupancy_grid_update, log_odds_to_occupancy_probability
            grid_shape = (100, 100)
            grid_bounds = ((-2, 2), (-2, 2))
            ogm_grid = np.zeros(grid_shape)
            pose = (px, py, ptheta)  # Use (X, Y, theta) to match scatter
            ogm_grid = incremental_occupancy_grid_update(
                ogm_grid, pose, z_lidar, grid_shape, grid_bounds,
                min_theta=-np.pi, max_theta=np.pi, max_range=15.0, sigma_hit=0.01
            )
            occ = log_odds_to_occupancy_probability(ogm_grid)
            # Plot OGM as background image in the same axes
            ax.imshow(occ.T, origin='lower', extent=[-2, 2, -2, 2], cmap='gray_r', vmin=0, vmax=1, alpha=0.7, zorder=0)
            # Plot LIDAR scatter on top
            ax.scatter(ys, xs, c='magenta', s=10, alpha=0.8, label='LIDAR scan', zorder=1)
            # Draw global 0-angle reference line (from robot position, length 1.0 m)
            try:
                zero_len = 1.0
                # Note: axes use horizontal=Y, vertical=X so plot as [py, py] (x), [px, px+zero_len] (y)
                ax.plot([py, py], [px, px + zero_len], color='green', linestyle='--', linewidth=2, zorder=2)
                ax.text(py, px + zero_len + 0.02, '0 rad', color='green', fontsize=8, ha='center', va='bottom')
            except Exception:
                # If px/py not available for any reason, skip drawing the reference
                pass
            ax.legend()


def on_key(event):
    if event.key == 'm':
        print("\n[OGM] Computing occupancy grid map from history...")
        if len(pose_history) < 2 or len(scan_history) < 2:
            print("Not enough data for OGM.")
            return
        # OGM grid parameters
        grid_shape = (100, 100)
        grid_bounds = ((0, 1.8), (0, 2.0))
        # Dummy compute_from_map (not used in this OGM version)
        def compute_from_map(x, y, angle, map_data, max_range):
            return max_range
        m, grid_coords = occupancy_grid_mapping(
            scan_history, pose_history, grid_shape, grid_bounds, compute_from_map, max_iterations=1, verbose=True
        )
        occ = log_odds_to_occupancy_probability(m)
        plt.figure("Occupancy Grid Map")
        plt.imshow(occ, origin='lower', extent=[-2,2,-2,2], cmap='gray_r')
        plt.title('Occupancy Grid Map')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.colorbar(label='Occupancy Probability')
        plt.show()


def receive_data():
    global latest_msg
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((PC_IP, PC_PORT))
        server_socket.listen(1)
        print(f"Listening for SLAM stream on {PC_IP}:{PC_PORT} ...")
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
                                    msg = json.loads(message)
                                    with msg_lock:
                                        latest_msg = msg
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


def main():
    global fig
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 7 + "SLAM Visualization Dashboard" + " " * 13 + "║")
    print("╚" + "=" * 48 + "╝\n")
    receiver_thread = threading.Thread(target=receive_data, daemon=True)
    receiver_thread.start()
    fig, ax = plt.subplots(figsize=(4, 4))
    main.ax = ax
    main.colorbar = None
    ani = FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)
    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nVisualization closed")

if __name__ == '__main__':
    main()