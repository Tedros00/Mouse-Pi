"""
EKF-SLAM Frontier-Based Coverage Exploration
--------------------------------------------
- Persistent coverage map (saved to disk)
- Frontier-driven exploration
- Blended obstacle avoidance + goal seeking
- Goal hysteresis + escape-to-different-frontier
- Robust collision checking (swept path + robot radius) to prevent wall tunneling
"""
#this version works way better than the previous ones, covers approximately 70% of the entire map in one try but its slower
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from EKF_SLAM import EKF_SLAM

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
SAFE_DISTANCE = 15.0
MAX_SPEED = 50.0
MAX_TURN = 2.5
LIDAR_MAX_RANGE = 1500.0
DT = 0.1

CELL_SIZE_CM = 2.0
ROBOT_RADIUS_CM = 6.0  # IMPORTANT: collision inflation

GOAL_REACHED_DIST = 10.0

GOAL_HOLD_MAX = 30        # hysteresis frames
ESCAPE_STUCK_LIMIT = 40   # frames without progress
ESCAPE_NEAREST_K = 8      # pick among K nearest frontiers to break cycles
PROGRESS_EPS = 1.0        # cm movement threshold considered "no progress"

# ------------------------------------------------------------------
# Load maze (GROUND TRUTH – simulation only)
# ------------------------------------------------------------------
maze_path = os.path.join(os.path.dirname(__file__), '..', 'SimulationEnv', 'TestMaze.JPG')
maze_img = np.array(Image.open(maze_path).convert('L'))
maze_binary = (maze_img > 127).astype(np.uint8)
maze_h_px, maze_w_px = maze_binary.shape

# ------------------------------------------------------------------
# World scaling
# ------------------------------------------------------------------
REAL_WIDTH = 180.0
REAL_HEIGHT = 200.0
scale_x = REAL_WIDTH / maze_w_px
scale_y = REAL_HEIGHT / maze_h_px

grid_width = int(REAL_WIDTH / CELL_SIZE_CM)
grid_height = int(REAL_HEIGHT / CELL_SIZE_CM)

# ------------------------------------------------------------------
# Coverage grid (PERSISTENT)
# -1 unknown | 0 free | 1 wall
# ------------------------------------------------------------------
if os.path.exists("coverage.npy"):
    coverage = np.load("coverage.npy")
else:
    coverage = -np.ones((grid_height, grid_width), dtype=np.int8)

# Shadow map: per-scan only (for map-quality; NOT used to forbid frontiers)
shadowed = np.zeros_like(coverage, dtype=bool)

# ------------------------------------------------------------------
# LiDAR raycasting (SIMULATION ONLY)
# ------------------------------------------------------------------
def compute_expected_measurements(x, y, theta, map_data, max_range, num_beams=180):
    x_px = x / scale_x
    y_px = y / scale_y
    max_px = max_range / min(scale_x, scale_y)

    ranges = np.zeros(num_beams)
    angles = np.linspace(-np.pi, np.pi, num_beams)

    for i, a in enumerate(angles):
        ca, sa = np.cos(theta + a), np.sin(theta + a)
        for d in np.arange(0.5, max_px, 0.5):
            cx = int(x_px + d * ca)
            cy = int(y_px + d * sa)

            if cx < 0 or cy < 0 or cx >= map_data.shape[1] or cy >= map_data.shape[0]:
                ranges[i] = d * min(scale_x, scale_y)
                break

            if map_data[cy, cx] == 0:
                ranges[i] = d * min(scale_x, scale_y)
                break
        else:
            ranges[i] = max_range

    return ranges

# ------------------------------------------------------------------
# Obstacle avoidance (proportional, avoids spin-lock)
# ------------------------------------------------------------------
def obstacle_avoidance(lidar):
    idx = np.argmin(lidar)
    angles = np.linspace(-np.pi, np.pi, len(lidar))
    obs_angle = angles[idx]
    v = 6.0
    w = -1.0 * obs_angle
    return np.array([v, w], dtype=float)

# ------------------------------------------------------------------
# Coverage update (keeps same structure)
# ------------------------------------------------------------------
def update_coverage(pose, lidar):
    # shadowed should be per-frame only
    shadowed[:] = False

    angles = np.linspace(-np.pi, np.pi, len(lidar))
    for r, a in zip(lidar, angles):
        steps = int(r / CELL_SIZE_CM)
        for k in range(steps):
            wx = pose[0] + k * CELL_SIZE_CM * np.cos(pose[2] + a)
            wy = pose[1] + k * CELL_SIZE_CM * np.sin(pose[2] + a)

            cx = int(wx / CELL_SIZE_CM)
            cy = int(wy / CELL_SIZE_CM)
            if not (0 <= cx < grid_width and 0 <= cy < grid_height):
                break

            mx = int(np.clip(wx / scale_x, 0, maze_binary.shape[1] - 1))
            my = int(np.clip(wy / scale_y, 0, maze_binary.shape[0] - 1))

            if maze_binary[my, mx] == 0:
                coverage[cy, cx] = 1

                # mark unknown behind wall as shadowed for this scan
                for kk in range(k + 1, steps):
                    wx2 = pose[0] + kk * CELL_SIZE_CM * np.cos(pose[2] + a)
                    wy2 = pose[1] + kk * CELL_SIZE_CM * np.sin(pose[2] + a)
                    cx2 = int(wx2 / CELL_SIZE_CM)
                    cy2 = int(wy2 / CELL_SIZE_CM)
                    if 0 <= cx2 < grid_width and 0 <= cy2 < grid_height:
                        if coverage[cy2, cx2] == -1:
                            shadowed[cy2, cx2] = True
                    else:
                        break
                break

            coverage[cy, cx] = 0

# ------------------------------------------------------------------
# Frontier listing / selection (free cell adjacent to unknown)
# ------------------------------------------------------------------
def list_frontiers():
    ys, xs = np.where(coverage == 0)
    frontiers = []
    for x, y in zip(xs, ys):
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                if coverage[ny, nx] == -1:
                    frontiers.append((x, y))
                    break
    return frontiers

def nearest_frontier(pose):
    frontiers = list_frontiers()
    if not frontiers:
        return None
    rx = pose[0] / CELL_SIZE_CM
    ry = pose[1] / CELL_SIZE_CM
    dists = [(fx - rx) ** 2 + (fy - ry) ** 2 for fx, fy in frontiers]
    fx, fy = frontiers[int(np.argmin(dists))]
    return np.array([fx * CELL_SIZE_CM, fy * CELL_SIZE_CM], dtype=float)

def random_frontier_nearby(pose, k=8):
    frontiers = list_frontiers()
    if not frontiers:
        return None
    rx = pose[0] / CELL_SIZE_CM
    ry = pose[1] / CELL_SIZE_CM
    dists = np.array([(fx - rx) ** 2 + (fy - ry) ** 2 for fx, fy in frontiers])
    order = np.argsort(dists)
    pick = order[:min(k, len(order))]
    fx, fy = frontiers[int(np.random.choice(pick))]
    return np.array([fx * CELL_SIZE_CM, fy * CELL_SIZE_CM], dtype=float)

# ------------------------------------------------------------------
# Goal controller
# ------------------------------------------------------------------
def go_to_goal(pose, goal):
    dx = goal[0] - pose[0]
    dy = goal[1] - pose[1]
    target = np.arctan2(dy, dx)
    err = np.arctan2(np.sin(target - pose[2]), np.cos(target - pose[2]))

    v = MAX_SPEED * (1.0 - 0.5 * min(abs(err), 1.0))
    v = min(v, 20.0)  # keep stable for DT=0.2 in tight areas
    w = np.clip(2.0 * err, -MAX_TURN, MAX_TURN)
    return np.array([v, w], dtype=float)

# ------------------------------------------------------------------
# Collision checking (fixes wall tunneling)
# ------------------------------------------------------------------
def in_wall_with_radius(x, y):
    """True if a disk of radius ROBOT_RADIUS_CM around (x,y) intersects a wall."""
    r_cells = int(np.ceil(ROBOT_RADIUS_CM / CELL_SIZE_CM))
    cx = int(np.clip(x / CELL_SIZE_CM, 0, grid_width - 1))
    cy = int(np.clip(y / CELL_SIZE_CM, 0, grid_height - 1))

    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            nx = cx + dx
            ny = cy + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                # circle mask (optional but helps)
                if (dx * CELL_SIZE_CM) ** 2 + (dy * CELL_SIZE_CM) ** 2 > ROBOT_RADIUS_CM ** 2:
                    continue
                wx = nx * CELL_SIZE_CM
                wy = ny * CELL_SIZE_CM
                mx = int(np.clip(wx / scale_x, 0, maze_binary.shape[1] - 1))
                my = int(np.clip(wy / scale_y, 0, maze_binary.shape[0] - 1))
                if maze_binary[my, mx] == 0:
                    return True
    return False

def will_hit_wall_swept(pose, u):
    """Check the entire motion segment for collision (swept path)."""
    x, y, theta = pose
    v, w = u

    # Number of samples along the motion segment
    dist = abs(v) * DT
    steps = int(max(2, np.ceil(dist / (0.5 * CELL_SIZE_CM))))  # sample at <= 1 cm-ish

    for i in range(1, steps + 1):
        t = i / steps
        xi = x + t * v * np.cos(theta) * DT
        yi = y + t * v * np.sin(theta) * DT
        if in_wall_with_radius(xi, yi):
            return True
    return False

def collision_safe_control(pose, u, lidar):
    """
    Adjust control to prevent passing through walls:
    - cap speed near obstacles
    - if collision predicted, reduce speed progressively
    - if still colliding, rotate in place away from closest obstacle direction
    """
    u = u.astype(float).copy()

    # Cap speed near obstacles (prevents tunneling/instability)
    if np.min(lidar) < 2.0 * SAFE_DISTANCE:
        u[0] = min(u[0], 20.0)

    # Progressive slowdown if swept collision predicted
    if will_hit_wall_swept(pose, u):
        for v_cap in [10.0, 8.0, 5.0, 3.0]:
            u_try = u.copy()
            u_try[0] = np.sign(u_try[0]) * min(abs(u_try[0]), v_cap)
            if not will_hit_wall_swept(pose, u_try):
                return u_try

        # If even v=0 collides (we are too close), rotate away from nearest obstacle
        idx = np.argmin(lidar)
        angles = np.linspace(-np.pi, np.pi, len(lidar))
        obs_angle = angles[idx]
        w = np.clip(-np.sign(obs_angle) * MAX_TURN, -MAX_TURN, MAX_TURN)
        return np.array([0.0, w], dtype=float)

    return u

# ------------------------------------------------------------------
# EKF-SLAM init
# ------------------------------------------------------------------
ekf = EKF_SLAM(
    grid_shape=(grid_height, grid_width),
    grid_bounds=((0, REAL_WIDTH), (0, REAL_HEIGHT)),
    compute_from_map=compute_expected_measurements,
    dt=DT,
    motion_noise_std=(0.01, 0.005),
    measurement_noise_std=0.3
)
ekf.set_initial_pose(np.array([100.0, 175.0, 0.0]), uncertainty=0.1)

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, REAL_WIDTH)
ax.set_ylim(0, REAL_HEIGHT)
ax.invert_yaxis()
ax.set_aspect('equal')

ax.imshow(
    maze_binary,
    extent=[0, REAL_WIDTH, REAL_HEIGHT, 0],
    cmap='gray',
    alpha=0.35
)
# --- Live coverage text (top-left of plot) ---
coverage_text = ax.text(
    0.02, 0.98, "",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

def make_overlay():
    img = np.zeros_like(coverage, dtype=float)
    alpha = np.zeros_like(coverage, dtype=float)

    img[coverage == -1] = 0.0
    alpha[coverage == -1] = 1.0

    img[coverage == 0] = 1.0
    alpha[coverage == 0] = 0.0

    img[coverage == 1] = 0.5
    alpha[coverage == 1] = 0.6

    return np.dstack([img, img, img, alpha])

coverage_img = ax.imshow(
    make_overlay(),
    extent=[0, REAL_WIDTH, REAL_HEIGHT, 0]
)

robot_dot, = ax.plot([], [], 'ro')
heading_line, = ax.plot([], [], 'r-')

# ------------------------------------------------------------------
# Exploration state
# ------------------------------------------------------------------
current_goal = None
goal_hold = 0
stuck_counter = 0
prev_pose = None
# ------------------------------------------------------------------
# Coverage percentage
# ------------------------------------------------------------------
def coverage_percentage():
    """
    Percentage of grid cells that are no longer unknown.
    """
    known = np.count_nonzero(coverage != -1)
    total = coverage.size
    return 100.0 * known / total

# ------------------------------------------------------------------
# Animation loop
# ------------------------------------------------------------------
def update(frame):
    global current_goal, goal_hold, stuck_counter, prev_pose

    pose = ekf.get_estimated_pose()

    lidar = compute_expected_measurements(
        pose[0], pose[1], pose[2],
        maze_binary, LIDAR_MAX_RANGE
    )

    update_coverage(pose, lidar)
    cov = coverage_percentage()
    coverage_text.set_text(f"Coverage: {cov:.1f}%")

    if cov >= 95.0:
        print("Coverage ≥ 95%, stopping exploration.")
        np.save("coverage.npy", coverage)
        return robot_dot, heading_line
    reached = (current_goal is not None and np.linalg.norm(current_goal - pose[:2]) < GOAL_REACHED_DIST)
    
    if current_goal is None or reached or goal_hold > GOAL_HOLD_MAX:
        current_goal = nearest_frontier(pose)
        goal_hold = 0
    else:
        goal_hold += 1

    if current_goal is None:
        np.save("coverage.npy", coverage)
        return robot_dot, heading_line, coverage_text

    # Controls
    u_goal = go_to_goal(pose, current_goal)
    u_avoid = obstacle_avoidance(lidar)

    alpha = np.clip((SAFE_DISTANCE - np.min(lidar)) / SAFE_DISTANCE, 0, 1)
    u = (1 - alpha) * u_goal + alpha * u_avoid

    # Collision safety (prevents going through walls)
    u = collision_safe_control(pose, u, lidar)

    # EKF update
    z = np.clip(lidar + np.random.normal(0, 1.0, len(lidar)), 0.1, LIDAR_MAX_RANGE)
    ekf.update(u, z)

    # Pose after update for progress detection + rendering
    pose_after = ekf.get_estimated_pose()

    # Stuck detection (correct: after update)
    if prev_pose is not None:
        if np.linalg.norm(pose_after[:2] - prev_pose[:2]) < PROGRESS_EPS:
            stuck_counter += 1
        else:
            stuck_counter = 0

    # Escape: switch to a different nearby frontier if stuck
    if stuck_counter > ESCAPE_STUCK_LIMIT:
        escape_goal = random_frontier_nearby(pose_after, k=ESCAPE_NEAREST_K)
        if escape_goal is not None:
            current_goal = escape_goal
            goal_hold = 0
        stuck_counter = 0

    prev_pose = pose_after.copy()

    # Draw robot
    robot_dot.set_data([pose_after[0]], [pose_after[1]])
    heading_line.set_data(
        [pose_after[0], pose_after[0] + 15 * np.cos(pose_after[2])],
        [pose_after[1], pose_after[1] + 15 * np.sin(pose_after[2])]
    )

    coverage_img.set_data(make_overlay())
    return robot_dot, heading_line

ani = FuncAnimation(fig, update, interval=40, blit=False)
plt.show()

# Persist map for next trial
np.save("coverage.npy", coverage)