#!/usr/bin/env python3
"""
EKF-SLAM Coverage-Based Exploration (Gazebo)
-------------------------------------------
- Uses REAL Gazebo LiDAR
- Uses REAL Gazebo world pose
- Builds coverage grid online
- Stops robot when exploration completes
"""

import os
import sys
import time
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATH SETUP
# ============================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PI_SRC = os.path.join(REPO_ROOT, "Pi", "src")
sys.path.insert(0, PI_SRC)

from EKF_SLAM import EKF_SLAM  # DO NOT MODIFY

# ============================================================
# GAZEBO NAMES
# ============================================================
WORLD_NAME = "maze_world"
ROBOT_NAME = "robot"

# ============================================================
# PARAMETERS
# ============================================================
DT = 0.1

MAX_SPEED = 0.25
MAX_TURN = 2.0
SAFE_DISTANCE = 0.18       # meters
LIDAR_MAX_RANGE = 3.5      # meters

CELL_SIZE = 0.05           # meters per cell
GRID_W = 80
GRID_H = 80

# ============================================================
# COVERAGE GRID
# -1 = unknown, 0 = free, 1 = wall
# ============================================================
coverage = -np.ones((GRID_H, GRID_W), dtype=np.int8)

# ============================================================
# GAZEBO IO
# ============================================================
def gz_pub_cmd_vel(v, w):
    msg = f"linear: {{x: {v:.3f}}} angular: {{z: {w:.3f}}}"
    subprocess.run(
        ["gz", "topic", "-t", f"/model/{ROBOT_NAME}/cmd_vel",
         "-m", "gz.msgs.Twist", "-p", msg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def gz_read_world_pose():
    topic = f"/world/{WORLD_NAME}/pose/info"
    try:
        out = subprocess.check_output(
            ["gz", "topic", "-e", "-t", topic,
             "-m", "gz.msgs.Pose_V", "-n", "1"],
            timeout=0.5
        ).decode()
    except Exception:
        return None

    key = f'name: "{ROBOT_NAME}"'
    i = out.find(key)
    if i < 0:
        return None
    s = out[i:i+800]

    def val(k):
        j = s.find(k)
        if j < 0:
            return None
        t = ""
        for c in s[j+len(k):]:
            if c in "0123456789+-.eE":
                t += c
            elif t:
                break
        return float(t)

    x = val("x:")
    y = val("y:")
    zq = val("z:")
    wq = val("w:")

    if None in (x, y, zq, wq):
        return None

    yaw = 2 * math.atan2(zq, wq)
    return np.array([x, y, yaw])


def gz_read_lidar():
    try:
        out = subprocess.check_output(
            ["gz", "topic", "-e", "-t", f"/model/{ROBOT_NAME}/scan",
             "-m", "gz.msgs.LaserScan", "-n", "1"],
            timeout=0.5
        ).decode()
    except Exception:
        return None

    ranges = []
    a_min = a_max = None

    for l in out.splitlines():
        if "angle_min:" in l:
            a_min = float(l.split(":")[1])
        elif "angle_max:" in l:
            a_max = float(l.split(":")[1])
        elif "ranges:" in l:
            ranges.append(float(l.split(":")[1]))

    if a_min is None or a_max is None or not ranges:
        return None

    return np.array(ranges), a_min, a_max

# ============================================================
# COVERAGE UPDATE (REAL LIDAR)
# ============================================================
def update_coverage(pose, ranges, a_min, a_max):
    angles = np.linspace(a_min, a_max, len(ranges))

    for r, a in zip(ranges, angles):
        r = min(r, LIDAR_MAX_RANGE)
        steps = int(r / CELL_SIZE)

        for k in range(steps):
            wx = pose[0] + k * CELL_SIZE * math.cos(pose[2] + a)
            wy = pose[1] + k * CELL_SIZE * math.sin(pose[2] + a)

            cx = int(wx / CELL_SIZE)
            cy = int(wy / CELL_SIZE)

            if not (0 <= cx < GRID_W and 0 <= cy < GRID_H):
                break

            coverage[cy, cx] = 0

        if r < LIDAR_MAX_RANGE:
            wx = pose[0] + r * math.cos(pose[2] + a)
            wy = pose[1] + r * math.sin(pose[2] + a)
            cx = int(wx / CELL_SIZE)
            cy = int(wy / CELL_SIZE)
            if 0 <= cx < GRID_W and 0 <= cy < GRID_H:
                coverage[cy, cx] = 1

# ============================================================
# PLANNING
# ============================================================
def nearest_uncovered(pose):
    ys, xs = np.where(coverage == -1)
    if len(xs) == 0:
        return None

    rx = int(pose[0] / CELL_SIZE)
    ry = int(pose[1] / CELL_SIZE)

    d = (xs - rx)**2 + (ys - ry)**2
    i = np.argmin(d)
    return np.array([xs[i]*CELL_SIZE, ys[i]*CELL_SIZE])


def obstacle_avoidance(ranges):
    idx = np.argmin(ranges)
    ang = np.linspace(-math.pi, math.pi, len(ranges))[idx]
    return 0.05, -1.8 * np.sign(ang)


def go_to_goal(pose, goal):
    dx, dy = goal - pose[:2]
    target = math.atan2(dy, dx)
    err = math.atan2(math.sin(target-pose[2]), math.cos(target-pose[2]))

    v = MAX_SPEED * math.exp(-abs(err))
    w = max(-MAX_TURN, min(MAX_TURN, 2.5 * err))
    return v, w

# ============================================================
# EKF-SLAM INIT
# ============================================================
ekf = EKF_SLAM(
    grid_shape=(GRID_H, GRID_W),
    grid_bounds=((0, GRID_W*CELL_SIZE), (0, GRID_H*CELL_SIZE)),
    dt=DT,
    motion_noise_std=(0.02, 0.01),
    measurement_noise_std=0.3
)

print("[Explorer] Waiting for first pose...")
while True:
    p = gz_read_world_pose()
    if p is not None:
        ekf.set_initial_pose(p, uncertainty=0.05)
        break
    time.sleep(0.1)

# ============================================================
# VISUALIZATION
# ============================================================
plt.ion()
fig, ax = plt.subplots()
img = ax.imshow(coverage, cmap="gray", vmin=-1, vmax=1)
ax.set_title("Coverage Grid")
plt.show(block=False)

# ============================================================
# MAIN LOOP
# ============================================================
print("[Explorer] Starting exploration...")
while True:
    pose = gz_read_world_pose()
    scan = gz_read_lidar()

    if pose is None or scan is None:
        time.sleep(DT)
        continue

    ranges, a_min, a_max = scan
    update_coverage(pose, ranges, a_min, a_max)

    goal = nearest_uncovered(pose)
    if goal is None:
        print("[Explorer] Coverage complete. Stopping robot.")
        for _ in range(10):
            gz_pub_cmd_vel(0, 0)
            time.sleep(0.05)
        break

    if np.min(ranges) < SAFE_DISTANCE:
        v, w = obstacle_avoidance(ranges)
    else:
        v, w = go_to_goal(pose, goal)

    gz_pub_cmd_vel(v, w)
    ekf.update(np.array([v, w]), ranges)

    img.set_data(coverage)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    time.sleep(DT)

print("[Explorer] Done.")
