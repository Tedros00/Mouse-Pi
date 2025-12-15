#!/usr/bin/env python3
import os
import sys
import time
import math
import subprocess
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

# Goal in WORLD coordinates (meters)
GOAL_WORLD = (1.00, -1.00)

# Topics (change if your robot uses different names)
ODOM_TOPIC = "/model/robot/odom"
SCAN_TOPIC = "/model/robot/scan"   # if yours differs, run: gz topic -l | grep -i scan

# OGM grid bounds in WORLD coordinates (meters)
# This is NOT "map knowledge" of obstacles; it’s just the workspace bounds.
X_BOUNDS = (-1.35, 1.35)
Y_BOUNDS = (-1.35, 1.35)

# OGM resolution (m/cell)
MAP_RESOLUTION = 0.02

UNKNOWN_AS_OCCUPIED = False

# Occupancy thresholds
P_OCC = 0.65   # above => occupied
P_FREE = 0.35  # below => free

# Robot inflation radius for planning
ROBOT_RADIUS_M = 0.05

# Replanning rate (seconds)
REPLAN_DT = 0.30

# Controller params
V_MAX = 0.22
W_MAX = 2.0
LOOKAHEAD = 0.12
DT_CTRL = 0.05

# Stop behavior
GOAL_TOLERANCE = 0.06
STOP_REPEAT = 12
STOP_DT = 0.04

# ============================================================
# Import your existing modules WITHOUT modifying Pi/src
# ============================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PI_SRC = os.path.join(REPO_ROOT, "Pi", "src")
sys.path.insert(0, PI_SRC)

from path_finding import find_path                 # existing planner
from OGM import incremental_occupancy_grid_update  # existing OGM update


# ============================================================
# Helpers: world <-> grid (y-down grid for planner/image)
# ============================================================
def make_meta(x_bounds, y_bounds, resolution):
    min_x, max_x = x_bounds
    min_y, max_y = y_bounds
    width = int(math.ceil((max_x - min_x) / resolution))
    height = int(math.ceil((max_y - min_y) / resolution))
    return {
        "min_x": min_x, "max_x": max_x,
        "min_y": min_y, "max_y": max_y,
        "resolution": resolution,
        "width": width,
        "height": height,
    }


def world_to_grid(wx, wy, meta):
    gx = int((wx - meta["min_x"]) / meta["resolution"])
    gy_up = int((wy - meta["min_y"]) / meta["resolution"])
    gy = (meta["height"] - 1) - gy_up
    return gx, gy


def grid_to_world(gx, gy, meta):
    wx = meta["min_x"] + (gx + 0.5) * meta["resolution"]
    gy_up = (meta["height"] - 1) - gy
    wy = meta["min_y"] + (gy_up + 0.5) * meta["resolution"]
    return wx, wy


def in_bounds(gx, gy, meta):
    return 0 <= gx < meta["width"] and 0 <= gy < meta["height"]


# ============================================================
# Gazebo IO (gz CLI)
# ============================================================
def gz_pub_cmd_vel(v, w):
    msg = f"linear: {{x: {v:.4f}, y: 0, z: 0}} angular: {{x: 0, y: 0, z: {w:.4f}}}"
    subprocess.run(
        ["gz", "topic", "-t", "/model/robot/cmd_vel", "-m", "gz.msgs.Twist", "-p", msg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def hard_stop():
    for _ in range(STOP_REPEAT):
        gz_pub_cmd_vel(0.0, 0.0)
        time.sleep(STOP_DT)


def gz_read_odom_once(timeout_s=0.5):
    try:
        out = subprocess.check_output(
            ["gz", "topic", "-e", "-t", ODOM_TOPIC, "-m", "gz.msgs.Odometry", "-n", "1"],
            stderr=subprocess.DEVNULL,
            timeout=timeout_s
        ).decode("utf-8", errors="ignore")
    except Exception:
        return None

    def first_after(key, start=0):
        idx = out.find(key, start)
        if idx < 0:
            return None, -1
        s = out[idx + len(key):]
        tok = ""
        for ch in s:
            if ch in "0123456789+-.eE":
                tok += ch
            elif tok:
                break
        try:
            return float(tok), idx + len(key)
        except Exception:
            return None, -1

    pidx = out.find("pose {")
    if pidx < 0:
        return None

    x, ix = first_after("position {\n    x:", pidx)
    if x is None:
        x, ix = first_after("x:", pidx)
    y, iy = first_after("y:", ix if ix > 0 else pidx)

    oz, iz = first_after("orientation {\n    z:", iy if iy > 0 else pidx)
    if oz is None:
        oz, iz = first_after("z:", iy if iy > 0 else pidx)
    ow, _ = first_after("w:", iz if iz > 0 else (iy if iy > 0 else pidx))

    if x is None or y is None or oz is None or ow is None:
        return None

    yaw = 2.0 * math.atan2(oz, ow)
    return x, y, yaw


def gz_read_scan_once(timeout_s=0.8):
    """
    Reads gz.msgs.LaserScan from SCAN_TOPIC and returns:
      ranges (np.ndarray), angle_min (float), angle_max (float), range_max (float)
    """
    try:
        out = subprocess.check_output(
            ["gz", "topic", "-e", "-t", SCAN_TOPIC, "-m", "gz.msgs.LaserScan", "-n", "1"],
            stderr=subprocess.DEVNULL,
            timeout=timeout_s
        ).decode("utf-8", errors="ignore")
    except Exception:
        return None

    # Minimal parsing based on typical gz.msgs.LaserScan text format
    def get_float_after(key):
        idx = out.find(key)
        if idx < 0:
            return None
        s = out[idx + len(key):]
        tok = ""
        for ch in s:
            if ch in "0123456789+-.eE":
                tok += ch
            elif tok:
                break
        try:
            return float(tok)
        except Exception:
            return None

    angle_min = get_float_after("angle_min:")
    angle_max = get_float_after("angle_max:")
    range_max = get_float_after("range_max:")

    # ranges often appear as repeated "ranges: <val>"
    ranges = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("ranges:"):
            try:
                ranges.append(float(line.split("ranges:")[1].strip()))
            except Exception:
                pass

    if angle_min is None or angle_max is None or range_max is None or len(ranges) < 5:
        # Some Gazebo versions use "scan { angle_min:" nesting; fallback:
        # Try finding angle_min/angle_max again in entire output; ranges still from "ranges:"
        if len(ranges) < 5:
            return None

    ranges = np.array(ranges, dtype=np.float32)
    return ranges, float(angle_min), float(angle_max), float(range_max)


# ============================================================
# Map conversion: log-odds -> planner image
# ============================================================
def logodds_to_probs(L):
    return 1.0 / (1.0 + np.exp(-L))


def ogm_image_from_logodds(L, unknown_as_occupied=True):
    """
    Return uint8 image: 255 free, 0 occupied.
    Unknown cells (between P_FREE and P_OCC) treated per flag.
    """
    p = logodds_to_probs(L)
    occ = (p >= P_OCC)
    free = (p <= P_FREE)

    if unknown_as_occupied:
        img = np.where(free, 255, 0).astype(np.uint8)
    else:
        img = np.where(occ, 0, 255).astype(np.uint8)

    return img


# ============================================================
# Plotting (live + final)
# ============================================================
def init_live_plot(meta):
    plt.ion()
    fig, ax = plt.subplots()

    ax.set_title("Online OGM + Robot Trajectory (live)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")

    xs, ys = [], []

    # Map image placeholder
    extent = [meta["min_x"], meta["max_x"], meta["min_y"], meta["max_y"]]
    img_artist = ax.imshow(
        np.zeros((meta["height"], meta["width"]), dtype=np.uint8),
        origin="lower",
        cmap="gray",
        extent=extent,
        interpolation="nearest",
        vmin=0, vmax=255
    )

    (traj_line,) = ax.plot([], [], linewidth=2, label="Trajectory")
    (robot_dot,) = ax.plot([], [], marker="o", markersize=6, label="Robot")
    (plan_line,) = ax.plot([], [], linestyle="--", linewidth=2, label="Planned")

    ax.legend(loc="upper right")
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, img_artist, robot_dot, traj_line, plan_line, xs, ys


def update_plot(fig, img_artist, robot_dot, traj_line, plan_line,
                xs, ys, x, y, ogm_img=None, planned_world=None, every_n=2):
    xs.append(float(x))
    ys.append(float(y))

    if len(xs) % every_n != 0:
        return

    if ogm_img is not None:
        # flip image for display (y-up world)
        img_artist.set_data(np.flipud(ogm_img))

    robot_dot.set_data([xs[-1]], [ys[-1]])
    traj_line.set_data(xs, ys)

    if planned_world is not None and len(planned_world) > 1:
        px = [p[0] for p in planned_world]
        py = [p[1] for p in planned_world]
        plan_line.set_data(px, py)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def save_outputs(fig, xs, ys, out_png="trajectory.png", out_csv="trajectory.csv"):
    fig.savefig(out_png, dpi=200)
    with open(out_csv, "w") as f:
        f.write("t_index,x,y\n")
        for i, (x, y) in enumerate(zip(xs, ys)):
            f.write(f"{i},{x},{y}\n")


# ============================================================
# Simple heading controller (same as your plan.py style)
# ============================================================
def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def drive_towards(target_x, target_y, odom):
    x, y, th = odom
    ang = math.atan2(target_y - y, target_x - x)
    e = wrap(ang - th)

    v = V_MAX * max(0.0, 1.0 - abs(e) / 1.2)
    w = max(-W_MAX, min(W_MAX, 2.5 * e))
    gz_pub_cmd_vel(v, w)


# ============================================================
# Main loop: online map -> replan -> drive
# ============================================================
def main():
    meta = make_meta(X_BOUNDS, Y_BOUNDS, MAP_RESOLUTION)
    grid_shape = (meta["height"], meta["width"])
    grid_bounds = (X_BOUNDS, Y_BOUNDS)

    # Log-odds map initialized unknown (0 => p=0.5)
    L = np.zeros(grid_shape, dtype=np.float32)

    print(f"[UnknownPlan] Grid: {meta['width']} x {meta['height']} @ {MAP_RESOLUTION:.3f} m/cell")
    print(f"[UnknownPlan] Odom: {ODOM_TOPIC}")
    print(f"[UnknownPlan] Scan: {SCAN_TOPIC}")
    print(f"[UnknownPlan] Goal(world): {GOAL_WORLD}")

    # Wait for odom
    print("[UnknownPlan] Waiting for odom...")
    while True:
        od = gz_read_odom_once()
        if od is not None:
            break
    print(f"[UnknownPlan] Start(world): ({od[0]:.3f}, {od[1]:.3f})")


    start_time = time.time()
    WARMUP_S = 2.0   # seconds to build initial OGM before planning



    fig, ax, img_artist, robot_dot, traj_line, plan_line, xs, ys = init_live_plot(meta)

    last_plan_time = 0.0
    current_waypoints = None

    while True:
        od = gz_read_odom_once()
        if od is None:
            time.sleep(DT_CTRL)
            continue

        x, y, th = od

    # ------------------------------------------------------------
    # Mapping warm-up: do NOT plan yet
    # ------------------------------------------------------------
        if time.time() - start_time < WARMUP_S:
            # rotate slowly to collect scans
            gz_pub_cmd_vel(0.0, 0.35)
            time.sleep(DT_CTRL)
            continue


        # Stop condition
        dist_goal = math.hypot(GOAL_WORLD[0] - x, GOAL_WORLD[1] - y)
        if dist_goal <= GOAL_TOLERANCE:
            print(f"[UnknownPlan] Goal reached (dist={dist_goal:.3f} m). Stopping.")
            hard_stop()
            save_outputs(fig, xs, ys, out_png="trajectory.png", out_csv="trajectory.csv")
            plt.ioff()
            plt.show(block=False)
            return

        # Read scan and update OGM
        scan = gz_read_scan_once()
        if scan is not None:
            ranges, angle_min, angle_max, range_max = scan

            # incremental_occupancy_grid_update expects "measurement" = ranges
            # and angles assumed evenly spaced between min_theta and max_theta.
            L = incremental_occupancy_grid_update(
                current_map=L,
                pose=(x, y, th),
                measurement=ranges,
                grid_shape=grid_shape,
                grid_bounds=grid_bounds,
                min_theta=angle_min,
                max_theta=angle_max,
                max_range=range_max,
                sigma_hit=0.2,
            )

        # Build planner image
        ogm_img = ogm_image_from_logodds(L, unknown_as_occupied=UNKNOWN_AS_OCCUPIED)

        # Replan periodically
        now = time.time()
        planned_world = None

        if (now - last_plan_time) >= REPLAN_DT:
            last_plan_time = now

            sx, sy = world_to_grid(x, y, meta)
            gx, gy = world_to_grid(GOAL_WORLD[0], GOAL_WORLD[1], meta)

            if not in_bounds(sx, sy, meta) or not in_bounds(gx, gy, meta):
                print("[UnknownPlan] Start/goal out of grid bounds. Adjust X_BOUNDS/Y_BOUNDS.")
                hard_stop()
                return

            R = int(math.ceil(ROBOT_RADIUS_M / meta["resolution"]))

            # Plan on current map (unknown treated as occupied/free via ogm_img)

            # Force robot cell and a small radius around it to be free (bootstrap)
            sx, sy = world_to_grid(x, y, meta)

            r0 = 2  # cells
            for dy in range(-r0, r0 + 1):
                for dx in range(-r0, r0 + 1):
                    gx, gy = sx + dx, sy + dy
                    if 0 <= gx < meta["width"] and 0 <= gy < meta["height"]:
                        ogm_img[gy, gx] = 255


            print(f"Forced FREE around start cell ({sx},{sy}) before planning.")


            path_px = find_path(
                ogm_image=ogm_img,
                start_x=sx,
                start_y=sy,
                end_x=gx,
                end_y=gy,
                diagonal=True,
                R=R,
                downsample_factor=1
                )
            




            if path_px is not None and len(path_px) >= 2:
                planned_world = [grid_to_world(px, py, meta) for (px, py) in path_px]

                # Use a short-horizon waypoint (receding horizon)
                step = min(10, len(planned_world) - 1)
                current_waypoints = planned_world[step]
            else:
                # If no path yet (map incomplete), rotate gently to explore
                current_waypoints = None

        # Plot update
        update_plot(fig, img_artist, robot_dot, traj_line, plan_line,
                    xs, ys, x, y, ogm_img=ogm_img, planned_world=planned_world, every_n=2)

        # Control
        if current_waypoints is None:
            # “explore”: rotate slowly to gather scans (helps OGM fill in)
            gz_pub_cmd_vel(0.0, 0.35)
        else:
            tx, ty = current_waypoints
            drive_towards(tx, ty, od)

        time.sleep(DT_CTRL)


if __name__ == "__main__":
    main()
