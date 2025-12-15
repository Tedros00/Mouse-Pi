#!/usr/bin/env python3
import os
import sys
import time
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

GOAL_WORLD = (0.50, 0.00)

USE_MANUAL_START = False
MANUAL_START_WORLD = (-1.20, -1.20)

MAP_RESOLUTION = 0.02
MAP_MARGIN = 0.15

ROBOT_RADIUS_M = 0.05

V_MAX = 0.22
W_MAX = 2.0
LOOKAHEAD = 0.12
DT = 0.05
WAYPOINT_STEP = 6

GOAL_TOLERANCE = 0.06
STOP_REPEAT = 12
STOP_DT = 0.04

WORLD_NAME = "maze_world"
MODEL_NAME = "robot"

# ============================================================
# Import your existing planner WITHOUT modifying Pi/src
# ============================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PI_SRC = os.path.join(REPO_ROOT, "Pi", "src")
sys.path.insert(0, PI_SRC)

from path_finding import find_path  # DO NOT MODIFY Pi/src


# ============================================================
# WORLD pose reader (ground truth) - fixes mismatch with video
# ============================================================
def gz_read_world_pose_once(timeout_s=0.5):
    topic = f"/world/{WORLD_NAME}/pose/info"
    try:
        out = subprocess.check_output(
            ["gz", "topic", "-e", "-t", topic, "-m", "gz.msgs.Pose_V", "-n", "1"],
            stderr=subprocess.DEVNULL,
            timeout=timeout_s
        ).decode("utf-8", errors="ignore")
    except Exception:
        return None

    key = f'name: "{MODEL_NAME}"'
    i = out.find(key)
    if i < 0:
        return None
    sub = out[i:i+1400]

    def get_float_after(s, k):
        j = s.find(k)
        if j < 0:
            return None
        s2 = s[j+len(k):]
        tok = ""
        for ch in s2:
            if ch in "0123456789+-.eE":
                tok += ch
            elif tok:
                break
        try:
            return float(tok)
        except Exception:
            return None

    # position x,y
    x = get_float_after(sub, "position {\n    x:")
    if x is None:
        x = get_float_after(sub, "x:")
    y = get_float_after(sub, "y:")

    # orientation quaternion: use qz, qw for planar yaw
    qz = get_float_after(sub, "orientation {\n    x:")  # anchor only
    qz = get_float_after(sub, "z:")
    qw = get_float_after(sub, "w:")

    if x is None or y is None or qz is None or qw is None:
        return None

    yaw = 2.0 * math.atan2(qz, qw)
    return x, y, yaw


# ============================================================
# SDF parsing + OGM build (NOW includes wood_maze model pose)
# ============================================================
def parse_pose(text):
    vals = [float(x) for x in text.strip().split()]
    if len(vals) == 6:
        return vals
    if len(vals) == 3:
        return vals + [0, 0, 0]
    return [0, 0, 0, 0, 0, 0]


def build_ogm_from_maze_sdf(world_sdf_path, resolution=0.02, margin=0.15):
    tree = ET.parse(world_sdf_path)
    root = tree.getroot()

    boxes = []
    for model in root.iter("model"):
        if model.attrib.get("name") != "wood_maze":
            continue

        # MODEL pose (apply to every link pose)
        model_pose_el = model.find("pose")
        mp = parse_pose(model_pose_el.text) if model_pose_el is not None else [0,0,0,0,0,0]
        mx, my, _, _, _, myaw = mp

        for link in model.iter("link"):
            pose_el = link.find("pose")
            lp = parse_pose(pose_el.text) if pose_el is not None else [0,0,0,0,0,0]
            lx, ly, _, _, _, lyaw = lp

            # Compose (approx): world = model + link (good if walls are axis-aligned links)
            x = mx + lx
            y = my + ly
            yaw = myaw + lyaw  # kept for completeness (we still raster axis-aligned below)

            col = link.find("collision")
            if col is None:
                continue
            geom = col.find("geometry")
            if geom is None:
                continue
            box = geom.find("box")
            if box is None:
                continue
            size_el = box.find("size")
            if size_el is None:
                continue
            sx, sy, _ = [float(v) for v in size_el.text.strip().split()]

            boxes.append((x, y, yaw, sx, sy))

    if not boxes:
        raise RuntimeError("No wall boxes found in wood_maze. Check Gazebo/worlds/maze.sdf.")

    xs, ys = [], []
    for (x, y, yaw, sx, sy) in boxes:
        xs += [x - sx / 2, x + sx / 2]
        ys += [y - sy / 2, y + sy / 2]

    min_x = min(xs) - margin
    max_x = max(xs) + margin
    min_y = min(ys) - margin
    max_y = max(ys) + margin

    width = int(math.ceil((max_x - min_x) / resolution))
    height = int(math.ceil((max_y - min_y) / resolution))

    occ = np.ones((height, width), dtype=np.uint8)  # 1 free, 0 obstacle

    def world_to_pixel_local(wx, wy):
        px = int((wx - min_x) / resolution)
        py_up = int((wy - min_y) / resolution)
        return px, py_up

    # NOTE: we raster axis-aligned AABBs. If any wall link is rotated, we can add rotated-box rastering.
    for (x, y, yaw, sx, sy) in boxes:
        x0, y0 = x - sx / 2, y - sy / 2
        x1, y1 = x + sx / 2, y + sy / 2
        px0, py0 = world_to_pixel_local(x0, y0)
        px1, py1 = world_to_pixel_local(x1, y1)

        px0, px1 = sorted([max(0, px0), min(width - 1, px1)])
        py0, py1 = sorted([max(0, py0), min(height - 1, py1)])
        occ[py0:py1 + 1, px0:px1 + 1] = 0

    ogm = (occ * 255).astype(np.uint8)
    meta = {
        "resolution": resolution,
        "min_x": min_x,
        "min_y": min_y,
        "width": width,
        "height": height,
    }
    return ogm, meta


# ============================================================
# world <-> grid mapping (planner grid: y DOWN)
# ============================================================
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


# ============================================================
# Inflation + snapping + reachability
# ============================================================
def inflate_obstacles_binary(free_mask: np.ndarray, R: int) -> np.ndarray:
    if R <= 0:
        return free_mask.copy()

    obs = ~free_mask
    try:
        from scipy.ndimage import binary_dilation
        inflated_obs = binary_dilation(obs, iterations=R)
        return ~inflated_obs
    except Exception:
        inflated_obs = obs.copy()
        for _ in range(R):
            new = inflated_obs.copy()
            new[1:, :] |= inflated_obs[:-1, :]
            new[:-1, :] |= inflated_obs[1:, :]
            new[:, 1:] |= inflated_obs[:, :-1]
            new[:, :-1] |= inflated_obs[:, 1:]
            inflated_obs = new
        return ~inflated_obs


def in_bounds(px, py, w, h):
    return 0 <= px < w and 0 <= py < h


def snap_to_nearest_free(px, py, free_mask, max_radius=200):
    h, w = free_mask.shape
    if in_bounds(px, py, w, h) and free_mask[py, px]:
        return px, py

    q = deque([(px, py)])
    seen = {(px, py)}
    while q:
        x, y = q.popleft()
        if in_bounds(x, y, w, h) and free_mask[y, x]:
            return x, y
        if max(abs(x - px), abs(y - py)) >= max_radius:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    raise RuntimeError(f"Could not find free cell near ({px},{py}). Increase MAP_MARGIN or reduce ROBOT_RADIUS_M.")


def reachable_mask_from_start(free_mask, start):
    h, w = free_mask.shape
    sx, sy = start
    reach = np.zeros((h, w), dtype=bool)
    if not in_bounds(sx, sy, w, h) or not free_mask[sy, sx]:
        return reach
    q = deque([(sx, sy)])
    reach[sy, sx] = True
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny, w, h) and (not reach[ny, nx]) and free_mask[ny, nx]:
                reach[ny, nx] = True
                q.append((nx, ny))
    return reach


def snap_goal_to_reachable(goal_px, reachable):
    gx, gy = goal_px
    h, w = reachable.shape
    if in_bounds(gx, gy, w, h) and reachable[gy, gx]:
        return gx, gy

    q = deque([(gx, gy)])
    seen = {(gx, gy)}
    while q:
        x, y = q.popleft()
        if in_bounds(x, y, w, h) and reachable[y, x]:
            return x, y
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    raise RuntimeError("Goal is not reachable from start component.")


# ============================================================
# Gazebo cmd_vel publisher
# ============================================================
def gz_pub_cmd_vel(v, w):
    msg = f"linear: {{x: {v:.4f}, y: 0, z: 0}} angular: {{x: 0, y: 0, z: {w:.4f}}}"
    subprocess.run(
        ["gz", "topic", "-t", "/model/robot/cmd_vel", "-m", "gz.msgs.Twist", "-p", msg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ============================================================
# Plotting
# ============================================================
def init_live_plot(ogm, meta, planned_world_path=None):
    plt.ion()
    fig, ax = plt.subplots()

    ogm_plot = np.flipud(ogm)

    extent = [
        meta["min_x"],
        meta["min_x"] + meta["width"] * meta["resolution"],
        meta["min_y"],
        meta["min_y"] + meta["height"] * meta["resolution"],
    ]

    ax.imshow(
        ogm_plot,
        origin="lower",
        cmap="gray",
        extent=extent,
        interpolation="nearest",
    )

    if planned_world_path is not None and len(planned_world_path) > 1:
        pxs = [p[0] for p in planned_world_path]
        pys = [p[1] for p in planned_world_path]
        ax.plot(pxs, pys, linestyle="--", linewidth=2, label="Planned path")

    ax.set_title("Robot Trajectory (live) - WORLD frame")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")

    xs, ys = [], []
    (line,) = ax.plot([], [], linewidth=2, label="Actual trajectory")
    (dot,) = ax.plot([], [], marker="o", markersize=6, label="Robot")

    ax.legend(loc="upper right")
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, dot, line, xs, ys


def update_live_plot(fig, dot, line, xs, ys, x, y, every_n=2):
    xs.append(float(x))
    ys.append(float(y))

    if len(xs) % every_n != 0:
        return

    dot.set_data([xs[-1]], [ys[-1]])
    line.set_data(xs, ys)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def save_trajectory(fig, xs, ys, out_png="trajectory.png", out_csv="trajectory.csv"):
    fig.savefig(out_png, dpi=200)
    with open(out_csv, "w") as f:
        f.write("t_index,x,y\n")
        for i, (x, y) in enumerate(zip(xs, ys)):
            f.write(f"{i},{x},{y}\n")


# ============================================================
# Controller (guaranteed stop)
# ============================================================
def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def hard_stop():
    for _ in range(STOP_REPEAT):
        gz_pub_cmd_vel(0.0, 0.0)
        time.sleep(STOP_DT)


def follow_waypoints(waypoints, ogm, meta, planned_world_path=None):
    fig, ax, dot, line, xs, ys = init_live_plot(ogm, meta, planned_world_path=planned_world_path)

    goal_x, goal_y = waypoints[-1]
    idx = 0

    while True:
        pose = gz_read_world_pose_once()
        if pose is None:
            time.sleep(DT)
            continue
        x, y, th = pose

        update_live_plot(fig, dot, line, xs, ys, x, y, every_n=2)

        dist_to_goal = math.hypot(goal_x - x, goal_y - y)
        if dist_to_goal <= GOAL_TOLERANCE:
            print(f"[Adapter] Goal reached (dist={dist_to_goal:.3f} m). Stopping.")
            hard_stop()
            save_trajectory(fig, xs, ys, out_png="trajectory.png", out_csv="trajectory.csv")
            plt.ioff()
            plt.show(block=False)
            return

        while idx < len(waypoints):
            tx, ty = waypoints[idx]
            if math.hypot(tx - x, ty - y) < LOOKAHEAD:
                idx += 1
            else:
                break
        idx = min(idx, len(waypoints) - 1)

        tx, ty = waypoints[idx]
        ang = math.atan2(ty - y, tx - x)
        e = wrap(ang - th)

        v = V_MAX * max(0.0, 1.0 - abs(e) / 1.2)
        w = max(-W_MAX, min(W_MAX, 2.5 * e))

        gz_pub_cmd_vel(v, w)
        time.sleep(DT)


# ============================================================
# Main
# ============================================================
def main():
    world_sdf = os.path.join(REPO_ROOT, "Gazebo", "worlds", "maze.sdf")
    ogm, meta = build_ogm_from_maze_sdf(world_sdf, resolution=MAP_RESOLUTION, margin=MAP_MARGIN)

    R = int(math.ceil(ROBOT_RADIUS_M / meta["resolution"]))
    free_mask = (ogm == 255)
    inflated_free = inflate_obstacles_binary(free_mask, R)

    print(f"[Adapter] Map dimensions: {meta['width']} x {meta['height']}")
    print(f"[Adapter] Inflation: ROBOT_RADIUS_M={ROBOT_RADIUS_M:.3f}m -> R={R} pixels")

    # Get start in WORLD frame (matches Gazebo video)
    if USE_MANUAL_START:
        sx_w, sy_w = MANUAL_START_WORLD
        print(f"[Adapter] Using MANUAL start (world): {sx_w:.3f}, {sy_w:.3f}")
        # yaw not needed for planning
    else:
        print(f"[Adapter] Waiting for /world/{WORLD_NAME}/pose/info ...")
        while True:
            pose = gz_read_world_pose_once()
            if pose is not None:
                break
        sx_w, sy_w, _ = pose
        print(f"[Adapter] WORLD start (world): {sx_w:.3f}, {sy_w:.3f}")

    gx_w, gy_w = GOAL_WORLD
    print(f"[Adapter] Goal (world): {gx_w:.3f}, {gy_w:.3f}")

    sx, sy = world_to_grid(sx_w, sy_w, meta)
    gx, gy = world_to_grid(gx_w, gy_w, meta)

    sx2, sy2 = snap_to_nearest_free(sx, sy, inflated_free)
    if (sx2, sy2) != (sx, sy):
        print(f"[Adapter] Start was blocked at ({sx},{sy}); snapped to ({sx2},{sy2})")

    reach = reachable_mask_from_start(inflated_free, (sx2, sy2))
    print(f"[Adapter] Reachable cells from start (inflated): {int(reach.sum())}")

    gx2, gy2 = snap_to_nearest_free(gx, gy, inflated_free)
    gx3, gy3 = snap_goal_to_reachable((gx2, gy2), reach)
    if (gx3, gy3) != (gx, gy):
        print(f"[Adapter] Goal adjusted to reachable: ({gx},{gy}) -> ({gx3},{gy3})")

    path_px = find_path(
        ogm_image=ogm,
        start_x=sx2,
        start_y=sy2,
        end_x=gx3,
        end_y=gy3,
        diagonal=True,
        R=R,
        downsample_factor=1
    )

    if path_px is None or len(path_px) < 2:
        raise RuntimeError("No path found. Ensure robot is inside maze corridors and/or reduce ROBOT_RADIUS_M.")

    planned_world_path = [grid_to_world(px, py, meta) for (px, py) in path_px]

    waypoints = [grid_to_world(px, py, meta) for (px, py) in path_px[::WAYPOINT_STEP]]
    waypoints.append(grid_to_world(path_px[-1][0], path_px[-1][1], meta))

    print(f"[Adapter] Path pixels: {len(path_px)} | waypoints: {len(waypoints)}")
    print("[Adapter] Driving... (WORLD pose, no mismatch)")
    follow_waypoints(waypoints, ogm, meta, planned_world_path=planned_world_path)


if __name__ == "__main__":
    main()
