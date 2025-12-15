#!/usr/bin/env python3
import math
import time
import subprocess
import argparse
import re
import sys

# ===================== PARAMETERS =====================
SAFE_DIST = 0.35      # meters (obstacle threshold)
V_MAX     = 0.22      # m/s
W_MAX     = 1.5       # rad/s
DT        = 0.05      # control period
GOAL_TOL  = 0.10      # meters

# =====================================================
# Utility
# =====================================================
def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

# =====================================================
# Gazebo I/O
# =====================================================
def gz_pub_cmd_vel(v, w):
    msg = f"linear: {{x: {v}}} angular: {{z: {w}}}"
    subprocess.run(
        ["gz", "topic",
         "-t", "/model/robot/cmd_vel",
         "-m", "gz.msgs.Twist",
         "-p", msg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def read_odom():
    out = subprocess.check_output(
        ["gz", "topic", "-e",
         "-t", "/model/robot/odom",
         "-m", "gz.msgs.Odometry",
         "-n", "1"],
        stderr=subprocess.DEVNULL
    ).decode(errors="ignore")

    def find(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else 0.0

    x  = find(r"x:\s*([-0-9.eE]+)")
    y  = find(r"y:\s*([-0-9.eE]+)")
    oz = find(r"z:\s*([-0-9.eE]+)")
    ow = find(r"w:\s*([-0-9.eE]+)")

    yaw = 2.0 * math.atan2(oz, ow)
    return x, y, yaw

def read_lidar():
    out = subprocess.check_output(
        ["gz", "topic", "-e",
         "-t", "/scan",
         "-m", "gz.msgs.LaserScan",
         "-n", "1"],
        stderr=subprocess.DEVNULL
    ).decode(errors="ignore")

    ranges = []
    for m in re.finditer(r"ranges:\s*([-0-9.eE]+)", out):
        r = float(m.group(1))
        if r > 0.01:
            ranges.append(r)
    return ranges

# =====================================================
# Reactive Planner (Bug-style)
# =====================================================
def main(goal):
    print("[planner] Reactive navigation started")
    print(f"[planner] Goal = ({goal[0]:.2f}, {goal[1]:.2f})")

    while True:
        try:
            x, y, th = read_odom()
            ranges = read_lidar()
        except Exception:
            time.sleep(DT)
            continue

        dx = goal[0] - x
        dy = goal[1] - y
        dist = math.hypot(dx, dy)

        # Goal reached
        if dist < GOAL_TOL:
            gz_pub_cmd_vel(0.0, 0.0)
            print("[planner] Goal reached")
            return

        goal_ang = math.atan2(dy, dx)
        heading_err = wrap(goal_ang - th)

        # Lidar sectors
        n = len(ranges)
        if n == 0:
            continue

        front = min(ranges[n//3 : 2*n//3])
        left  = min(ranges[2*n//3 :])
        right = min(ranges[: n//3])

        # Decision logic
        if front > SAFE_DIST:
            v = V_MAX * max(0.0, 1.0 - abs(heading_err))
            w = max(-W_MAX, min(W_MAX, 2.5 * heading_err))
        else:
            # Obstacle ahead → wall follow
            v = 0.05
            w = W_MAX if left > right else -W_MAX

        gz_pub_cmd_vel(v, w)
        time.sleep(DT)

# =====================================================
# CLI
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", nargs=2, type=float, metavar=("GX", "GY"),
                        required=True, help="Goal position in world coordinates")
    args = parser.parse_args()

    main((args.goal[0], args.goal[1]))
