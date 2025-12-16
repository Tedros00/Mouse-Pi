#!/usr/bin/env python3
import math
import random
import subprocess
import time

WORLD = "maze_world"
ROBOT_NAME = "robot"
ROBOT_URI = "model://robot"

# IMPORTANT: Set this to a region you know is FREE in your maze.
# Start with something conservative, then expand after you confirm it works.
FREE_X = (-0.9, -0.6)
FREE_Y = ( 0.6,  0.9)

def run(cmd):
    print("[cmd]", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("[out]", p.stdout.strip())
    return p.returncode, p.stdout

def quat_from_yaw(yaw):
    # Gazebo messages expect quaternion. We'll use z,w for yaw rotation.
    return (0.0, 0.0, math.sin(yaw/2.0), math.cos(yaw/2.0))

def main():
    random.seed()

    # Give Gazebo a brief moment if you run this right after ./run.sh
    time.sleep(0.2)

    remove_srv = f"/world/{WORLD}/remove"
    create_srv = f"/world/{WORLD}/create"

    # 1) Remove existing robot (ok if it returns false sometimes)
    run([
        "gz", "service",
        "-s", remove_srv,
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "2000",
        "--req", f"name: '{ROBOT_NAME}'"
    ])

    time.sleep(0.1)

    # 2) Sample random pose in known-free region
    x = random.uniform(*FREE_X)
    y = random.uniform(*FREE_Y)
    yaw = random.uniform(-math.pi, math.pi)
    qx, qy, qz, qw = quat_from_yaw(yaw)

    print(f"[spawn] {ROBOT_NAME} @ x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

    req = (
        f"name: '{ROBOT_NAME}' "
        f"sdf_filename: '{ROBOT_URI}' "
        f"pose: {{ "
        f"  position: {{ x: {x}, y: {y}, z: 0.06 }} "
        f"  orientation: {{ x: {qx}, y: {qy}, z: {qz}, w: {qw} }} "
        f"}}"
    )

    run([
        "gz", "service",
        "-s", create_srv,
        "--reqtype", "gz.msgs.EntityFactory",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "2000",
        "--req", req
    ])

if __name__ == "__main__":
    main()
