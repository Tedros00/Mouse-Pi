from dataclasses import dataclass
from typing import Tuple

# Gazebo topics (NO ROS)
TOPIC_CMD_VEL = "/model/robot/cmd_vel"
TOPIC_ODOM    = "/model/robot/odom"
TOPIC_SCAN    = "/scan"

@dataclass
class Params:
    # motion limits
    v_max: float = 0.22
    w_max: float = 1.6

    # safety / obstacle thresholds
    safe_front: float = 0.35        # front clearance threshold (m)
    side_target: float = 0.30       # desired wall-follow side distance (m)
    hard_stop: float = 0.18         # emergency stop distance (m)

    # goal tolerance
    goal_tol: float = 0.10

    # control timing
    dt: float = 0.05

    # wall follow gains
    k_wall: float = 2.0
    k_heading: float = 2.6

    # stuck detection
    stuck_window_s: float = 1.5
    stuck_min_motion: float = 0.03   # meters in window
    recovery_spin_s: float = 0.9

    # exploration
    explore_time_s: float = 60.0
    novelty_grid: float = 0.20       # meters per visited cell
    novelty_min_new_frac: float = 0.02  # if new cells fraction too low -> change behavior
    novelty_check_s: float = 8.0

    # random goal sampling bounds (world meters): xmin,xmax,ymin,ymax
    goal_box: Tuple[float, float, float, float] = (-1.2, 1.2, -1.2, 1.2)
