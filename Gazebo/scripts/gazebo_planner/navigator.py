import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import Params
from gazebo_io import gz_pub_cmd_vel, gz_read_odom, gz_read_scan
from utils import wrap, clamp

def now_s() -> float:
    return time.time()

class State:
    GO_TO_GOAL  = "GO_TO_GOAL"
    FOLLOW_WALL = "FOLLOW_WALL"
    EXPLORE     = "EXPLORE"
    RECOVERY    = "RECOVERY"

class ReactiveNavigator:
    def __init__(self, params: Params, mode: str, goal: Optional[Tuple[float, float]], seed: int):
        self.p = params
        self.mode = mode
        random.seed(seed)

        self.goal = goal
        self.state = State.EXPLORE if mode == "explore" else State.GO_TO_GOAL

        self.follow_left = True

        # stuck detection
        self._pose_hist: List[Tuple[float, float, float]] = []
        self._t_hist: List[float] = []
        self._recovery_until = 0.0
        self._recovery_w = 0.0

        # exploration novelty proxy
        self.visited = set()
        self._last_novelty_check = now_s()
        self._last_visited_count = 0

        # wall-follow leave condition support
        self._best_goal_dist = float("inf")

    def _sector_min(self, ranges: List[float], a0: float, a1: float) -> float:
        n = len(ranges)
        i0 = int(clamp(a0, 0.0, 1.0) * (n - 1))
        i1 = int(clamp(a1, 0.0, 1.0) * (n - 1))
        if i1 <= i0:
            i1 = min(n - 1, i0 + 1)
        seg = ranges[i0:i1]
        return min(seg) if seg else min(ranges)

    def _read_pose_and_scan(self):
        od = gz_read_odom()
        sc = gz_read_scan()
        if od is None or sc is None:
            return None
        return od, sc

    def _update_visited(self, x: float, y: float) -> None:
        g = self.p.novelty_grid
        cx = int(math.floor(x / g))
        cy = int(math.floor(y / g))
        self.visited.add((cx, cy))

    def _maybe_pick_random_goal(self) -> None:
        if self.goal is not None:
            return
        xmin, xmax, ymin, ymax = self.p.goal_box
        gx = random.uniform(xmin, xmax)
        gy = random.uniform(ymin, ymax)
        self.goal = (gx, gy)
        print(f"[planner] Random goal sampled: ({gx:.2f}, {gy:.2f}) within box {self.p.goal_box}")

    def _stuck(self) -> bool:
        if len(self._pose_hist) < 2:
            return False
        t_now = now_s()
        while self._t_hist and (t_now - self._t_hist[0]) > self.p.stuck_window_s:
            self._t_hist.pop(0)
            self._pose_hist.pop(0)
        if len(self._pose_hist) < 2:
            return False
        x0, y0, _ = self._pose_hist[0]
        x1, y1, _ = self._pose_hist[-1]
        moved = math.hypot(x1 - x0, y1 - y0)
        return moved < self.p.stuck_min_motion

    def _enter_recovery(self) -> None:
        self.state = State.RECOVERY
        self._recovery_until = now_s() + self.p.recovery_spin_s
        self._recovery_w = -self.p.w_max if self.follow_left else self.p.w_max
        print("[planner] Entering RECOVERY (stuck).")

    def _control_recovery(self):
        if now_s() >= self._recovery_until:
            self.state = State.EXPLORE if self.mode == "explore" else State.GO_TO_GOAL
            print("[planner] Recovery complete; resuming.")
            return 0.0, 0.0
        return 0.0, self._recovery_w

    def _control_go_to_goal(self, pose, ranges):
        assert self.goal is not None
        x, y, th = pose
        gx, gy = self.goal
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        if dist <= self.p.goal_tol:
            print("[planner] Goal reached.")
            return 0.0, 0.0

        goal_ang = math.atan2(dy, dx)
        e = wrap(goal_ang - th)

        front = self._sector_min(ranges, 0.40, 0.60)
        left  = self._sector_min(ranges, 0.70, 0.95)
        right = self._sector_min(ranges, 0.05, 0.30)

        if front < self.p.hard_stop:
            self.follow_left = (left > right)
            self.state = State.FOLLOW_WALL
            self._best_goal_dist = dist
            return 0.0, self.p.w_max if self.follow_left else -self.p.w_max

        if front < self.p.safe_front:
            self.follow_left = (left > right)
            self.state = State.FOLLOW_WALL
            self._best_goal_dist = dist
            print(f"[planner] Switching to FOLLOW_WALL (front={front:.2f}). follow_left={self.follow_left}")
            return 0.05, self.p.w_max * (0.8 if self.follow_left else -0.8)

        w = clamp(self.p.k_heading * e, -self.p.w_max, self.p.w_max)
        v = self.p.v_max * max(0.0, 1.0 - abs(e) / 1.2)
        return v, w

    def _control_follow_wall(self, pose, ranges):
        assert self.goal is not None
        x, y, _ = pose
        gx, gy = self.goal

        front   = self._sector_min(ranges, 0.45, 0.55)
        front_l = self._sector_min(ranges, 0.60, 0.75)
        front_r = self._sector_min(ranges, 0.25, 0.40)
        side_l  = self._sector_min(ranges, 0.80, 0.98)
        side_r  = self._sector_min(ranges, 0.02, 0.20)

        dist_goal = math.hypot(gx - x, gy - y)
        self._best_goal_dist = min(self._best_goal_dist, dist_goal)

        if front > (self.p.safe_front + 0.10) and dist_goal <= (self._best_goal_dist + 0.05):
            self.state = State.GO_TO_GOAL
            print("[planner] Leaving wall -> GO_TO_GOAL")
            return 0.0, 0.0

        if self.follow_left:
            side = side_l
            ahead_side = front_l
            turn_sign = +1.0
        else:
            side = side_r
            ahead_side = front_r
            turn_sign = -1.0

        if front < self.p.hard_stop or ahead_side < self.p.hard_stop:
            return 0.02, turn_sign * self.p.w_max

        e_side = (self.p.side_target - side)
        if side > 2.0:
            e_side = -0.30

        w = turn_sign * clamp(self.p.k_wall * e_side, -self.p.w_max, self.p.w_max)

        v = self.p.v_max * 0.55
        if front < (self.p.safe_front + 0.10):
            v *= 0.5
        if abs(w) > 1.0:
            v *= 0.7

        return v, w

    def _control_explore(self, pose, ranges):
        x, y, _ = pose

        front = self._sector_min(ranges, 0.45, 0.55)
        left  = self._sector_min(ranges, 0.70, 0.95)
        right = self._sector_min(ranges, 0.05, 0.30)

        if front < self.p.safe_front:
            self.follow_left = (left > right)

        t = now_s()
        if (t - self._last_novelty_check) > self.p.novelty_check_s:
            new_cells = len(self.visited) - self._last_visited_count
            frac = new_cells / max(1, self._last_visited_count)
            self._last_visited_count = len(self.visited)
            self._last_novelty_check = t
            if frac < self.p.novelty_min_new_frac:
                self.follow_left = random.choice([True, False])
                self._enter_recovery()
                return 0.0, 0.0

        if front > (self.p.safe_front + 0.25):
            if left > right + 0.20:
                return self.p.v_max * 0.6, +0.45
            if right > left + 0.20:
                return self.p.v_max * 0.6, -0.45

        # Reuse wall-follow control. Provide a dummy goal (not used for explore).
        if self.goal is None:
            self.goal = (x, y)
        return self._control_follow_wall(pose, ranges)

    def step(self) -> bool:
        data = self._read_pose_and_scan()
        if data is None:
            time.sleep(self.p.dt)
            return True

        pose, ranges = data
        x, y, _ = pose

        self._update_visited(x, y)

        self._pose_hist.append(pose)
        self._t_hist.append(now_s())

        if self.state != State.RECOVERY and self._stuck():
            self._enter_recovery()

        if self.mode == "goal":
            self._maybe_pick_random_goal()

        if self.state == State.RECOVERY:
            v, w = self._control_recovery()
        elif self.mode == "explore":
            v, w = self._control_explore(pose, ranges)
        else:
            assert self.goal is not None
            if self.state == State.GO_TO_GOAL:
                v, w = self._control_go_to_goal(pose, ranges)
            else:
                v, w = self._control_follow_wall(pose, ranges)

        gz_pub_cmd_vel(v, w)

        if self.mode == "goal" and self.goal is not None:
            gx, gy = self.goal
            if math.hypot(gx - x, gy - y) <= self.p.goal_tol:
                gz_pub_cmd_vel(0.0, 0.0)
                return False

        time.sleep(self.p.dt)
        return True
