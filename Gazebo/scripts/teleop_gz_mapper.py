#!/usr/bin/env python3
"""
Self-contained Matplotlib teleop + LiDAR mapping using an EMBEDDED SDF string (no external files).

Controls (click plot first):
  W/S = forward/back
  A/D = turn left/right
  Shift = boost
  Space = stop
  Esc = quit

Outputs:
  map.png, map.npy

Video output (NEW):
  run.mp4  (frame-by-frame recording while you teleop)
"""

import math
import time
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Optional, Deque
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter


# ============================================================
# EMBEDDED SDF (your maze.sdf pasted here)
# ============================================================
SDF_TEXT = r"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="maze_world">

    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <pose>0 0 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>20 20</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>20 20</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- ================= MAZE ================= -->
    <model name="wood_maze">
      <static>true</static>

      <link name="outer_bottom">
        <pose>0 -1.2 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>2.4 0.02 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>2.4 0.02 0.30</size></box></geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient></material>
        </visual>
      </link>

      <link name="outer_top">
        <pose>0 1.2 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>2.4 0.02 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>2.4 0.02 0.30</size></box></geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient></material>
        </visual>
      </link>

      <link name="outer_left">
        <pose>-1.2 0 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>0.02 2.4 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>0.02 2.4 0.30</size></box></geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient></material>
        </visual>
      </link>

      <link name="outer_right">
        <pose>1.2 0 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>0.02 2.4 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>0.02 2.4 0.30</size></box></geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient></material>
        </visual>
      </link>

      <link name="top_left_short_h">
        <pose>-0.75 0.6 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>0.9 0.02 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>0.9 0.02 0.30</size></box></geometry>
          <material><ambient>0.8 0.6 0.4 1</ambient></material>
        </visual>
      </link>

      <link name="top_center_vertical">
        <pose>0.0 0.9 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>0.02 0.6 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>0.02 0.6 0.30</size></box></geometry>
          <material><ambient>0.8 0.6 0.4 1</ambient></material>
        </visual>
      </link>

      <link name="top_right_long_h">
        <pose>0.6 0.6 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>1.2 0.02 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>1.2 0.02 0.30</size></box></geometry>
          <material><ambient>0.8 0.6 0.4 1</ambient></material>
        </visual>
      </link>

      <link name="top_right_cross_v">
        <pose>0.9 0.6 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>0.02 0.6 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>0.02 0.6 0.30</size></box></geometry>
          <material><ambient>0.8 0.6 0.4 1</ambient></material>
        </visual>
      </link>

      <link name="mid_left_cross_v">
        <pose>-0.6 0.0 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>0.02 0.9 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>0.02 0.9 0.30</size></box></geometry>
          <material><ambient>0.8 0.6 0.4 1</ambient></material>
        </visual>
      </link>

      <link name="mid_left_cross_h">
        <pose>-0.6 0.0 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>0.9 0.02 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>0.9 0.02 0.30</size></box></geometry>
          <material><ambient>0.8 0.6 0.4 1</ambient></material>
        </visual>
      </link>

      <link name="bottom_vertical">
        <pose>0.5 -0.6 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>0.02 1.2 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>0.02 1.2 0.30</size></box></geometry>
          <material><ambient>0.8 0.6 0.4 1</ambient></material>
        </visual>
      </link>

      <link name="bottom_horizontal_low">
        <pose>0.5 -0.9 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>0.8 0.02 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>0.8 0.02 0.30</size></box></geometry>
          <material><ambient>0.8 0.6 0.4 1</ambient></material>
        </visual>
      </link>

      <link name="bottom_horizontal_high">
        <pose>0.5 -0.3 0.15 0 0 0</pose>
        <collision name="col">
          <geometry><box><size>0.8 0.02 0.30</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>0.8 0.02 0.30</size></box></geometry>
          <material><ambient>0.8 0.6 0.4 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- ================= Robot ================= -->
    <include>
      <uri>model://robot</uri>
      <name>robot</name>
      <pose>0.25 0.25 0.1 0 0 0</pose>
    </include>

  </world>
</sdf>
"""


# =========================
# Configuration
# =========================
@dataclass
class Config:
    res_m: float = 0.02
    padding_m: float = 0.5
    dt: float = 0.05
    v_max: float = 0.8
    w_max: float = 2.5
    n_rays: int = 360
    r_max: float = 3.0           # your maze is ~2.4m wide; 3m is enough
    p0: float = 0.50
    p_occ: float = 0.70
    p_free: float = 0.30
    show_gt_underlay: bool = True
    mapped_threshold: float = 0.10

    # ---------- VIDEO RECORDING (NEW) ----------
    record_video: bool = True          # set False if you don't want video
    video_path: str = "run.mp4"
    video_fps: int = 20                # frames per second in the output
    video_dpi: int = 200               # dpi for saved frames
    video_bitrate: int = 3500          # quality-ish; bigger = larger files
    max_video_seconds: Optional[float] = None
    # if None: record until you close the window (Esc / close button)
    # if a number (e.g., 60): auto-stop after that many seconds


# =========================
# Utilities
# =========================
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def prob_to_logodds(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))

def logodds_to_prob(L: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-L))

def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float32)

def bresenham(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    if dy <= dx:
        err = dx / 2.0
        while x != x1:
            yield x, y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        yield x1, y1
    else:
        err = dy / 2.0
        while y != y1:
            yield x, y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        yield x1, y1


# =========================
# SDF → ground truth grid
# =========================
@dataclass
class Box2D:
    cx: float
    cy: float
    yaw: float
    sx: float
    sy: float

def parse_pose_text(pose_text: str) -> Tuple[float, float, float]:
    vals = [float(v) for v in pose_text.strip().split()]
    # SDF pose: x y z roll pitch yaw
    x = vals[0] if len(vals) > 0 else 0.0
    y = vals[1] if len(vals) > 1 else 0.0
    yaw = vals[5] if len(vals) > 5 else 0.0
    return x, y, yaw

def load_boxes_and_robot_from_embedded_sdf(sdf_text: str):
    root = ET.fromstring(sdf_text)

    # plane size
    plane_size = None
    plane = root.find(".//model[@name='ground_plane']//plane/size")
    if plane is not None and plane.text:
        sx, sy = [float(v) for v in plane.text.strip().split()[:2]]
        plane_size = (sx, sy)

    # robot include pose
    robot_pose = (0.0, 0.0, 0.0)
    for inc in root.findall(".//include"):
        name_el = inc.find("name")
        if name_el is not None and (name_el.text or "").strip() == "robot":
            pose_el = inc.find("pose")
            if pose_el is not None and pose_el.text:
                robot_pose = parse_pose_text(pose_el.text)
            break

    boxes = []
    # parse all link collision boxes; use link pose as box pose (matches your file)
    for link in root.findall(".//model[@name='wood_maze']//link"):
        pose_el = link.find("pose")
        lx = ly = lyaw = 0.0
        if pose_el is not None and pose_el.text:
            lx, ly, lyaw = parse_pose_text(pose_el.text)

        for col in link.findall("./collision"):
            geom = col.find("geometry")
            if geom is None:
                continue
            box = geom.find("box")
            if box is None:
                continue
            size_el = box.find("size")
            if size_el is None or not size_el.text:
                continue
            sx, sy, *_ = [float(v) for v in size_el.text.strip().split()]
            boxes.append(Box2D(cx=lx, cy=ly, yaw=lyaw, sx=sx, sy=sy))

    return boxes, plane_size, robot_pose

def rasterize_boxes(boxes: List[Box2D], cfg: Config, world_sx: float, world_sy: float):
    xmin = -world_sx / 2.0 - cfg.padding_m
    xmax =  world_sx / 2.0 + cfg.padding_m
    ymin = -world_sy / 2.0 - cfg.padding_m
    ymax =  world_sy / 2.0 + cfg.padding_m

    W = int(math.ceil((xmax - xmin) / cfg.res_m))
    H = int(math.ceil((ymax - ymin) / cfg.res_m))
    gt = np.zeros((H, W), dtype=np.uint8)

    def world_to_grid(x, y):
        gx = int((x - xmin) / cfg.res_m)
        gy = int((y - ymin) / cfg.res_m)
        return gx, gy

    for b in boxes:
        R = rot2d(b.yaw)
        Rt = R.T
        half = np.array([0.5 * b.sx, 0.5 * b.sy], dtype=np.float32)

        corners_local = np.array([
            [-half[0], -half[1]],
            [-half[0], +half[1]],
            [+half[0], -half[1]],
            [+half[0], +half[1]],
        ], dtype=np.float32)
        corners_world = corners_local @ R.T + np.array([b.cx, b.cy], dtype=np.float32)
        x0, y0 = corners_world.min(axis=0)
        x1, y1 = corners_world.max(axis=0)

        gx0, gy0 = world_to_grid(float(x0), float(y0))
        gx1, gy1 = world_to_grid(float(x1), float(y1))

        gx0 = max(0, gx0); gy0 = max(0, gy0)
        gx1 = min(W - 1, gx1); gy1 = min(H - 1, gy1)

        for gy in range(gy0, gy1 + 1):
            y = ymin + (gy + 0.5) * cfg.res_m
            for gx in range(gx0, gx1 + 1):
                x = xmin + (gx + 0.5) * cfg.res_m
                p = np.array([x - b.cx, y - b.cy], dtype=np.float32)
                pl = p @ Rt
                if abs(pl[0]) <= half[0] and abs(pl[1]) <= half[1]:
                    gt[gy, gx] = 1

    extent = (xmin, xmax, ymin, ymax)
    return gt, extent

def compute_reachable_free_mask(gt: np.ndarray, start_gx: int, start_gy: int) -> np.ndarray:
    """
    Accurate "coverage denominator":
    reachable free space from the robot start, 4-connected, on gt==0 cells.
    """
    H, W = gt.shape
    reachable = np.zeros((H, W), dtype=bool)
    if not (0 <= start_gx < W and 0 <= start_gy < H):
        return reachable
    if gt[start_gy, start_gx] != 0:
        return reachable

    q: Deque[Tuple[int, int]] = deque()
    q.append((start_gx, start_gy))
    reachable[start_gy, start_gx] = True

    while q:
        x, y = q.popleft()
        for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= nx < W and 0 <= ny < H and (not reachable[ny, nx]) and gt[ny, nx] == 0:
                reachable[ny, nx] = True
                q.append((nx, ny))
    return reachable


# =========================
# Sim: teleop + lidar + mapping
# =========================
class MappingSim:
    def __init__(self, gt: np.ndarray, extent, robot_pose, cfg: Config):
        self.cfg = cfg
        self.gt = gt
        self.xmin, self.xmax, self.ymin, self.ymax = extent
        self.H, self.W = gt.shape

        self.x, self.y, self.th = robot_pose

        self.v_cmd = 0.0
        self.w_cmd = 0.0

        self.keys = {"w": False, "a": False, "s": False, "d": False, "shift": False}

        self.l0 = prob_to_logodds(cfg.p0)
        self.l_occ = prob_to_logodds(cfg.p_occ)
        self.l_free = prob_to_logodds(cfg.p_free)
        self.L = np.full((self.H, self.W), self.l0, dtype=np.float32)
        self.L_min = prob_to_logodds(0.02)
        self.L_max = prob_to_logodds(0.98)

        self.angles = np.linspace(-math.pi, math.pi, cfg.n_rays, endpoint=False).astype(np.float32)

        if self.is_wall(self.x, self.y):
            self.find_nearest_free_start()

        # compute reachable free mask ONCE from start (accurate denominator)
        gx0, gy0 = self.world_to_grid(self.x, self.y)
        self.reachable_free = compute_reachable_free_mask(self.gt, gx0, gy0)
        self.reachable_free_count = int(np.sum(self.reachable_free))

        self.last_percent = 0.0

    def world_to_grid(self, x, y):
        gx = int((x - self.xmin) / self.cfg.res_m)
        gy = int((y - self.ymin) / self.cfg.res_m)
        return gx, gy

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.W and 0 <= gy < self.H

    def is_wall(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if not self.in_bounds(gx, gy):
            return True
        return self.gt[gy, gx] == 1

    def find_nearest_free_start(self):
        for r in np.linspace(0.0, 0.6, 200):
            for a in np.linspace(0, 2*math.pi, 90, endpoint=False):
                xx = self.x + r * math.cos(a)
                yy = self.y + r * math.sin(a)
                if not self.is_wall(xx, yy):
                    self.x, self.y = xx, yy
                    return

    def update_cmd_from_keys(self):
        boost = 1.5 if self.keys["shift"] else 1.0
        v = 0.0
        w = 0.0
        if self.keys["w"]:
            v += self.cfg.v_max * boost
        if self.keys["s"]:
            v -= self.cfg.v_max * boost
        if self.keys["a"]:
            w += self.cfg.w_max * boost
        if self.keys["d"]:
            w -= self.cfg.w_max * boost
        self.v_cmd = v
        self.w_cmd = w

    def step_robot(self):
        dt = self.cfg.dt
        th_new = wrap_pi(self.th + self.w_cmd * dt)
        x_new = self.x + self.v_cmd * math.cos(th_new) * dt
        y_new = self.y + self.v_cmd * math.sin(th_new) * dt
        if not self.is_wall(x_new, y_new):
            self.x, self.y = x_new, y_new
        self.th = th_new

    def cast_lidar(self) -> np.ndarray:
        ranges = np.empty((len(self.angles),), dtype=np.float32)
        step = max(0.5 * self.cfg.res_m, 0.01)
        for i, a in enumerate(self.angles):
            ang = self.th + float(a)
            r = 0.0
            hit = False
            while r < self.cfg.r_max:
                r += step
                px = self.x + r * math.cos(ang)
                py = self.y + r * math.sin(ang)
                if self.is_wall(px, py):
                    hit = True
                    break
            ranges[i] = r if hit else self.cfg.r_max
        return ranges

    def update_map(self, ranges: np.ndarray):
        gx0, gy0 = self.world_to_grid(self.x, self.y)
        if not self.in_bounds(gx0, gy0):
            return

        for a, dist in zip(self.angles, ranges):
            dist = float(min(dist, self.cfg.r_max))
            if dist <= 1e-3:
                continue
            ex = self.x + dist * math.cos(self.th + float(a))
            ey = self.y + dist * math.sin(self.th + float(a))
            gx1, gy1 = self.world_to_grid(ex, ey)
            if not self.in_bounds(gx1, gy1):
                continue

            for gx, gy in bresenham(gx0, gy0, gx1, gy1):
                if (gx, gy) == (gx1, gy1):
                    break
                self.L[gy, gx] = np.clip(self.L[gy, gx] + (self.l_free - self.l0), self.L_min, self.L_max)

            if dist < self.cfg.r_max - 1e-6:
                self.L[gy1, gx1] = np.clip(self.L[gy1, gx1] + (self.l_occ - self.l0), self.L_min, self.L_max)

    def percent_mapped(self) -> float:
        """
        Accurate coverage: only over reachable free cells.
        A cell is "covered/mapped" if its probability moved away from 0.5 enough.
        """
        if self.reachable_free_count <= 0:
            return 0.0
        P = logodds_to_prob(self.L)
        mapped = (np.abs(P - 0.5) > self.cfg.mapped_threshold)
        covered = mapped & self.reachable_free
        return 100.0 * float(np.sum(covered)) / float(self.reachable_free_count)

    def step(self):
        self.update_cmd_from_keys()
        self.step_robot()
        ranges = self.cast_lidar()
        self.update_map(ranges)
        self.last_percent = self.percent_mapped()


# =========================
# Main UI
# =========================
def main():
    cfg = Config()

    boxes, plane_size, robot_pose = load_boxes_and_robot_from_embedded_sdf(SDF_TEXT)
    world_sx, world_sy = plane_size if plane_size is not None else (20.0, 20.0)

    gt, extent = rasterize_boxes(boxes, cfg, world_sx, world_sy)
    sim = MappingSim(gt, extent, robot_pose, cfg)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    # FIXED title (never updated inside the loop)
    ax.set_title("Teleop LiDAR Mapping")

    # GT underlay
    if cfg.show_gt_underlay:
        gt_img = 1.0 - gt.astype(np.float32)  # walls=0, free=1
        ax.imshow(
            gt_img,
            origin="lower",
            extent=[extent[0], extent[1], extent[2], extent[3]],
            alpha=0.25,
            vmin=0.0,
            vmax=1.0,
        )

    # occupancy overlay
    P0 = logodds_to_prob(sim.L)
    occ_im = ax.imshow(
        P0,
        origin="lower",
        extent=[extent[0], extent[1], extent[2], extent[3]],
        vmin=0.0,
        vmax=1.0,
        alpha=0.85,
    )

    robot_dot, = ax.plot([sim.x], [sim.y], marker="o")
    heading_line, = ax.plot([sim.x, sim.x + 0.5], [sim.y, sim.y], linewidth=2)

    hud = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # LIVE TIMER start
    t0 = time.perf_counter()

    # ---------- VIDEO WRITER SETUP (NEW) ----------
    writer = None
    frames_written = 0

    if cfg.record_video:
        try:
            writer = FFMpegWriter(
                fps=cfg.video_fps,
                metadata={"title": "Teleop Mapping", "artist": "matplotlib"},
                bitrate=cfg.video_bitrate,
            )
            writer.setup(fig, cfg.video_path, dpi=cfg.video_dpi)
            print(f"[INFO] Recording video to {cfg.video_path} at {cfg.video_fps} fps ...")
        except Exception as e:
            writer = None
            print("[WARN] Could not start FFMpegWriter. Is ffmpeg installed?")
            print(f"[WARN] {e}")

    def on_key_press(event):
        if event.key is None:
            return
        k = event.key.lower()
        if k in ("w", "a", "s", "d"):
            sim.keys[k] = True
        elif k == "shift":
            sim.keys["shift"] = True
        elif k in (" ", "space"):
            sim.keys["w"] = sim.keys["a"] = sim.keys["s"] = sim.keys["d"] = False
            sim.v_cmd = 0.0
            sim.w_cmd = 0.0
        elif k == "escape":
            plt.close(fig)

    def on_key_release(event):
        if event.key is None:
            return
        k = event.key.lower()
        if k in ("w", "a", "s", "d"):
            sim.keys[k] = False
        elif k == "shift":
            sim.keys["shift"] = False

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    def update(_frame):
        nonlocal frames_written, writer

        sim.step()

        occ_im.set_data(logodds_to_prob(sim.L))
        robot_dot.set_data([sim.x], [sim.y])
        hx = sim.x + 0.7 * math.cos(sim.th)
        hy = sim.y + 0.7 * math.sin(sim.th)
        heading_line.set_data([sim.x, hx], [sim.y, hy])

        elapsed = time.perf_counter() - t0

        hud.set_text(
            f"Time: {elapsed:6.2f} s\n"
            f"Pose: x={sim.x:+.2f} m, y={sim.y:+.2f} m, yaw={math.degrees(sim.th):+.1f}°\n"
            f"Cmd : v={sim.v_cmd:+.2f} m/s, w={sim.w_cmd:+.2f} rad/s\n"
            f"Covered (reachable free): {sim.last_percent:6.2f}%\n"
            f"Grid: {sim.W}×{sim.H} @ {cfg.res_m:.2f} m/cell | LiDAR: {cfg.n_rays} rays, r_max={cfg.r_max:.1f} m"
        )

        # Write the current frame to video (frame-by-frame) AFTER artists update
        if writer is not None:
            writer.grab_frame()
            frames_written += 1

            if cfg.max_video_seconds is not None:
                if elapsed >= cfg.max_video_seconds:
                    plt.close(fig)

        return occ_im, robot_dot, heading_line, hud

    ani = FuncAnimation(fig, update, interval=int(cfg.dt * 1000), blit=False)
    plt.show()

    # Finish video cleanly
    if writer is not None:
        try:
            writer.finish()
            print(f"[INFO] Video saved: {cfg.video_path} (frames: {frames_written})")
        except Exception as e:
            print(f"[WARN] Could not finalize video: {e}")

    # Save outputs
    P = logodds_to_prob(sim.L)
    np.save("map.npy", P)
    plt.figure()
    plt.imshow(P, origin="lower", vmin=0.0, vmax=1.0,
               extent=[extent[0], extent[1], extent[2], extent[3]])
    plt.title("Final Occupancy Probability Map")
    plt.colorbar(label="P(occupied)")
    plt.savefig("map.png", dpi=200)
    print("[INFO] Saved map.npy and map.png")


if __name__ == "__main__":
    main()
