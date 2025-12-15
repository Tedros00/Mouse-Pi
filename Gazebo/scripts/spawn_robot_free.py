#!/usr/bin/env python3
import os
import math
import random
import subprocess
import xml.etree.ElementTree as ET

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORLD_SDF = os.path.join(REPO, "worlds", "maze.sdf")

def parse_pose(text):
    v = [float(x) for x in text.strip().split()]
    return v if len(v) == 6 else (v + [0, 0, 0])

def boxes_from_world(world_sdf):
    tree = ET.parse(world_sdf)
    root = tree.getroot()
    boxes = []
    for model in root.iter("model"):
        if model.attrib.get("name") != "wood_maze":
            continue
        for link in model.iter("link"):
            p = link.find("pose")
            x, y, _, _, _, _ = parse_pose(p.text) if p is not None else [0]*6
            col = link.find("collision")
            if col is None:
                continue
            geom = col.find("geometry")
            if geom is None:
                continue
            box = geom.find("box")
            if box is None:
                continue
            size = box.find("size")
            if size is None:
                continue
            sx, sy, _ = [float(s) for s in size.text.strip().split()]
            boxes.append((x, y, sx, sy))
    return boxes

def inside_any_box(x, y, boxes, pad=0.08):
    for bx, by, sx, sy in boxes:
        if (bx - sx/2 - pad) <= x <= (bx + sx/2 + pad) and \
           (by - sy/2 - pad) <= y <= (by + sy/2 + pad):
            return True
    return False

def world_bounds(boxes, margin=0.15):
    xs, ys = [], []
    for x, y, sx, sy in boxes:
        xs += [x - sx/2, x + sx/2]
        ys += [y - sy/2, y + sy/2]
    return min(xs)-margin, max(xs)+margin, min(ys)-margin, max(ys)+margin

def gz(cmd):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    boxes = boxes_from_world(WORLD_SDF)
    if not boxes:
        raise SystemExit("No maze geometry found")

    minx, maxx, miny, maxy = world_bounds(boxes)

    for _ in range(2000):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        yaw = random.uniform(-math.pi, math.pi)

        if inside_any_box(x, y, boxes):
            continue

        # Remove old robot if present
        gz([
            "gz", "service", "-s", "/world/default/remove",
            "--reqtype", "gz.msgs.Entity",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", "name: 'robot' type: MODEL"
        ])

        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)

        req = (
            "sdf_filename: 'model://robot' "
            "pose: { "
            f"position: {{ x: {x}, y: {y}, z: 0.05 }}, "
            f"orientation: {{ w: {qw}, z: {qz} }} "
            "} "
            "name: 'robot'"
        )

        gz([
            "gz", "service", "-s", "/world/default/create",
            "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", req
        ])

        print(f"[spawn] robot @ x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        return

    raise SystemExit("Failed to find a free spawn pose")

if __name__ == "__main__":
    main()
