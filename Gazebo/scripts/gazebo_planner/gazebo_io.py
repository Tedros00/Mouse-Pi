import math
import re
import subprocess
from typing import List, Optional, Tuple

from config import TOPIC_CMD_VEL, TOPIC_ODOM, TOPIC_SCAN

def gz_pub_cmd_vel(v: float, w: float) -> None:
    msg = f"linear: {{x: {v:.4f}, y: 0, z: 0}} angular: {{x: 0, y: 0, z: {w:.4f}}}"
    subprocess.run(
        ["gz", "topic", "-t", TOPIC_CMD_VEL, "-m", "gz.msgs.Twist", "-p", msg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def gz_read_once(topic: str, msg_type: str, timeout_s: float = 0.6) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["gz", "topic", "-e", "-t", topic, "-m", msg_type, "-n", "1"],
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        ).decode("utf-8", errors="ignore")
        return out
    except Exception:
        return None

def parse_odom_text(out: str) -> Optional[Tuple[float, float, float]]:
    # pose->position
    mpos = re.search(
        r"position\s*\{\s*[^}]*x:\s*([-0-9.eE]+)\s*[^}]*y:\s*([-0-9.eE]+)\s*[^}]*z:\s*([-0-9.eE]+)",
        out
    )
    # pose->orientation quaternion
    mori = re.search(
        r"orientation\s*\{\s*[^}]*x:\s*([-0-9.eE]+)\s*[^}]*y:\s*([-0-9.eE]+)\s*[^}]*z:\s*([-0-9.eE]+)\s*[^}]*w:\s*([-0-9.eE]+)",
        out
    )
    if not mpos or not mori:
        return None

    x = float(mpos.group(1))
    y = float(mpos.group(2))

    qx = float(mori.group(1))
    qy = float(mori.group(2))
    qz = float(mori.group(3))
    qw = float(mori.group(4))

    # yaw from quaternion (robust)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return x, y, yaw

def gz_read_odom() -> Optional[Tuple[float, float, float]]:
    out = gz_read_once(TOPIC_ODOM, "gz.msgs.Odometry")
    if out is None:
        return None
    return parse_odom_text(out)

def parse_scan_text(out: str) -> List[float]:
    ranges = []
    for m in re.finditer(r"ranges:\s*([-0-9.eE]+)", out):
        r = float(m.group(1))
        if r > 0.0:
            ranges.append(r)
    return ranges

def gz_read_scan() -> Optional[List[float]]:
    out = gz_read_once(TOPIC_SCAN, "gz.msgs.LaserScan")
    if out is None:
        return None
    ranges = parse_scan_text(out)
    if not ranges:
        return None
    return ranges
