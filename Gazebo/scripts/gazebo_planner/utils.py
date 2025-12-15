import math

def wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
