import argparse
import time

from config import Params
from navigator import ReactiveNavigator
from gazebo_io import gz_pub_cmd_vel

def parse_args():
    ap = argparse.ArgumentParser(description="Gazebo-only reactive navigation (NO ROS, NO OGM).")
    ap.add_argument("--mode", choices=["goal", "explore"], default="goal",
                    help="goal: Task B reach a goal | explore: Task A exploration")
    ap.add_argument("--goal", nargs=2, type=float, default=None, metavar=("GX", "GY"),
                    help="Goal (world meters). If omitted in goal mode, a random goal is sampled.")
    ap.add_argument("--goal_box", nargs=4, type=float, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="Bounds for random goal sampling.")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed for repeatability.")
    ap.add_argument("--time", type=float, default=None,
                    help="Time limit in seconds. (explore mode defaults to params.explore_time_s)")
    ap.add_argument("--vmax", type=float, default=None, help="Override max linear speed.")
    ap.add_argument("--wmax", type=float, default=None, help="Override max angular speed.")
    ap.add_argument("--safe", type=float, default=None, help="Override safe front distance.")
    return ap.parse_args()

def main():
    args = parse_args()

    p = Params()
    if args.goal_box is not None:
        p.goal_box = (args.goal_box[0], args.goal_box[1], args.goal_box[2], args.goal_box[3])
    if args.vmax is not None:
        p.v_max = args.vmax
    if args.wmax is not None:
        p.w_max = args.wmax
    if args.safe is not None:
        p.safe_front = args.safe

    time_limit = args.time
    if time_limit is None:
        time_limit = p.explore_time_s if args.mode == "explore" else None

    nav = ReactiveNavigator(params=p, mode=args.mode,
                            goal=tuple(args.goal) if args.goal else None,
                            seed=args.seed)

    print(f"[planner] mode={args.mode} seed={args.seed}")
    if args.mode == "explore":
        print(f"[planner] explore_time_s={time_limit}")
    else:
        if args.goal:
            print(f"[planner] goal=({args.goal[0]:.2f},{args.goal[1]:.2f})")
        else:
            print(f"[planner] goal=None -> will sample within {p.goal_box}")

    t0 = time.time()
    try:
        while True:
            if time_limit is not None and (time.time() - t0) > time_limit:
                print("[planner] time limit reached; stopping.")
                break
            cont = nav.step()
            if not cont:
                break
    except KeyboardInterrupt:
        pass
    finally:
        gz_pub_cmd_vel(0.0, 0.0)

if __name__ == "__main__":
    main()
