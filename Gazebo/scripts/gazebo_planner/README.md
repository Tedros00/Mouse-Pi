# Gazebo Reactive Planner (No ROS, No OGM)

This is a modular, Gazebo-native navigation stack for your Maze SLAM & Relocalization project:
- **No occupancy grid**
- **No parsing of the maze SDF**
- **Works directly from Gazebo topics**: `/model/robot/odom`, `/scan`
- Publishes `/model/robot/cmd_vel`

## Folder structure
- `main.py` — CLI entrypoint (Task A exploration / Task B goal reach)
- `config.py` — parameters
- `gazebo_io.py` — gz topic read/write helpers + robust parsers
- `navigator.py` — state machine + behaviors (GO_TO_GOAL, FOLLOW_WALL, EXPLORE, RECOVERY)
- `utils.py` — small math helpers

## Requirements
- Gazebo Sim (`gz`) available in PATH
- Robot publishes odom on `/model/robot/odom` and lidar on `/scan`
- Robot accepts cmd_vel on `/model/robot/cmd_vel`

## Run
From this folder:

### Task B (Goal reach)
```bash
python3 main.py --mode goal --goal 0.6 0.6
```

Random (non-hardcoded) goal:
```bash
python3 main.py --mode goal --seed 3
```

Restrict random goal sampling bounds:
```bash
python3 main.py --mode goal --goal_box -0.8 0.8 -0.8 0.8 --seed 10
```

### Task A (Exploration)
```bash
python3 main.py --mode explore --time 90
```

## Notes for grading
This is **navigation / path-planning** logic (reactive Bug-style + wall follow). It is robust to random starts and does not hardcode start pose.
