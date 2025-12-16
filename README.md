# Mouse-Pi

A compact, self-contained repository for the Mouse-Pi autonomous mobile robotics project. This README summarizes the repository purpose, quick-start instructions, and the folder layout so contributors and maintainers can find relevant code, models, and hardware files quickly.

## Quick start
- Install Python dependencies: run `pip install -r requirements.txt` in a suitable venv.
- Simulation: open the Gazebo worlds in `Gazebo/worlds/` and run the scripts in `Gazebo/scripts/`.
- Development: most robot algorithms and helpers live in `src/` and `examples/` — open those to run localization, SLAM, and planning examples.
- Embedded firmware: use PlatformIO workspace in `PlatformIO/` to build and flash microcontroller code.

## Folder structure
- `3D parts/` : SolidWorks assemblies, parts and exported STLs for 3D printing and mechanical design.
	- `SW/` contains individual SolidWorks part files.
- `Gazebo/` : Simulation assets (worlds, models, and helper scripts).
	- `models/robot/` contains the robot model used by Gazebo.
	- `scripts/` contains launch and utility scripts (e.g., `ekf.py`, `plan.py`).
- `PC/` : Host/PC-specific code and tools (if present, tools for mapping/visualization).
- `Pi/` : Pi-side scripts and deployment notes for running on Raspberry Pi.
- `examples/` : Example scripts grouped by purpose: Sensors, Interfaces, Models, SimulationEnv — useful starting points and demos.
- `PlatformIO/` : Embedded firmware source and configuration (`platformio.ini`, `src/`).
- `requirements.txt` : Python dependencies for simulation and algorithms.
- `results/` : Output artifacts such as generated maps and logs (e.g., `map.npy`, `trajectory.csv`).

