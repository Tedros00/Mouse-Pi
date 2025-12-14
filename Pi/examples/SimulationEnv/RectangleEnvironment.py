import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import sys
import os

class RectangularRoomSimulation:
    """
    Simulation of a robot moving in a rectangular room with LiDAR measurements.
    """
    
    def __init__(self, 
                 room_width: float = 10.0,
                 room_height: float = 8.0,
                 ellipse_a_ratio: float = 0.7,  # Ratio of ellipse major axis to room width
                 ellipse_b_ratio: float = 0.7,  # Ratio of ellipse minor axis to room height
                 num_beams: int = 181,           # 60 degree FoV with 0.33 degree resolution
                 lidar_fov: float = np.pi/3,     # 60 degrees in radians
                 max_range: float = 15.0,
                 num_trajectory_points: int = 1000,
                 measurement_noise_std: float = 0.3,
                 motion_noise_std: tuple = (0.1, 0.05)):
        """
        Initialize the simulation.
        
        Parameters:
        -----------
        room_width : float
            Width of the rectangular room (X dimension)
        room_height : float
            Height of the rectangular room (Y dimension)
        ellipse_a_ratio : float
            Ratio of ellipse major axis to room width
        ellipse_b_ratio : float
            Ratio of ellipse minor axis to room height
        num_beams : int
            Number of LiDAR beams
        lidar_fov : float
            Field of view of LiDAR in radians
        max_range : float
            Maximum range of LiDAR
        num_trajectory_points : int
            Number of points in the trajectory
        measurement_noise_std : float
            Gaussian noise std for range measurements
        motion_noise_std : tuple
            Gaussian noise std for (linear velocity, angular velocity)
        """
        self.room_width = room_width
        self.room_height = room_height
        self.room_center = np.array([room_width / 2, room_height / 2])
        self.measurement_noise_std = measurement_noise_std
        self.motion_noise_std = motion_noise_std
        
        # Ellipse parameters
        self.ellipse_a = (room_width / 2) * ellipse_a_ratio   # Semi-major axis (X)
        self.ellipse_b = (room_height / 2) * ellipse_b_ratio  # Semi-minor axis (Y)
        
        # LiDAR parameters
        self.num_beams = num_beams
        self.lidar_fov = lidar_fov
        self.max_range = max_range
        self.beam_angles = np.linspace(-lidar_fov/2, lidar_fov/2, num_beams)
        
        # Trajectory parameters
        self.num_trajectory_points = num_trajectory_points
        
        # Storage
        self.trajectory = None      # (N, 3) - poses [x, y, theta]
        self.controls = None        # (N-1, 2) - [v, w]
        self.measurements = None    # (N, num_beams) - range measurements
        self.time_steps = None      # (N,) - time values
    
    def generate_trajectory(self, dt: float = 0.1):
        """
        Generate elliptical trajectory with the robot starting at center.
        The robot moves counterclockwise around an ellipse.
        
        Parameters:
        -----------
        dt : float
            Time step between trajectory points
        """
        # Parameter t goes from 0 to 2*pi for full ellipse
        t = np.linspace(0, 2 * np.pi, self.num_trajectory_points)
        
        # Elliptical trajectory centered at room center
        x = self.room_center[0] + self.ellipse_a * np.cos(t)
        y = self.room_center[1] + self.ellipse_b * np.sin(t)
        
        # Compute heading (tangent to ellipse)
        dx_dt = -self.ellipse_a * np.sin(t)
        dy_dt = self.ellipse_b * np.cos(t)
        theta = np.arctan2(dy_dt, dx_dt)
        
        self.trajectory = np.column_stack([x, y, theta])
        self.time_steps = np.arange(self.num_trajectory_points) * dt
        
        # Compute controls (velocity and angular velocity)
        self.controls = np.zeros((self.num_trajectory_points - 1, 2))
        
        for i in range(len(self.trajectory) - 1):
            curr_pose = self.trajectory[i]
            next_pose = self.trajectory[i + 1]
            
            # Linear distance
            dist = np.linalg.norm(next_pose[:2] - curr_pose[:2])
            v = dist / dt  # Linear velocity
            
            # Angular difference
            dtheta = next_pose[2] - curr_pose[2]
            # Normalize to [-pi, pi]
            while dtheta > np.pi:
                dtheta -= 2 * np.pi
            while dtheta < -np.pi:
                dtheta += 2 * np.pi
            w = dtheta / dt  # Angular velocity
            
            # Add Gaussian noise to controls
            v_noisy = v + np.random.normal(0, self.motion_noise_std[0])
            w_noisy = w + np.random.normal(0, self.motion_noise_std[1])
            
            self.controls[i] = [v_noisy, w_noisy]
    
    def _point_to_segment_distance(self, point: np.ndarray, 
                                   seg_start: np.ndarray, 
                                   seg_end: np.ndarray) -> float:
        """
        Compute distance from point to line segment.
        
        Parameters:
        -----------
        point : np.ndarray
            Point in 2D
        seg_start : np.ndarray
            Segment start point
        seg_end : np.ndarray
            Segment end point
        
        Returns:
        --------
        float
            Distance to segment
        """
        # Vector from start to end
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq == 0:
            return np.linalg.norm(point - seg_start)
        
        # Parameter t along segment
        t = max(0, min(1, np.dot(point - seg_start, seg_vec) / seg_len_sq))
        closest_point = seg_start + t * seg_vec
        
        return np.linalg.norm(point - closest_point)
    
    def _ray_cast(self, pose: np.ndarray, beam_angle: float) -> float:
        """
        Ray cast from robot position to find intersection with room walls.
        
        Parameters:
        -----------
        pose : np.ndarray
            Robot pose [x, y, theta]
        beam_angle : float
            Angle of beam relative to robot heading
        
        Returns:
        --------
        float
            Range to nearest obstacle (wall)
        """
        x, y, theta = pose
        robot_pos = np.array([x, y])
        
        # Absolute angle of beam
        abs_angle = theta + beam_angle
        beam_dir = np.array([np.cos(abs_angle), np.sin(abs_angle)])
        
        min_range = self.max_range
        
        # Room walls (4 lines)
        walls = [
            # Bottom wall (y=0)
            (np.array([0, 0]), np.array([self.room_width, 0])),
            # Top wall (y=room_height)
            (np.array([0, self.room_height]), np.array([self.room_width, self.room_height])),
            # Left wall (x=0)
            (np.array([0, 0]), np.array([0, self.room_height])),
            # Right wall (x=room_width)
            (np.array([self.room_width, 0]), np.array([self.room_width, self.room_height]))
        ]
        
        for wall_start, wall_end in walls:
            # Check intersection: ray from robot_pos in direction beam_dir with wall segment
            # Ray: P = robot_pos + t * beam_dir (t >= 0)
            # Wall: Q = wall_start + s * (wall_end - wall_start) (0 <= s <= 1)
            
            wall_vec = wall_end - wall_start
            
            # Solve: robot_pos + t * beam_dir = wall_start + s * wall_vec
            # Rearrange: t * beam_dir - s * wall_vec = wall_start - robot_pos
            
            # Matrix form: [beam_dir | -wall_vec] [t, s]^T = wall_start - robot_pos
            A = np.column_stack([beam_dir, -wall_vec])
            b = wall_start - robot_pos
            
            try:
                ts = np.linalg.solve(A, b)
                t, s = ts
                
                # Check validity
                if t >= 0 and 0 <= s <= 1:
                    # Valid intersection
                    range_val = t
                    if range_val < min_range:
                        min_range = range_val
            except np.linalg.LinAlgError:
                # Parallel lines, no intersection
                pass
        
        return min(min_range, self.max_range)
    
    def generate_measurements(self):
        """
        Generate LiDAR measurements for each pose in the trajectory.
        Adds Gaussian noise to simulate sensor uncertainty.
        """
        self.measurements = np.zeros((len(self.trajectory), self.num_beams))
        
        for i, pose in enumerate(self.trajectory):
            for j, beam_angle in enumerate(self.beam_angles):
                # Get ideal measurement from ray casting
                ideal_range = self._ray_cast(pose, beam_angle)
                # Add Gaussian noise
                noisy_range = ideal_range + np.random.normal(0, self.measurement_noise_std)
                # Clip to valid range
                self.measurements[i, j] = np.clip(noisy_range, 0, self.max_range)
    
    def get_pose(self, step: int) -> np.ndarray:
        """Get robot pose at a specific step."""
        if 0 <= step < len(self.trajectory):
            return self.trajectory[step].copy()
        return self.trajectory[-1].copy()
    
    def get_measurement(self, step: int) -> np.ndarray:
        """Get LiDAR measurement at a specific step."""
        if 0 <= step < len(self.measurements):
            return self.measurements[step].copy()
        return self.measurements[-1].copy()
    
    def get_control(self, step: int) -> np.ndarray:
        """Get control input at a specific step."""
        if 0 <= step < len(self.controls):
            return self.controls[step].copy()
        return np.array([0, 0])


class SimulationVisualizer:
    """
    Animated visualization of robot trajectory and LiDAR measurements.
    """
    
    def __init__(self, simulation: RectangularRoomSimulation, 
                 update_interval: int = 10):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        simulation : RectangularRoomSimulation
            The simulation object
        update_interval : int
            Update every N frames (for slower animation)
        """
        self.sim = simulation
        self.update_interval = update_interval
        self.current_step = 0
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 6))
        
        # Main trajectory view
        self.ax_main = self.fig.add_subplot(121)
        self.ax_main.set_xlim(-0.5, simulation.room_width + 0.5)
        self.ax_main.set_ylim(-0.5, simulation.room_height + 0.5)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlabel('X (m)')
        self.ax_main.set_ylabel('Y (m)')
        self.ax_main.set_title('Robot Trajectory and LiDAR Scan')
        
        # LiDAR polar view
        self.ax_lidar = self.fig.add_subplot(122, projection='polar')
        self.ax_lidar.set_ylim(0, simulation.max_range)
        self.ax_lidar.set_title('LiDAR Polar View')
        
        # Draw room walls
        wall_color = 'black'
        wall_width = 2
        self.ax_main.plot([0, simulation.room_width], [0, 0], color=wall_color, linewidth=wall_width)
        self.ax_main.plot([0, simulation.room_width], [simulation.room_height, simulation.room_height], 
                         color=wall_color, linewidth=wall_width)
        self.ax_main.plot([0, 0], [0, simulation.room_height], color=wall_color, linewidth=wall_width)
        self.ax_main.plot([simulation.room_width, simulation.room_width], [0, simulation.room_height], 
                         color=wall_color, linewidth=wall_width)
        
        # Draw trajectory as faint background
        self.ax_main.plot(simulation.trajectory[:, 0], simulation.trajectory[:, 1], 
                         'b--', alpha=0.2, label='Planned Trajectory', linewidth=1)
        
        # Robot representation
        self.robot_circle, = self.ax_main.plot([], [], 'ro', markersize=8, label='Robot')
        self.heading_line, = self.ax_main.plot([], [], 'r-', linewidth=2, label='Robot Heading')
        
        # LiDAR scan visualization
        self.scan_lines = LineCollection([], colors='green', linewidths=0.5)
        self.ax_main.add_collection(self.scan_lines)
        
        # LiDAR polar plot
        self.lidar_line, = self.ax_lidar.plot([], [], 'g-', linewidth=1, label='Measurements')
        
        # Text information
        self.info_text = self.ax_main.text(0.02, 0.98, '', transform=self.ax_main.transAxes,
                                           verticalalignment='top', fontfamily='monospace',
                                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax_main.legend(loc='upper right')
    
    def update_frame(self, frame: int) -> list:
        """
        Update animation frame.
        
        Parameters:
        -----------
        frame : int
            Frame number
        
        Returns:
        --------
        list
            Updated artists
        """
        step = frame // self.update_interval
        
        if step >= len(self.sim.trajectory):
            step = len(self.sim.trajectory) - 1
        
        self.current_step = step
        
        # Get current state
        pose = self.sim.get_pose(step)
        measurement = self.sim.get_measurement(step)
        control = self.sim.get_control(step)
        
        x, y, theta = pose
        
        # Update robot position and heading
        self.robot_circle.set_data([x], [y])
        
        # Draw heading arrow
        arrow_length = 0.5
        arrow_end_x = x + arrow_length * np.cos(theta)
        arrow_end_y = y + arrow_length * np.sin(theta)
        self.heading_line.set_data([x, arrow_end_x], [y, arrow_end_y])
        
        # Draw LiDAR scan rays
        scan_segments = []
        lidar_angles = []
        lidar_ranges = []
        
        for i, (beam_angle, range_val) in enumerate(zip(self.sim.beam_angles, measurement)):
            abs_angle = theta + beam_angle
            
            # Ray endpoint
            end_x = x + range_val * np.cos(abs_angle)
            end_y = y + range_val * np.sin(abs_angle)
            
            scan_segments.append([(x, y), (end_x, end_y)])
            
            # Polar coordinates for polar plot
            lidar_angles.append(beam_angle)
            lidar_ranges.append(range_val)
        
        self.scan_lines.set_segments(scan_segments)
        
        # Update polar plot
        lidar_angles = np.array(lidar_angles)
        lidar_ranges = np.array(lidar_ranges)
        self.lidar_line.set_data(lidar_angles, lidar_ranges)
        
        # Update info text
        v, w = control
        info_str = (f"Step: {step}/{len(self.sim.trajectory)-1}\n"
                   f"Position: ({x:.2f}, {y:.2f}) m\n"
                   f"Heading: {np.degrees(theta):.1f}°\n"
                   f"Velocity: {v:.2f} m/s\n"
                   f"Angular Vel: {np.degrees(w):.1f} °/s\n"
                   f"LiDAR FoV: {np.degrees(self.sim.lidar_fov):.1f}°\n"
                   f"Num Beams: {self.sim.num_beams}")
        self.info_text.set_text(info_str)
        
        return [self.robot_circle, self.heading_line, self.scan_lines, self.lidar_line, self.info_text]
    
    def animate(self, save_path: str = None):
        """
        Run animation.
        
        Parameters:
        -----------
        save_path : str, optional
            If provided, save animation to this path
        """
        total_frames = (len(self.sim.trajectory) - 1) * self.update_interval
        
        anim = FuncAnimation(self.fig, self.update_frame, frames=total_frames,
                            interval=50, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='ffmpeg', fps=20)
            print(f"Animation saved to {save_path}")
        
        plt.show()
    
    def save_frame(self, step: int, filepath: str):
        """Save a single frame at a specific step."""
        self.update_frame(step * self.update_interval)
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Frame saved to {filepath}")


def main():
    """Run the simulation and visualization."""
    print("=" * 70)
    print("Robot Trajectory and LiDAR Simulation")
    print("=" * 70)
    
    # Create simulation
    print("\nGenerating simulation...")
    sim = RectangularRoomSimulation(
        room_width=10.0,
        room_height=8.0,
        ellipse_a_ratio=0.7,
        ellipse_b_ratio=0.7,
        num_beams=181,              # 60° FoV with ~0.33° angular resolution
        lidar_fov=np.pi/3,           # 60 degrees
        max_range=15.0,
        num_trajectory_points=1000
    )
    
    print("  Generating elliptical trajectory...")
    sim.generate_trajectory(dt=0.01)  # 10 ms time step
    
    print("  Generating LiDAR measurements (ray-casting)...")
    sim.generate_measurements()
    
    print(f"\nSimulation Parameters:")
    print(f"  Room size: {sim.room_width} x {sim.room_height} m")
    print(f"  Ellipse axes: a={sim.ellipse_a:.2f} m, b={sim.ellipse_b:.2f} m")
    print(f"  Trajectory points: {len(sim.trajectory)}")
    print(f"  LiDAR beams: {sim.num_beams}")
    print(f"  LiDAR FoV: {np.degrees(sim.lidar_fov):.1f}°")
    print(f"  Max range: {sim.max_range} m")
    
    # Create and run visualization
    print("\nCreating visualization...")
    visualizer = SimulationVisualizer(sim, update_interval=5)
    
    print("\nStarting animation (close window to exit)...")
    visualizer.animate()


if __name__ == "__main__":
    main()
