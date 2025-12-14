import numpy as np
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add exam2 directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exam2'))

from ProbabilisticSensorModel import beam_range_finder_model


def occupancy_grid_mapping(Z, X, grid_shape, grid_bounds, 
                           compute_from_map, beam_model_params=None, 
                           convergence_threshold=1e-3, max_iterations=100, verbose=False):
    """
    Occupancy Grid Mapping using inverse measurement model.
    
    Uses the beam range finder model to compute p(z|x,m) for each cell.
    Applies inverse sensor model to update log-odds representation.
    
    Parameters:
    -----------
    Z : list of arrays
        Measurements z_1, ..., z_T. Each element is a measurement vector at time t.
    X : list of tuples
        Poses x_1, ..., x_T. Each element is (x, y, theta).
    grid_shape : tuple
        Shape of the occupancy grid (rows, cols).
    grid_bounds : tuple
        Bounds of the grid ((min_x, max_x), (min_y, max_y)).
    compute_from_map : function
        Function to compute expected range given pose and map.
        Signature: compute_from_map(x, y, angle, map_data, max_range).
    beam_model_params : dict, optional
        Parameters for beam range finder model.
    convergence_threshold : float
        Convergence threshold for log-odds change.
    max_iterations : int
        Maximum number of EM iterations.
    
    Returns:
    --------
    m : ndarray
        Occupancy grid with log-odds values.
    grid_coords : tuple
        ((cell_size_x, cell_size_y), (min_x, max_x, min_y, max_y))
        Information to convert back to world coordinates.
    """
    
    # Set default beam model parameters
    if beam_model_params is None:
        beam_model_params = {
            'min_theta': -np.pi/2,
            'max_theta': np.pi/2,
            'max_range': 10.0,
            'sigma_hit': 0.2,
            'lambda_short': 0.1,
            'z_hit': 0.7,
            'z_short': 0.1,
            'z_max': 0.1,
            'z_rand': 0.1
        }
    
    # Initialize log-odds grid
    rows, cols = grid_shape
    (min_x, max_x), (min_y, max_y) = grid_bounds
    cell_size_x = (max_x - min_x) / cols
    cell_size_y = (max_y - min_y) / rows
    
    # Log-odds representation: L(m_i) = 0 initially (uniform prior)
    m = np.zeros(grid_shape)
    
    # Prior in log-odds (log(0.5 / 0.5) = 0, meaning 50% occupancy prior)
    L0 = 0.0
    
    # EM iterations
    if verbose:
        print(f"\n[OGM] Starting iterative updates (max {max_iterations} iterations)...")
    for iteration in range(max_iterations):
        m_prev = m.copy()
        
        # For each grid cell
        for i in range(rows):
            for j in range(cols):
                # Convert grid cell (i, j) to world coordinates
                cell_x = min_x + (j + 0.5) * cell_size_x
                cell_y = min_y + (i + 0.5) * cell_size_y
                
                # Log-odds update for this cell
                log_odds_update = 0.0
                
                # For each measurement-pose pair
                for t in range(len(Z)):
                    z_t = Z[t]
                    x_t = X[t]
                    x_pose, y_pose, theta_pose = x_t
                    
                    # Inverse sensor model parameters
                    min_theta = beam_model_params['min_theta']
                    max_theta = beam_model_params['max_theta']
                    num_beams = len(z_t)
                    sigma_hit = beam_model_params.get('sigma_hit', 0.2)
                    
                    # Range and angle to cell from sensor
                    range_to_cell = np.sqrt((cell_x - x_pose)**2 + (cell_y - y_pose)**2)
                    angle_to_cell = np.arctan2(cell_y - y_pose, cell_x - x_pose)
                    
                    # For each beam ray
                    for beam_idx in range(num_beams):
                        # Beam angle from sensor
                        beam_angle = theta_pose + min_theta + beam_idx * (max_theta - min_theta) / (num_beams - 1)
                        z_measured = z_t[beam_idx]
                        
                        # Normalize angle differences
                        angle_diff = angle_to_cell - beam_angle
                        while angle_diff > np.pi:
                            angle_diff -= 2 * np.pi
                        while angle_diff < -np.pi:
                            angle_diff += 2 * np.pi
                        
                        # Only process if beam points roughly toward this cell
                        beam_width = (max_theta - min_theta) / num_beams
                        if abs(angle_diff) < 1.5 * beam_width:
                            # Inverse sensor model (simplified occupancy grid inverse)
                            # Key insight: 
                            # - If z_measured is close to range_to_cell, cell is likely at the hit -> OCCUPIED
                            # - If z_measured < range_to_cell, cell is beyond hit -> FREE
                            # - If z_measured > range_to_cell, cell is before hit -> OCCUPIED
                            
                            margin = 3 * sigma_hit
                            
                            if abs(z_measured - range_to_cell) <= margin:
                                # Cell is at beam hit point -> OCCUPIED (positive log-odds)
                                p_occ = 0.7  # Hit probability
                                log_odds_update += np.log(p_occ / (1 - p_occ))
                            elif z_measured > range_to_cell:
                                # Beam passed through cell without hitting -> FREE (negative log-odds)
                                p_occ = 0.2  # Low occupancy
                                log_odds_update += np.log(p_occ / (1 - p_occ))
                            # else: z_measured < range_to_cell (unknown, no update)
                
                # Update log-odds for this cell
                m[i, j] = L0 + log_odds_update
        
        # Check convergence
        delta = np.sum(np.abs(m - m_prev))
        
        if verbose and (iteration % max(1, max_iterations // 5) == 0 or iteration < 3):
            print(f"  Iteration {iteration + 1}: Delta = {delta:.6e}, Log-odds range: [{np.min(m):.2f}, {np.max(m):.2f}]")
        
        if delta < convergence_threshold:
            if verbose:
                print(f"✓ Converged at iteration {iteration + 1}")
            break
    
    # Store grid information for coordinate conversion
    grid_coords = ((cell_size_x, cell_size_y), (min_x, max_x, min_y, max_y))
    
    return m, grid_coords


def log_odds_to_occupancy_probability(log_odds):
    """
    Convert log-odds representation to occupancy probability [0, 1].
    
    Parameters:
    -----------
    log_odds : ndarray
        Log-odds grid values.
    
    Returns:
    --------
    prob : ndarray
        Occupancy probabilities in [0, 1].
    """
    return 1.0 - 1.0 / (1.0 + np.exp(log_odds))


def incremental_occupancy_grid_update(current_map, pose, measurement, grid_shape, grid_bounds,
                                      min_theta=-np.pi, max_theta=np.pi, max_range=300.0,
                                      sigma_hit=0.2):
    """
    Incremental inverse sensor model update using raycasting.
    
    Updates the occupancy grid by raycasting from robot to measurement points.
    Accumulates log-odds evidence properly without recomputing from scratch.
    
    Parameters:
    -----------
    current_map : ndarray
        Current occupancy grid (log-odds)
    pose : tuple or array
        Robot pose [x, y, theta]
    measurement : ndarray
        Range measurements (downsampled beams)
    grid_shape : tuple
        Shape of occupancy grid (rows, cols)
    grid_bounds : tuple
        Bounds of grid ((min_x, max_x), (min_y, max_y))
    min_theta : float
        Minimum beam angle (default: -π)
    max_theta : float
        Maximum beam angle (default: π)
    max_range : float
        Maximum sensor range (default: 300.0)
    sigma_hit : float
        Measurement noise std (default: 0.2)
    
    Returns:
    --------
    ndarray
        Updated occupancy grid (log-odds)
    """
    rows, cols = grid_shape
    (min_x, max_x), (min_y, max_y) = grid_bounds
    cell_size_x = (max_x - min_x) / cols
    cell_size_y = (max_y - min_y) / rows
    
    updated_grid = current_map.copy()
    x, y, theta = pose[0], pose[1], pose[2]
    z = measurement
    num_beams = len(z)
    
    # Log-odds for hit and free cells
    l_occ = np.log(0.7 / 0.3)      # Occupied: ~0.847
    l_free = np.log(0.2 / 0.8)     # Free: ~-1.386 (stronger evidence for free)
    
    # Process each beam
    for beam_idx in range(num_beams):
        z_measured = z[beam_idx]
        
        # Skip invalid measurements
        if z_measured >= max_range or z_measured <= 0:
            continue
        
        # Beam angle in world frame
        beam_angle = theta + min_theta + beam_idx * (max_theta - min_theta) / (num_beams - 1)
        beam_dx = np.cos(beam_angle)
        beam_dy = np.sin(beam_angle)
        
        # Hit point in world coordinates
        hit_x = x + z_measured * beam_dx
        hit_y = y + z_measured * beam_dy
        
        # Raycast from robot to hit point
        max_steps = int(z_measured / min(cell_size_x, cell_size_y)) + 1
        for step in range(max_steps):
            # Interpolate along ray
            alpha = step / max(1, max_steps - 1) if max_steps > 1 else 0.0
            cell_x = x + alpha * (hit_x - x)
            cell_y = y + alpha * (hit_y - y)
            
            # Convert to grid indices
            col = int((cell_x - min_x) / cell_size_x)
            row = int((cell_y - min_y) / cell_size_y)
            
            # Check bounds
            if row < 0 or row >= rows or col < 0 or col >= cols:
                continue
            
            # Distance from robot to this cell
            dist_to_cell = np.sqrt((cell_x - x)**2 + (cell_y - y)**2)
            
            # Determine if cell is occupied or free
            if abs(dist_to_cell - z_measured) <= 2 * sigma_hit:
                # Cell at hit point -> OCCUPIED
                updated_grid[row, col] += l_occ
            elif dist_to_cell < z_measured - 2 * sigma_hit:
                # Cell in free space before hit
                updated_grid[row, col] += l_free
            # else: beyond hit point, no update
    
    # Clip log-odds to prevent numerical issues
    updated_grid = np.clip(updated_grid, -10.0, 10.0)
    
    return updated_grid


def save_ogm_to_pgm(log_odds_grid, filepath):
    """
    Save occupancy grid (log-odds) to PGM file.
    
    Parameters:
    -----------
    log_odds_grid : np.ndarray
        Log-odds occupancy grid
    filepath : str
        Path to save the PGM file
    """
    # Convert log-odds to occupancy probability [0, 1]
    occupancy_prob = log_odds_to_occupancy_probability(log_odds_grid)
    
    # Convert to grayscale [0, 255]
    # Occupied (1) -> black (0), Free (0) -> white (255)
    pgm_data = (255 * (1 - occupancy_prob)).astype(np.uint8)
    
    # Save as PGM (ASCII format for portability)
    with open(filepath, 'w') as f:
        f.write('P5\n')  # PGM binary format
        f.write(f'{pgm_data.shape[1]} {pgm_data.shape[0]}\n')  # width height
        f.write('255\n')  # max value
    
    # Append binary data
    with open(filepath, 'ab') as f:
        f.write(pgm_data.tobytes())


def load_ogm_from_pgm(filepath, grid_shape):
    """
    Load occupancy grid from PGM file.
    
    Parameters:
    -----------
    filepath : str
        Path to PGM file
    grid_shape : tuple
        Expected grid shape (rows, cols)
    
    Returns:
    --------
    np.ndarray
        Log-odds occupancy grid
    """
    with open(filepath, 'rb') as f:
        # Read header
        magic = f.readline()
        dimensions = f.readline()
        max_val = f.readline()
        
        # Read binary data
        pgm_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(grid_shape)
    
    # Convert from grayscale back to occupancy probability
    occupancy_prob = 1 - (pgm_data / 255.0)
    
    # Convert occupancy probability back to log-odds
    # p(x=1) = occupancy_prob, log-odds = log(p / (1-p))
    log_odds = np.log(np.clip(occupancy_prob, 0.01, 0.99) / np.clip(1 - occupancy_prob, 0.01, 0.99))
    
    return log_odds
