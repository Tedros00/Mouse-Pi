"""
Generic path finding module for maze solving using OGM images.
Provides reusable functions that work with any OGM jpg image.
"""

from PIL import Image
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import os
from scipy import ndimage
import matplotlib.pyplot as plt


def find_path(
    ogm_image_path: str = None,
    ogm_image: np.ndarray = None,
    start_x: int = None,
    start_y: int = None,
    end_x: int = None,
    end_y: int = None,
    diagonal: bool = True,
    R: int = 0,
    downsample_factor: int = 1
) -> Optional[List[Tuple[int, int]]]:
    """
    Find the shortest path in a maze represented by an OGM image using BFS.
    
    The OGM image is expected to be binary where:
    - White/bright pixels (>127): free space (navigable)
    - Black/dark pixels (<=127): obstacles (walls)
    
    Args:
        ogm_image_path: Path to the OGM image file (jpg, png, etc.) OR None if using ogm_image
        ogm_image: Pre-loaded numpy array of OGM image OR None if using ogm_image_path
        start_x: X coordinate of start position
        start_y: Y coordinate of start position
        end_x: X coordinate of end position
        end_y: Y coordinate of end position
        diagonal: If True, allow diagonal movement; otherwise only cardinal directions
        R: Obstacle inflation radius in pixels for obstacle avoidance (default: 0)
        downsample_factor: Factor to downsample grid (e.g., 4 = 1/4 resolution)
    
    Returns:
        List of (x, y) tuples representing the path from start to end,
        or None if no path exists
    
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If image cannot be loaded or positions are invalid
    """
    # Load or use provided image
    if ogm_image is not None:
        # Use provided numpy array
        image = ogm_image
        print(f"[PathFinder] Using provided OGM image: {image.shape[1]}x{image.shape[0]} pixels")
    elif ogm_image_path is not None:
        # Load from file
        if not os.path.exists(ogm_image_path):
            raise FileNotFoundError(f"OGM image not found: {ogm_image_path}")
        
        try:
            image = np.array(Image.open(ogm_image_path).convert('L'))
        except Exception:
            raise ValueError(f"Failed to load image: {ogm_image_path}")
        
        print(f"[PathFinder] Loaded OGM image: {ogm_image_path}")
    else:
        raise ValueError("Must provide either ogm_image_path or ogm_image")
    
    height_orig, width_orig = image.shape
    
    # Create binary occupancy map
    # Free space: value > 127, Obstacles: value <= 127
    occupancy_map = (image > 127).astype(np.uint8)
    
    # Downsample grid for faster pathfinding
    if downsample_factor > 1:
        occupancy_map = occupancy_map[::downsample_factor, ::downsample_factor]
        # Scale start and end coordinates to downsampled grid
        start_x = start_x // downsample_factor
        start_y = start_y // downsample_factor
        end_x = end_x // downsample_factor
        end_y = end_y // downsample_factor
        print(f"[PathFinder] Downsampled grid by factor {downsample_factor}: {width_orig}x{height_orig} -> {occupancy_map.shape[1]}x{occupancy_map.shape[0]}")
    
    # Inflate obstacles by R pixels for obstacle avoidance
    if R > 0:
        obstacles = 1 - occupancy_map  # Invert: obstacles = 1, free space = 0
        inflated_obstacles = ndimage.binary_dilation(obstacles, iterations=R).astype(np.uint8)
        occupancy_map = 1 - inflated_obstacles  # Invert back: free space = 1, obstacles = 0
        print(f"[PathFinder] Inflated obstacles by R={R} pixels")
    
    # Get dimensions of occupancy map (after downsampling)
    height, width = occupancy_map.shape
    print(f"[PathFinder] Map dimensions: {width} x {height}")
    
    # Helper function to check if a cell is free
    def is_free(x: int, y: int) -> bool:
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        return occupancy_map[y, x] == 1
    
    # Helper function to get neighbors
    def get_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if diagonal:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_free(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    # Validate start and end positions
    if not is_free(start_x, start_y):
        raise ValueError(f"Start position ({start_x}, {start_y}) is blocked")
    
    if not is_free(end_x, end_y):
        raise ValueError(f"End position ({end_x}, {end_y}) is blocked")
    
    # BFS to find shortest path
    queue = deque([(start_x, start_y)])
    visited = {(start_x, start_y)}
    parent = {(start_x, start_y): None}
    
    print(f"[BFS] Starting path search from ({start_x}, {start_y}) to ({end_x}, {end_y})...")
    
    while queue:
        current_x, current_y = queue.popleft()
        
        # Goal reached
        if (current_x, current_y) == (end_x, end_y):
            print("[BFS] Path found!")
            # Reconstruct path
            path = []
            current = (end_x, end_y)
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            
            # Scale path back to original resolution if downsampled
            if downsample_factor > 1:
                path = [(x * downsample_factor, y * downsample_factor) for x, y in path]
            
            return path
        
        # Explore neighbors
        for nx, ny in get_neighbors(current_x, current_y):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                parent[(nx, ny)] = (current_x, current_y)
                queue.append((nx, ny))
    
    print("[BFS] No path found!")
    return None


def visualize_path(ogm_image_path: str, path: List[Tuple[int, int]]):
    """
    Visualize the found path on the OGM image.
    
    Args:
        ogm_image_path: Path to the OGM image file
        path: List of (x, y) tuples representing the path
    """
    image = np.array(Image.open(ogm_image_path).convert('L'))
    occupancy_map = (image > 127).astype(np.uint8)
    
    start_x, start_y = path[0]
    end_x, end_y = path[-1]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display the occupancy map
    ax.imshow(occupancy_map, cmap='gray', origin='upper')
    
    # Draw the path with thick lines
    if len(path) > 1:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=4, label='Path')
    
    # Draw start point
    ax.plot(start_x, start_y, 'go', markersize=12, label='Start', markeredgewidth=2, markeredgecolor='darkgreen')
    
    # Draw end point
    ax.plot(end_x, end_y, 'r*', markersize=20, label='End', markeredgewidth=2, markeredgecolor='darkred')
    
    ax.legend(fontsize=12)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Maze Path Finding', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
