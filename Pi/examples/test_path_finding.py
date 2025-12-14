#!/usr/bin/env python3
"""
Example: Path Finding in Maze using OGM Image
Demonstrates how to use the generic path finding function with any OGM jpg.

Usage:
    python test_path_finding.py <ogm_image_path> <start_x> <start_y> <end_x> <end_y>

Example:
    python test_path_finding.py maze.jpg 10 10 100 100
"""

import sys
import os
import numpy as np
from PIL import Image

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from path_finding import find_path, visualize_path


# Configuration variables
ogm_path = r"C:\Users\User\Projects\Mouse-Pi\Pi\examples\TestMaze.JPG"

# Downsample factor for coarser path planning (larger = fewer waypoints)

start_x = 150 
start_y = 1200 
end_x = 1000 
end_y = 350 


def downsample_image(image_path: str, factor: int) -> str:
    """
    Downsample OGM image for coarser path planning.
    Creates a temporary downsampled version.
    
    Args:
        image_path: Path to original OGM image
        factor: Downsample factor (e.g., 4 = 1/4 resolution)
    
    Returns:
        Path to downsampled image
    """
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    
    # Downsample
    new_size = (img.width // factor, img.height // factor)
    img_downsampled = img.resize(new_size, Image.NEAREST)
    
    # Save temporarily
    temp_path = "TestMaze_downsampled.jpg"
    img_downsampled.save(temp_path)
    print(f"[PathFinding] Created downsampled image: {new_size[0]}x{new_size[1]} (factor: {factor}x)")
    
    return temp_path


def main():
    """Main entry point for the example."""
    
    try:
        # Downsample OGM for coarser path planning
        #downsampled_ogm = downsample_image(ogm_path, downsample_factor)
        
        # Find path using downsampled image
        path = find_path(ogm_image_path=ogm_path, start_x=start_x, start_y=start_y, 
                        end_x=end_x, end_y=end_y, diagonal=True, R=40)
        
        if path:
            # Scale path back to original resolution
            path_original_res = [(x , y ) for x, y in path]
            
            print(f"\n[Result] Path found with {len(path)} waypoints (downsampled)")
            print(f"         = {len(path_original_res)} waypoints at original resolution")
            print("\nX Y (one per line - original resolution):")
            
            
            # Save path to file (original resolution)
            output_file = "path_output.txt"
            with open(output_file, 'w') as f:
                for x, y in path_original_res:
                    f.write(f"{x} {y}\n")
            print(f"\n[Output] Path saved to {output_file}")
            
            # Visualize the path on original image
            visualize_path(ogm_path, path_original_res)
        else:
            print("\n[Error] No path found between the two points!")
            sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
