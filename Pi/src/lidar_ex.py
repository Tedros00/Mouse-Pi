#!/usr/bin/env python3
'''Save LIDAR animation to file - optimized for lightweight data'''
from rplidar import RPLidar
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

PORT_NAME = '/dev/ttyUSB1'
DMAX = 4000
IMIN = 0
IMAX = 50

# Create output directory
os.makedirs('lidar_animations', exist_ok=True)

frame_list = []
frame_skip = 2  # Skip every N frames to reduce data
frame_counter = 0

def update_line(num, iterator, line):
    global frame_counter
    try:
        frame_counter += 1
        
        # Skip frames to reduce data volume
        for _ in range(frame_skip):
            scan = next(iterator)
        
        # Downsample points - only keep every other point
        scan = list(scan)[::2]
        
        if scan:
            offsets = np.array([(np.radians(meas[1]), meas[2]) for meas in scan if meas[2] > 0])
            if len(offsets) > 0:
                line.set_offsets(offsets)
                intens = np.array([meas[0] for meas in scan if meas[2] > 0])
                line.set_array(intens)
                frame_list.append(num)
        
        if len(frame_list) % 5 == 0:
            print(f"Captured {len(frame_list)} frames (processed {frame_counter} total)...")
    except StopIteration:
        print(f"End of scan data")
    except Exception as e:
        print(f"Error in frame {num}: {e}")
    return line,

def run():
    print("Connecting to LIDAR...")
    lidar = RPLidar(PORT_NAME)
    
    try:
        lidar.connect()
        print("LIDAR connected")
        print("(Downsampling: skipping frames and reducing points)\n")
        
        fig = plt.figure(figsize=(8, 8), dpi=80)  # Lower DPI = smaller file
        ax = plt.subplot(111, projection='polar')
        line = ax.scatter([0, 0], [0, 0], s=3, c=[IMIN, IMAX],  # Smaller markers
                               cmap=plt.cm.Greys_r, lw=0)
        ax.set_rmax(DMAX)
        ax.grid(True, alpha=0.3)  # Lower alpha = less data
        
        print("Capturing animation frames...")
        iterator = lidar.iter_scans()
        ani = animation.FuncAnimation(fig, update_line,
            fargs=(iterator, line), interval=100, frames=200, repeat=False)  # Fewer frames
        
        # Save with compression - MP4 codec
        print("Saving compressed animation...")
        try:
            # High compression for MP4
            ani.save('lidar_animations/lidar_animation.mp4', 
                    fps=10, dpi=80, codec='libx264', bitrate=2000)
            print("✓ Animation saved successfully!")
            
            # Show file size
            size_mb = os.path.getsize('lidar_animations/lidar_animation.mp4') / (1024*1024)
            print(f"  File size: {size_mb:.2f} MB")
        except Exception as e:
            print(f"MP4 save failed: {e}")
            print("Trying WebM (better compression)...")
            try:
                ani.save('lidar_animations/lidar_animation.webm', fps=10, dpi=80)
                size_mb = os.path.getsize('lidar_animations/lidar_animation.webm') / (1024*1024)
                print(f"✓ WebM saved! File size: {size_mb:.2f} MB")
            except Exception as e2:
                print(f"WebM save failed: {e2}")
        
        plt.close(fig)
        print(f"\nCapture complete: {len(frame_list)} frames captured")
        
    except KeyboardInterrupt:
        print(f"\n\nStopped by user. Saved {len(frame_list)} frames.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Stopping LIDAR...")
        lidar.stop()
        lidar.disconnect()
        print("Done!")

if __name__ == '__main__':
    run()
