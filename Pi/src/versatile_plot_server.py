import socket
import time
import json
from collections import deque
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Configuration
HOST = '0.0.0.0'
PORT = 5005
MAX_POINTS = 200

# Data storage - dictionary to hold data for each variable
data_storage = {}
plot_lines = {}
plot_axes = {}
fig = None
thread = None

def create_dynamic_plots(data_dict):
    """Dynamically create subplots based on data structure"""
    global fig, plot_lines, plot_axes
    
    # Count how many plots we need
    num_plots = 0
    plot_config = {}  # Maps plot_idx to list of keys
    
    for key, value in data_dict.items():
        if isinstance(value, (list, tuple)) and len(value) > 1:
            # Multi-element tuple/list gets one plot
            plot_config[num_plots] = [key]
            num_plots += 1
        elif isinstance(value, (int, float)):
            # Scalar value - we'll group similar ones
            pass
    
    # Count scalar plots (each key gets its own plot for simplicity)
    scalar_keys = [k for k, v in data_dict.items() if isinstance(v, (int, float))]
    num_plots += len(scalar_keys)
    
    # Create figure with subplots
    if num_plots == 0:
        num_plots = 1
    
    cols = min(2, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4*rows))
    
    # Flatten axes for easier iteration
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    fig.suptitle('Real-Time Data Plotter', fontsize=14)
    
    plot_lines = {}
    plot_axes = {}
    plot_idx = 0
    
    # Create lines for tuple/list parameters
    for key, value in data_dict.items():
        if isinstance(value, (list, tuple)) and len(value) > 1:
            ax = axes[plot_idx]
            ax.set_title(f'{key}')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Sample')
            
            plot_lines[key] = []
            plot_axes[key] = ax
            
            # Create a line for each element in the tuple
            for i in range(len(value)):
                line, = ax.plot([], [], linewidth=2, label=f'{key}[{i}]')
                plot_lines[key].append(line)
            ax.legend(loc='upper left')
            plot_idx += 1
    
    # Create lines for scalar parameters
    for key in scalar_keys:
        ax = axes[plot_idx]
        ax.set_title(f'{key}')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Sample')
        
        line, = ax.plot([], [], 'b-', linewidth=2, label=key)
        plot_lines[key] = [line]
        plot_axes[key] = ax
        ax.legend(loc='upper left')
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, plot_lines, plot_axes

def receive_data():
    """Run in background thread to receive data from Pi"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"Server listening on port {PORT}...")
    print("Waiting for connection from Pi...")
    
    conn, addr = server_socket.accept()
    print(f"Connected to: {addr}")
    
    buffer = ""
    plots_created = False
    
    try:
        while True:
            try:
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    print("Connection closed by client")
                    break
                
                buffer += data
                # Process complete JSON lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        msg = json.loads(line.strip())
                        
                        # Create plots on first message
                        if not plots_created:
                            print(f"Creating plots for: {list(msg.keys())}")
                            create_dynamic_plots(msg)
                            plots_created = True
                        
                        # Store data
                        for key, value in msg.items():
                            if key not in data_storage:
                                if isinstance(value, (list, tuple)):
                                    data_storage[key] = deque(maxlen=MAX_POINTS)
                                else:
                                    data_storage[key] = deque(maxlen=MAX_POINTS)
                            
                            data_storage[key].append(value)
                            
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        print(f"Error parsing data: {e}")
            except Exception as e:
                print(f"Error receiving data: {e}")
                break
    except Exception as e:
        print(f"Thread error: {e}")
    finally:
        try:
            conn.close()
            server_socket.close()
        except:
            pass
        print("Connection closed")

def update_plot():
    """Update all plots with new data"""
    if not plot_lines:
        return
    
    for key, lines in plot_lines.items():
        if key in data_storage:
            ax = plot_axes[key]
            data = list(data_storage[key])
            
            if isinstance(data[0] if data else None, (list, tuple)):
                # Multi-element data
                for i, line in enumerate(lines):
                    values = [d[i] if i < len(d) else 0 for d in data]
                    line.set_data(range(len(values)), values)
            else:
                # Scalar data
                lines[0].set_data(range(len(data)), data)
            
            # Auto-scale axes
            if data:
                ax.set_xlim(0, max(len(data), 10))
                all_values = []
                if isinstance(data[0], (list, tuple)):
                    all_values = [v for d in data for v in d if isinstance(v, (int, float))]
                else:
                    all_values = [v for v in data if isinstance(v, (int, float))]
                
                if all_values:
                    min_val = min(all_values)
                    max_val = max(all_values)
                    margin = (max_val - min_val) * 0.1 if max_val != min_val else 1
                    ax.set_ylim(min_val - margin, max_val + margin)

def animate(frame):
    try:
        update_plot()
    except Exception as e:
        print(f"Error updating plot: {e}")
    return []

# Start receiving thread
thread = threading.Thread(target=receive_data, daemon=True)
thread.start()

# Wait a bit for first connection and data
time.sleep(2)

if not data_storage:
    print("Waiting for data from Pi...")
    while not data_storage:
        time.sleep(0.5)

# Create animation
ani = FuncAnimation(fig, animate, interval=100, blit=False, cache_frame_data=False)

try:
    plt.show()
except Exception as e:
    print(f"Plot error: {e}")
