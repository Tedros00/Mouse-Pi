#necessary imports
import sys
import numpy as np

#source path
sys.path.insert(0, '/home/mouse/AMR/src')

#import necessary modules
from kinematics import pot_readings_to_velocities, IK, FK, velocities_to_pwm
from nano_interface import get_encoder_counts, send_vel_cmd, initialize_serial, init_encoders
from ProbabilisticMotionModel import sample_motion_velocity_model, command_correction

from lidar import connect_lidar, init_lidar, capture_map, disconnect_lidar

#Global vars
WHEEL_DIAMETER = 0.037 #meters
ROBOT_WIDTH = 0.081   #meters
