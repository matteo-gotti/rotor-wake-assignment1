from BEM_functions import *
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------Define the blade geometry---------------------------------------------------------------------------
n_blades = 3
rotor_radius = 50
root_location_over_R = 0.2
tip_location_over_R = 1
delta_r_over_R = 0.01
r_over_R = np.arange(root_location_over_R, tip_location_over_R, delta_r_over_R)
n_annuli = len(r_over_R) - 1
airfoil_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "DU95W180.cvs")
pitch = -2  # degrees
chord_distribution = 3 * (1 - r_over_R) + 1  # meters
twist_distribution = 14 * (1 - r_over_R) + pitch  # degrees
sigma = n_blades * chord_distribution / (2 * np.pi * r_over_R * rotor_radius)

# ----Define flow conditions--------------------------------------------------------------------------------
u_inf = 10  # unperturbed wind speed in m/s
tip_speed_ratio = 8  # tip speed ratio
yaw_angle = np.radians(0)
Omega = u_inf * tip_speed_ratio / rotor_radius
