from BEM_functions import *
from BEM_plots import plot_mesh_convergence
import numpy as np
import pandas as pd
import os

# ----Discretization method-----------------------------------------------------------------------------------
number_of_annuli = np.arange(20, 420, 20)  # number of annuli [-]

# ------Define the blade geometry-----------------------------------------------------------------------------
n_blades = 3  # number of blades [-]
root_location_over_R = 0.2  # location of the blade root divided by blade radius [-]
tip_location_over_R = 1  # location of the blade tip divided by blade radius [-]
pitch = -2  # pitch angle [deg]
def chord_dist(r_over_R): return 3 * (1 - r_over_R) + 1  # chord at each annulus [m]
def twist_dist(r_over_R, pitch): return 14 * (1 - r_over_R) + pitch  # twist at each annulus [deg]


# ----Define flow conditions----------------------------------------------------------------------------------
u_inf = 10    # unperturbed wind speed [m/s]
tip_speed_ratio = 8.0  # tip speed ratio [-]
rotor_radius = 50  # rotor radius [m]
Omega = u_inf * tip_speed_ratio / rotor_radius  # rotor rotation speed [rad/s]

# ----Import polar data---------------------------------------------------------------------------------------
airfoil_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "DU95W180.cvs")
polar_data = pd.read_csv(airfoil_data_path, header=0, names=[
                         "alfa", "cl", "cd", "cm"], sep='\s+')
polar_alpha = polar_data['alfa'][:]
polar_cl = polar_data['cl'][:]
polar_cd = polar_data['cd'][:]

# ----Compute results for each number of annuli---------------------------------------------------------------
CT_uniform = np.zeros(len(number_of_annuli))
CT_cosine = np.zeros(len(number_of_annuli))
for j in range(len(number_of_annuli)):

    n_annuli = number_of_annuli[j]
    N_vec = np.arange(0, n_annuli + 1, 1)
    r_over_R_cosine = (tip_location_over_R + root_location_over_R)/2 + (tip_location_over_R -
                                                                        root_location_over_R)/2 * np.cos((n_annuli - N_vec) * np.pi/n_annuli)
    r_over_R_uniform = np.linspace(root_location_over_R, tip_location_over_R, n_annuli + 1)

    dr_uniform = np.array([(r_over_R_uniform[1:] - r_over_R_uniform[:-1]) * rotor_radius]).T
    chord_distribution_uniform = chord_dist(r_over_R_uniform)
    twist_distribution_uniform = twist_dist(r_over_R_uniform, pitch)
    current_result_uniform = BEM_cycle(u_inf, r_over_R_uniform, root_location_over_R, tip_location_over_R, Omega, rotor_radius, n_blades,
                                       chord_distribution_uniform, twist_distribution_uniform, 0.0, tip_speed_ratio, polar_alpha, polar_cl, polar_cd)

    dr_cosine = np.array([(r_over_R_cosine[1:] - r_over_R_cosine[:-1]) * rotor_radius]).T
    chord_distribution_cosine = chord_dist(r_over_R_cosine)
    twist_distribution_cosine = twist_dist(r_over_R_cosine, pitch)
    current_result_cosine = BEM_cycle(u_inf, r_over_R_cosine, root_location_over_R, tip_location_over_R, Omega, rotor_radius, n_blades,
                                      chord_distribution_cosine, twist_distribution_cosine, 0.0, tip_speed_ratio, polar_alpha, polar_cl, polar_cd)

    CT_uniform[j] = np.sum(dr_uniform * current_result_uniform['normal_force'] *
                           (0.5 * u_inf**2 * rotor_radius) * n_blades) / (0.5 * u_inf ** 2 * np.pi * rotor_radius ** 2)
    CT_cosine[j] = np.sum(dr_cosine * current_result_cosine['normal_force'] *
                          (0.5 * u_inf**2 * rotor_radius) * n_blades) / (0.5 * u_inf ** 2 * np.pi * rotor_radius ** 2)

rel_error_cosine = np.abs(CT_cosine[0:-1] - CT_cosine[-1]) / CT_cosine[-1]
rel_error_uniform = np.abs(CT_uniform[0:-1] - CT_uniform[-1]) / CT_uniform[-1]

# ----Plot results-------------------------------------------------------------------------------------------
plot_mesh_convergence(number_of_annuli, CT_uniform, CT_cosine, rel_error_uniform, rel_error_cosine)
