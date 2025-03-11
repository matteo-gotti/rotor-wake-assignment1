# Import necessary libraries and functions
from BEM_functions import *
from BEM_plots import *
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
TODO:
1. ADD POSSIBILITY OF NON-ZERO YAW
2. ADD POSSIBILITY OF COMPUTING DIFFERENT TIP SPEED RATIOS
3. For a given CT=0.75, change the pitch or the chord distribution or the twist distribution 
    in order to maximise the Cp in axial flow at tip speed ratio 8 (eight). You can choose your 
    own design approach. Compare with the expected result from actuator disk theory. 
    Discuss the rationale for your design, including the twist and chord distributions.
4. Plots with explanation of results (alpha/inflow/a/a'/Ct/Cn/Cq vs r/R)
        - Span-wise distribution of angle of attack and inflow angle
        - Span-wise distribution of axial and azimuthal inductions
        - Span-wise distribution of thrust and azimuthal loading
        - Total thrust and torque versus tip-speed ratio/advance ratio
        - For the cases of yawed rotor (optional), also plot the azimuthal variation (suggestion: polar contour plot)
5. Plots with explanation of the influence of the tip correction
6. (optional): Plots with explanation of influence of number of annuli, spacing method (constant, cosine) 
    and convergence history for total thrust.
7. (optional): Explanation of the design approach used for maximizing the Cp or efficiency
8. (optional): Plots with explanation of the new designs
9. Plot the distribution of stagnation pressure as a function of radius at four locations: infinity upwind, 
    at the rotor (upwind side), at the rotor (downwind side), infinity downwind.
10. (optional): Plot a representation of the system of circulation. Discuss the generation and release 
    of vorticity in relation to the loading and circulation over the blade.
"""

# ------Define the blade geometry-----------------------------------------------------------------------------
n_blades = 3  # number of blades [-]
# location of the blade root divided by blade radius [-]
root_location_over_R = 0.2
# location of the blade tip divided by blade radius [-]
tip_location_over_R = 1
delta_r_over_R = 0.01  # length of each annulus divided by blade radius [-]
r_over_R = np.arange(root_location_over_R, tip_location_over_R + delta_r_over_R/2,
                     delta_r_over_R)  # location of each annulus divided by blade radius [-]
airfoil_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "DU95W180.cvs")
delta_psi = 0.01  # azimuthal discretization step [rad]
psi_vec = np.arange(0, 2 * np.pi, delta_psi)  # azimuthal discretization [rad]

# blade shape
pitch = -2  # pitch angle [deg]
chord_distribution = 3 * (1 - r_over_R) + 1  # chord at each annulus [m]
twist_distribution = 14 * (1 - r_over_R) + pitch  # twist at each annulus [deg]

# ----Define flow conditions----------------------------------------------------------------------------------
u_inf = 10    # unperturbed wind speed [m/s]
tip_speed_ratios = np.array([6, 8, 10], dtype=float)  # tip speed ratio [-]
rotor_radius = 50  # rotor radius [m]
yaw_angles = np.array([0, 15, 30], dtype=float)  # yaw angle [deg]
Omega = u_inf * tip_speed_ratios / rotor_radius  # rotor rotation speed [rad/s]

# ----Check the validity of the input data--------------------------------------------------------------------
if not (any(x == 8 for x in tip_speed_ratios)):
    raise Exception(
        "The tip speed ratio 8 is not in the list of tip speed ratios")

if not (any(x == 0 for x in yaw_angles)):
    raise Exception("The yaw angle 0 is not in the list of yaw angles")

# ----Plot induction factor to show Glauert correction--------------------------------------------------------
fig_glauert = plot_glauert_correction()

# ----Applying Prandtl tip-speed correction-------------------------------------------------------------------
fig_prandtl_single_tsr = plot_prandtl_correction(
    r_over_R, root_location_over_R, tip_location_over_R, tip_speed_ratios[1], n_blades)
fig_prandtl = plot_prandtl_correction(
    r_over_R, root_location_over_R, tip_location_over_R, tip_speed_ratios, n_blades)

# ----Import polar data---------------------------------------------------------------------------------------
polar_data = pd.read_csv(airfoil_data_path, header=0, names=[
                         "alfa", "cl", "cd", "cm"], sep='\s+')
polar_alpha = polar_data['alfa'][:]
polar_cl = polar_data['cl'][:]
polar_cd = polar_data['cd'][:]

# -----Plot polars of the airfoil C-alfa and Cl-Cd------------------------------------------------------------
fig_polar = plot_polar_data(polar_alpha, polar_cl, polar_cd)

# -----Solve BEM model for each requested case (corrected)----------------------------------------------------------------
results_corrected = {}
for yaw_angle in yaw_angles:
    psi_vector = psi_vec if yaw_angle != 0 else []

    for j, tip_speed_ratio in enumerate(tip_speed_ratios):
        if yaw_angle != 0 and tip_speed_ratio != 8:
            continue
        key = f'yaw_{yaw_angle}_TSR_{tip_speed_ratio}'
        results_corrected[key] = BEM_cycle(
            u_inf, r_over_R, root_location_over_R, tip_location_over_R, Omega[j], rotor_radius, n_blades,
            chord_distribution, twist_distribution, yaw_angle, tip_speed_ratio, polar_alpha, polar_cl, polar_cd, psi_vector)

# areas = (r_over_R[1:] ** 2 - r_over_R[:-1] ** 2) * np.pi * rotor_radius ** 2
non_yawed_keys = [key for key in results_corrected.keys() if 'yaw_0.0' in key]
yawed_keys = [key for key in results_corrected.keys() if 'yaw_0.0' not in key]

# -----Compute CT, CP, CQ for non yawed corrected case------------------------------------------------------------------
non_yawed_corrected_results = {key: results_corrected[key] for key in non_yawed_keys}
CT_corr = {}
CP_corr = {}
CQ_corr = {}
dr = (r_over_R[1:] - r_over_R[:-1]) * rotor_radius
for j, tip_speed_ratio in enumerate(tip_speed_ratios):
    key = non_yawed_keys[j]
    CT_corr[key] = np.sum(dr * results_corrected[key]['normal_force'] *
                          n_blades / (0.5 * u_inf ** 2 * np.pi * rotor_radius ** 2))
    CP_corr[key] = np.sum(dr * results_corrected[key]['tangential_force'] * results_corrected[key]['r_over_R']
                          * n_blades * rotor_radius * Omega[j] / (0.5 * u_inf ** 3 * np.pi * rotor_radius ** 2))
    CQ_corr[key] = np.sum(dr * results_corrected[key]['tangential_force'] *
                          n_blades / (0.5 * u_inf ** 2 * np.pi * rotor_radius ** 2))

# -----Plot results for non yawed case------------------------------------------------------------------------
n_tsr = len(tip_speed_ratios)
n_yaw = len(yaw_angles)
figs_non_yawed = plots_non_yawed(non_yawed_corrected_results, CT_corr, CP_corr,
                                 CQ_corr, rotor_radius, tip_speed_ratios, u_inf, Omega, n_blades)
