# Import necessary libraries and functions
from BEM_functions import *
from BEM_plots import *
import pandas as pd
import os
from blade_optimization import blade_optimization

# ----Plotting flags------------------------------------------------------------------------------------------
plot_glauert = False    # plot the Glauert correction
plot_prandtl_single_tsr = False    # plot the Prandtl correction for a single tip speed ratio
plot_prandtl = False    # plot the Prandtl correction for all tip speed ratios
plot_polar = False    # plot the airfoil polars
plot_non_yawed_corrected = False    # plot the results for the non yawed case with Prandtl correction
plot_non_yawed_comparison = False   # plot the comparison of results with and without Prandtl correction for the non yawed case
plot_yawed = False    # plot the results for the yawed case with Prandtl correction
plot_p_tot = False    # plot the stagnation pressure distribution
plot_optimization_results = False    # plot the results of the blade optimization

# Flag for blade optimization
perform_blade_optimization = False

# ----Discretization -----------------------------------------------------------------------------------
number_of_annuli = 80  # number of annuli [-]

# ------Define the blade geometry-----------------------------------------------------------------------------
n_blades = 3  # number of blades [-]
root_location_over_R = 0.2  # location of the blade root divided by blade radius [-]
tip_location_over_R = 1  # location of the blade tip divided by blade radius [-]
r_over_R = np.linspace(root_location_over_R, tip_location_over_R, number_of_annuli + 1)

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
if plot_glauert:
    fig_glauert = plot_glauert_correction()

# ----Applying Prandtl tip-speed correction-------------------------------------------------------------------
if plot_prandtl_single_tsr:
    fig_prandtl_single_tsr = plot_prandtl_correction(
        r_over_R, root_location_over_R, tip_location_over_R, tip_speed_ratios[1], n_blades)

if plot_prandtl:
    fig_prandtl = plot_prandtl_correction(
        r_over_R, root_location_over_R, tip_location_over_R, tip_speed_ratios, n_blades)

# ----Import polar data---------------------------------------------------------------------------------------
polar_data = pd.read_csv(airfoil_data_path, header=0, names=[
                         "alfa", "cl", "cd", "cm"], sep='\s+')
polar_alpha = polar_data['alfa'][:]
polar_cl = polar_data['cl'][:]
polar_cd = polar_data['cd'][:]

# -----Plot polars of the airfoil C-alfa and Cl-Cd------------------------------------------------------------
if plot_polar:
    fig_polar = plot_polar_data(polar_alpha, polar_cl, polar_cd)

# -----Solve BEM model for each requested case (corrected)----------------------------------------------------
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

non_yawed_keys = [key for key in results_corrected.keys() if 'yaw_0.0' in key]
yawed_keys = [key for key in results_corrected.keys() if 'yaw_0.0' not in key]

# -----Compute CT, CP, CQ for non yawed corrected case--------------------------------------------------------
non_yawed_corrected_results = {key: results_corrected[key] for key in non_yawed_keys}
CT_corr = {}
CP_corr = {}
CQ_corr = {}
dr = np.array([(r_over_R[1:] - r_over_R[:-1]) * rotor_radius]).T
for j, tip_speed_ratio in enumerate(tip_speed_ratios):
    key = non_yawed_keys[j]
    CT_corr[key] = np.sum(dr * results_corrected[key]['normal_force'] * (0.5 * u_inf**2 *
                          rotor_radius) * n_blades) / (0.5 * u_inf ** 2 * np.pi * rotor_radius ** 2)
    CP_corr[key] = np.sum(dr * results_corrected[key]['tangential_force'] * (0.5 * u_inf**2 * rotor_radius) * results_corrected[key]['r_over_R']
                          * n_blades * rotor_radius * Omega[j] / (0.5 * u_inf ** 3 * np.pi * rotor_radius ** 2))
    CQ_corr[key] = np.sum(dr * results_corrected[key]['r_over_R'] * results_corrected[key]['tangential_force'] * (0.5 * u_inf**2 * rotor_radius) * n_blades
                          / (0.5 * u_inf ** 2 * np.pi * rotor_radius ** 2))

# ----Solve BEM model for non-corrected non-yawed case TSR = 8, yaw_angle = 0---------------------------------
results_uncorrected = BEM_cycle(
    u_inf, r_over_R, root_location_over_R, tip_location_over_R, Omega[1], rotor_radius, n_blades,
    chord_distribution, twist_distribution, 0.0, 8, polar_alpha, polar_cl, polar_cd, prandtl_correction=False)

# -----Compute stagnation pressure for TSR = 8, yaw_angle = 0-------------------------------------------------
p_tot = 101325 + (0.5 * u_inf**2) * np.ones_like(results_corrected['yaw_0.0_TSR_8.0']['r_over_R'])
v_norm = u_inf * (1 - results_corrected['yaw_0.0_TSR_8.0']['a'])
v_tan = Omega[1] * results_corrected['yaw_0.0_TSR_8.0']['r_over_R'] * \
    rotor_radius * (1 + results_corrected['yaw_0.0_TSR_8.0']['a_line'])
dp_tot_behind_rotor_norm = (0.5 * u_inf**2 * rotor_radius) * \
    (results_corrected['yaw_0.0_TSR_8.0']['normal_force'] * n_blades * dr /
     (2*np.pi*results_corrected['yaw_0.0_TSR_8.0']['r_over_R']*rotor_radius * dr))

dp_tot = dp_tot_behind_rotor_norm
p_tot_behind_rotor = (p_tot - dp_tot)/p_tot
p_tot = p_tot/p_tot

# ----If requested perform blade optimization-----------------------------------------------------------------
if perform_blade_optimization:
    CT_ref = 0.75
    CP_max_CT075, optim_chord_distribution, optim_twist_distribution = blade_optimization(
        tip_location_over_R, root_location_over_R, r_over_R, rotor_radius, n_blades, pitch, u_inf, 8.0,
        CT_ref, polar_alpha, polar_cl, polar_cd, plot_contour=True)
    results_optim = BEM_cycle(
        u_inf, r_over_R, root_location_over_R, tip_location_over_R, Omega[1], rotor_radius, n_blades,
        optim_chord_distribution, optim_twist_distribution, 0.0, 8, polar_alpha, polar_cl, polar_cd)

# -----Plot stagnation pressure distribution------------------------------------------------------------------
if plot_p_tot:
    plot_polar_pressure_distribution(results_corrected['yaw_0.0_TSR_8.0']['r_over_R'], p_tot, p_tot_behind_rotor)

# -----Plot results for non yawed case------------------------------------------------------------------------
n_tsr = len(tip_speed_ratios)
n_yaw = len(yaw_angles)
centroids = np.array((r_over_R[1:] + r_over_R[:-1]) / 2).reshape(-1, 1)
if plot_non_yawed_corrected or plot_non_yawed_comparison:
    plots_non_yawed(non_yawed_corrected_results, results_uncorrected, tip_speed_ratios,
                    polar_alpha, polar_cd, polar_cl, chord_distribution, plot_non_yawed_corrected, plot_non_yawed_comparison)

# -----Plot results for yawed case----------------------------------------------------------------------------
if plot_yawed:
    n_psi = len(psi_vec)
    results_yawed = {key: results_corrected[key] for key in yawed_keys}
    results_to_add = results_corrected['yaw_0.0_TSR_8.0']
    results_to_add['a'] = np.tile(results_to_add['a'], (1, n_psi))
    results_to_add['a_line'] = np.tile(results_to_add['a_line'], (1, n_psi))
    results_to_add['r_over_R'] = np.tile(results_to_add['r_over_R'], (1, n_psi))
    results_to_add['normal_force'] = np.tile(results_to_add['normal_force'], (1, n_psi))
    results_to_add['tangential_force'] = np.tile(results_to_add['tangential_force'], (1, n_psi))
    results_to_add['gamma'] = np.tile(results_to_add['gamma'], (1, n_psi))
    results_to_add['alpha'] = np.tile(results_to_add['alpha'], (1, n_psi))
    results_to_add['inflow_angle'] = np.tile(results_to_add['inflow_angle'], (1, n_psi))
    results_to_add['c_thrust'] = np.tile(results_to_add['c_thrust'], (1, n_psi))
    results_to_add['c_torque'] = np.tile(results_to_add['c_torque'], (1, n_psi))
    results_to_add['c_power'] = np.tile(results_to_add['c_power'], (1, n_psi))
    results_yawed['yaw_0.0_TSR_8.0'] = results_to_add

    variables_to_plot = ['alpha', 'inflow_angle', 'a', 'a_line',
                         'normal_force', 'tangential_force', 'gamma']
    labels = [r'$\alpha$ [deg]', r'$\phi$ [deg]', r'$a$ [-]', r"$a'$ [-]",
              r'$C_n$ [-]', r'$C_t$ [-]', r'$\Gamma$ [-]']
    plots_yawed(results_yawed, yaw_angles, centroids, psi_vec, variables_to_plot, labels)

# -----Plot optimization results-------------------------------------------------------------------------------
if plot_optimization_results and perform_blade_optimization:
    plots_optimization(results_optim, optim_chord_distribution, optim_twist_distribution,
                       chord_distribution, twist_distribution, results_corrected['yaw_0.0_TSR_8.0'],
                       results_corrected['yaw_0.0_TSR_8.0']['r_over_R'], r_over_R, polar_alpha, polar_cl, polar_cd)

print('Done')
