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

# ------Define the blade geometry---------------------------------------------------------------------------
n_blades = 3
root_location_over_R = 0.2
tip_location_over_R = 1
delta_r_over_R = 0.01
r_over_R = np.arange(root_location_over_R, tip_location_over_R, delta_r_over_R)
airfoil_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "DU95W180.cvs")

# blade shape
pitch = -2  # degrees
chord_distribution = 3 * (1 - r_over_R) + 1  # meters
twist_distribution = 14 * (1 - r_over_R) + pitch  # degrees

# ----Define flow conditions--------------------------------------------------------------------------------
u_inf = 10  # unperturbed wind speed in m/s
tip_speed_ratios = np.array([6, 8, 10], dtype = float)  # tip speed ratio
rotor_radius = 50
Omega = u_inf * tip_speed_ratios / rotor_radius

# ----Plot induction factor to show Glauert correction------------------------------------------------------
fig_glauert = plot_glauert_correction()

#----Applying Prandtl tip-speed correction------------------------------------------------------------------
fig_prandtl = plot_prandtl_correction(r_over_R, root_location_over_R, tip_location_over_R, tip_speed_ratios, n_blades)

#----Import polar data--------------------------------------------------------------------------------------
polar_data = pd.read_csv(airfoil_data_path, header=0, names=["alfa", "cl", "cd", "cm"], sep='\s+')
polar_alpha = polar_data['alfa'][:]
polar_cl = polar_data['cl'][:]
polar_cd = polar_data['cd'][:]

# -----Plot polars of the airfoil C-alfa and Cl-Cd----------------------------------------------------------
fig_polar = plot_polar_data(polar_alpha, polar_cl, polar_cd)

# -----Solve BEM model--------------------------------------------------------------------------------------
results = np.zeros([len(r_over_R) - 1, 6])
for i in range(len(r_over_R) - 1):
    chord = np.interp((r_over_R[i] + r_over_R[i + 1]) / 2, r_over_R, chord_distribution)
    twist = np.interp((r_over_R[i] + r_over_R[i + 1]) / 2, r_over_R, twist_distribution)
    results[i, :] = solve_stream_tube(u_inf, r_over_R[i], r_over_R[i + 1], root_location_over_R, tip_location_over_R,
                                      Omega, rotor_radius, n_blades, chord, twist, polar_alpha, polar_cl, polar_cd)

areas = (r_over_R[1:] ** 2 - r_over_R[:-1] ** 2) * np.pi * rotor_radius ** 2
dr = (r_over_R[1:] - r_over_R[:-1]) * rotor_radius
CT = np.sum(dr * results[:, 3] * n_blades / (0.5 * u_inf ** 2 * np.pi * rotor_radius ** 2))
CP = np.sum(dr * results[:, 4] * results[:, 2] * n_blades * rotor_radius * Omega / (0.5 * u_inf ** 3 * np.pi * rotor_radius ** 2))

fig3 = plt.figure(figsize=(12, 6))
plt.title('Axial and tangential induction')
plt.plot(results[:, 2], results[:, 0], 'r-', label=r'$a$')
plt.plot(results[:, 2], results[:, 1], 'g--', label=r'$a^,$')
plt.grid()
plt.xlabel('r/R')
plt.legend()
plt.show()

fig4 = plt.figure(figsize=(12, 6))
plt.title(r'Normal and tangential force, non-dimensioned by $\frac{1}{2} \rho U_\infty^2 R$')
plt.plot(results[:, 2], results[:, 3] / (0.5 * u_inf ** 2 * rotor_radius), 'r-', label=r'normal force')
plt.plot(results[:, 2], results[:, 4] / (0.5 * u_inf ** 2 * rotor_radius), 'g--', label=r'tangential force')
plt.grid()
plt.xlabel('r/R')
plt.legend()
plt.show()

fig5 = plt.figure(figsize=(12, 6))
plt.title(r'Circulation distribution, non-dimensioned by $\frac{\pi U_\infty^2}{\Omega * NBlades } $')
plt.plot(results[:, 2], results[:, 5] / (np.pi * u_inf ** 2 / (n_blades * Omega)), 'r-', label=r'$\Gamma$')
plt.grid()
plt.xlabel('r/R')
plt.legend()
plt.show()
