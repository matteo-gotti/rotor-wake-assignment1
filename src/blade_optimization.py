# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.interpolate import griddata
from BEM_functions import *

airfoil_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "DU95W180.cvs")
polar_data = pd.read_csv(airfoil_data_path, header=0, names=["alfa", "cl", "cd", "cm"], sep='\s+')
polar_alpha = polar_data['alfa'][:]
polar_cl = polar_data['cl'][:]
polar_cd = polar_data['cd'][:]


def blade_optimization(tip_location_over_R, root_location_over_R, r_over_R, rotor_radius, n_blades, Pitch, u_inf, tip_speed_ratio,
                       CT_ref, polar_alpha, polar_cl, polar_cd, plot_contour=True):

    Omega = u_inf * tip_speed_ratio / rotor_radius  # rotational speed
    yaw_angle = 0.0                            # yaw angle [deg]
    tol = 1e-3  # [0.75-tol : 0.75+tol]

    # coefficients used to get the sampling points for the chord distribution and twist distribution in the following for-cycles
    a = np.linspace(3, 8, 12)  # 3, 8, 12 for CT_ref = 0.75
    c = np.linspace(6, 15, 20)  # 6, 15, 20 for CT_ref = 0.75

    N = len(a) * len(c)  # total number of combinations being tested
    CT_values = np.zeros(N)  # initialization vector for CT
    CP_values = np.zeros(N)  # initialization vector for CP
    index = np.empty((N, 2))
    counter = 0
    x_data = np.zeros(N)
    y_data = np.zeros(N)

    # iterates over the possible configurations (varying a and c) representing the sampling points
    for i1 in range(len(a)):
        for i2 in range(len(c)):
            b = -1.25 * a[i1] + 3.75    # condition (given a) to keep the selected value for the r_root = 3.4
            chord_distribution = a[i1] * (1 - r_over_R) + b * (1 - r_over_R) ** 2 + \
                1  # constant term = 1 to set r_tip = 1
            twist_distribution = c[i2] * (1 - r_over_R) + Pitch

            results = BEM_cycle(u_inf, r_over_R, root_location_over_R, tip_location_over_R, Omega, rotor_radius, n_blades,
                                chord_distribution, twist_distribution, yaw_angle, tip_speed_ratio, polar_alpha, polar_cl, polar_cd)

            dr = np.array([(r_over_R[1:] - r_over_R[:-1]) * rotor_radius]).T

            CT = np.sum(dr * results['normal_force'] * (0.5 * u_inf**2 *
                                                        rotor_radius) * n_blades) / (0.5 * u_inf ** 2 * np.pi * rotor_radius ** 2)
            CP = np.sum(dr * results['tangential_force'] * (0.5 * u_inf**2 * rotor_radius) * results['r_over_R']
                        * n_blades * rotor_radius * Omega / (0.5 * u_inf ** 3 * np.pi * rotor_radius ** 2))

            # data storing
            CT_values[counter] = CT
            CP_values[counter] = CP
            x_data[counter] = a[i1]
            y_data[counter] = c[i2]
            index[counter] = np.array([i1, i2])
            counter += 1

    # ind = np.where(np.abs(CT_values - CT_ref) < tol)  # indices where CT is approx 0.75

    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()

    xi = np.linspace(x_min, x_max, 100)
    yi = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x_data, y_data), CT_values, (X, Y), method='cubic')
    inds_CT_is_CT_ref = np.where(np.abs(Z - CT_ref) < tol)
    Z2 = griddata((x_data, y_data), CP_values, (X, Y), method='cubic')

    CP_values_CT075 = Z2[inds_CT_is_CT_ref]
    print(CP_values_CT075)
    CP_max_CT075 = np.max(CP_values_CT075)
    print("Maximum CP value:", CP_max_CT075)
    ind_max_CP = np.argmax(CP_values_CT075)

    X_values = X[inds_CT_is_CT_ref]
    Y_values = Y[inds_CT_is_CT_ref]
    a_optim = X_values[ind_max_CP]
    c_optim = Y_values[ind_max_CP]
    print("a optimum:", a_optim)
    print("c optimum:", c_optim)

    optim_chord_distribution = a_optim * (1 - r_over_R) + (-1.25 * a_optim + 3.75) * (1 - r_over_R) ** 2 + 1
    optim_twist_distribution = c_optim * (1 - r_over_R) + Pitch

    if plot_contour:
        plt.figure(figsize=(8, 6))
        contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
        cbar = plt.colorbar(contour, orientation='horizontal')
        cbar.set_label(label=r'$C_T$ [-]', fontsize=12)
        plt.contour(X, Y, Z, levels=[CT_ref], colors='red', linewidths=2)
        plt.xlabel(r'$a$')
        plt.ylabel(r'$c$')
        # plt.title('Interpolated CT contour')

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'$a$')
        ax.set_ylabel(r'$c$')
        ax.set_zlabel(r'$CT$')
        # ax.set_title('3D Surface Plot of  interpolated CT')
        ax.scatter(X[inds_CT_is_CT_ref], Y[inds_CT_is_CT_ref], Z[inds_CT_is_CT_ref],
                   color='red', s=50)  # highlight points with CT approx 0.75

        plt.figure(figsize=(8, 6))
        contour_rbf = plt.contour(X, Y, Z2, levels=50, cmap='viridis')
        cbar = plt.colorbar(contour_rbf, orientation='horizontal')
        cbar.set_label(label=r'$C_P$ [-]', fontsize=12)
        plt.scatter(X[inds_CT_is_CT_ref], Y[inds_CT_is_CT_ref], color='red', s=10)  # points with CT approx 0.75
        plt.scatter(a_optim, c_optim, color='blue', s=50)    # optimum point found
        plt.scatter(x_data, y_data, color='black', s=2)  # sampling points to construct interpolation
        plt.xlabel(r'$a$')
        plt.ylabel(r'$c$')
        # plt.title(r'$CP$')
        plt.show()

    return CP_max_CT075, optim_chord_distribution, optim_twist_distribution
