import matplotlib.pyplot as plt
from BEM_functions import *
import numpy as np
import matplotlib.cm as cm


def plot_glauert_correction():
    a = np.arange(0.0, 1.0, 0.01)
    c_t_uncorrected = compute_c_t(a)  # CT without correction
    c_t_glauert = compute_c_t(a, True)  # CT with Glauert's correction

    fig = plt.figure(figsize=(12, 6))
    plt.plot(a, c_t_uncorrected, 'k-', label='$C_T$')
    plt.plot(a, c_t_glauert, 'b--', label='$C_T$ Glauert')
    plt.plot(a, c_t_glauert * (1 - a), 'g--', label='$C_P$ Glauert')
    plt.xlabel(r'$a$ [-]')
    plt.ylabel(r'$C_T$ and $C_P$ [-]')
    plt.grid()
    plt.legend()
    plt.show()

    return fig


def plot_prandtl_correction(r_over_R, root_location_over_R, tip_location_over_R, tip_speed_ratio, n_blades):
    a = np.zeros(np.shape(r_over_R)) + 0.3
    n_tsr = 1 if type(tip_speed_ratio) is np.float64 else len(tip_speed_ratio)
    colormap = cm.get_cmap('viridis', n_tsr)
    fig = plt.figure(figsize=(12, 6))

    if n_tsr == 1:
        prandtl, prandtl_tip, prandtl_root = prandtl_tip_root_correction(
            r_over_R, root_location_over_R, tip_location_over_R, tip_speed_ratio, n_blades, a)
        plt.plot(r_over_R, prandtl_tip, 'g.', label='Prandtl tip')
        plt.plot(r_over_R, prandtl_root, 'b.', label='Prandtl root')
        color = colormap(0)
        plt.plot(r_over_R, prandtl, color=color, label=f'Prandtl TSR={tip_speed_ratio}')
    else:
        for i, tsr in enumerate(tip_speed_ratio):
            prandtl, prandtl_tip, prandtl_root = prandtl_tip_root_correction(
                r_over_R, root_location_over_R, tip_location_over_R, tsr, n_blades, a)
            color = colormap(i)
            plt.plot(r_over_R, prandtl, color=color, label=f'Prandtl TSR={tsr}')
    plt.grid()
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.legend()
    plt.show()

    return fig


def plot_polar_data(polar_alpha, polar_cl, polar_cd):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(polar_alpha, polar_cl)
    axs[0].set_xlim([-30, 30])
    axs[0].set_xlabel(r'$\alpha$ [deg]')
    axs[0].set_ylabel(r'$C_l$ [-]')
    axs[0].grid()
    axs[1].plot(polar_cd, polar_cl)
    axs[1].set_xlim([0, .1])
    axs[1].set_xlabel(r'$C_d$ [-]')
    axs[1].grid()
    plt.show()

    return fig


def plots_non_yawed(corrected_results, uncorrected_results, CT, CP, CQ, rotor_radius, tip_speed_ratios, u_inf, Omega, n_blades):
    n_tsr = len(tip_speed_ratios)
    # ----Spanwise distribution of angle of attack------------------------------------------------------------
    fig_alpha = plt.figure()
    colormap = cm.get_cmap('viridis', n_tsr)
    for i, TSR in enumerate(tip_speed_ratios):
        alpha = corrected_results[f'yaw_0.0_TSR_{TSR}']['alpha']
        r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']
        color = colormap(i)
        plt.plot(r_R, alpha, color=color, label=f'TSR={TSR}')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$\alpha$ [deg]')
    # plt.title('Alpha vs r/R')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ----Spanwise distribution of inflow angle---------------------------------------------------------------
    fig_phi = plt.figure()
    colormap = cm.get_cmap('viridis', n_tsr)
    for i, TSR in enumerate(tip_speed_ratios):
        inflow_angle = corrected_results[f'yaw_0.0_TSR_{TSR}']['inflow_angle']
        r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']
        color = colormap(i)
        plt.plot(r_R, inflow_angle, color=color, label=f'TSR={TSR}')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$\phi$ [deg]')
    # plt.title('Inflow angle vs r/R')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ----Spanwise distribution of axial induction factor-----------------------------------------------------
    fig_a = plt.figure()
    colormap = cm.get_cmap('viridis', n_tsr)
    for i, TSR in enumerate(tip_speed_ratios):
        a = corrected_results[f'yaw_0.0_TSR_{TSR}']['a']
        r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

        color = colormap(i)
        plt.plot(r_R, a, color=color, label=f'TSR={TSR}')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$a$ [-]')
    # plt.title('Axial Induction Factors vs r/R')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ----Spanwise distribution of axial induction factor-----------------------------------------------------
    fig_a_line = plt.figure()
    colormap = cm.get_cmap('viridis', n_tsr)
    for i, TSR in enumerate(tip_speed_ratios):
        a_line = corrected_results[f'yaw_0.0_TSR_{TSR}']['a_line']
        r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

        color = colormap(i)
        plt.plot(r_R, a_line, color=color, label=f'TSR={TSR}')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$a$\'[-]')
    # plt.title('Tangential Induction Factors vs r/R')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ----Spanwise distribution of normal loading-----------------------------------------------------
    fig_c_t = plt.figure()
    colormap = cm.get_cmap('viridis', n_tsr)
    for i, TSR in enumerate(tip_speed_ratios):
        c_t = corrected_results[f'yaw_0.0_TSR_{TSR}']['normal_force'] / (0.5 * u_inf**2 * rotor_radius)
        r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

        color = colormap(i)
        plt.plot(r_R, c_t, color=color, label=f'TSR={TSR}')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$f_{norm}$ [-]')
    # plt.title('Normal force coefficient vs r/R')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ----Spanwise distribution of tangentual loading-----------------------------------------------------
    fig_c_q = plt.figure()
    colormap = cm.get_cmap('viridis', n_tsr)
    for i, TSR in enumerate(tip_speed_ratios):
        c_q = corrected_results[f'yaw_0.0_TSR_{TSR}']['tangential_force'] / (0.5 * u_inf**2 * rotor_radius)
        r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

        color = colormap(i)
        plt.plot(r_R, c_q, color=color, label=f'TSR={TSR}')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$f_{tan}$ [-]')
    # plt.title('Tangential force coefficient vs r/R')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ----Spanwise distribution of circulation------------------------------------------------------------------------------
    fig_c_q = plt.figure()
    colormap = cm.get_cmap('viridis', n_tsr)
    for i, TSR in enumerate(tip_speed_ratios):
        c_q = corrected_results[f'yaw_0.0_TSR_{TSR}']['gamma'] / (np.pi * u_inf**2 / (n_blades * Omega[i]))
        r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

        color = colormap(i)
        plt.plot(r_R, c_q, color=color, label=f'TSR={TSR}')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$\Gamma$ [-]')
    # plt.title('Circulation vs r/R')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ----CP, CQ, CT vs TSR------------------------------------------------------------------------------
    # fig_CP_CQ_CT = plt.figure()
    # colormap = cm.get_cmap('viridis', n_tsr)
    # for i, TSR in enumerate(tip_speed_ratios):
    #     color = colormap(0)
    #     plt.plot(TSR, CP[f'yaw_0.0_TSR_{TSR}'], color=color, label=f'TSR={TSR}', linestyle='-')
    #     plt.plot(TSR, CT[f'yaw_0.0_TSR_{TSR}'], color=color, label=f'TSR={TSR}', linestyle='--')
    #     plt.plot(TSR, CQ[f'yaw_0.0_TSR_{TSR}'], color=color, label=f'TSR={TSR}', linestyle=':')

    # plt.xlabel(r'$\frac{r}{R} [-]$')
    # plt.ylabel('Inflow Angle (degrees)')
    # # plt.title('Inflow angle vs r/R')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # ----Comparison between corrected and uncorrected results------------------------------------------------
    # ----Spanwise distribution of angle of attack------------------------------------------------------------
    fig_alpha_comparison = plt.figure()
    colormap = cm.get_cmap('viridis', n_tsr)
    for i, TSR in enumerate(tip_speed_ratios):
        alpha = corrected_results[f'yaw_0.0_TSR_{TSR}']['alpha']
        r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']
        color = colormap(i)
        plt.plot(r_R, alpha, color=color, label=f'TSR={TSR}')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$\alpha$ [deg]')
    # plt.title('Alpha vs r/R')
    plt.grid(True)
    plt.legend()
    plt.show()
    return fig_alpha, fig_phi, fig_a, fig_a_line, fig_c_t, fig_c_q
