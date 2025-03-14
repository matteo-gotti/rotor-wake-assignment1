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


def plots_non_yawed(corrected_results, uncorrected_results, rotor_radius, tip_speed_ratios, u_inf,
                    Omega, n_blades, plot_corrected=True, plot_comparison=True):
    n_tsr = len(tip_speed_ratios)
    # ----Spanwise distribution of angle of attack------------------------------------------------------------

    if plot_corrected:
        plt.figure()
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

        # ----Spanwise distribution of inflow angle---------------------------------------------------------------
        plt.figure()
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

        # ----Spanwise distribution of axial induction factor-----------------------------------------------------
        plt.figure()
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

        # ----Spanwise distribution of axial induction factor-----------------------------------------------------
        plt.figure()
        colormap = cm.get_cmap('viridis', n_tsr)
        for i, TSR in enumerate(tip_speed_ratios):
            a_line = corrected_results[f'yaw_0.0_TSR_{TSR}']['a_line']
            r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

            color = colormap(i)
            plt.plot(r_R, a_line, color=color, label=f'TSR={TSR}')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r"$a'$[-]")
        # plt.title('Tangential Induction Factors vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of normal loading-----------------------------------------------------
        plt.figure()
        colormap = cm.get_cmap('viridis', n_tsr)
        for i, TSR in enumerate(tip_speed_ratios):
            c_t = corrected_results[f'yaw_0.0_TSR_{TSR}']['normal_force']
            r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

            color = colormap(i)
            plt.plot(r_R, c_t, color=color, label=f'TSR={TSR}')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_n$ [-]')
        # plt.title('Normal force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of tangential loading-----------------------------------------------------
        plt.figure()
        colormap = cm.get_cmap('viridis', n_tsr)
        for i, TSR in enumerate(tip_speed_ratios):
            c_t = corrected_results[f'yaw_0.0_TSR_{TSR}']['tangential_force']
            r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

            color = colormap(i)
            plt.plot(r_R, c_t, color=color, label=f'TSR={TSR}')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_t$ [-]')
        # plt.title('Tangential force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of CT-----------------------------------------------------
        plt.figure()
        colormap = cm.get_cmap('viridis', n_tsr)
        for i, TSR in enumerate(tip_speed_ratios):
            c_T = corrected_results[f'yaw_0.0_TSR_{TSR}']['c_thrust']
            r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

            color = colormap(i)
            plt.plot(r_R, c_T, color=color, label=f'TSR={TSR}')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_T(r)$ [-]')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of CQ-----------------------------------------------------
        plt.figure()
        colormap = cm.get_cmap('viridis', n_tsr)
        for i, TSR in enumerate(tip_speed_ratios):
            c_Q = corrected_results[f'yaw_0.0_TSR_{TSR}']['c_torque']
            r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

            color = colormap(i)
            plt.plot(r_R, c_Q, color=color, label=f'TSR={TSR}')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_Q(r)$ [-]')
        # plt.title('Tangential force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of circulation------------------------------------------------------------------------------
        plt.figure()
        colormap = cm.get_cmap('viridis', n_tsr)
        for i, TSR in enumerate(tip_speed_ratios):
            c_q = corrected_results[f'yaw_0.0_TSR_{TSR}']['gamma']
            r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

            color = colormap(i)
            plt.plot(r_R, c_q, color=color, label=f'TSR={TSR}')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$\Gamma$ [-]')
        # plt.title('Circulation vs r/R')
        plt.grid(True)
        plt.legend()

        plt.show()

    # ----Comparison between corrected and uncorrected results------------------------------------------------
    if plot_comparison:
        TSR = tip_speed_ratios[1]
        r_R_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']
        r_R_uncorr = uncorrected_results['r_over_R']
        TSR = tip_speed_ratios[1]
        colormap = cm.get_cmap('viridis', 2)

    # ----Spanwise distribution of angle of attack------------------------------------------------------------
        plt.figure()
        alpha_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['alpha']
        alpha_uncorr = uncorrected_results['alpha']
        plt.plot(r_R_corr, alpha_corr, color=colormap(0), label=f'with Pradtl correction')
        plt.plot(r_R_uncorr, alpha_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$\alpha$ [deg]')
        # plt.title('Alpha vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of inflow angle---------------------------------------------------------------
        plt.figure()
        inflow_angle_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['inflow_angle']
        inflow_angle_uncorr = uncorrected_results['inflow_angle']
        plt.plot(r_R_corr, inflow_angle_corr, color=colormap(0), label=f'with Pradtl correction')
        plt.plot(r_R_uncorr, inflow_angle_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$\phi$ [deg]')
        # plt.title('Inflow angle vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of axial induction factor-----------------------------------------------------
        plt.figure()
        a_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['a']
        a_uncorr = uncorrected_results['a']
        plt.plot(r_R_corr, a_corr, color=colormap(0), label=f'with Pradtl correction')
        plt.plot(r_R_uncorr, a_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$a$ [-]')
        # plt.title('Axial Induction Factors vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of axial induction factor-----------------------------------------------------
        plt.figure()
        a_line_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['a_line']
        a_line_uncorr = uncorrected_results['a_line']
        plt.plot(r_R_corr, a_line_corr, color=colormap(0), label=f'with Pradtl correction')
        plt.plot(r_R_uncorr, a_line_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r"$a'$[-]")
        # plt.title('Tangential Induction Factors vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of normal loading-----------------------------------------------------
        plt.figure()
        c_t_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['normal_force']
        c_t_uncorr = uncorrected_results['normal_force']
        plt.plot(r_R_corr, c_t_corr, color=colormap(0), label=f'with Pradtl correction')
        plt.plot(r_R_uncorr, c_t_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_n$ [-]')
        # plt.title('Normal force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of tangential loading-----------------------------------------------------
        plt.figure()
        c_q_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['tangential_force']
        c_q_uncorr = uncorrected_results['tangential_force']
        plt.plot(r_R_corr, c_q_corr, color=colormap(0), label=f'with Pradtl correction')
        plt.plot(r_R_uncorr, c_q_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_t$ [-]')
        # plt.title('Tangential force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of C_T-----------------------------------------------------
        plt.figure()
        c_T_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['c_thrust']
        c_T_uncorr = uncorrected_results['c_thrust']
        plt.plot(r_R_corr, c_T_corr, color=colormap(0), label=f'with Pradtl correction')
        plt.plot(r_R_uncorr, c_T_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_T(r)$ [-]')
        # plt.title('Normal force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of C_Q-----------------------------------------------------
        plt.figure()
        c_q_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['c_torque']
        c_q_uncorr = uncorrected_results['c_torque']
        plt.plot(r_R_corr, c_q_corr, color=colormap(0), label=f'with Pradtl correction')
        plt.plot(r_R_uncorr, c_q_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_Q(r)$ [-]')
        # plt.title('Tangential force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of circulation------------------------------------------------------------------------------
        plt.figure()
        c_q_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['gamma']
        c_q_uncorr = uncorrected_results['gamma']
        plt.plot(r_R_corr, c_q_corr, color=colormap(0), label=f'with Pradtl correction')
        plt.plot(r_R_uncorr, c_q_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$\Gamma$ [-]')
        # plt.title('Circulation vs r/R')
        plt.grid(True)
        plt.legend()

        plt.show()

    return


def plots_yawed(results, rotor_radius, yaw_angles, u_inf, Omega, n_blades, r_over_R, psi, variables_to_plot, labels):

    if len(variables_to_plot) != len(labels):
        raise ValueError('The number of variables to plot must match the number of labels')
    n_yaw = len(yaw_angles)
    R, Psi = np.meshgrid(r_over_R, psi)
    for i, var in enumerate(variables_to_plot):
        if var not in results['yaw_0.0_TSR_8.0']:
            raise ValueError(f'Variable {var} not found in results')

        fig, axs = plt.subplots(1, n_yaw, subplot_kw=dict(polar=True), figsize=(19, 5))
        colormap = plt.get_cmap('viridis')

        vmax = max([np.max(np.array(results[f'yaw_{yaw}_TSR_8.0'][var])) for yaw in yaw_angles])
        vmin = min([np.min(np.array(results[f'yaw_{yaw}_TSR_8.0'][var])) for yaw in yaw_angles])

        for j, ax in enumerate(axs):
            key = f'yaw_{yaw_angles[j]}_TSR_8.0'
            Y = np.array(results[key][var]).T
            c = ax.contourf(Psi, R, Y, levels=30, cmap=colormap, vmin=vmin, vmax=vmax)

            ax.set_title(f'Yaw angle {yaw_angles[j]}°', fontsize=14, pad=30)
            ax.set_yticklabels([])
            ax.grid(True)
            # Add a colorbar for each subplot
            cbar = fig.colorbar(c, ax=ax, orientation='vertical', fraction=0.04, pad=0.1)
            cbar.set_label(labels[i], fontsize=10)
            cbar.ax.tick_params(axis='y', labelrotation=45)  # Adjust angle as needed
        # cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
        # cb = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
        # cb.set_label(labels[i], fontsize=12)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.92, wspace=0.3)
        # plt.subplots_adjust(left=0.05, right=0.85, bottom=0.15, top=0.92, wspace=0.2)

    plt.show()

    return
