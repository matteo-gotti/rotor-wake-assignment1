import matplotlib.pyplot as plt
from BEM_functions import *
import numpy as np
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter


def plot_glauert_correction():
    a = np.arange(0.0, 1.0, 0.01)
    colormap = cm.get_cmap('brg', 3)
    c_t_uncorrected = compute_c_t(a)  # CT without correction
    c_t_glauert = compute_c_t(a, True)  # CT with Glauert's correction

    fig = plt.figure(figsize=(12, 6))
    plt.plot(a, c_t_uncorrected, color=colormap(0), linestyle='-', label='$C_T$')
    plt.plot(a, c_t_glauert, 'b--', color=colormap(1), linestyle='--', label='$C_T$ Glauert')
    plt.plot(a, c_t_glauert * (1 - a), color=colormap(2), linestyle='--', label='$C_P$ Glauert')
    plt.xlabel(r'$a$ [-]')
    plt.ylabel(r'$C_T$ and $C_P$ [-]')
    plt.grid()
    plt.legend()
    plt.show()

    return fig


def plot_prandtl_correction(r_over_R, root_location_over_R, tip_location_over_R, tip_speed_ratio, n_blades):
    a = np.zeros(np.shape(r_over_R)) + 0.3
    n_tsr = 1 if type(tip_speed_ratio) is np.float64 else len(tip_speed_ratio)
    colormap = cm.get_cmap('brg', n_tsr)
    fig = plt.figure(figsize=(12, 6))

    if n_tsr == 1:
        prandtl, prandtl_tip, prandtl_root = prandtl_tip_root_correction(
            r_over_R, root_location_over_R, tip_location_over_R, tip_speed_ratio, n_blades, a)
        plt.plot(r_over_R, prandtl_tip, 'b.', label='Prandtl tip')
        plt.plot(r_over_R, prandtl_root, 'r.', label='Prandtl root')
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
    plt.ylabel('Prandtl correction factor [-]')
    plt.legend()
    plt.show()

    return fig


def plot_polar_data(polar_alpha, polar_cl, polar_cd):
    # Compute efficiency (L/D ratio)
    efficiency = polar_cl / polar_cd
    max_idx = np.argmax(efficiency)

    # Extract max efficiency values
    alpha_maxE = polar_alpha[max_idx]
    CL_maxE = polar_cl[max_idx]
    CD_maxE = polar_cd[max_idx]

    # Create plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # CL vs Alpha plot
    axs[0].plot(polar_alpha, polar_cl)
    axs[0].scatter(alpha_maxE, CL_maxE, color='red', label="Max Efficiency", zorder=3)
    axs[0].set_xlim([-30, 30])
    axs[0].set_xlabel(r'$\alpha$ [deg]')
    axs[0].set_ylabel(r'$C_l$ [-]')
    # axs[0].legend()
    axs[0].grid()

    # CL vs CD (Drag Polar) plot
    axs[1].plot(polar_cd, polar_cl)
    axs[1].scatter(CD_maxE, CL_maxE, color='red', label="Max Efficiency", zorder=3)
    axs[1].set_xlim([0, .1])
    axs[1].set_xlabel(r'$C_d$ [-]')
    axs[1].set_ylabel(r'$C_l$ [-]')
    axs[1].legend()
    axs[1].grid()

    plt.show()

    return fig


def plots_non_yawed(corrected_results, uncorrected_results, tip_speed_ratios, polar_alpha, polar_cd, polar_cl, plot_corrected=True, plot_comparison=True):
    n_tsr = len(tip_speed_ratios)
    # ----Spanwise distribution of angle of attack------------------------------------------------------------

    if plot_corrected:
        plt.figure()
        colormap = cm.get_cmap('brg', n_tsr)
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
        colormap = cm.get_cmap('brg', n_tsr)
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
        colormap = cm.get_cmap('brg', n_tsr)
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
        colormap = cm.get_cmap('brg', n_tsr)
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
        colormap = cm.get_cmap('brg', n_tsr)
        for i, TSR in enumerate(tip_speed_ratios):
            c_n = corrected_results[f'yaw_0.0_TSR_{TSR}']['normal_force']
            r_R = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']

            color = colormap(i)
            plt.plot(r_R, c_n, color=color, label=f'TSR={TSR}')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_n$ [-]')
        # plt.title('Normal force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of tangential loading-----------------------------------------------------
        plt.figure()
        colormap = cm.get_cmap('brg', n_tsr)
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
        colormap = cm.get_cmap('brg', n_tsr)
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
        colormap = cm.get_cmap('brg', n_tsr)
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
        colormap = cm.get_cmap('brg', n_tsr)
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

        # ----Spanwise distribution of efficiency------------------------------------------------------------------------------
        plt.figure()
        efficiency = np.zeros_like(corrected_results[f'yaw_0.0_TSR_{tip_speed_ratios[1]}']['r_over_R'])
        for i, AoA in enumerate(corrected_results[f'yaw_0.0_TSR_{tip_speed_ratios[1]}']['alpha']):
            cl = np.interp(AoA, polar_alpha, polar_cl)
            cd = np.interp(AoA, polar_alpha, polar_cd)
            efficiency[i] = cl/cd

        plt.plot(corrected_results[f'yaw_0.0_TSR_{tip_speed_ratios[1]}']
                 ['r_over_R'], efficiency, color=colormap(0))
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$E [-]$')
        # plt.title('Circulation vs r/R')
        plt.grid(True)

        plt.show()

    # ----Comparison between corrected and uncorrected results------------------------------------------------
    if plot_comparison:
        TSR = tip_speed_ratios[1]
        r_R_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['r_over_R']
        r_R_uncorr = uncorrected_results['r_over_R']
        TSR = tip_speed_ratios[1]
        colormap = cm.get_cmap('brg', 2)

    # ----Spanwise distribution of angle of attack------------------------------------------------------------
        plt.figure()
        alpha_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['alpha']
        alpha_uncorr = uncorrected_results['alpha']
        plt.plot(r_R_corr, alpha_corr, color=colormap(0), label=f'with Prandtl correction')
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
        plt.plot(r_R_corr, inflow_angle_corr, color=colormap(0), label=f'with Prandtl correction')
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
        plt.plot(r_R_corr, a_corr, color=colormap(0), label=f'with Prandtl correction')
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
        plt.plot(r_R_corr, a_line_corr, color=colormap(0), label=f'with Prandtl correction')
        plt.plot(r_R_uncorr, a_line_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r"$a'$[-]")
        # plt.title('Tangential Induction Factors vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of normal loading-----------------------------------------------------
        plt.figure()
        c_n_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['normal_force']
        c_n_uncorr = uncorrected_results['normal_force']
        plt.plot(r_R_corr, c_n_corr, color=colormap(0), label=f'with Prandtl correction')
        plt.plot(r_R_uncorr, c_n_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_n$ [-]')
        # plt.title('Normal force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of tangential loading-----------------------------------------------------
        plt.figure()
        c_t_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['tangential_force']
        c_t_uncorr = uncorrected_results['tangential_force']
        plt.plot(r_R_corr, c_t_corr, color=colormap(0), label=f'with Prandtl correction')
        plt.plot(r_R_uncorr, c_t_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_t$ [-]')
        # plt.title('Tangential force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of C_T-----------------------------------------------------
        plt.figure()
        c_T_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['c_thrust']
        c_T_uncorr = uncorrected_results['c_thrust']
        plt.plot(r_R_corr, c_T_corr, color=colormap(0), label=f'with Prandtl correction')
        plt.plot(r_R_uncorr, c_T_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_T(r)$ [-]')
        # plt.title('Normal force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of C_Q-----------------------------------------------------
        plt.figure()
        c_Q_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['c_torque']
        c_Q_uncorr = uncorrected_results['c_torque']
        plt.plot(r_R_corr, c_Q_corr, color=colormap(0), label=f'with Prandtl correction')
        plt.plot(r_R_uncorr, c_Q_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$C_Q(r)$ [-]')
        # plt.title('Tangential force coefficient vs r/R')
        plt.grid(True)
        plt.legend()

        # ----Spanwise distribution of circulation------------------------------------------------------------------------------
        plt.figure()
        gamma_corr = corrected_results[f'yaw_0.0_TSR_{TSR}']['gamma']
        gamma_uncorr = uncorrected_results['gamma']
        plt.plot(r_R_corr, gamma_corr, color=colormap(0), label=f'with Prandtl correction')
        plt.plot(r_R_uncorr, gamma_uncorr, color=colormap(1), label=f'without Prandtl correction')
        plt.xlabel(r'$\frac{r}{R}$ [-]')
        plt.ylabel(r'$\Gamma$ [-]')
        # plt.title('Circulation vs r/R')
        plt.grid(True)
        plt.legend()

        plt.show()

    return


def plots_yawed(results, yaw_angles, r_over_R, psi, variables_to_plot, labels):

    if len(variables_to_plot) != len(labels):
        raise ValueError('The number of variables to plot must match the number of labels')
    n_yaw = len(yaw_angles)
    R, Psi = np.meshgrid(r_over_R, psi)
    for i, var in enumerate(variables_to_plot):
        if var not in results['yaw_0.0_TSR_8.0']:
            raise ValueError(f'Variable {var} not found in results')

        fig, axs = plt.subplots(1, n_yaw, subplot_kw=dict(polar=True), figsize=(19, 6))
        colormap = plt.get_cmap('RdBu')
        # colormap = plt.get_cmap('turbo')

        vmax = max([np.max(np.array(results[f'yaw_{yaw}_TSR_8.0'][var])) for yaw in yaw_angles])
        vmin = min([np.min(np.array(results[f'yaw_{yaw}_TSR_8.0'][var])) for yaw in yaw_angles])

        for j, ax in enumerate(axs):
            key = f'yaw_{yaw_angles[j]}_TSR_8.0'
            Y = np.array(results[key][var]).T
            c = ax.contourf(Psi, R, Y, levels=30, cmap=colormap, vmin=vmin, vmax=vmax)

            ax.set_title(r'$\gamma$ = ' + f'{int(yaw_angles[j])} deg', fontsize=12, pad=30)
            ax.set_yticklabels([])
            ax.grid(True)
            cbar = fig.colorbar(c, ax=ax, orientation='horizontal', fraction=0.04, pad=0.1)
            cbar.set_label(labels[i], fontsize=12)

            if var == 'alpha' or var == 'inflow_angle':
                cbar.formatter = FormatStrFormatter('%.0f')
            else:
                cbar.formatter = FormatStrFormatter('%.2f')

            grid_lines = ax.get_yticks()  # Get grid circle positions
            angle = np.deg2rad(315)  # 45 degrees in radians
            for r in grid_lines:
                if not np.isclose(r, 1.0):  # Skip plotting if the value is 1
                    ax.text(angle, r, f'{r:.1f}', fontsize=10, ha='center',
                            va='center', color='white', alpha=1)

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.92, wspace=0.3)

    plt.show()

    return


def plot_mesh_convergence(number_of_annuli_vec, CT_uniform, CT_cosine, rel_error_uniform, rel_error_cosine):
    plt.figure()
    plt.plot(number_of_annuli_vec, CT_uniform, 'k-', label='Uniform')
    plt.plot(number_of_annuli_vec, CT_cosine, 'k--', label='Cosine')
    plt.xlabel('N [-]')
    plt.ylabel('$C_T$ [-]')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.loglog(number_of_annuli_vec[0:-1], rel_error_uniform, 'k-', label='Uniform')
    plt.loglog(number_of_annuli_vec[0:-1], rel_error_cosine, 'k--', label='Cosine')
    plt.loglog(number_of_annuli_vec[0:-1], 1e-4 * np.ones(len(number_of_annuli_vec[0:-1])), 'r:', label='Threshold')
    plt.xlabel('N [-]')
    plt.ylabel(r'$e_{rel}$ [-]')
    plt.grid()
    plt.legend()
    plt.show()

    return


def plot_polar_pressure_distribution(centroids, p_tot, p_tot_behind_rotor):
    # Define the stations
    plt.figure()
    plt.plot(centroids, p_tot, 'r-', label='Total pressure in front of rotor')
    plt.plot(centroids, p_tot_behind_rotor, 'b-', label='Total pressure behind rotor')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$P_{tot}$ [Pa]')
    plt.grid()
    plt.legend()

    # Display the plot
    plt.show()

    return


def plots_optimization(results_opt, opt_chord_distribution, opt_twist_distribution,
                       orig_chord_distribution, orig_twist_distribution, results_orig, centroids,  r_R):
    colormap = cm.get_cmap('brg', 2)

# ----Spanwise distribution of angle of attack------------------------------------------------------------
    plt.figure()
    alpha_orig = results_orig['alpha']
    alpha_opt = results_opt['alpha']
    plt.plot(centroids, alpha_orig, color=colormap(0), label=f'Original')
    plt.plot(centroids, alpha_opt, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$\alpha$ [deg]')
    # plt.title('Alpha vs r/R')
    plt.grid(True)
    plt.legend()

    # ----Spanwise distribution of inflow angle---------------------------------------------------------------
    plt.figure()
    inflow_angle_orig = results_orig['inflow_angle']
    inflow_angle_opt = results_opt['inflow_angle']
    plt.plot(centroids, inflow_angle_orig, color=colormap(0), label=f'Original')
    plt.plot(centroids, inflow_angle_opt, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$\phi$ [deg]')
    # plt.title('Inflow angle vs r/R')
    plt.grid(True)
    plt.legend()

    # ----Spanwise distribution of axial induction factor-----------------------------------------------------
    plt.figure()
    a_orig = results_orig['a']
    a_opt = results_opt['a']
    plt.plot(centroids, a_orig, color=colormap(0), label=f'Original')
    plt.plot(centroids, a_opt, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$a$ [-]')
    # plt.title('Axial Induction Factors vs r/R')
    plt.grid(True)
    plt.legend()

    # ----Spanwise distribution of axial induction factor-----------------------------------------------------
    plt.figure()
    a_line_orig = results_orig['a_line']
    a_line_opt = results_opt['a_line']
    plt.plot(centroids, a_line_orig, color=colormap(0), label=f'Original')
    plt.plot(centroids, a_line_opt, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r"$a'$[-]")
    # plt.title('Tangential Induction Factors vs r/R')
    plt.grid(True)
    plt.legend()

    # ----Spanwise distribution of normal loading-----------------------------------------------------
    plt.figure()
    c_n_orig = results_orig['normal_force']
    c_n_opt = results_opt['normal_force']
    plt.plot(centroids, c_n_orig, color=colormap(0), label=f'Original')
    plt.plot(centroids, c_n_opt, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$C_n$ [-]')
    # plt.title('Normal force coefficient vs r/R')
    plt.grid(True)
    plt.legend()

    # ----Spanwise distribution of tangential loading-----------------------------------------------------
    plt.figure()
    c_t_orig = results_orig['tangential_force']
    c_t_opt = results_opt['tangential_force']
    plt.plot(centroids, c_t_orig, color=colormap(0), label=f'Original')
    plt.plot(centroids, c_t_opt, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$C_t$ [-]')
    # plt.title('Tangential force coefficient vs r/R')
    plt.grid(True)
    plt.legend()

    # ----Spanwise distribution of C_T-----------------------------------------------------
    plt.figure()
    c_T_orig = results_orig['c_thrust']
    c_T_opt = results_opt['c_thrust']
    plt.plot(centroids, c_T_orig, color=colormap(0), label=f'Original')
    plt.plot(centroids, c_T_opt, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$C_T(r)$ [-]')
    # plt.title('Normal force coefficient vs r/R')
    plt.grid(True)
    plt.legend()

    # ----Spanwise distribution of C_Q-----------------------------------------------------
    plt.figure()
    c_Q_orig = results_orig['c_torque']
    c_Q_opt = results_opt['c_torque']
    plt.plot(centroids, c_Q_orig, color=colormap(0), label=f'Original')
    plt.plot(centroids, c_Q_opt, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$C_Q(r)$ [-]')
    # plt.title('Tangential force coefficient vs r/R')
    plt.grid(True)
    plt.legend()

    # ----Spanwise distribution of circulation------------------------------------------------------------------------------
    plt.figure()
    gamma_orig = results_orig['gamma']
    gamma_opt = results_opt['gamma']
    plt.plot(centroids, gamma_orig, color=colormap(0), label=f'Original')
    plt.plot(centroids, gamma_opt, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$\Gamma$ [-]')
    # plt.title('Circulation vs r/R')
    plt.grid(True)
    plt.legend()

    # ----Spanwise distribution of chord------------------------------------------------------------------------------
    plt.figure()
    plt.plot(r_R, orig_chord_distribution, color=colormap(0), label=f'Original')
    plt.plot(r_R, opt_chord_distribution, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$C$ [m]')
    # plt.title('Circulation vs r/R')
    plt.grid(True)
    plt.legend()

    # ----Spanwise distribution of twist------------------------------------------------------------------------------
    plt.figure()
    plt.plot(r_R, orig_twist_distribution, color=colormap(0), label=f'Original')
    plt.plot(r_R, opt_twist_distribution, color=colormap(1), label=f'Optimized')
    plt.xlabel(r'$\frac{r}{R}$ [-]')
    plt.ylabel(r'$\Theta$ [deg]')
    # plt.title('Circulation vs r/R')
    plt.grid(True)
    plt.legend()

    return
