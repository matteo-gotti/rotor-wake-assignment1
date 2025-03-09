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
     plt.xlabel('a')
     plt.ylabel(r'$C_T$ and $C_P$')
     plt.grid()
     plt.legend()
     plt.show()
     
     return fig

def plot_prandtl_correction(r_over_R, root_location_over_R, tip_location_over_R, tip_speed_ratio, n_blades):
     a = np.zeros(np.shape(r_over_R)) + 0.3
     n_tsr = len(tip_speed_ratio)
     colormap = cm.get_cmap('viridis', n_tsr)
     fig = plt.figure(figsize=(12, 6))
     
     for i, tsr in enumerate(tip_speed_ratio):
          prandtl, prandtl_tip, prandtl_root = prandtl_tip_root_correction(
               r_over_R, root_location_over_R, tip_location_over_R, tsr, n_blades, a)
          color = colormap(i)
          plt.plot(r_over_R, prandtl, color=color, label=f'Prandtl TSR={tsr}')

     if n_tsr == 1:
          plt.plot(r_over_R, prandtl_tip, 'g.', label='Prandtl tip')
          plt.plot(r_over_R, prandtl_root, 'b.', label='Prandtl root')
     
     plt.xlabel('r/R')
     plt.legend()
     plt.show()

     return fig

def plot_polar_data(polar_alpha, polar_cl, polar_cd):
     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
     axs[0].plot(polar_alpha, polar_cl)
     axs[0].set_xlim([-30, 30])
     axs[0].set_xlabel(r'$\alpha$')
     axs[0].set_ylabel(r'$C_l$')
     axs[0].grid()
     axs[1].plot(polar_cd, polar_cl)
     axs[1].set_xlim([0, .1])
     axs[1].set_xlabel(r'$C_d$')
     axs[1].grid()
     plt.show()
     
     return fig
