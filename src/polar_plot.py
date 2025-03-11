import numpy as np
import matplotlib.pyplot as plt

# Define the grid
r = np.linspace(0, 1, 100)
theta = np.radians(np.linspace(0, 360, 360))
R, Theta = np.meshgrid(r, theta)

# Synthetic axial induction distribution (replace with real data as needed)
def generate_axial_induction(R, Theta, yaw_angle_deg):
    yaw_angle_rad = np.radians(yaw_angle_deg)
    a = 0.3 + 0.15 * (np.cos(Theta - yaw_angle_rad) * (1 - R**2))
    return a

# Generate values
yaw_angles = [0, 15, 30]
a_values = [generate_axial_induction(R, Theta, yaw) for yaw in yaw_angles]

# Set up the figure
fig, axs = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(18, 7))

vmin, vmax = 0.15, 0.48
cmap = plt.get_cmap('viridis')

# Create the polar plots
for i, ax in enumerate(axs):
    c = ax.contourf(Theta, R, a_values[i], levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'Yaw angle {yaw_angles[i]}°', fontsize=14, pad=30)
    ax.set_yticklabels([])
    ax.grid(True)

# Add a colorbar manually at a custom position
# [left, bottom, width, height] — all in figure coordinates (0 to 1)
cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])  # Lowered the bottom value for more space
cb = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
cb.set_label('a [-]', fontsize=12)

# Main title
plt.suptitle('Axial Induction Polar Plots for Yaw Angles 0°, 15°, and 30°', fontsize=18, y=0.96)

plt.tight_layout(rect=[0, 0.15, 1, 0.92])  # Leave more bottom space for the colorbar
plt.show()