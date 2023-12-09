"""Illustration of a 2D loss function"""

import matplotlib.pyplot as plt
import numpy as np

# Parameters
plt.rcParams["figure.figsize"] = 2, 2
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["lines.linewidth"] = 1


n_samples = 15
x, y = np.meshgrid(
    np.linspace(-3, 3, n_samples), np.linspace(-3, 3, n_samples)
)
z = (1 - x / 3 + x**5 - y**3) * np.exp(-(x**2) / 2 - y**2 / 2)

# Plot loss in 3D
fig = plt.figure(figsize=(4, 4))
ax = plt.axes(projection="3d")
ax.plot_surface(x, y, z[::-1, ::-1], cmap="magma_r", edgecolor="black")

# Labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel(r"$\theta_1$", labelpad=-10)
ax.set_ylabel(r"$\theta_2$", labelpad=-10)
ax.set_zlabel(
    r"$\mathcal{L}(\theta_1, \theta_2)$", labelpad=-10, rotation="horizontal"
)
ax.view_init(15, 35)


# Save
plt.savefig(
    "gradient_descent_3d.svg", dpi=300, bbox_inches="tight", pad_inches=0.2
)
