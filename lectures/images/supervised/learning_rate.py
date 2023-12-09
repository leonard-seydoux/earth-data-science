"""Illustration of the learning rate"""

import matplotlib.pyplot as plt
import numpy as np

# The loss is ax parabola
thetas = np.linspace(-1, 1, 100)
theta_0 = 0.9
loss = thetas**2
d_loss = 2 * thetas
loss_0 = theta_0**2

# Prepare figure
fig, axes = plt.subplots(1, 3, figsize=(7, 2))

# Plot loss
for ax in axes:
    # Plot loss
    ax.plot(thetas, loss, color="0.5")

    # Format axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Mark theta_0
    ax.vlines(theta_0, 0, loss_0, color="C1", linestyle="--")
    ax.plot(theta_0, loss_0, color="C1", marker="o", markersize=5)
    ax.text(theta_0, -0.09, r"$\theta_0$", va="top", ha="center", color="C1")
    ax.set_xlabel(r"$\theta$")

# Label only the first y-axis
axes[0].set_ylabel(r"$\mathcal{L}(\theta)$")

# Plot fast learning rate as bouncing ball
learning_rate = 0.9
theta = theta_0
for _ in range(5):
    # Update theta
    theta_index = np.abs(thetas - theta).argmin()
    theta_new = theta - learning_rate * d_loss[theta_index] + 0.1

    # Get sign of update
    sign = "+" if np.sign(theta_new - theta) < 0 else "-"

    # Annotate
    arrowprops = dict(
        arrowstyle="->",
        color="C0",
        connectionstyle=f"arc3,rad={sign}0.4",
        shrinkA=5,
        shrinkB=5,
    )
    axes[0].annotate(
        "",
        xy=(theta_new, theta_new**2),
        xytext=(theta, theta**2),
        arrowprops=arrowprops,
    )

    # Update theta
    axes[0].plot(theta_new, theta_new**2, color="C0", marker="o", ms=5)

    # Update theta
    theta = theta_new

# Plot slow learning rate as bouncing ball
learning_rate = 0.07
theta = theta_0
for _ in range(4):
    # Update theta
    theta_index = np.abs(thetas - theta).argmin()
    theta_new = theta - learning_rate * d_loss[theta_index]

    # Get sign of update
    sign = "+" if np.sign(theta_new - theta) < 0 else "-"

    # Annotate
    arrowprops = dict(
        arrowstyle="->",
        color="C0",
        connectionstyle=f"arc3,rad={sign}3",
        shrinkA=6,
        shrinkB=7,
    )
    axes[1].annotate(
        "",
        xy=(theta_new, theta_new**2),
        xytext=(theta, theta**2),
        arrowprops=arrowprops,
    )

    # Update theta
    axes[1].plot(theta_new, theta_new**2, color="C0", marker="o", ms=5)

    # Update theta
    theta = theta_new

# Plot perfect learning rate as bouncing ball
learning_rate = 0.2
theta = theta_0
for _ in range(3):
    # Update theta
    theta_index = np.abs(thetas - theta).argmin()
    theta_new = theta - learning_rate * d_loss[theta_index]

    # Get sign of update
    sign = "+" if np.sign(theta_new - theta) < 0 else "-"

    # Annotate
    arrowprops = dict(
        arrowstyle="->",
        color="C0",
        connectionstyle=f"arc3,rad={sign}2.2",
        shrinkA=8,
        shrinkB=4,
    )
    axes[2].annotate(
        "",
        xy=(theta_new, theta_new**2),
        xytext=(theta, theta**2),
        arrowprops=arrowprops,
    )

    # Update theta
    axes[2].plot(theta_new, theta_new**2, color="C0", marker="o", ms=5)

    # Update theta
    theta = theta_new

# Titles
axes[0].set_title("Too large")
axes[1].set_title("Too small")
axes[2].set_title("Perfect")

# Save
plt.savefig("learning_rate.svg", dpi=300, bbox_inches="tight", pad_inches=0.2)
