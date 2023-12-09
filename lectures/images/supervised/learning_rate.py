"""Illustration of the learning rate"""

import matplotlib.pyplot as plt
import numpy as np

# Parameters
plt.rcParams["figure.figsize"] = 2, 2
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["lines.linewidth"] = 1


# The loss is a parabola
x = np.linspace(-1, 1, 100)
y = x**2

# Plot loss
fig, ax = plt.subplots(1, 3, figsize=(7, 2))

# Plot loss
theta = 0.9
for a in ax:
    a.plot(x, y, color="black")
    a.set_xticks([])
    a.set_yticks([])
    a.axvline(theta, color="C1")
    a.plot(theta, theta**2, color="C1", marker="o", markersize=5)
    a.text(theta, -0.09, r"$\theta_0$", va="top", ha="center", color="C1")
    a.set_xlabel(r"$\theta$")

ax[0].set_ylabel(r"$\mathcal{L}(\theta)$")

# Plot fast learning rate as bouncing ball
learning_rate = 0.9
derivative = 2 * x
theta = 0.9
for _ in range(5):
    theta_0 = theta
    theta -= learning_rate * derivative[np.argmin(np.abs(x - theta))] + 0.1
    sign = np.sign(theta - theta_0)
    sign = "-" if sign < 0 else "+"
    ax[0].annotate(
        "",
        xy=(theta_0, theta_0**2),
        xytext=(theta, theta**2),
        arrowprops=dict(
            arrowstyle="<-",
            color="C0",
            connectionstyle=f"arc3,rad={sign}0.4",
            shrinkA=5,
            shrinkB=5,
        ),
    )
    ax[0].plot(theta, theta**2, color="C0", marker="o", markersize=5)

# Plot slow learning rate as bouncing ball
learning_rate = 0.05
derivative = 2 * x
theta = 0.9
for _ in range(4):
    theta_0 = theta
    theta -= learning_rate * derivative[np.argmin(np.abs(x - theta))]
    sign = np.sign(theta - theta_0)
    sign = "-" if sign < 0 else "+"
    ax[1].annotate(
        "",
        xy=(theta_0, theta_0**2),
        xytext=(theta, theta**2),
        arrowprops=dict(
            arrowstyle="<-",
            color="C0",
            connectionstyle=f"arc3,rad={sign}2.5",
            shrinkA=4,
            shrinkB=6,
        ),
    )
    ax[1].plot(theta, theta**2, color="C0", marker="o", markersize=5)

# Plot good learning rate as bouncing ball
learning_rate = 0.1
derivative = 2 * x
theta = 0.9
for _ in range(4):
    theta_0 = theta
    theta -= learning_rate * derivative[np.argmin(np.abs(x - theta))] + 0.1
    amp = np.abs(theta - theta_0)
    sign = np.sign(theta - theta_0)
    sign = "-" if sign < 0 else "+"
    amp = f"{(8 - 15 * amp) / 3:.1f}"
    print(amp)
    ax[2].annotate(
        "",
        xy=(theta_0, theta_0**2),
        xytext=(theta, theta**2),
        arrowprops=dict(
            arrowstyle="<-",
            color="C0",
            connectionstyle=f"arc3,rad={sign}{amp}",
            shrinkA=4,
            shrinkB=6,
        ),
    )
    ax[2].plot(theta, theta**2, color="C0", marker="o", markersize=5)

ax[0].set_title("Fast learning rate")
ax[1].set_title("Slow learning rate")
ax[2].set_title("Good learning rate")

ax[-1].set_xlabel(r"$\theta$")
ax[-1].text(0.9, -0.09, r"$\theta_0$", va="top", ha="center", color="C1")

# Save
plt.savefig("learning_rate.svg", dpi=300, bbox_inches="tight", pad_inches=0.2)
