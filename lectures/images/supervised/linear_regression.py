"""Illustration of linear regresison"""

import matplotlib.pyplot as plt
import numpy as np


# Parameters
plt.rcParams["figure.figsize"] = 2, 2
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["lines.linewidth"] = 1


def linear_data(
    slope=1, intercept=-0.5, n_samples=50, noise_level=0.1, random_state=1
):
    """Generate linear data.

    Parameters
    ----------
    slope : float, optional
        Slope of the line.
    intercept : float, optional
        Intercept of the line.
    n_samples : int, optional
        Number of samples.
    noise_level : float, optional
        Noise level.

    Returns
    -------
    x : array, shape (n_samples, 1)
        Input data.
    y : array, shape (n_samples, 1)
        Continuous label.
    """
    # Seed the random generator
    np.random.seed(random_state)

    # Generate data
    x = np.random.rand(n_samples)
    y = slope * x + intercept + noise_level * np.random.randn(n_samples)

    return x.reshape(-1, 1), y.reshape(-1, 1)


def gaussian_blob(center=(0, 0), radius=1, n_samples=50, random_state=1):
    """Generate a gaussian blob.

    Parameters
    ----------
    center: tuple, optional
        Center of the blob.
    radius: float, optional
        Radius of the blob.

    Returns
    -------
    x: array, shape (n_samples,)
        Input data.
    """
    # Seed the random generator
    np.random.seed(random_state)

    return radius * np.random.randn(n_samples, 2) + center


def main():
    # Regression
    a = 1
    b = -0.5
    x, y = linear_data(a, b)
    fig, ax = plt.subplots()
    ax.plot(x, y, ".")
    ax.axline((0, b), slope=a, color="C1")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.8, 0.7)
    ax.set_aspect("equal")
    fig.savefig("linear_regression.svg")
    plt.close()

    # Regression
    a = 1
    b = -0.5
    x, y = linear_data(a, b)
    fig, ax = plt.subplots()
    ax.plot(x, y, ".", label=r"$\mathcal{D}$")
    ax.axline((0, b), slope=a, color="C1", label=r"$f_\theta(x)$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.8, 0.7)
    ax.legend()
    ax.set_aspect("equal")
    fig.savefig("linear_regression_math.svg")
    plt.close()

    # Brute force
    a = 1
    b = -0.5
    x, y = linear_data(a, b)
    fig, ax = plt.subplots()
    ax.plot(x, y, ".")
    cmap = plt.get_cmap("RdPu_r")
    for a in np.tan(np.pi * np.linspace(-0.5, 0.5, 5))[:-1]:
        for b in np.linspace(-5, 5, 50):
            # Calculate r^2 between 0 and 1
            residuals = np.abs(y - (a * x + b)) / (np.max(y) - np.min(y))
            residuals = np.mean(residuals)
            ax.axline((0, b), slope=a, color=cmap(np.sqrt(residuals)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.8, 0.7)
    ax.set_aspect("equal")
    fig.savefig("linear_regression_brute_force.svg")
    plt.close()

    # Random
    a = 1
    b = -0.5
    x, y = linear_data(a, b)
    fig, ax = plt.subplots()
    ax.plot(x, y, ".")
    cmap = plt.get_cmap("YlOrRd_r")
    for a, b in zip(
        np.tan(np.pi * np.linspace(-0.5, 0.5, 100)),
        5 * np.random.rand(100) - 2,
    ):
        residuals = np.abs(y - (a * x + b)) / (np.max(y) - np.min(y))
        residuals = np.mean(residuals)
        ax.axline((0, b), slope=a, color=cmap(np.sqrt(residuals)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.8, 0.7)
    ax.set_aspect("equal")
    fig.savefig("linear_regression_random.svg")
    plt.close()

    # Simulate gradient descent
    a = 1
    b = -0.5
    x, y = linear_data(a, b)
    fig, ax = plt.subplots()
    ax.plot(x, y, ".")
    cmap = plt.get_cmap("autumn")
    for a, b, c in zip(
        np.linspace(-2, 1, 40),
        np.linspace(-2, -0.5, 40),
        np.linspace(-1, 0, 40),
    ):
        residuals = np.abs(y - (a * x + b)) / (np.max(y) - np.min(y))
        residuals = np.mean(residuals)
        ax.axline((c, b), slope=a, color=cmap(np.sqrt(residuals)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.8, 0.7)
    ax.set_aspect("equal")
    fig.savefig("linear_regression_gradient_descent.svg")

    # Classification
    fig, ax = plt.subplots()
    x = gaussian_blob(center=(0.6, 0.3), radius=0.15)
    ax.plot(*x.T, ".", label="Target 1")
    x = gaussian_blob(center=(0.1, -0.1), radius=0.12)
    ax.plot(*x.T, ".", label="Target 2", c="lightblue")
    ax.set_xlabel("Feature #1")
    ax.set_ylabel("Feature #2")
    ax.legend(fontsize="small")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.8, 0.7)
    ax.axline((0, 0.5), slope=-1, color="C1")
    ax.set_aspect("equal")
    fig.savefig("linear_classification.svg")


if __name__ == "__main__":
    main()
