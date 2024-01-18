"""Read, process and plot lidar data.

The initial version of this module was written by Antoine Lucas and Grégory
Sainton. The code was updated by Alexandre Fournier with parallel processing
using the mpi4py library. The code was then cleaned up and modified by Leonard
Seydoux, and now uses the multiprocessing library and the "core points" trick to
speed up the feature extraction.

Authors:
    Léonard Seydoux, Grégory Sainton, Antoine Lucas, and Alexandre Fournier.

Last update:
    June 2023
"""

from multiprocessing import Pool
from functools import partial

import laspy
from matplotlib import pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree


def read_xyz(filepath: str) -> np.ndarray:
    """Read xyz file.

    Read a lidar data file (.xyz) and return the points coordinates in meters.

    Parameters
    ----------
    filepath : str
        Full path to the file to read.

    Returns
    -------
    points : np.ndarray
        Array of points coordinates in meters, of shape (n_points, 3)
    """
    # Read file
    points = np.loadtxt(filepath)

    # Remove NaNs
    points = points[~np.any(np.isnan(points), axis=1)]

    return points[:, :3]


def read_las(filepath: str) -> np.ndarray:
    """Read LAS (for LASer) file.

    Read a lidar data file (.las, or .laz) and return the points coordinates in
    meters. This function is a wrapper around laspy.read() and the xyz
    attribute of the returned object.

    Parameters
    ----------
    filepath : str
        Full path to the file to read.

    Returns
    -------
    points : np.ndarray
        Array of points coordinates in meters, of shape (n_points, 3)
    """
    return laspy.read(filepath).xyz


def sample(points: np.ndarray, n_points: int) -> np.ndarray:
    """Select a random sample of array elements.

    Given an array of points, select a random sample of n_points elements.
    The function works on the first axis of the array.

    Parameters
    ----------
    points : array-like
        Array of points coordinates in meters, of shape (3, n_points).
    n_points : int
        Number of points to sample.

    Returns
    -------
    points : array-like
        Array of points coordinates in meters, of shape (n_points, 3).
    """
    # Get number of points
    n_total = points.shape[0]

    # Get indices of sampled points
    indices = sample_indices(n_total, n_points)

    return points[indices]


def sample_indices(n_total: int, n_points: int) -> np.ndarray:
    """Select a random sample of array indices.

    Given an array of points, select a random sample of n_points elements.
    The function works on the first axis of the array.

    Parameters
    ----------
    n_total : int
        Total number of points.
    n_points : int
        Number of points to sample.

    Returns
    -------
    indexes : array-like
        Array of indexes, of shape (n_points,).
    """
    # Check that the number of points is not larger than the array size
    n_points = min(n_points, n_total)

    # Sample points
    indices = np.arange(n_total)
    indices = np.random.choice(indices, size=n_points)

    return indices


def plot_scene(points: np.ndarray, ax=None, n_points: int = 10000, **kwargs):
    """Show a 3D scene with points.

    Parameters
    ----------
    points: np.ndarray
        Points to plot.
    ax: Axes3DSubplot, optional
        Axes of the figure, by default None.
    sample : int, optional
        Number of points to plot, by default 1000.
    **kwargs : dict, optional
        Keyword arguments for matplotlib.pyplot.scatter.

    Returns
    -------
    ax : Axes3DSubplot
        Axes of the figure.
    """
    # Get axes
    ax = ax or plt.axes(projection="3d")

    # Default arguments
    kwargs.setdefault("rasterized", True)

    # Decimate points
    if n_points is not None:
        # Sample points
        indices = sample_indices(points.shape[0], n_points)

        # Decimate colors and sizes
        if "c" in kwargs and isinstance(kwargs["c"], np.ndarray):
            kwargs["c"] = kwargs["c"][indices]
        if "s" in kwargs and isinstance(kwargs["s"], np.ndarray):
            kwargs["s"] = kwargs["s"][indices]

        # Decimate points
        points = points[indices]

    # Plot points
    ax.scatter(*points.T, **kwargs)

    # Label axes
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_aspect("equal")

    return ax


def plot_ternary(x, y, ax=None, **kwargs):
    """Plot a ternary diagram.

    Parameters
    ----------
    x : array
        Array of x coordinates.
    y : array
        Array of y coordinates.
    ax : Axes, optional
        Axes of the figure, by default None.
    sort : bool, optional
        Sort points by density, by default True.
    bins : int, optional
        Number of bins, by default 20.
    **kwargs : dict, optional
        Keyword arguments for matplotlib.pyplot.scatter.

    Returns
    -------
    ax : Axes
        Axes of the figure.
    """
    # Axes instance
    ax = ax or plt.gca()

    # Reject NaNs
    points = np.vstack((x, y))
    points = points[:, ~np.any(np.isnan(points), axis=0)]
    x, y = points

    # Estimate density with a Gaussian kernel density estimator
    try:
        density = gaussian_kde(points)(points)
    except:
        density = np.ones_like(x)

    # Default arguments
    kwargs.setdefault("rasterized", True)

    # Plot
    mappable = ax.scatter(x, y, c=density, s=0.5, **kwargs)

    # Colorbar
    plt.colorbar(
        mappable,
        orientation="horizontal",
        label="Density of points",
        shrink=0.5,
        pad=0.05,
    )

    # Triangular grid
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) * 0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=3)
    ax.triplot(trimesh, color="0.5", linestyle="-", linewidth=0.5, zorder=0)

    # Create outline frame
    ax.plot(
        np.hstack((corners[:, 0], corners[0, 0])),
        np.hstack((corners[:, 1], corners[0, 1])),
        color="k",
        linestyle="-",
        linewidth=plt.rcParams["axes.linewidth"],
        zorder=0,
    )

    # Labels
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-0.015, -0.015, "1D", ha="right", va="top")
    ax.text(1.015, -0.015, "2D", ha="left", va="top")
    ax.text(0.5, np.sqrt(3) * 0.5 + 0.015, "3D", ha="center", va="bottom")

    return ax


def eigs(
    point: np.ndarray,
    points: np.ndarray = None,
    diameter: float = None,
    n_min_points: int = 10,
    tree: KDTree = None,
) -> np.ndarray:
    """Calculate the covariance matrix eigenvalues of a point neighborhood.

    Arguments:
    ----------
    point: array-like
        Point of interest of shape (3,).
    points: array-like
        Point cloud of shape (n, 3).
    diameter: float
        Diameter of the neighborhood.
    n_min_points: int, optional
        Minimum number of points in the neighborhood, by default 10.
    tree: KDTree, optional
        KDTree of the point cloud, by default None.

    Returns:
    --------
    covariance: array-like
        Covariance matrix of shape (3, 3).
    """
    # Calculate the distance to the point of interest
    neighbors = points[tree.query_ball_point(point, diameter / 2, workers=-1)]

    # Reject cases where the number of points is too small to calculate the
    # covariance matrix. In this case, return NaNs.
    if neighbors.shape[0] < n_min_points:
        return np.nan * np.ones(3)
    else:
        covariance = np.cov(neighbors, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        return eigenvalues


def calculate_eigenvalues(
    selected_points: np.ndarray,
    points: np.ndarray,
    diameter: float,
    n_min_points: int = 10,
) -> np.ndarray:
    """Calculate the eigenvalues of a point cloud.

    This function calculates the eigenvalues of the covariance matrix of a
    point cloud. The covariance matrix is calculated for each point in
    parallel.

    Arguments:
    ----------
    selected_points: array-like
        Points of interest of shape (m < n, 3).
    points: array-like
        Point cloud of shape (n, 3).
    diameter: float
        Diameter of the neighborhood.
    n_min_points: int, optional
        Minimum number of points in the neighborhood, by default 10.

    Returns:
    --------
    eigenvalues: array-like
        Eigenvalues of the covariance matrix.
    """
    # Get the tree of the point cloud
    tree = KDTree(points)

    # Define the function to calculate the covariance matrix of a point
    eigs_partial = partial(
        eigs,
        points=points,
        diameter=diameter,
        n_min_points=n_min_points,
        tree=tree,
    )

    # Calculate the covariance in parallel
    with Pool(processes=10) as pool:
        eigenvalues = pool.map(eigs_partial, selected_points)

    return np.array(eigenvalues)


def calculate_barycentric_coordinates(eigenvalues):
    """Calculate barycentric coordinates from eigenvalues.

    The barycentric coordinates are calculated from the eigenvalues of the
    covariance matrix of a point cloud. The eigenvalues are assumed to be
    normalized (i.e. they sum to 1). The barycentric coordinates are calculated as follows:

    .. math::

        x = \\frac{1}{2} \\frac{2b + c}{a + b + c}
        y = \\frac{\\sqrt{3}}{2} \\frac{c}{a + b + c}

    where :math:`a`, :math:`b`, and :math:`c` are the eigenvalues of the
    covariance matrix, and :math:`a \\geq b \\geq c`.

    Parameters
    ----------
    eigenvalues : array
        Array of eigenvalues.

    Returns
    -------
    tertiary : array
        Array of tertiary coordinates.
    """
    delta = np.sqrt(2) / 2 * (eigenvalues[:, 0] - eigenvalues[:, 1])
    Delta = np.sqrt(2) / 2
    c = eigenvalues[:, 2] / (1 / 3)
    a = (1 - c) * delta / Delta
    b = 1 - (a + c)
    x = 1 / 2 * (2 * b + c) / (a + b + c)
    y = np.sqrt(3) / 2 * c / (a + b + c)
    return np.vstack((x, y))
