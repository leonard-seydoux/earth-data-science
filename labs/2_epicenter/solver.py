"""Define functions to compute the likelihood of a problem.

Made in 2023 by Leonard Seydoux.
"""

import cartopy
from matplotlib import pyplot as plt
import numpy as np
import obspy


def monte_carlo_gaussian(
    stream, uncertainty=20, wavespeed=3, shots=100, extent=[4, 6, 44, 46]
):
    """
    Compute the likelihood of a set of points given a set of travel times.

    Parameters
    ----------
    stream : obspy.Stream
        Stream containing the travel times.
    uncertainty : float
        Uncertainty on the travel times.
    wavespeed : float
        Wavespeed in km/s.
    shots : int
        Number of random points to shoot.
    extent : list
        Extent of the domain in the form [min_longitude, max_longitude,
        min_latitude, max_latitude]

    Returns
    -------
    latitudes : numpy.ndarray
        Array of latitudes.
    longitudes : numpy.ndarray
        Array of longitudes.
    likelihood : numpy.ndarray
        Likelihood of each point.
    """
    # Sample random points in the domain
    longitudes = np.random.uniform(extent[0], extent[1], shots)
    latitudes = np.random.uniform(extent[2], extent[3], shots)

    # Compute the expected travel time for each point
    t_hat = []
    for trace in stream:
        lat = trace.stats.coordinates["latitude"]
        lon = trace.stats.coordinates["longitude"]
        distance = obspy.geodetics.base.locations2degrees(
            lat, lon, latitudes, longitudes
        )
        t_hat.append(distance * 111.19 / wavespeed)

    # Turn into a numpy array
    t_hat = np.array(t_hat)

    # Get observed travel times
    t_observed = np.array([trace.stats.onset for trace in stream])

    # Compute the likelihood
    likelihood = np.exp(
        -0.5
        * np.sum((t_hat - t_observed[:, None]) ** 2 / uncertainty**2, axis=0)
    )

    # Normalize the likelihood
    likelihood /= np.max(likelihood)

    return latitudes, longitudes, likelihood


def monte_carlo_laplacian(
    stream, uncertainty=20, wavespeed=3, shots=100, extent=[4, 6, 44, 46]
):
    """
    Compute the likelihood of a set of points given a set of travel times.

    Parameters
    ----------
    stream : obspy.Stream
        Stream containing the travel times.
    uncertainty : float
        Uncertainty on the travel times.
    wavespeed : float
        Wavespeed in km/s.
    shots : int
        Number of random points to shoot.
    extent : list
        Extent of the domain in the form [min_longitude, max_longitude,
        min_latitude, max_latitude]

    Returns
    -------
    latitudes : numpy.ndarray
        Array of latitudes.
    longitudes : numpy.ndarray
        Array of longitudes.
    likelihood : numpy.ndarray
        Likelihood of each point.
    """
    # Sample random points in the domain
    longitudes = np.random.uniform(extent[0], extent[1], shots)
    latitudes = np.random.uniform(extent[2], extent[3], shots)

    # Compute the expected travel time for each point
    t_hat = []
    for trace in stream:
        lat = trace.stats.coordinates["latitude"]
        lon = trace.stats.coordinates["longitude"]
        distance = obspy.geodetics.base.locations2degrees(
            lat, lon, latitudes, longitudes
        )
        t_hat.append(distance * 111.19 / wavespeed)

    # Turn into a numpy array
    t_hat = np.array(t_hat)

    # Get observed travel times
    t_observed = np.array([trace.stats.onset for trace in stream])

    # Compute the likelihood
    likelihood = np.exp(
        -np.sum(np.abs(t_hat - t_observed[:, None]) / uncertainty, axis=0)
    )

    # Normalize the likelihood
    likelihood /= np.max(likelihood)

    return latitudes, longitudes, likelihood


def basemap(extent=[4, 6, 44, 46], stream=None, event=None):
    """
    Create a basemap of the region.

    Parameters
    ----------
    extent : list
        Extent of the domain in the form [min_longitude, max_longitude,
        min_latitude, max_latitude]
    resolution : str
        Resolution of the basemap.

    Returns
    -------
    basemap : cartopy.mpl.geoaxes.GeoAxesSubplot
        Basemap of the region.
    """
    # Create a figure
    fig = plt.figure()

    # Create a basemap
    ax = fig.add_subplot(111, projection=cartopy.crs.PlateCarree())

    # Add gridlines
    ax.gridlines()

    # Add features
    ax.add_feature(cartopy.feature.OCEAN, color="0.9", zorder=10, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS, edgecolor="royalblue", zorder=10, alpha=0.5)
    ax.coastlines()

    # Set extent
    ax.set_extent(extent)

    # Add stations
    if stream is not None:
        for trace in stream:
            ax.plot(
                trace.stats.coordinates["longitude"],
                trace.stats.coordinates["latitude"],
                "kv",
            )

    # Add event
    if event is not None:
        ax.plot(
            event.origins[0].longitude,
            event.origins[0].latitude,
            "r*",
            mec="k",
            ms=9,
        )

    # Add labels        
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    return ax