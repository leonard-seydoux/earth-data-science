"""Define functions to compute the likelihood of a problem.

Made in 2023 by Leonard Seydoux.
"""

import cartopy
from matplotlib import pyplot as plt
import numpy as np
import obspy


DEGREE_TO_KM = 111.19


def predict_travel_times(stream, latitudes, longitudes, wavespeed=3):
    """
    Compute the expected travel time for each point.

    Parameters
    ----------
    stream : obspy.Stream
        Stream containing the travel times.
    longitudes : numpy.ndarray
        Array of longitudes.
    latitudes : numpy.ndarray
        Array of latitudes.
    wavespeed : float
        Wavespeed in km/s.

    Returns
    -------
    predicted_travel_times : numpy.ndarray
        Array of predicted travel times.
    """
    # Make input arrays at least 1D
    latitudes = np.atleast_1d(latitudes)
    longitudes = np.atleast_1d(longitudes)

    # Compute the expected travel time for each point
    predicted_travel_times = []
    for trace in stream:
        lat = trace.stats.coordinates["latitude"]
        lon = trace.stats.coordinates["longitude"]
        distance = obspy.geodetics.base.locations2degrees(
            lat, lon, latitudes, longitudes
        )
        predicted_travel_times.append(distance * DEGREE_TO_KM / wavespeed)

    # Turn into a numpy array
    predicted_travel_times = np.array(predicted_travel_times)

    return predicted_travel_times.squeeze()


def random_coordinates(extent=[4, 6, 44, 46], shots=100):
    """
    Sample random coordinates in a given extent.

    Parameters
    ----------
    extent : list
        Extent of the domain in the form [min_longitude, max_longitude,
        min_latitude, max_latitude]
    shots : int
        Number of random points to shoot.

    Returns
    -------
    latitudes : numpy.ndarray
        Array of latitudes.
    longitudes : numpy.ndarray
        Array of longitudes.
    """
    # Sample random points in the domain
    longitudes = np.random.uniform(extent[0], extent[1], shots)
    latitudes = np.random.uniform(extent[2], extent[3], shots)

    return latitudes, longitudes


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
    random_grid_test = random_coordinates(extent=extent, shots=shots)

    # Compute the travel time for each point
    predicted_travel_times = predict_travel_times(
        stream, *random_grid_test, wavespeed=wavespeed
    )

    # Get observed travel times
    observed_travel_times = np.array([trace.stats.onset for trace in stream])

    # Compute the likelihood
    error = np.abs(predicted_travel_times - observed_travel_times[:, None])
    likelihoods = np.exp(-0.5 * np.sum(error**2 / uncertainty**2, axis=0))

    # Normalize the likelihood
    likelihoods /= np.max(likelihoods)

    return *random_grid_test, likelihoods


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
    predicted_travel_times = []
    for trace in stream:
        lat = trace.stats.coordinates["latitude"]
        lon = trace.stats.coordinates["longitude"]
        distance = obspy.geodetics.base.locations2degrees(
            lat, lon, latitudes, longitudes
        )
        predicted_travel_times.append(distance * DEGREE_TO_KM / wavespeed)

    # Turn into a numpy array
    predicted_travel_times = np.array(predicted_travel_times)

    # Get observed travel times
    observed_travel_times = np.array([trace.stats.onset for trace in stream])

    # Compute the likelihood
    likelihood = np.exp(
        -np.sum(
            np.abs(predicted_travel_times - observed_travel_times[:, None])
            / uncertainty,
            axis=0,
        )
    )

    # Normalize the likelihood
    likelihood /= np.max(likelihood)

    return latitudes, longitudes, likelihood


def basemap(extent=[4, 6, 44, 46], stream=None, event=None, figsize=None):
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
    fig = plt.figure(figsize=figsize)

    # Create a basemap
    ax = fig.add_subplot(111, projection=cartopy.crs.PlateCarree())

    # Add gridlines
    ax.gridlines()

    # Add features
    ax.add_feature(cartopy.feature.OCEAN, color="0.9", zorder=10, alpha=0.5)
    ax.add_feature(
        cartopy.feature.RIVERS, edgecolor="royalblue", zorder=10, alpha=0.5
    )
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
            ms=10,
        )

    # Add labels
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    return ax
