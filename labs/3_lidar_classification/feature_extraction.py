"""Extract features from the point cloud.

This script extracts features from a lidar point cloud. The features are
calculated for a set of scales and are saved in a dataframe. The dataframe
is saved in a pickle file.

Author:
    Léonard Seydoux

Date:
    June 2023
"""

import pandas as pd
import tqdm

import lidar

SCALES_IN_M = range(1, 15)
N_MIN_POINTS = 10
CORE_POINTS_DECIM = 25
FILEPATH_LAS = "./data/unlabeled/scene.las"
FILEPATH_FEATURES = "./data/features/features.pkl"

# Read again the entire dataset
dataset = lidar.read_las(FILEPATH_LAS)

# Core points (deterministic sampling for reproducibility)
core_points = dataset[::CORE_POINTS_DECIM, :]
print(f"Number of core points: {core_points.shape[0]:,}")

# Prepare dataframe for features
scale_x_features = [f"scale_{scale}_x" for scale in SCALES_IN_M]
scale_y_features = [f"scale_{scale}_y" for scale in SCALES_IN_M]
features = pd.DataFrame(columns=scale_x_features + scale_y_features)

for scale in tqdm.tqdm(SCALES_IN_M):

    # Calculate eigenvalues (in parallel)
    eigenvalues = lidar.calculate_eigenvalues(
        core_points, dataset, scale, N_MIN_POINTS
    )

    # Calculate tertiary coordinates
    x, y = lidar.calculate_barycentric_coordinates(eigenvalues)

    # Add to the dataframe
    features[f"scale_{scale}_x"] = x
    features[f"scale_{scale}_y"] = y

# Save the dataframe
features.to_pickle(FILEPATH_FEATURES)
