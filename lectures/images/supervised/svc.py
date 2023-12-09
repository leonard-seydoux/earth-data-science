# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Parameters
model = SVC(kernel="linear", C=0.025, random_state=42)

# Generate data
X, y = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Extreme points
x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7

# just plot the dataset first
colors = ["C0", "C1"]
colors_w = [colors[0]] + ["w"] + [colors[1]]
cm = LinearSegmentedColormap.from_list("my_list", colors_w, N=256)
cm_bright = ListedColormap(colors)

# Prepare plot
fig, ax = plt.subplots(figsize=(2, 1.5))

# Plot the testing points
img = ax.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap=cm_bright,
    alpha=0.6,
    edgecolors="face",
    s=17,
)

# add colorbar
cbar = fig.colorbar(img, ax=ax)
cbar.set_ticks([0.25, 0.75])
cbar.set_ticklabels(["$y=0$", "$y=1$"], color="0.5")
cbar.ax.tick_params(size=0, labelsize="small")


# Plot the training points
ax.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap=cm_bright,
    edgecolors="k",
    s=17,
)

# Labels
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

# Train
clf = make_pipeline(StandardScaler(), model)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=cm,
    alpha=0.6,
    ax=ax,
    eps=0.7,
    zorder=0,
    grid_resolution=200,
    levels=10,
)

# Draw boundary
a = model.coef_[0]
b = model.intercept_[0]
ax.axline((0, b), slope=-a[0] / a[1], color="k", lw=1, zorder=1)

# Save
plt.savefig("svc.svg")
