"""Illustration of overfitting"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Parameters
plt.rcParams["figure.figsize"] = 2, 2
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["lines.linewidth"] = 1


np.random.seed(2)
x = np.linspace(-1, 1, 20)
y = x**3 + np.random.randn(20) * 0.2

# Plot
fig, ax = plt.subplots(1, 3, figsize=(7, 2), sharex=True, sharey=True)

# Linear regression
model = LinearRegression()
model.fit(x[:, None], y)
x_pred = np.linspace(-1, 1, 100)
y_pred = model.predict(x_pred[:, None])
train = mean_squared_error(y, model.predict(x[:, None]))
ax[0].plot(x_pred, y_pred, color="C1")
ax[0].text(
    0,
    0.9,
    f"Loss: {train:.2f}",
    name="Cascadia Code",
    size="x-small",
    ha="center",
)

# Polynomial regression
model = SVR(kernel="poly", degree=3)
model.fit(x[:, None], y)
x_pred = np.linspace(-1, 1, 100)
y_pred = model.predict(x_pred[:, None])
ax[1].plot(x_pred, y_pred, color="C1")
train = mean_squared_error(y, model.predict(x[:, None]))
ax[1].text(
    0,
    0.9,
    f"Loss: {train:.2f}",
    name="Cascadia Code",
    size="x-small",
    ha="center",
)

# Random forest
model = RandomForestRegressor(n_estimators=200)
model.fit(x[:, None], y)
x_pred = np.linspace(-1, 1, 100)
y_pred = model.predict(x_pred[:, None])
train = mean_squared_error(y, model.predict(x[:, None]))
ax[2].plot(x_pred, y_pred, color="C1")
ax[2].text(
    0,
    0.9,
    f"Loss: {train:.2f}",
    name="Cascadia Code",
    size="x-small",
    ha="center",
)

for a in ax:
    a.scatter(x, y, s=5)
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel(r"$x$")
    a.set_aspect("equal")
ax[0].set_ylabel(r"$y$")

ax[0].set_title("Underfitting")
ax[1].set_title("Just right")
ax[2].set_title("Overfitting")

# Save figure
plt.savefig("overfitting.svg", dpi=300)
