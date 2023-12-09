"""Illustration of overfitting"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


# Parameters
plt.rcParams["figure.figsize"] = 2, 2
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["lines.linewidth"] = 1


# Generate data
np.random.seed(0)
x = np.linspace(-1, 1, 75)
y = x**3 + np.random.randn(75) * 0.2

# Split data
x, x_test, y, y_test = train_test_split(x, y, test_size=0.3)

# Plot
fig, ax = plt.subplots(1, 3, figsize=(7, 2), sharex=True, sharey=True)

# Linear regression
model = MLPRegressor(random_state=42)
model.fit(x[:, None], y)
x_pred = np.linspace(-1, 1, 100)
y_pred = model.predict(x_pred[:, None])
train = mean_squared_error(y, model.predict(x[:, None]))
test = mean_squared_error(y_test, model.predict(x_test[:, None]))
ax[0].plot(x_pred, y_pred, color="C1")
ax[0].text(
    0,
    0.98,
    f"Train loss: {train:.2f}",
    name="Cascadia Code",
    size="x-small",
    ha="center",
)
ax[0].text(
    0,
    0.83,
    f"Test loss: {test:.2f}",
    name="Cascadia Code",
    size="x-small",
    ha="center",
)

# Polynomial regression
model = MLPRegressor(
    hidden_layer_sizes=(100, 100, 300, 100), alpha=0.01, random_state=42
)
model.fit(x[:, None], y)
x_pred = np.linspace(-1, 1, 100)
y_pred = model.predict(x_pred[:, None])
train = mean_squared_error(y, model.predict(x[:, None]))
test = mean_squared_error(y_test, model.predict(x_test[:, None]))
ax[1].plot(x_pred, y_pred, color="C1")
ax[1].text(
    0,
    0.98,
    f"Train loss: {train:.2f}",
    name="Cascadia Code",
    size="x-small",
    ha="center",
)
ax[1].text(
    0,
    0.83,
    f"Test loss: {test:.2f}",
    name="Cascadia Code",
    size="x-small",
    ha="center",
)

# Random forest
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(x[:, None], y)
x_pred = np.linspace(-1, 1, 100)
y_pred = model.predict(x_pred[:, None])
train = mean_squared_error(y, model.predict(x[:, None]))
test = mean_squared_error(y_test, model.predict(x_test[:, None]))
ax[2].plot(x_pred, y_pred, color="C1")
ax[2].text(
    0,
    0.98,
    f"Train loss: {train:.2f}",
    name="Cascadia Code",
    size="x-small",
    ha="center",
)
ax[2].text(
    0,
    0.83,
    f"Test loss: {test:.2f}",
    name="Cascadia Code",
    size="x-small",
    ha="center",
)

for a in ax:
    a.scatter(x, y, s=5)
    a.scatter(x_test, y_test, s=5, color="C0", alpha=0.2)
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel(r"$x$")
    a.set_aspect("equal")
ax[0].set_ylabel(r"$y$")

ax[0].set_title("Too simple")
ax[1].set_title("Just right")
ax[2].set_title("Too complex")

# Save figure
plt.savefig("regularization.svg", dpi=300)
