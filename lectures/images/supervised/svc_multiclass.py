import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = ["C0", "C1", "C2"]
colors_w = [colors[0]] + ["w"] + [colors[1]] + ["w"] + [colors[2]]
cm = LinearSegmentedColormap.from_list("my_list", colors_w, N=256)
cm_bright = ListedColormap(colors)

# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 10  # SVM regularization parameter
models = (
    svm.SVC(kernel="linear", C=C),
    svm.SVC(kernel="rbf", gamma=0.7, C=C),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
)
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = (
    "Linear kernel",
    "RBF kernel",
    "Cubic kernel",
)

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(1, 3, figsize=(6, 1.5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]

for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=cm,
        alpha=0.5,
        ax=ax,
        xlabel="Sepal length ($x_1$)",
        ylabel="Sepal width ($x_2$)",
        grid_resolution=200,
        levels=30,
    )
    ax.scatter(X0, X1, c=y, cmap=cm_bright, s=10, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

    # Evaluate the model
    accuracy = clf.score(X, y)
    ax.text(
        0.5,
        0.06,
        f"accuracy: {accuracy:.2f}",
        transform=ax.transAxes,
        ha="center",
        name="Cascadia Code",
        fontsize="small",
    )


plt.savefig("svc_multiclass.svg")
