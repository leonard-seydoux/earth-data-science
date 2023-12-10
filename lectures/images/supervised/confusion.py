import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# import the confusion matrix function
from sklearn.metrics import confusion_matrix

colors = ["C0", "C1", "C2"]
colors_w = [colors[0]] + ["w"] + [colors[1]] + ["w"] + [colors[2]]
cm = LinearSegmentedColormap.from_list("my_list", colors_w, N=256)
cm_bright = ListedColormap(colors)

# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
names = iris.target_names

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
    # Confusion matrix
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)

    # Plot the confusion matrix
    im = ax.imshow(cm, interpolation="nearest", cmap="RdPu", vmin=0, vmax=70)
    ax.set(xticks=[0, 1, 2], yticks=[0, 1, 2], xlabel="Predicted")
    ax.tick_params(size=0, labelsize="small")

    # Show the number of observations in each box
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="k",
                size="small",
                name="Cascadia Code",
            )

sub[0].set_ylabel("True")


plt.savefig("svc_confusion.svg")
