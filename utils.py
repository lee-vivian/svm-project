import numpy as np
import matplotlib.pyplot as plt

data_colors = [(1, 0, 0), (0, 0, 1)]

# we use the plot_data and plot_decision_function methods here to help with plotting the graphs of our 
# datasets and the identified decision boundaries in our report
# this file is publicly available and attributed to the GitHub repository mentioned below 
# https://github.com/spmallick/learnopencv/blob/master/SVM-using-Python/utils.py

#Reads points from a file and returns an array of those points
def read_points_file(filename):
    pts = []
    with open(filename, "r") as f:
        for pt in f:
            pt = pt.strip("\n").split()
            pts.append([float(pt[0]), float(pt[1])])
    return pts


def read_data(class_0_file, class_1_file):
    pts_0 = read_points_file(class_0_file)
    pts_1 = read_points_file(class_1_file)

    x = pts_0 + pts_1
    labels = [0] * len(pts_0) + [1] * len(pts_1)
    x = np.array(x)
    return (x, labels)


def plot_data(X_train, y_train, X_test, y_test):
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    colors = get_colors(y)
    colors_train = get_colors(y_train)
    colors_test = get_colors(y_test)

    plt.figure(figsize=(12, 4), dpi=150)

    # Plot all data plot
    plt.subplot(131)
    plt.axis('equal')
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=10, edgecolors=colors)
    plt.title("All Samples (100%)")

    # training data plot
    plt.subplot(132)
    plt.axis('equal')
    # plt.axis('off')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors_train, s=10, edgecolors=colors_train)
    plt.title("Training Data (80%)")

    # testing data plot
    plt.subplot(133)
    plt.axis('equal')
    # plt.axis('off')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors_test, s=10, edgecolors=colors_test)
    plt.title("Test Data (20%)")
    plt.tight_layout()

    plt.show()


def get_colors(y):
    return [data_colors[item] for item in y]


def plot_decision_function(X_train, y_train, X_test, y_test, clf):
    plt.figure(figsize=(8, 4), dpi=150)
    plt.subplot(121)
    plt.title("Training data")
    plot_decision_function_helper(X_train, y_train, clf)
    plt.subplot(122)
    plt.title("Test data")
    plot_decision_function_helper(X_test, y_test, clf, True)
    plt.show()


def plot_decision_function_helper(X, y, clf, show_only_decision_function=False):
    colors = get_colors(y)
    plt.axis('equal')
    plt.tight_layout()

    plt.scatter(X[:, 0], X[:, 1], c=colors, s=10, edgecolors=colors)
    ax = plt.gca()
    xlimit = ax.get_xlim()
    ylimit = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlimit[0], xlimit[1], 30)
    yy = np.linspace(ylimit[0], ylimit[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    if show_only_decision_function:
        # Plot decision boundary
        ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5,
                   linestyles=['-'])
    else:
        # Plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])



