import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn import linear_model
from sklearn.svm import SVC

# -------------------------------------------------------
# create the data sets
# -------------------------------------------------------

# data set 1
X, y = make_blobs(n_samples=500, centers = 2, n_features = 2, random_state=2)
train_X_1 = X[0:400,]
train_y_1 = y[0:400]
test_X_1 = X[100:,]
test_y_1 = y[100:]

# data set 2
X, y = make_blobs(n_samples=50, centers = 2, n_features = 2, random_state=4)
train_X_2 = X[0:40,]
train_y_2 = y[0:40]
test_X_2 = X[40:,]
test_y_2 = y[40:]

# data set 3
X, y = make_blobs(n_samples=500, centers = 2, n_features = 2, random_state=0)
train_X_3 = X[0:400,]
train_y_3 = y[0:400]
test_X_3 = X[400:,]
test_y_3 = y[400:]

# data set 4
X, y = make_blobs(n_samples=50, centers = 2, n_features = 2, random_state=3)
train_X_4 = X[0:40,]
train_y_4 = y[0:40]
test_X_4 = X[40:,]
test_y_4 = y[40:]

# data set 5
X, y = make_blobs(n_samples=500, centers = 2, n_features = 2, random_state=5)
train_X_5 = X[0:400,]
train_y_5 = y[0:400]
test_X_5 = X[400:,]
test_y_5 = y[400:]

# -------------------------------------------------------
# our self-impl optimization and prediction function
# -------------------------------------------------------

# classify based on which side of the decision boundary X is on
# returns 0 or 1 instead of -1 or 1 for plotting function purposes
def svm_gd_hinge_predict(w, X):
    predictions = []

    for test_sample in X:
        # insert 1 for the intercept b
        x = np.insert(test_sample, 0, 1)
        pred = np.dot(w, x)
        classification = 1 if pred >= 0 else 0
        predictions += [classification]

    return np.array(predictions)


def svm_gd_hinge_fit(X, y, alpha=0.01, max_iter=10000, C=1.0):

    num_samples = X.shape[0]
    num_features = X.shape[1]
    learning_rate = alpha

    # initialize weights vector +1 for b
    w = np.array([0.0] * (num_features + 1))

    for i in range(max_iter):

        # if i % 10 == 0:
        #     print("iteration: " + str(i))

        total_hinge_loss = np.array([0.0] * (num_features + 1))

        for j in range(num_samples):

            x = np.insert(X[j], 0, 1)

            label = -1 if y[j] == 0 else 1

            # incur penalty if margin < 1.0
            if label * np.dot(w, x) < 1.0:
                total_hinge_loss += learning_rate * (w - C * label * x)
                w = w - learning_rate * (w + C * -label * x)
            else:
                total_hinge_loss += (learning_rate * w)

        # update weights
        avg_total_hinge_loss = total_hinge_loss / num_samples
        w = w - avg_total_hinge_loss

        # update learning rate
        learning_rate = learning_rate * 0.95

    return w

# -------------------------------------------------------
# global variables to plot learning curves
# -------------------------------------------------------

# Learning Curve Plots:

c_params = np.array([1.0,2.0,5.0,10.0])
print("c: " + str(c_params))

alphas = np.logspace(-5, 0, num=10)
print("learning rates: " + str(alphas))

num_iters = [int(x) for x in np.linspace(2, 100, num=75)]

alpha_colors_dict = dict()
alpha_colors_dict[alphas[0]] = "red"
alpha_colors_dict[alphas[1]] = "orange"
alpha_colors_dict[alphas[2]] = "yellow"
alpha_colors_dict[alphas[3]] = "green"
alpha_colors_dict[alphas[4]] = "blue"
alpha_colors_dict[alphas[5]] = "purple"
alpha_colors_dict[alphas[6]] = "magenta"
alpha_colors_dict[alphas[7]] = "grey"
alpha_colors_dict[alphas[8]] = "brown"
alpha_colors_dict[alphas[9]] = "cyan"

# -------------------------------------------------------
# function to plot learning curves and tune hyperparams
# -------------------------------------------------------

def plot_optimize_hyperparams(dataset_idx, colors_dict,
                              alpha_list, c_list, num_iters_list,
                              train_x, train_y, test_x, test_y,
                              classifier="our"):

    for c in c_list:

        title = str(classifier) + "_dataset_" + str(dataset_idx) +  "_c_" + str(c)

        print("--------current stage: " + str(title) + "----------")

        alpha_accuracy_dict = dict()

        for a in alpha_list:

            print("c: " + str(c) + ", a: " + str(a))

            a_num_iters_accuracy_dict = dict()

            for n in num_iters_list:

                pred = []

                if classifier == "sgd":
                    clf_sgd = linear_model.SGDClassifier(max_iter=n, tol=1e-3, random_state=0, alpha=a)
                    clf_sgd.fit(train_x, train_y)
                    pred = clf_sgd.predict(test_x)

                elif classifier == "svc":
                    clf_svc = SVC(C=c, kernel='linear', random_state=0, max_iter=n)
                    clf_svc.fit(train_x, train_y)
                    pred = clf_svc.predict(test_x)

                elif classifier == "our":
                    weights = svm_gd_hinge_fit(train_x, train_y, C=c, alpha=a, max_iter=n)
                    pred = svm_gd_hinge_predict(weights, test_x)

                num_predictions_correct = sum(pred == test_y)
                num_predictions_made = len(test_y) * 1.0
                a_num_iters_accuracy_dict[n] = num_predictions_correct / num_predictions_made

            alpha_accuracy_dict[a] = a_num_iters_accuracy_dict

        figure = plt.figure()
        plt.xlabel("num iterations")
        plt.ylabel("accuracy rate")
        plt.title(title)

        # plot each alpha line
        for alpha in alpha_accuracy_dict.keys():
            num_iters_accuracy_dict = alpha_accuracy_dict[alpha]
            x_values = num_iters_accuracy_dict.keys()
            y_values = num_iters_accuracy_dict.values()
            color = colors_dict[alpha]
            label = str(alpha)
            plt.plot(x_values, y_values, c=color, label=label)

        plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
        plt.tight_layout()
        figure.savefig("learning_curves/" + str(classifier) + "_classifier/" +
                       "dataset_" + str(dataset_idx) + "/" +
                       str(title) + ".png")
        print("----------checkpoint------------")

    return

# plots learning curves for specified list of datasets

datasets_to_plot = [1,2,3,4,5]

datasets_dict = dict()
datasets_dict[1] = {"x_train": train_X_1, "y_train": train_y_1, "x_test": test_X_1, "y_test": test_y_1}
datasets_dict[2] = {"x_train": train_X_2, "y_train": train_y_2, "x_test": test_X_2, "y_test": test_y_2}
datasets_dict[3] = {"x_train": train_X_3, "y_train": train_y_3, "x_test": test_X_3, "y_test": test_y_3}
datasets_dict[4] = {"x_train": train_X_4, "y_train": train_y_4, "x_test": test_X_4, "y_test": test_y_4}
datasets_dict[5] = {"x_train": train_X_5, "y_train": train_y_5, "x_test": test_X_5, "y_test": test_y_5}

for d in datasets_to_plot:

    training_x = datasets_dict[d]["x_train"]
    training_y = datasets_dict[d]["y_train"]
    testing_x = datasets_dict[d]["x_test"]
    testing_y = datasets_dict[d]["y_test"]

    plot_optimize_hyperparams(dataset_idx=str(d), colors_dict=alpha_colors_dict,
                              alpha_list=alphas, c_list=c_params, num_iters_list=num_iters,
                              train_x=training_x, train_y=training_y, test_x=testing_x, test_y=testing_y,
                              classifier="our")


# -------------------------------------------------------
# playground ... playing around and testing code
# -------------------------------------------------------

# clf_sgd = linear_model.SGDClassifier(max_iter=10000, tol=1e-3, random_state=0, alpha = 2.0)
# clf_sgd.fit(train_X_1, train_y_1)
# sgd_theta = np.insert(clf_sgd.coef_[0], 0, clf_sgd.intercept_[0]).tolist()

# clf_svc = SVC(C=1.0, kernel='linear', random_state=0, max_iter=10000)
# clf_svc.fit(train_X_1, train_y_1)
# svc_theta = np.concatenate([clf_svc.intercept_, clf_svc.coef_[0]])

# our_theta = svm_gd_hinge_fit(train_X_1, train_y_1, alpha=0.001, max_iter=30, C=1.0)
#
# pred = svm_gd_hinge_predict(our_theta, test_X_1)
#
# num_predictions_correct = sum(pred == test_y_1)
# num_predictions_made = len(test_y_1) * 1.0
# print("accuracy: " + str(num_predictions_correct / num_predictions_made))


# ------------------------------------------------------------

# clf_sgd.coef_ = np.array(our_weights[1:]).reshape(1,-1)
# clf_sgd.intercept_ = np.array([our_weights[0]]).reshape(1,-1)
#
# pred2  = clf_sgd.predict(test_X_1)
#
# num_predictions_correct_1 = sum(pred1  == test_y_1)
# num_predictions_correct_2 = sum(pred2  == test_y_1)
# num_predictions_made = len(test_y_1) * 1.0
# print("sgd prediction fn with our weights accuracy")
# print(num_predictions_correct_2 / num_predictions_made)
# print("sgd classifier weights accuracy")
# print(num_predictions_correct_1 / num_predictions_made)
#
#
# pred3 = svm_gd_hinge_predict(our_weights, test_X_1)
# num_predictions_correct_3 = sum(pred3  == test_y_1)
#
# print("our prediction fn with our weights accuracy")
# print(num_predictions_correct_3 / num_predictions_made)
#
# pred4  = svm_gd_hinge_predict(sgd_weights, test_X_1)
# num_predictions_correct_4 = sum(pred4  == test_y_1)
# print("our prediction fn with sgd weights accuracy")
# print(num_predictions_correct_4 / num_predictions_made)