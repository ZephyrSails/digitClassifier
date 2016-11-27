# import packages (including scikit-learn packages)
# a
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier # Use this function for adaboosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn import metrics


def boosting_A(training_set, training_labels, testing_set, testing_labels):
	'''
	Input Parameters:
		- training_set: a 2-D numpy array that contains training examples (size: the number of training examples X the number of attributes)
		  (NOTE: If a training example is 10x10 images, the number of attributes will be 100. You need to reshape your training example)
		- training_labels: a 1-D numpy array that labels of training examples (size: the number of training examples)
		- testing_set: a 2-D numpy array that contains testing examples  (size: the number of testing examples X the number of attributes)
		- testing_labels: a 1-D numpy array that labels of testing examples (size: the number of testing examples)

	Returns:
		- predicted_labels: a 1-D numpy array that contains the labels predicted by the classifier. Labels in this array should be sorted in the same order as testing_labels
		- confusion_matrix: a 2-D numpy array of confusion matrix (size: the number of classes X the number of classes)
	'''

    # Build boosting algorithm for question 6-A
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
	bdt.fit(X, y)

	bdt.fit(X, y)

	# plot_colors = "br"
	# plot_step = 0.02
	# class_names = "AB"
	#
	# plt.figure(figsize=(10, 5))
	#
	# # Plot the decision boundaries
	# plt.subplot(121)
	# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	# xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
	#                      np.arange(y_min, y_max, plot_step))

	Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
	confusion_matrix = metrics.confusion_matrix(testing_labels, Z)

	Z = Z.reshape(xx.shape)
	# cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
	# plt.axis("tight")
	#
	# # Plot the training points
	# for i, n, c in zip(range(2), class_names, plot_colors):
	#     idx = np.where(y == i)
	#     plt.scatter(X[idx, 0], X[idx, 1],
	#                 c=c, cmap=plt.cm.Paired,
	#                 label="Class %s" % n)
	# plt.xlim(x_min, x_max)
	# plt.ylim(y_min, y_max)
	# plt.legend(loc='upper right')
	# plt.xlabel('x')
	# plt.ylabel('y')
	# plt.title('Decision Boundary')
	#
	# # Plot the two-class decision scores
	# twoclass_output = bdt.decision_function(X)
	# plot_range = (twoclass_output.min(), twoclass_output.max())
	# plt.subplot(122)
	# for i, n, c in zip(range(2), class_names, plot_colors):
	#     plt.hist(twoclass_output[y == i],
	#              bins=10,
	#              range=plot_range,
	#              facecolor=c,
	#              label='Class %s' % n,
	#              alpha=.5)
	# x1, x2, y1, y2 = plt.axis()
	# plt.axis((x1, x2, y1, y2 * 1.2))
	# plt.legend(loc='upper right')
	# plt.ylabel('Samples')
	# plt.xlabel('Score')
	# plt.title('Decision Scores')
	#
	# plt.tight_layout()
	# plt.subplots_adjust(wspace=0.35)
	# plt.show()


    return predicted_labels, confusion_matrix

def boosting_B(training_set, training_labels, testing_set, testing_labels):
	'''
	Input Parameters:
		- training_set: a 2-D numpy array that contains training examples (size: the number of training examples X the number of attributes)
		(NOTE: If a training example is 10x10 images, the number of attributes will be 100. You need to reshape your training example)
		- training_labels: a 1-D numpy array that labels of training examples (size: the number of training examples)
		- testing_set: a 2-D numpy array that contains testing examples  (size: the number of testing examples X the number of attributes)
		- testing_labels: a 1-D numpy array that labels of testing examples (size: the number of testing examples)

	Returns:
		- predicted_labels: a 1-D numpy array that contains the labels predicted by the classifier. Labels in this array should be sorted in the same order as testing_labels
		- confusion_matrix: a 2-D numpy array of confusion matrix (size: the number of classes X the number of classes)
	'''
    # Build boosting algorithm for question 6-B


    return predicted_labels, confusion_matrix

def main():
    """
    This function runs boosting_A() and boosting_B() for problem 7.
    Load data set and perform adaboosting using boosting_A() and boosting_B()
    """
	images, labels = load_mnist(digits=range(0, 10), path='.')
    # preprocessing
    images = preprocess(images)

    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA
    training_set = images[:6000]
    training_labels = labels[:6000]

	boosting_A(training_set, training_labels, testing_set, testing_labels)

	print("Confusion matrix:\n%s" % )



if __name__ == '__main__':
    main()
