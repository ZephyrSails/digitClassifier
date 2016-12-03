import pickle
import sklearn
from sklearn import svm, metrics # this is an example of using SVM
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from mnist import load_mnist
from helper import *


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def build_classifier(images, labels, _gamma=0.001, _C=1.0):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = svm.SVC(gamma=_gamma, C=_C)
    classifier.fit(images, labels)
    return classifier


def classify(images, classifier):
    #runs the classifier on a set of images.
    clsres = classifier.predict(images)
    return clsres


def test(training_set, training_labels, testing_set, testing_labels, _gamma=0.001, _C=1.0):
    classifier = build_classifier(training_set, training_labels, _gamma, _C)
    save_classifier(classifier, training_set, training_labels)
    # classifier = pickle.load(open('classifier_1.p'))
    predicted = classify(testing_set, classifier)

    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(testing_labels, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(testing_labels, predicted))

    # for i, item in enumerate(testing_labels):
    #     print predicted[i], '\t', testing_labels[i]

    error = error_measure(predicted, testing_labels)
    print 'error rate %f' % error
    return error


def cmpParams(X, y):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)

    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)

    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))

    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    fig.savefig('hotColormap')
    plt.show()


if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(0, 10), path='.')
    # preprocessing
    images = preprocess(images)

    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA
    training_set = images[:]
    training_labels = labels[:]
    # testing_set = images[-100:]
    # testing_labels = labels[-100:]
    # cmpParams(training_set, training_labels)

    classifier = build_classifier(training_set, training_labels, _gamma=0.01, _C=100.0)
    save_classifier(classifier, training_set, training_labels)

    # test(training_set, training_labels, testing_set, testing_labels)
    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
