import pickle
import sklearn
from sklearn import svm, metrics# this is an example of using SVM
import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from helper import *

def build_classifier(images, labels, _k = 3):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = KNeighborsClassifier(n_neighbors=_k)
    classifier.fit(images, labels)
    return classifier


def classify(images, classifier):
    #runs the classifier on a set of images.
    clsres = classifier.predict(images)
    return clsres


def test(training_set, training_labels, testing_set, testing_labels):
    classifier = build_classifier(training_set, training_labels)
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

def test(training_set, training_labels, testing_set, testing_labels, _k = 3):
    classifier = build_classifier(training_set, training_labels, _k)
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

    k_range = np.array([1, 2, 4, 8, 16, 64, 128])
    param_grid = dict(n_neighbors = k_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv)

    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    print grid.cv_results_

    sorted(grid.cv_results_.keys())


if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(0, 10), path='.')
    # preprocessing
    images = preprocess(images)

    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA
    training_set = images[0:6000]
    training_labels = labels[0:6000]
    # testing_set = images[-100:]
    # testing_labels = labels[-100:]
    cmpParams(training_set, training_labels)

    # test(training_set, training_labels, testing_set, testing_labels)
    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
