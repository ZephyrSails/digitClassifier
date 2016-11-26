import pickle
import sklearn
from sklearn import svm, metrics# this is an example of using SVM
import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
from helper import *


def build_classifier(images, labels):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = svm.SVC(gamma=0.001)
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


if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(0, 10), path='.')
    # preprocessing
    images = preprocess(images)

    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA
    training_set = images[0:1000]
    training_labels = labels[0:1000]
    testing_set = images[-100:]
    testing_labels = labels[-100:]

    test(training_set, training_labels, testing_set, testing_labels)
    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
