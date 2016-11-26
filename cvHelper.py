import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
import itertools
import random
import sys
import classifier_1


def cv(k, method, directory='.'):
    """
    :type directory: String
    :type k: Int
    :rtype: None
    """
    images, labels = load_mnist(digits=range(0, 10), path=directory)

    images = preprocess(images)
    groupedPairs = cvDivide(images, labels, k)

    for i in xrange(k):
        # trainingImages = np.concatenate(groupedImges[:i], groupedImges[:i]
        testingPairs = zip(*groupedPairs[i])
        trainingPairs = zip(*list(itertools.chain(*groupedPairs[:i])) + list(itertools.chain(*groupedPairs[i+1:])))


        testingImages = np.array(testingPairs[0])
        testingLabels = np.array(testingPairs[1])
        trainingImages = np.array(trainingPairs[0])
        trainingLabels = np.array(trainingPairs[1])
        # testingImages = testingPairs[0]
        # testingLabels = testingPairs[1]
        # trainingImages = trainingPairs[0]
        # trainingLabels = trainingPairs[1]

        print np.shape(testingImages), np.shape(testingLabels), np.shape(trainingImages), np.shape(trainingLabels)
        if method == 0:
            classifier_1.test(trainingImages, trainingLabels, testingImages, testingLabels)

        # print np.shape(testingImages), np.shape(testingLabels), np.shape(trainingImages), np.shape(trainingLabels)


def cvDivide(datas, labels, k):
    """
    :type images:           List of images
    :type labels:           List of labels
    :type k:                K fold cross validation
    :rtype groupedPairs:    K Lists of evenly divided grouped data-lable pairs
    """
    pairs = zip(datas, labels)
    random.shuffle(pairs)

    return list(chunks(pairs, k))


# def eva(classfied, actualLabels):
#     acc =


# evenly divid a list to n part
def chunks(lst, n):
    l = [(float(len(lst))/n) * i for i in xrange(n+1)]
    for f, t in zip(l, l[1:]):
        yield lst[int(f):int(t)]


def preprocess(images):
    #this function is suggested to help build your classifier.
    #You might want to do something with the images before
    #handing them to the classifier. Right now it does nothing.
    return [i.flatten() for i in images]


if __name__ == '__main__':
    cv(10, int(sys.argv[1]), '.')
