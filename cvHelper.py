import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
import itertools
import random
import sys
import classifier_1
import classifier_2
import os


def cv(k, method, directory='.'):
    """
    :type directory: String
    :type k: Int
    :rtype: None
    """
    images, labels = load_mnist(digits=range(0, 10), path=directory)

    images = preprocess(images)
    groupedPairs = cvDivide(images, labels, k)
    erros_1 = []
    erros_2 = []

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
            erros_1.append(classifier_1.test(trainingImages, trainingLabels, testingImages, testingLabels))
        elif method == 1:
            erros_2.append(classifier_2.test(trainingImages, trainingLabels, testingImages, testingLabels))

    print erros_1, erros_2

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

def outputImage(predicted, expected, testing_images):
    # create 10 folders

    for i in range(10):
        directory = str(i)
    
        if not os.path.exists(directory):
            os.mkdir(directory)

    # if a image has other labels but is misclassified as '1', then put the image into file '1'
    for i, label in enumerate(expected):
        if label != predicted[i]:
            # put the relevent image into folder 'predicted[i]'
            plt.imsave(str(label) + str(i) + '.png', testing_images[i], cmap = 'gray')


if __name__ == '__main__':
    cv(10, int(sys.argv[1]), '.')
