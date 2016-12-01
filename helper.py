import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess(images):
    #this function is suggested to help build your classifier.
    #You might want to do something with the images before
    #handing them to the classifier. Right now it does nothing.
    return [i.flatten() for i in images]


def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))


##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'w'))
    pickle.dump(training_set, open('training_set.p', 'w'))
    pickle.dump(training_labels, open('training_labels.p', 'w'))

def outputImage(predicted, expected, testing_images):

    # create 10 folders
    for i in range(10):
        # directory = os.path.join('misclassified', str(i))
        directory = str(i)
        if not os.path.exists(directory):
            os.mkdir(directory)

    # if a image has other labels but is misclassified as '1', then put the image into file '1'
    for i, label in enumerate(expected):
        if label != predicted[i]:
            # put the relevent image into folder 'predicted[i]'
            # pathName = os.path.join('misclassified', str(predicted[i]), 'label_' + str(label) + '_' + str(i) + '.png')
            pathName = os.path.join(str(predicted[i]), 'label_' + str(label) + '_' + str(i) + '.png')
            plt.imsave(pathName, testing_images[i], cmap = 'gray')