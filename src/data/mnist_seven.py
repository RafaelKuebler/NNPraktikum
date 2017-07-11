# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import shuffle
from data.data_set import DataSet


class MNISTSeven(object):
    """
    Small subset (5000 instances) of MNIST data to recognize the digit 7

    Parameters
    ----------
    dataPath : string
        Path to a CSV file with delimiter ',' and unint8 values.
    numTrain : int
        Number of training examples.
    numValid : int
        Number of validation examples.
    numTest : int
        Number of test examples.

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    """

    # dataPath = "data/mnist_seven.csv"

    def __init__(self, dataPath, 
                        numTrain=3000, 
                        numValid=1000,
                        numTest=1000,
                        oneHot=True,
                        targetDigit='7'):

        self.trainingSet = []
        self.validationSet = []
        self.testSet = []

        self.load(dataPath, numTrain, numValid, numTest, oneHot, targetDigit)

    def load(self, dataPath, numTrain, numValid, numTest, oneHot, targetDigit):
        """Load the data."""
        print("Loading data from " + dataPath + "...")

        data = np.genfromtxt(dataPath, delimiter=",", dtype="uint8")

        # The last numTest instances ALWAYS comprise the test set.
        train, test = data[:numTrain+numValid], data[numTrain+numValid:]
        shuffle(train)

        train, valid = train[:numTrain], train[numTrain:]

        self.trainingSet = DataSet(train, oneHot, targetDigit)
        self.validationSet = DataSet(valid, oneHot, targetDigit)
        self.testSet = DataSet(test, oneHot, targetDigit)

        print("Data loaded.")
