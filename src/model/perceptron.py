# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
from PIL import Image

from util.activation_functions import Activation
from model.classifier import Classifier


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test,
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Write your code to train the perceptron here

        if verbose:
            self.logAccuracy(-1)

        for epoch in range(self.epochs):

            # run the training epoch on training set
            for i in range(len(self.trainingSet.input)):
                instance = self.trainingSet.input[i]
                target = float(self.trainingSet.label[i])

                # float(bool) == 1. if true | 0. if false
                out = float(self.classify(instance))

                error = target - out

                self.updateWeights(instance, error)

            if verbose:
                self.logAccuracy(epoch)
                self.visualizeWeights(epoch)


    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Write your code to do the classification on an input image
        return self.fire(testInstance)


    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))


    def updateWeights(self, input, error):
        # Write your code to update the weights of the perceptron here
        # w_i = w_i + eta * (t - o) * x
        for i in range(len(self.weight)):
            update = self.learningRate * error * input[i]
            self.weight[i] += update


    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))


    def logAccuracy(self, epoch):
        # calculate accuracy and false positive rate on validation set
        hits, falsePos = 0, 0
        setSize = len(self.validationSet.input)

        for i in range(setSize):
            instance = self.validationSet.input[i]
            target = self.validationSet.label[i]

            out = self.classify(instance)

            if out == target:
                hits += 1
            elif out:  # if false pos
                falsePos += 1

        accuracy = float(hits) / setSize
        falsePosRate = float(falsePos) / setSize

        # print newline (and thereby ensure flush)
        logging.debug("Epoch {}: Acc: {}% | FPR: {}%".format(
            epoch, accuracy * 100, falsePosRate * 100))


    def visualizeWeights(self, epoch=-1):
        img = Image.new("L", (28, 28))
        data = map(lambda w: 255*w, self.weight)
        img.putdata(data)
        img.save("epoch_{}.png".format(epoch))


