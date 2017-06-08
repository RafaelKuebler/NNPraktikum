# -*- coding: utf-8 -*-

import sys
import os
import logging

import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

        self.accuracyValid = np.zeros(self.epochs + 1)
        self.accuracyTest = np.zeros(self.epochs + 1)
        self.error = np.zeros(self.epochs)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        use_sse = False

        from util.loss_functions import DifferentError, SumSquaredError
        lossDE = DifferentError()
        lossSSE = SumSquaredError()

        self.accuracyValid = np.zeros(self.epochs + 1)
        self.accuracyTest = np.zeros(self.epochs + 1)
        self.error = np.zeros(self.epochs + 1)

        if verbose:
            self.trackAccuracy(0)

        # make epoch go 1..epochs (not 0..epochs-1)
        for epoch in range(self.epochs+1)[1:]:
            gradient = np.zeros(self.weight.shape[0])

            sumError = 0

            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):

                output = self.fire(input)
                target = label

                if use_sse:

                    # use SSE
                    error = lossSSE.calculateError(target, output)

                    # gradient for SSE+sigmoid:
                    # dE/dw = - sum(x in X) of [ (t_x - o_x) * o_x * (1 - o_x) * x ]
                    gradient += - (target - output) * output * (1 - output) * input

                else:
                    # classical approach with different error
                    error = lossDE.calculateError(label, output)
                    gradient += - error * input

                sumError += abs(error)
                self.updateWeights(gradient)

            if verbose:
                self.trackAccuracy(epoch)
                self.trackError(epoch, sumError)

        if verbose:
            self.plotAccuracy()
            self.plotError()


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
        return self.fire(testInstance) >= 0.5

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

    def updateWeights(self, gradient):
        self.weight += - self.learningRate * gradient

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))


    def trackAccuracy(self, epoch, log=True):
        """Calculates the accuracy, stores it for the given epoch
        and returns it.
        If log is true, the accuracy is logged.
        """

        validAcc = metrics.accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
        testAcc = metrics.accuracy_score(self.testSet.label,
                                         self.evaluate(self.testSet))

        self.accuracyValid[epoch] = validAcc
        self.accuracyTest[epoch] = testAcc

        if log:
            logging.info("Epoch {}: Accuracy = {}% (on test: {}%)"
                         .format(epoch, validAcc * 100, testAcc * 100))

        return validAcc

    def plotAccuracy(self):
        epoch_axis = np.linspace(0, self.epochs, self.epochs + 1, endpoint=True)
        plt.plot(epoch_axis, self.accuracyValid, label="Accuracy (validation set)")
        plt.plot(epoch_axis, self.accuracyTest, label="Accuracy (test set)")
        plt.legend(loc="lower right")

        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig("plots/accuracy.png", dpi=100)

        plt.show()


    def trackError(self, epoch, error):
        self.error[epoch] = error

    def plotError(self):
        epoch_axis = np.linspace(1, self.epochs, self.epochs, endpoint=True)
        plt.plot(epoch_axis, self.error[1:], label="Error")
        plt.legend(loc="lower right")

        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig("plots/error.png", dpi=100)

        plt.show()


