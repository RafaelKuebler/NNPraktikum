# -*- coding: utf-8 -*-

import sys
import logging
import time

import numpy as np
from PIL import Image

from model.classifier import Classifier
from report.evaluator import Evaluator
from util.activation_functions import Activation
from util.loss_functions import DifferentError

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


# change to true to generate weight visualizing images
_do_weight_visualization = False


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

        # change to true to generate weight visualizing images
        do_weight_visualization = verbose and False

        start_time = time.time()
        evaluator = Evaluator()
        differentError = DifferentError()

        if verbose:
            self.logPerformance(evaluator, -1)

        for epoch in range(self.epochs):

            # run the training epoch on training set
            for index in range(len(self.trainingSet.input)):
                instance = self.trainingSet.input[index]
                target = float(self.trainingSet.label[index])

                # float(bool) == 1. if true | 0. if false
                output = float(self.classify(instance))

                if output != target:
                    #error = target - output
                    error = differentError.calculateError(target, output)
                    self.updateWeights(instance, error)

            if verbose:
                self.logPerformance(evaluator, epoch)
                if _do_weight_visualization:
                    self.visualizeWeights(epoch)

        end_time = time.time()
        logging.debug("Elapsed time: {}".format(end_time - start_time))


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
        # w_i = w_i + eta * (t - o) * x
        self.weight += self.learningRate * error * input


    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))


    def logPerformance(self, evaluator, epoch):
        sys.stdout.write("Epoch {:>2}: ".format(epoch))
        evaluator.printAccuracy(self.validationSet,
                                self.evaluate(self.validationSet.input))


    def visualizeWeights(self, epoch=-1):
        img = Image.new("L", (28, 28))
        data = map(lambda w: 255*w, self.weight)
        img.putdata(data)
        img.save("weights_epoch_{}.png".format(epoch))


