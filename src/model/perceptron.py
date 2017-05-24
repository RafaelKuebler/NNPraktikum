# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import time

# used to visualize the weights (requires Python Imaging Library)
from PIL import Image 


from util.activation_functions import Activation
from model.classifier import Classifier
from report.evaluator import Evaluator
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
        """
        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        start_time = time.time()
        input_size = len(self.trainingSet.input)
        evaluator = Evaluator()
        error_calc = DifferentError()

        for epoch in range(self.epochs):
            if verbose:
                evaluator.printAccuracy(self.validationSet,
                                        self.evaluate(self.validationSet.input))

            if _do_weight_visualization:
                self.visualizeWeights(epoch)

            for index in range(input_size):
                current_input = self.trainingSet.input[index]
                current_label = float(self.trainingSet.label[index])

                classification_result = float(self.classify(current_input))

                if current_label != classification_result:
                    # error = label - result
                    error = error_calc.calculateError(current_label, classification_result)
                    self.updateWeights(current_input, error);

        end_time = time.time()
        logging.debug("{} {}".format("Elapsed time:", end_time - start_time))

        if _do_weight_visualization:
            self.visualizeWeights(self.epochs)

    def classify(self, testInstance):
        return self.fire(testInstance)

    def evaluate(self, test=None):
        if test is None:
            test = self.testSet.input
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        self.weight += self.learningRate*error*input
         
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))

    def visualizeWeights(self, epoch=-1):
        img = Image.new("L", (28, 28))
        data = map(lambda w: 255*w, self.weight)
        img.putdata(data)
        img.save("weights_epoch_{}.png".format(epoch))

