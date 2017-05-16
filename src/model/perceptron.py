# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from report import evaluator as rep

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
        '''
        self.trainingSet.label[i] -> label int
        self.trainingSet.input[i] -> input list len=784
        '''
        
        #add bias
        b = 0.01
        np.insert(self.weight, 0, b) ##add bias to weight vector
        
        
        for i in range(self.epochs):
            
            for j in range(len(self.trainingSet.input)):
                x = self.trainingSet.input[j] ##training inputvector x
                np.insert(x,0,1) ##add flat 1 for bias
                y = self.trainingSet.label[j] ##training labels           
                y_hat = Perceptron.classify(self, x) ##predicted labels
                
                sig = y - y_hat ##direction of update
                
                if y != y_hat: ##if classification is not correct, update weights
                    Perceptron.updateWeights(self, x, sig)
            
            if verbose:
                eva = rep.Evaluator()
                eva.printAccuracy(self.validationSet, 
                                  Perceptron.evaluate(self, self.validationSet.input))
            
        #pass



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
        return Perceptron.fire(self, testInstance)
        
        


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
        self.weight = (self.weight + self.learningRate * input * error)
        
        #pass
         
    
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))
