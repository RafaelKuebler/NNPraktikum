# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp
from numpy import divide

class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        return 1 / (1 + exp(-netOutput))


    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)
        return netOutput * (1 - netOutput)

    @staticmethod
    def tanh(netOutput):
        # tanh(x) = 1 - 2 / (e^(2x) + 1)
        return 1 - 2 / (exp(2*netOutput) + 1)
        
    @staticmethod
    def tanhPrime(netOutput):
        # Here you have to code the derivative of tanh function
        # d/dx tanh(x) = 1 - tanh^2(x)
        return 1 - Activation.tanh(netOutput) ** 2

    @staticmethod
    def rectified(netOutput):
        return lambda x: max(0.0, x)

    @staticmethod
    def rectifiedPrime(netOutput):
        # Here you have to code the derivative of rectified linear function
        # my reasoning:
        # f(x<=0) = 0
        # f(x>0)  = 1
        if netOutput <= 0:
            return 0
        else:
            return 1

    @staticmethod
    def identity(netOutput):
        return lambda x: x

    @staticmethod
    def identityPrime(netOutput):
        # Here you have to code the derivative of identity function
        return 1

    @staticmethod
    def softmax(netOutput):
        # Here you have to code the softmax function
        # how should we do that? -> we need netOutput of other units as well
        pass

    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
