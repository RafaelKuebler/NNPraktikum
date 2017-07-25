# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp
from numpy import divide
import numpy as np
from numpy import ones
from numpy import asarray


class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        # use e^x from numpy to avoid overflow
        return 1/(1+exp(-1.0*netOutput))

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)
        return Activation.sigmoid(netOutput) * (1 - Activation.sigmoid(netOutput))

    @staticmethod
    def tanh(netOutput):
        # return 2*Activation.sigmoid(2*netOutput)-1
        ex = exp(1.0*netOutput)
        exn = exp(-1.0*netOutput)
        return divide(ex-exn, ex+exn)  # element-wise division

    @staticmethod
    def tanhPrime(netOutput):
        # Here you have to code the derivative of tanh function
        # d/dx tanh(x) = 1 - tanh^2(x)
        return 1 - Activation.tanh(netOutput) ** 2

    @staticmethod
    def rectified(netOutput):
        return asarray([max(0.0, i) for i in netOutput])

    @staticmethod
    def rectifiedPrime(netOutput):
        # Here you have to code the derivative of rectified linear function
        if netOutput <= 0:
            return 0.
        else:
            return 1.

    @staticmethod
    def identity(netOutput):
        return netOutput

    @staticmethod
    def identityPrime(netOutput):
        # identityPrime = 1
        return ones(netOutput.size)

    @staticmethod
    def softmax(netOutput):
        # Here you have to code the softmax function

        # netOutputs = { net_i }
        # softmax: e^(net_j) / sum(k) of (e^net_k)
        # sum => scalar, / => element-wise division of e^net_j by sum(e^net_k)
        return np.exp(netOutput) / np.sum(np.exp(netOutput))

    @staticmethod
    def softmaxPrime(netOutput):
        # from 2016 slides (p. 20):
        # d phi(a_j) / d a_j = phi(a_j) * (1 - phi(a_j))
        # => compute softmax(x), elementwise application of formula
        softmax = Activation.softmax(netOutput)
        prime = softmax * (1 - softmax)
        return prime

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
        elif str == 'softmax':
            return Activation.softmaxPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
