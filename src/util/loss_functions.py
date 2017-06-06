# -*- coding: utf-8 -*-


"""
Loss functions.
"""

import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty


class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def errorString(self):
        pass

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'absolute'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return abs(target - output)


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'different'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return target - output


class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'mse'

    def calculateError(self, target, output):
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        # reuse SSE
        return SumSquaredError().calculateError(target, output) / len(target)


class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'sse'

    def calculateError(self, target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        # assume target and output are lists
        sum = 0
        n = len(target)
        for i in range(n):
            sum += (target[i] - output[i]) ** 2
        return sum



class BinaryCrossEntropyError(Error):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """
    def errorString(self):
        self.errorString = 'bce'

    def calculateError(self, target, output):
        # for one instance x: (taken from [ExerciseSlides, p. 3]
        # E_bce = - sum(k) of (t_k*log(o_k) + (1-t_k)*log(1-o_k))
        # => To calculate the error of one fixed instance,
        #    we would need target and output to contain the values of
        #    each output unit (=> be a list over output neurons)

        # actually, I have no idea what the difference between BCE and CE is

        def crossEntropyFixedK(target_k, output_k):
            """Compute one element of the sum"""
            return - (target_k * np.log(output_k) + (1-target_k) * np.log(1-output_k))

        if type(target) is list:
            sum = 0
            for k in range(len(target)):
                sum += crossEntropyFixedK(target[k], output[k])
            return sum
        else:
            return crossEntropyFixedK(target, output)



class CrossEntropyError(Error):
    """
    The Loss calculated by the more general Cross Entropy between two
    probabilistic distributions.
    """
    def errorString(self):
        self.errorString = 'crossentropy'

    def calculateError(self, target, output):
        # for one instance x: (taken from [2016Slides, p. 44])

        # E_ce = - sum(k) of (t_k*log(o_k) + (1-t_k)*log(1-o_k))
        # => To calculate the error of one fixed instance,
        #    we would need target and output to contain the values of
        #    each output unit (=> be a list over output neurons)

        # actually, I have no idea what the difference between BCE and CE is

        def crossEntropyFixedK(target_k, output_k):
            """Compute one element of the sum."""
            return - (target_k * np.log(output_k) + (1-target_k) * np.log(1-output_k))

        if type(target) is list:
            sum = 0
            for k in range(len(target)):
                sum += crossEntropyFixedK(target[k], output[k])
            return sum
        else:
            return crossEntropyFixedK(target, output)
