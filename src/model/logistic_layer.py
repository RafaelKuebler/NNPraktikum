
import time

import numpy as np

from util.activation_functions import Activation


class LogisticLayer:
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    deltas : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='sigmoid', isClassifierLayer=False):

        # Get activation function from string
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.activationDerivative = Activation.getDerivative(
                                    self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn+1, 1))
        self.input[0] = 1

        self.netOutputs = np.ndarray((nOut, 1))
        self.output = np.ndarray((nOut, 1))

        self.deltas = np.zeros((nOut, 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nIn + 1, nOut))-0.5
        else:
            assert(weights.shape == (nIn + 1, nOut))
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, input):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        input : ndarray
            a numpy array (nIn + 1,1) containing the input of the layer

        Change output
        -------
        output: ndarray
            a numpy array (nOut,1) containing the output of the layer
        """
        # numpy indexing: array[vertical][horizontal]
        #
        # [ 00, 01, 02 ]  |
        # [ 10, 11, 12 ] dim0
        # [ 20, 21, 22 ]  v
        #  --  dim1  -->


        # Visualization of input, weight, and output:
        # assume:
        # - 3 features X = [x1, x2, x3]
        # - 2 outputs  Y = [y1, y2]
        # - no bias

        # that means:
        # [y1]   [ w11 w12 w13 ]   [x1]
        # [  ] = [             ] * [x2]
        # [y2]   [ w21 w22 w23 ]   [x3]
        #
        # wji = weight from input i to output/unit j

        # therefore:
        #  <--  nIn  -->
        # [ w11 w12 w13 ] ^
        # [             ] nOut
        # [ w21 w22 w23 ] v
        #
        # np shape = (vertical, horizontal)
        # => weights.shape = (nOut, nIn)
        # => input.shape = (nIn, 1)
        # => output.shape = (nOut, 1)


        # Include bias => replace nIn with nIn+1
        #
        #                              [ 1]
        # [y1] = [ w10 w11 w12 w13 ] * [x1]
        # [y2]   [ w20 w21 w22 w23 ]   [x2]
        #                              [x3]
        # => weights.shape = (nOut, nIn + 1)
        # => input.shape = (nIn+1, 1)


        self.input = input
        self.netOutputs = np.dot(self.weights, self.input)
        self.output = self.activation(self.netOutputs)

        return self.output

    def computeDerivative(self, next_derivatives, next_weights):
        """
        Compute the derivatives (backward)

        Parameters
        ----------
        next_derivatives: ndarray
            a numpy array containing the derivatives from next layer
        next_weights : ndarray
            a numpy array containing the weights from next layer

        Change deltas
        -------
        deltas: ndarray
            a numpy array containing the partial derivatives on this layer
        """

        # Here the implementation of partial derivative calculation

        # In case of the output layer, next_weights is array of 1
        # and next_derivatives - the derivative of the error will be the errors
        # Please see the call of this method in LogisticRegression.
        # self.deltas = (self.outp *
        #              (1 - self.outp) *
        #               np.dot(next_derivatives, next_weights))

        # Or more general: output*(1-output) is the derivatives of sigmoid
        # (sigmoid_prime)
        # self.deltas = (Activation.sigmoid_prime(self.outp) *
        #                np.dot(next_derivatives, next_weights))

        # Or even more general: doesn't care which activation function is used
        # dado: derivative of activation function w.r.t the output
        # dado = self.activationDerivative(self.output)
        # self.deltas = (dado * np.dot(next_derivatives, next_weights))

        # Or you can explicitly calculate the derivatives for two cases
        # Page 40 Back-propagation slides

        if self.isClassifierLayer:
            self.deltas = self.activationPrime(self.netOutputs) \
                          * (next_derivatives - self.output)
        else:
            self.deltas = self.activationPrime(self.netOutputs) \
                          * np.dot(next_derivatives, next_weights)

        # Or you can have two computeDerivative methods, feel free to call
        # the other is computeOutputLayerDerivative or such.
        return self.deltas

    def updateWeights(self, learningRate):
        """
        Update the weights of the layer
        """
        # weight updating as gradient descent principle
        for neuron in range(0, self.nOut):
            self.weights[:, neuron] -= (learningRate *
                                        self.deltas[neuron] *
                                        self.input)
