
import time

import numpy as np

from util.activation_functions import Activation
#from model.layer import Layer


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
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='softmax', isClassifierLayer=True):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn+1, 1))
        self.input[0] = 1
        self.output = np.ndarray((nOut, 1))
        self.delta = np.zeros((nOut, 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1))-0.5
        else:
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

        Returns
        -------
        ndarray :
            a numpy array (nOut, 1) containing the output of the layer
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
        # [y1]   [ w11 w21 w31 ]   [x1]
        # [  ] = [             ] * [x2]
        # [y2]   [ w12 w22 w32 ]   [x3]
        #
        # wij = weight from input i to output j

        # therefore:
        #  <--  nIn  -->
        # [ w11 w21 w31 ] ^
        # [             ] nOut
        # [ w12 w22 w32 ] v
        #
        # np shape = (vertical, horizontal)
        # => weights.shape = (nOut, nIn)
        # => input.shape = (nIn, 1)
        # => output.shape = (nOut, 1)


        # Include bias => replace nIn with nIn+1
        #
        #                              [ 1]
        # [y1] = [ w01 w11 w21 w31 ] * [x1]
        # [y2]   [ w02 w12 w22 w32 ]   [x2]
        #                              [x3]
        # => weights.shape = (nOut, nIn + 1)
        # => input.shape = (nIn+1, 1)

        # add bias
        self.input[1:, 0] = input
        netOutputs = np.dot(self.weights, self.input)
        self.output = self.activation(netOutputs)

        return self.output

    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (back)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """
        pass

    def updateWeights(self):
        """
        Update the weights of the layer
        """
        pass
