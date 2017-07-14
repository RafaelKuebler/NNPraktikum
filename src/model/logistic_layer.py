
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
            a numpy array (nIn,1) containing the input of the layer

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

        # add bias
        self.input[1:, 0] = input[:, 0]
        netOutputs = np.dot(self.weights, self.input)
        self.output = self.activation(netOutputs)

        return self.output

    def computeDerivative(self, target, nextDerivatives, nextWeights):
        """
        Compute the derivatives (back)

        Parameters
        ----------
        target : ndarray
            a numpy array (nOut, 1) containing the target/label values of the training instance
        nextDerivatives: ndarray
            a numpy array (nOutputs, 1) containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array (nOutputs, 1) containing the weights from this layer to the next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """

        # Mitchell, p. 98 states: (MSE error function, sigmoid activation function)
        #
        # For each output unit k, calculate its error term delta_k:
        # delta_k <-- o_k (1 - o_k)(t_k - o_k)
        #
        # For each hidden unit h, calculate its error term delta_h:
        # delta_h <-- o_h (1 - o_h) * sum[k in outputs] (w_kh delta_k)
        #
        # where:
        #   o = output, t = target
        #   outputs = output layer
        #   w_kh = weight associated with k-th input to unit h

        self.delta = np.ndarray((self.nOut, 1))

        if self.isClassifierLayer:
            # output layer
            self.delta = self.output * (1 - self.output) * (target - self.output)


        else:
            # hidden layer

            # compute the sum (over k) for each h
            sums_over_k = np.array([np.dot(nextWeights[:,h], nextDerivatives) for h in range(self.nOut)])

            # np.dot(nextWeights, nextDerivatives) is equivalent to this:
            # ----
            # sum_over_k = 0
            # for k in range(nOutputs):
            #
            #     w_kh = nextWeights[k, h]
            #     delta_k = nextDerivatives[k]
            #
            #     sum_over_k += w_kh * delta_k
            # ----
            # => np.dot(nextWeights[:,h], nextDerivatives) == sum_over_k (experimentally checked)


            self.delta = self.output * (1 - self.output) * sums_over_k

            # equivalent to: (checked)
            # for h in range(self.nOut):
            #     self.delta[h] = self.output[h] * (1 - self.output[h]) * np.dot(nextWeights[:, h], nextDerivatives)

        return self.delta

    def updateWeights(self):
        """
        Update the weights of the layer
        """

        # Mitchell, p. 98:
        # Update each network weight w_ji
        #     w_ji <-- wji + DeltaW_ji
        # where
        #     DeltaW_ji = learning rate * delta_j * x_ji
        #
        # x_ji = i-th input to unit j


        learningRate = 0.01


        weightUpdate = np.ndarray(self.weights.shape)

        # j = unit
        # i = input
        for j in range(len(self.output)):
            for i in range(len(self.input)):
                weightUpdate[j,i] = learningRate * self.delta[j] * self.input[i]

        self.weights += weightUpdate


