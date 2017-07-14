#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.logistic_layer import LogisticLayer
from report.evaluator import Evaluator


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=True)

    inputDim = len(data.trainingSet.input[0])

    logisticLayerHidden = LogisticLayer(inputDim, 5, activation='sigmoid', isClassifierLayer=False)
    logisticLayerOutput = LogisticLayer(5, 1, activation='sigmoid', isClassifierLayer=True)


    input = np.ndarray((inputDim, 1))
    input[:,0] = data.trainingSet.input[0]
    target = data.trainingSet.label[0]


    print("\nTraining istance: \n - input:  [length {}]\n - target: {}".format(len(input), target))


    print("\nForward pass ...")

    hiddenOutput = logisticLayerHidden.forward(input)
    output = logisticLayerOutput.forward(hiddenOutput)

    print("Output of Hidden Layer:\n{}".format(hiddenOutput))
    print("Output of Output Layer:\n{}".format(output))


    print("\nBackward pass ...")

    outputDerivative = logisticLayerOutput.computeDerivative(target, None, None)
    hiddenDerivative = logisticLayerHidden.computeDerivative(None, outputDerivative, logisticLayerOutput.weights)

    print("Derivatives of Output Layer:\n{}".format(output))
    print("Derivatives of Hidden Layer:\n{}".format(hiddenOutput))


    print("\nWeight Update...")

    logisticLayerOutput.updateWeights()
    logisticLayerHidden.updateWeights()

    print("Done.")


if __name__ == '__main__':
    main()
