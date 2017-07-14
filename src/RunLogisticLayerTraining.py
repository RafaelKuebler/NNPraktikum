#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.logistic_layer import LogisticLayer
from report.evaluator import Evaluator
import sys

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=True)

    inputDim = len(data.trainingSet.input[0])
    hiddenDim = inputDim / 2
    outputDim = 2
    epochs = 30
    learningRate = 0.1

    logisticLayerHidden = LogisticLayer(inputDim, hiddenDim, activation='sigmoid',
                                        learningRate=learningRate, isClassifierLayer=False)
    logisticLayerOutput = LogisticLayer(hiddenDim, outputDim, activation='softmax',
                                        learningRate=learningRate, isClassifierLayer=True)

    print("Training parameters:")
    print("  Topolgoy: [{}] (input) -> [{}] (hidden) -> [{}] (outout)".format(inputDim, hiddenDim, outputDim))
    print("  Epochs: {}, Learning Rate: {}".format(epochs, learningRate))

    print("\n------------")
    print("| Training |")
    print("------------\n")
    for epoch in range(epochs):

        sys.stdout.write("Epoch {} ... ".format(epoch))


        sumSquaredError = 0.0
        correct = 0

        for input, label in zip(data.trainingSet.input, data.trainingSet.label):

            # make input shape (nIn, 1)
            input = np.array([input])
            # target = [seven, not seven]
            target = np.zeros((outputDim, 1), dtype=np.float)
            if label == 1:
                target[0, 0] = 1.
            else:
                target[1, 0] = 1.


            hiddenOutput = logisticLayerHidden.forward(input)
            output = logisticLayerOutput.forward(hiddenOutput)

            sumSquaredError += np.sum((target - output)**2)

            if label == 1 and output[0,0] > output[1,0] or label == 0 and output[0,0] < output[1,0]:
                correct += 1

            outputDerivative = logisticLayerOutput.computeDerivative(target, None, None)
            hiddenDerivative = logisticLayerHidden.computeDerivative(None, outputDerivative, logisticLayerOutput.weights)

            logisticLayerOutput.accumulateWeightUpdates()
            logisticLayerHidden.accumulateWeightUpdates()

            logisticLayerOutput.updateWeights()
            logisticLayerHidden.updateWeights()


        meanSquaredError = 0.5 * sumSquaredError
        print("\nMSE: {}".format(meanSquaredError))
        print("Accuracy: {}%".format(100. * correct / len(data.trainingSet.input)))


if __name__ == '__main__':
    main()
