#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np

from data.mnist_seven import MNISTSeven
from model.logistic_layer import LogisticLayer

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=True)

    inputDim = len(data.trainingSet.input[0])
    outputDim = 1
    epochs = 30
    learningRate = 0.1

    logisticLayer = LogisticLayer(inputDim, outputDim, activation='sigmoid',
                                        learningRate=learningRate, isClassifierLayer=True)

    print("Training parameters:")
    print("  Topolgoy: [{}] (input) -> [{}] (outout)".format(inputDim, outputDim))
    print("  Epochs: {}, Learning Rate: {}".format(epochs, learningRate))

    trainingData = zip(data.trainingSet.input, data.trainingSet.label)
    random.shuffle(trainingData)

    print("\n------------")
    print("| Training |")
    print("------------\n")
    for epoch in range(epochs):

        print("Epoch {} ... ".format(epoch))

        sumSquaredError = 0.0
        metrics = [[0, 0], [0, 0]]
        # metrics[result, label] =>
        # metrics[1,1] = truePos
        # metrics[1,0] = falsePos
        # metrics[0,1] = falseNeg
        # metrics[0,0] = trueNeg


        for input, label in trainingData:

            # make input shape (nIn, 1)
            input = np.array([input])
            # target = [seven, not seven]
            target = np.zeros((outputDim, 1), dtype=np.float)
            target[0, 0] = label

            output = logisticLayer.forward(input)
            # result == 1 if neuron predicted "seven"
            result = int(output[0, 0] > 0.5)

            print("Output: {} => result = {}".format(output[0,0],result))

            metrics[result][label] += 1
            sumSquaredError += np.sum((target - output)**2)

            outputDerivative = logisticLayer.computeDerivative(target, None, None)

            logisticLayer.accumulateWeightUpdates()

            logisticLayer.updateWeights()


        # metrics[result][label]
        truePos = metrics[1][1]
        falsePos = metrics[1][0]
        falseNeg = metrics[0][1]
        trueNeg = metrics[0][0]

        meanSquaredError = 0.5 * sumSquaredError
        print("\nMSE: {}".format(meanSquaredError))
        print("TP: {} | FP: {} | FN: {} | TN: {}".format(truePos, falsePos, falseNeg, trueNeg))
        print("Accuracy: {}%".format(100. * (truePos + trueNeg) / len(data.trainingSet.input)))


if __name__ == '__main__':
    main()
