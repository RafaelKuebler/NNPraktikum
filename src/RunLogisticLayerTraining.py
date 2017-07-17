#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

from data.mnist_seven import MNISTSeven
from model.logistic_layer import LogisticLayer


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)

    inputDim = len(data.trainingSet.input[0])
    hiddenDim = inputDim / 2
    outputDim = 10
    epochs = 30
    learningRate = 1.1

    logisticLayerHidden = LogisticLayer(inputDim, hiddenDim, activation='sigmoid',
                                        learningRate=learningRate, isClassifierLayer=False)
    logisticLayerOutput = LogisticLayer(hiddenDim, outputDim, activation='softmax',
                                        learningRate=learningRate, isClassifierLayer=True)

    print("\nTraining parameters:")
    print("  Topolgoy: [{}] (input) -> [{}] (hidden) -> [{}] (output)".format(inputDim, hiddenDim, outputDim))
    print("  Epochs: {}, Learning Rate: {}".format(epochs, learningRate))

    trainingData = zip(data.trainingSet.input, data.trainingSet.label)
    random.shuffle(trainingData)

    print("\n------------")
    print("| Training |")
    print("------------\n")
    for epoch in range(epochs):

        print("----------------------")
        print("Epoch {} ... ".format(epoch))

        sumSquaredError = 0.0
        hits = np.zeros(10, dtype=np.int) # hits per digit
        misses = np.zeros(10, dtype=np.int) # misses (per label)

        for input, label in trainingData:

            # make input shape (nIn, 1)
            input = np.array([input])
            # target = [seven, not seven]
            target = np.zeros((outputDim, 1), dtype=np.float)
            target[label, 0] = 1.


            hiddenOutput = logisticLayerHidden.forward(input)
            output = logisticLayerOutput.forward(hiddenOutput)

            sumSquaredError += np.sum((target - output)**2)

            result = np.argmax(output[:,0])

            # result == 1 if net predicted seven
            if result == label:
                hits[label] += 1
            else:
                misses[label] += 1


            outputDerivative = logisticLayerOutput.computeDerivative(target, None, None)
            hiddenDerivative = logisticLayerHidden.computeDerivative(None, outputDerivative, logisticLayerOutput.weights)

            logisticLayerOutput.accumulateWeightUpdates()
            logisticLayerHidden.accumulateWeightUpdates()

        logisticLayerOutput.updateWeights()
        logisticLayerHidden.updateWeights()


        # metrics[result][label]
        meanSquaredError = 0.5 * sumSquaredError
        print("\nMSE: {}".format(meanSquaredError))
        print("Accuracy: {}%".format(100. * np.sum(hits) / len(data.trainingSet.input)))
        print("Label | Hits | Misses")
        print("---------------------")

        for i in range(10):
            print("  {}   | {:>4} | {}".format(i, hits[i], misses[i]))


if __name__ == '__main__':
    main()
