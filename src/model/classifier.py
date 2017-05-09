# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class Classifier:
    """
    Abstract class of a classifier
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, trainingSet, validationSet):
        # train procedures of the classifier
        pass

    @abstractmethod
    def classify(self, testInstance):
        # classify an instance given the model of the classifier
        pass

    @abstractmethod
    def evaluate(self, test):
        # evaluate a whole test set given the model of the classifier
        pass
