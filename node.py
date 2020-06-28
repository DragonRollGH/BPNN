import numpy as np

class Node:
    def __init__(self, Dim):
        self.Dim = Dim
        self.weight = np.random.randn(Dim)*1
        self.input = np.zeros(Dim)
        self.inputSum = 0
        self.output = 0
        self.backPropagation = 0
        self.learnRate = 0.1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def Dsigmoid(self, x):
        return x * (1 - x)

    def getOutput(self):
        self.inputSum = np.sum(self.weight*self.input)
        self.output = self.sigmoid(self.inputSum)
        return self.output

    def setBP(self, BP):
        self.backPropagation = BP*self.Dsigmoid(self.output)

    def update(self):
        self.weight -= self.input*self.backPropagation*self.learnRate