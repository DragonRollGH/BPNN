from node import Node
import numpy as np

class Layer:
    def __init__(self, Dim, lastDim):
        self.Dim = Dim
        self.nodeList = [Node(lastDim) for i in range(Dim)]

    def __rshift__(self, other):                #magic function 模拟正向传递
        for updatingNode in other.nodeList:
            updatingNode.input = np.array([n.getOutput() for n in self.nodeList])

    def __lshift__(self, other):
        for i in range(len(self.nodeList)):     #magic function 模拟反向传递
            self.nodeList[i].setBP(np.sum([n.weight[i]*n.backPropagation for n in other.nodeList]))
        self.update()

    def computeError(self, goal):               #计算输出层与目标之间的误差
        for i in range(len(self.nodeList)):
            self.nodeList[i].setBP(self.nodeList[i].output-goal[i])
        self.update()

    def update(self):                           #更新这层所有节点的权重
        for updatingNode in self.nodeList:
            updatingNode.update()
