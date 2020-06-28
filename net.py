from layer import Layer

class Net:
    def __init__(self,DimList):
        self.Dim = len(DimList)-1
        self.layerList = [Layer(DimList[i+1], DimList[i]) for i in range(self.Dim)]

    def recognize(self, sample):
        for n in self.layerList[0].nodeList:
            n.input = sample
        for i in range(1, self.Dim):
            self.layerList[i-1]>>self.layerList[i]
        results = [n.getOutput() for n in self.layerList[-1].nodeList]
        return results

    def train(self, sample, goal):
        results = self.recognize(sample)
        self.layerList[-1].computeError(goal)
        for i in range(self.Dim-1, 0, -1):
            self.layerList[i-1]<<self.layerList[i]
        return results