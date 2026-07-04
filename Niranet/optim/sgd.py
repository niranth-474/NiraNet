from niranet.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:

            if param["params"] == "W":
                param["layer"].W -= self.lr * param["layer"].dW

            elif param["params"] == "b":
                param["layer"].b -= self.lr * param["layer"].db