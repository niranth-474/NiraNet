class Optimizer:

    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            if param["grads"] is not None:
                param["grads"].fill(0)