from .module import Module


class Sequential(Module):

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, dLoss):
        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss)
        return dLoss

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params