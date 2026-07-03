import numpy as np
from .module import Module


class ReLU(Module):

    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        dZ = dA.copy()
        dZ[self.Z <= 0] = 0
        return dZ


class LeakyReLU(Module):

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.where(Z > 0, Z, self.alpha * Z)

    def backward(self, dA):
        dZ = dA.copy()
        dZ[self.Z <= 0] *= self.alpha
        return dZ


class Sigmoid(Module):

    def __init__(self):
        self.output = None

    def forward(self, Z):
        self.output = 1 / (1 + np.exp(-Z))
        return self.output

    def backward(self, dA):
        return dA * self.output * (1 - self.output)


class Tanh(Module):

    def __init__(self):
        self.output = None

    def forward(self, Z):
        self.output = np.tanh(Z)
        return self.output

    def backward(self, dA):
        return dA * (1 - self.output ** 2)