'''
Linear Layer

Definition

A Linear Layer is a trainable neural network layer that performs a linear transformation on the input.

Mathematically,  z=xw+b, where w is the weight matrix, x is the input vector, and b is the bias vector.

'''



import numpy as np
from .module import Module

class Linear(Module):

    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        z = X @ self.W + self.b
        return z
    
    def backward(self, dZ):
        m = self.X.shape[0]
        self.dW = (self.X.T @ dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ self.W.T
        return dX