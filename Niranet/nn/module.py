class Module:
    """
    Base class for all neural network modules.
    """

    def forward(self, X):
        """
        Performs forward propagation.
        Must be implemented by child classes.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Performs backward propagation.
        Must be implemented by child classes.
        """
        raise NotImplementedError

    def parameters(self):
        """
        Returns trainable parameters.
        Layers without parameters return an empty list.
        """
        return []

    def __call__(self, X):
        """
        Enables calling the layer like a function.
        Example:
            output = layer(X)
        """
        return self.forward(X)