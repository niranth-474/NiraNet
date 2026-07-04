import numpy as np


class Loss:

    def forward(self, y_pred, y_true):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self):
        raise NotImplementedError("Backward method not implemented.")

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class MSELoss(Loss):

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((self.y_pred - self.y_true) ** 2)

    def backward(self):
        m = self.y_true.shape[0]
        return (2 / m) * (self.y_pred - self.y_true)


class BCELoss(Loss):

    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true

        return -np.mean(
            self.y_true * np.log(self.y_pred)
            + (1 - self.y_true) * np.log(1 - self.y_pred)
        )

    def backward(self):
        m = self.y_true.shape[0]

        return (
            (self.y_pred - self.y_true)
            / (self.y_pred * (1 - self.y_pred))
        ) / m


class CrossEntropyLoss(Loss):

    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true

        return -np.mean(
            np.sum(self.y_true * np.log(self.y_pred), axis=1)
        )

    def backward(self):
        m = self.y_true.shape[0]

        
        
        return (self.y_pred - self.y_true) / m