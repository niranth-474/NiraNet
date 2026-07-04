import numpy as np

from niranet.nn.linear import Linear
from niranet.nn.activations import ReLU, Sigmoid
from niranet.nn.sequential import Sequential
from niranet.nn.losses import BCELoss

from niranet.optim.sgd import SGD

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

model = Sequential(
    Linear(2, 3),
    ReLU(),
    Linear(3, 1),
    Sigmoid()
)

criterion = BCELoss()
optimizer = SGD(model.parameters(), lr=0.1)

epouch = 10000
for i in range(epouch):
    
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass
    dLoss = criterion.backward()
    model.backward(dLoss)

    # Update parameters
    optimizer.step()

    if i % 1000 == 0:
        print(f"Epoch {i}, Loss: {loss}")

