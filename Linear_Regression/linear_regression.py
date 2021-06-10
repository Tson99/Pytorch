import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

data = pd.read_csv('../Data/data_linear.csv').values

x = torch.tensor(data[:, 0])
y = torch.tensor(data[:, 1])

def model(x, a, b):
    return a*x + b

def loss_fn(y_hat, y):
    squared_diffs = (y_hat - y)**2
    return squared_diffs.mean()

def training_loop(n_epochs, learning_rate, params, x, y):
    a, b = params
    losses = []
    for epoch in range(1, n_epochs + 1):
        # a.requires_grad = True
        # b.requires_grad = True
        if a.grad is not None:
            a.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        y_hat = model(x, a, b)
        loss = loss_fn(y_hat, y)

        loss.backward()

        with torch.no_grad():
            a -= learning_rate*a.grad
            b -= learning_rate*b.grad
        if epoch % 1 == 0:
            losses.append(loss.item())
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return a, b, losses


if __name__ == '__main__':
    a = torch.ones((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    training_loop(n_epochs=10, learning_rate=0.00005, params=(a, b), x=x, y=y)
    print(a,b)
    x = torch.tensor(50)
    with torch.no_grad():
        y_hat = model(x, a, b)
        print(y_hat)