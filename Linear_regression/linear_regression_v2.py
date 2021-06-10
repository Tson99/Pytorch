import torch.nn as nn
import torch.optim as optim
import pandas
import torch
import matplotlib.pyplot as plt

df = pandas.read_csv('../Data/data_linear.csv').values
x = torch.tensor(df[:, 0], dtype=torch.float32)
y = torch.tensor(df[:, 1], dtype=torch.float32)
x = x.unsqueeze(1)
y = y.unsqueeze(1)

linear_model = nn.Linear(1, 1)
print(list(linear_model.parameters()))

loss_fn = nn.MSELoss()
optimizer = optim.SGD(linear_model.parameters(), lr=0.00005)


losses = []
for epoch in range(10):
    y_hat = linear_model(x)
    loss = loss_fn(y_hat, y)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
        losses.append(loss.item())
        print('Epoch %d, Loss %f' % (epoch, float(loss)))

print(list(linear_model.parameters()))

plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
