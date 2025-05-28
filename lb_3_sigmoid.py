import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class MLPptorchSigmoid(nn.Module):
    def __init__(self, in_size, h1, h2, h3, out_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, h1),
            nn.Sigmoid(),
            nn.Linear(h1, h2),
            nn.Sigmoid(),
            nn.Linear(h2, h3),
            nn.Sigmoid(),
            nn.Linear(h3, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train(net, x, y, num_iter=5000, lr=0.009):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    lossFn = nn.MSELoss()
    for i in range(num_iter):
        pred = net(x)
        loss = lossFn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(f"[Sigmoid] Итерация {i}, Ошибка: {loss.item():.4f}")
    return loss.item()


def evaluate(model, X_eval, Y_eval):
    pred = model(X_eval).detach().numpy()
    err = np.sum(np.abs((pred > 0.5) - Y_eval.numpy()))
    return err


# === Загрузка и подготовка данных ===
df = pd.read_csv('data.csv').sample(frac=1).reset_index(drop=True)

X = df.iloc[0:100, 0:3].values
y = df.iloc[0:100, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor': 3}).values.reshape(-1, 1)
Y = np.zeros((y.shape[0], 3))
for i in range(1, 4):
    Y[:, i - 1] = (y == i).astype(int).reshape(-1)

X_test = df.iloc[100:150, 0:3].values
y = df.iloc[100:150, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor': 3}).values.reshape(-1, 1)
Y_test = np.zeros((y.shape[0], 3))
for i in range(1, 4):
    Y_test[:, i - 1] = (y == i).astype(int).reshape(-1)

X_tensor = torch.from_numpy(X.astype(np.float32))
Y_tensor = torch.from_numpy(Y.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
Y_test_tensor = torch.from_numpy(Y_test.astype(np.float32))

# === Параметры и запуск ===
net = MLPptorchSigmoid(3, 50, 20, 10, 3)
train(net, X_tensor, Y_tensor)
print(f"[Sigmoid] Ошибки на обучении: {evaluate(net, X_tensor, Y_tensor)}")
print(f"[Sigmoid] Ошибки на тесте: {evaluate(net, X_test_tensor, Y_test_tensor)}")