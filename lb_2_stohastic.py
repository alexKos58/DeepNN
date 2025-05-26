import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def mse_loss(y_pred, y_true_idx):
    y_true = np.zeros(3)
    y_true[y_true_idx] = 1
    return np.sum((y_pred - y_true) ** 2), (y_pred - y_true)

# Параметры сети
input_size = 4
hidden_size = 6
output_size = 3
learning_rate = 0.08
epochs = 100
batch_size = 25

# Загрузка и подготовка данных
df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]
X = df.iloc[:, :4].values
y = df.iloc[:, 4].replace({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).values

# Инициализация весов
W1 = np.random.rand(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

data = list(zip(X, y))
error_history = []

for epoch in range(epochs):
    random.shuffle(data)
    total_error = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        grad_W1 = np.zeros_like(W1)
        grad_W2 = np.zeros_like(W2)

        for x, y_true in batch:
            z1 = x @ W1
            a1 = sigmoid(z1)
            z2 = a1 @ W2
            a2 = sigmoid(z2)

            loss, error = mse_loss(a2, y_true)
            total_error += loss

            d2 = error * sigmoid_derivative(z2)
            dW2 = np.outer(a1, d2)

            d1 = (W2 @ d2) * sigmoid_derivative(z1)
            dW1 = np.outer(x, d1)

            grad_W1 += dW1
            grad_W2 += dW2

        W1 -= learning_rate * grad_W1 / batch_size
        W2 -= learning_rate * grad_W2 / batch_size

    print(f"Epoch {epoch+1}, Error: {total_error / len(data):.4f}")
    error_history.append(total_error / len(data))

plt.plot(error_history)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Training Error Over Time")
plt.show()

# Проверка точности
correct = 0
for x, y_true in data:
    pred = sigmoid(sigmoid(x @ W1) @ W2)
    if np.argmax(pred) == y_true:
        correct += 1

print(f"Accuracy: {correct} / {len(data)} ({(correct / len(data)) * 100:.2f}%)")