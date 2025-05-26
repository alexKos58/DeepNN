import pandas as pd
import numpy as np

# Загрузка и перемешивание данных
data = pd.read_csv('data.csv')
data = data.sample(frac=1).reset_index(drop=True)

# Подготовка целевых меток и признаков
targets = data.iloc[:100, 4].values
targets = np.where(targets == "Iris-setosa", 1, -1)
features = data.iloc[:100, [0, 2]].values

# Параметры сети
n_inputs = features.shape[1]
hidden_neurons = 10
output_neurons = 1

# Инициализация весов
W_1 = np.zeros((1 + n_inputs, hidden_neurons))
W_1[0, :] = np.random.randint(0, 3, size=(hidden_neurons))
W_1[1:, :] = np.random.randint(-1, 2, size=(n_inputs, hidden_neurons))

W_2 = np.random.randint(0, 2, size=(1 + hidden_neurons, output_neurons)).astype(float)

# Предсказание
def forward_pass(X_sample):
    hidden_output = np.where((np.dot(X_sample, W_1[1:, :]) + W_1[0, :]) >= 0, 1, -1).astype(float)
    output = np.where((np.dot(hidden_output, W_2[1:, :]) + W_2[0, :]) >= 0, 1, -1).astype(float)
    return output, hidden_output

# Гиперпараметры
learning_rate = 0.01
n_iter = 0
weight_snapshots = []

# Обучение
while True:
    print(f"Итерация {n_iter}")
    n_iter += 1

    for x, y_true in zip(features, targets):
        y_pred, hidden = forward_pass(x)
        W_2[1:] += (learning_rate * (y_true - y_pred)) * hidden.reshape(-1, 1)
        W_2[0] += learning_rate * (y_true - y_pred)

    # Хранение снимков весов
    weight_snapshots.append(W_2.tobytes())

    y_preds, _ = forward_pass(features)
    total_error = np.sum(y_preds.reshape(-1) - targets)
    print('Суммарная ошибка:', total_error)

    # Условие остановки: всё классифицировано правильно
    if total_error == 0:
        print("Модель обучена: ошибки отсутствуют.")
        break

    # Условие остановки: повторение весов (зацикливание)
    if n_iter % 5 == 0:
        if len(set(weight_snapshots)) != len(weight_snapshots):
            print("Зафиксировано повторение весов — выход из цикла.")
            break