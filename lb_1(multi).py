import pandas as pd
import numpy as np

# Загрузка и перемешивание датасета
data = pd.read_csv('data.csv')
data = data.sample(frac=1).reset_index(drop=True)

# Классы: 0 = setosa, 1 = versicolor, 2 = virginica
targets = data.iloc[:100, 4].values
targets = np.where(targets == "Iris-setosa", 0, targets)
targets = np.where(targets == "Iris-versicolor", 1, targets)
targets = np.where(targets == "Iris-virginica", 2, targets).astype(int)

# Признаки: выбираем только 0 и 2 столбцы
features = data.iloc[:100, [0, 2]].values

# Архитектура сети
n_inputs = features.shape[1]
W_1_neurons = 10
output_neurons = 3

# Инициализация весов
W_1 = np.zeros((1 + n_inputs, W_1_neurons))
W_1[0, :] = np.random.randint(0, 3, size=(W_1_neurons))  # пороги
W_1[1:, :] = np.random.randint(-1, 2, size=(n_inputs, W_1_neurons))  # веса

W_2 = np.random.randint(0, 2, size=(1 + W_1_neurons, output_neurons)).astype(float)

# Прямой проход
def forward_pass(X_sample):
    W_1_output = np.where((np.dot(X_sample, W_1[1:, :]) + W_1[0, :]) >= 0, 1, -1).astype(float)
    output = np.where((np.dot(W_1_output, W_2[1:, :]) + W_2[0, :]) >= 0, 1, -1).astype(float)
    return output, W_1_output

# Подсчёт ошибки
def compute_error(prediction, true_class):
    target_vector = np.full(output_neurons, -1.0)
    target_vector[true_class] = 1.0
    return np.sum(target_vector - prediction)

# Гиперпараметры
learning_rate = 0.01
n_iter = 0
weight_history = []

# Обучение
while True:
    print(f"Итерация {n_iter}")
    n_iter += 1

    for x, y_class in zip(features, targets):
        predicted_output, W_1_output = forward_pass(x)
        target_vector = np.full(output_neurons, -1.0)
        target_vector[y_class] = 1.0
        W_2[1:] += learning_rate * (target_vector - predicted_output) * W_1_output[:, np.newaxis]
        W_2[0] += learning_rate * (target_vector - predicted_output)

    # Сохраняем снимок весов
    weight_history.append(W_2.tobytes())

    predictions, _ = forward_pass(features)
    total_error = 0
    for idx in range(len(targets)):
        total_error += compute_error(predictions[idx], targets[idx])
    print("Суммарная ошибка:", total_error)

    # Условие остановки: сходимость
    if total_error == 0:
        print("Модель успешно обучена — ошибок нет.")
        break

    # Условие остановки: повтор весов (зацикливание)
    if n_iter % 5 == 0:
        if len(set(weight_history)) != len(weight_history):
            print("Обнаружено повторение весов — выход из цикла.")
            break