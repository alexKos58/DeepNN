import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# Выбор устройства
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Преобразования для изображений
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Загрузка обучающих и тестовых данных
train_dataset = torchvision.datasets.ImageFolder(root='./animals/train', transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(root='./animals/val', transform=data_transforms)

# Ограничиваем выборку
train_subset, _ = torch.utils.data.random_split(train_dataset, [2000, len(train_dataset)-2000])
test_subset, _ = torch.utils.data.random_split(test_dataset, [1000, len(test_dataset)-1000])

# DataLoader
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=10, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=10, shuffle=True, num_workers=2)

# Названия классов
class_names = train_dataset.classes
print("Классы:", class_names)  

# Сверточная нейронная сеть
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(6 * 28 * 28, 512),  # 224 / 2 / 2 / 2 = 28
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Обучение
def train_model(model, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Эпоха [{epoch+1}/{epochs}], Шаг {i}, Потери: {loss.item():.4f}")

# Оценка точности
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Точность модели: {accuracy:.2f}%")

# Отображение предсказаний
def visualize_predictions(model):
    model.eval()
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    for i in range(len(inputs)):
        img = inputs[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Класс: {class_names[preds[i]]}")
        plt.axis("off")
        plt.pause(2)

# Основной блок
if __name__ == '__main__':
    model = CNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    epochs = 2

    print("Обучение модели...")
    start_time = time.time()
    train_model(model, criterion, optimizer, epochs)
    print("Время обучения:", time.time() - start_time)

    print("Оценка модели...")
    evaluate_model(model)

    print("Визуализация результатов...")
    visualize_predictions(model)

    torch.save(model.state_dict(), 'cnn_animals.pth')
