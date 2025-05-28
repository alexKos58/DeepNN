import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO


# Загрузка предобученной модели YOLOv8 для дообучения
model = YOLO("yolov8s.pt")

# Обучение модели на нашем датасете
model.train(
    data="animals.yaml",
    model="yolov8s.pt",
    epochs=1,
    imgsz=224,
    batch=16,
    project="animal_detection",
    name="exp1",
    verbose=True
)

# Список изображений для теста
test_images = [
    "cat1.jpg",
    "cat2.jpg",
    "dog1.jpg",
    "dog2.jpg",
    "dog_with_dog.jpg",
    "dog_with_human.jpg",
    "wild1.jpg",
    "wild2.jpg"
]

# Предсказание и отображение результатов
for img_path in test_images:
    results = model(img_path)
    result = results[0]
    cv2.imshow("Результат YOLOv8", result.plot())
    cv2.waitKey(1500)

cv2.destroyAllWindows()
