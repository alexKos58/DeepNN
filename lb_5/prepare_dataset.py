import os
from glob import glob
import shutil

def make_yolo_dataset_structure(base_path, categories):
    splits = ['train', 'val']
    for split in splits:
        split_path = os.path.join(base_path, split)
        img_dst = os.path.join(split_path, 'images')
        lbl_dst = os.path.join(split_path, 'labels')
        
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        for idx, cls_name in enumerate(categories):
            class_folder = os.path.join(split_path, cls_name)
            image_files = glob(os.path.join(class_folder, '*.jpg'))

            for img_path in image_files:
                fname = os.path.basename(img_path)
                txt_name = fname.replace('.jpg', '.txt')

                # копируем картинку
                shutil.copy(img_path, os.path.join(img_dst, fname))

                # создаем разметку YOLO — один объект, весь кадр
                label_path = os.path.join(lbl_dst, txt_name)
                with open(label_path, 'w') as lbl:
                    lbl.write(f"{idx} 0.5 0.5 1.0 1.0\n")  # весь кадр — один объект

if __name__ == '__main__':
    # путь до корневой директории с данными
    dataset_root = './animals'
    # список классов
    animal_classes = ['cat', 'dog', 'wild']

    make_yolo_dataset_structure(dataset_root, animal_classes)