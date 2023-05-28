import os
import shutil


def create_class_folders(val_annotations_file, val_folder):
    target_folder = os.path.join(val_folder, 'classified')
    os.makedirs(target_folder, exist_ok=True)
    annotations = {}
    with open(val_annotations_file, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split('\t')
            image_filename = parts[0]
            class_name = parts[1]
            annotations[image_filename] = class_name
    # 根据类别创建文件夹，并将对应类别的图像移动到相应文件夹中
    for image_filename, class_name in annotations.items():
        class_folder = os.path.join(target_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        source_path = os.path.join(val_folder, image_filename)
        target_path = os.path.join(class_folder, image_filename)
        shutil.move(source_path, target_path)


val_annotations_file = 'C:/Users/Adydio/Desktop/大二下/pythondl/tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt'
val_folder = 'C:/Users/Adydio/Desktop/大二下/pythondl/tiny-imagenet-200/tiny-imagenet-200/val/images'
create_class_folders(val_annotations_file, val_folder)
