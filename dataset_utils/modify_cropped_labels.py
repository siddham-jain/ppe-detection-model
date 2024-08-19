# script to remove a class from the labels of the dataset
# Usage: python modify_cropped_labels.py
import os

dataset_path = '/home/siddham/ml-models/ppe-detection/datasets/ppe_dataset'
class_id_to_drop = 4

labels_path = os.path.join(dataset_path, 'labels_initial')
output_path = os.path.join(dataset_path, 'labels')

if not os.path.exists(output_path):
    os.makedirs(output_path)

for label_file in os.listdir(labels_path):
    label_path = os.path.join(labels_path, label_file)
    
    with open(label_path, 'r') as file:
        lines = file.readlines()

    filtered_lines = [line for line in lines if int(line.split()[0]) != class_id_to_drop]

    if filtered_lines:
        with open(os.path.join(output_path, label_file), 'w') as file:
            file.writelines(filtered_lines)
    else:
        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(dataset_path, 'images', image_file)
        if os.path.exists(image_path):
            os.remove(image_path)
