# script to convert YOLO format annotations for person detection to cropped images
# Usage: python convert_annotations.py <input_dir> <output_dir>

import os
import cv2
import numpy as np
from pathlib import Path
import sys, argparse

def convert_annotations(input_dir, output_dir):
    input_images_dir = os.path.join(input_dir, 'images')
    input_labels_dir = os.path.join(input_dir, 'labels')
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    for img_file in os.listdir(input_images_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_images_dir, img_file)
            label_path = os.path.join(input_labels_dir, img_file.rsplit('.', 1)[0] + '.txt')

            if not os.path.exists(label_path):
                print(f"No label file found for {img_file}. Skipping.")
                continue

            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            with open(label_path, 'r') as f:
                annotations = [line.strip().split() for line in f]

            person_count = 0
            for ann in annotations:
                if ann[0] == '0':  # Assuming '0' is the class ID for person
                    person_count += 1
                    x_center, y_center, w, h = map(float, ann[1:])
                    x1 = int((x_center - w/2) * width)
                    y1 = int((y_center - h/2) * height)
                    x2 = int((x_center + w/2) * width)
                    y2 = int((y_center + h/2) * height)

                    # Crop the image
                    cropped_img = img[y1:y2, x1:x2]
                    crop_height, crop_width = cropped_img.shape[:2]

                    # Create new annotations for the cropped image
                    new_annotations = []
                    for item in annotations:
                        class_id = int(item[0])
                        if class_id != 0:  # Skip person class
                            item_x, item_y, item_w, item_h = map(float, item[1:])
                            item_x1 = (item_x - item_w/2) * width
                            item_y1 = (item_y - item_h/2) * height
                            item_x2 = (item_x + item_w/2) * width
                            item_y2 = (item_y + item_h/2) * height

                            # Check if the item is inside the person's bounding box
                            if x1 <= item_x1 < item_x2 <= x2 and y1 <= item_y1 < item_y2 <= y2:
                                # Adjust coordinates relative to the cropped image
                                new_x = (item_x1 - x1) / crop_width
                                new_y = (item_y1 - y1) / crop_height
                                new_w = item_w * width / crop_width
                                new_h = item_h * height / crop_height

                                new_annotations.append(f"{class_id-1} {new_x + new_w/2} {new_y + new_h/2} {new_w} {new_h}")

                    # Save the cropped image
                    output_img_path = os.path.join(output_images_dir, f"{img_file.rsplit('.', 1)[0]}_{person_count}.jpg")
                    cv2.imwrite(output_img_path, cropped_img)

                    # Save the new annotations
                    output_label_path = os.path.join(output_labels_dir, f"{img_file.rsplit('.', 1)[0]}_{person_count}.txt")
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(new_annotations))

    print("Conversion completed.")

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO format annotations for person detection to cropped images.")
    parser.add_argument('input_dir', type=str, help="Path to the input directory containing images and labels.")
    parser.add_argument('output_dir', type=str, help="Path to the output directory to save cropped images and labels.")
    args = parser.parse_args()

    if not args.input_dir or not args.output_dir:
        print("Error: Both input_dir and output_dir arguments are required.")
        print("Usage: python script_name.py <input_dir> <output_dir>")
        sys.exit(1)
    
    convert_annotations(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()