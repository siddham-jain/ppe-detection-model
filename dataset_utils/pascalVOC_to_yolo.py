# script to convert Pascal VOC annotations to YOLO format in place
# Usage: python pascalVOC_to_yolo.py <input_dir>

import os
import xml.etree.ElementTree as ET
import argparse

class_mapping = {
    "person": 0,
    "hard-hat": 1,
    "gloves": 2,
    "mask": 3,
    "glasses": 4,
    "boots": 5,
    "vest": 6,
    "ppe-suit": 7,
    "ear-protector": 8,
    "safety-harness": 9
}

def convert_pascal_voc_to_yolo(input_dir):
    for filename in os.listdir(input_dir):
        if not filename.endswith('.xml'):
            continue

        xml_path = os.path.join(input_dir, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        yolo_data = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                continue

            class_id = class_mapping[class_name]

            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # YOLO format: class_id center_x center_y width height
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

        # Write YOLO data to a new .txt file with the same name as the XML file
        output_file = os.path.join(input_dir, filename.replace('.xml', '.txt'))
        with open(output_file, 'w') as f:
            f.writelines(yolo_data)

        # Remove the original XML file
        os.remove(xml_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC annotations to YOLO format in place.")
    parser.add_argument('input_dir', type=str, help="Path to the directory containing Pascal VOC XML files.")

    args = parser.parse_args()

    convert_pascal_voc_to_yolo(args.input_dir)
