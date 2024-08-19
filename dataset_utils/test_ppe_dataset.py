import os, cv2, random, argparse, sys
import numpy as np

def draw_boxes(images_dir, labels_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_files = random.sample(os.listdir(images_dir), min(10, len(os.listdir(images_dir))))  # Limit to first 10 images for demo

    # Define color map for different classes (you can adjust these)
    color_map = {
        0: (255, 0, 0),    # Blue
        1: (0, 255, 0),    # Green
        2: (0, 0, 255),    # Red
        3: (255, 255, 0),  # Cyan
        4: (255, 0, 255),  # Magenta
        5: (0, 255, 255),  # Yellow
        6: (128, 0, 0),    # Navy
        7: (0, 128, 0)     # Dark Green
    }

    # Class names (adjust these to match your classes)
    class_names = ['hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector']

    for img_file in image_files:
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, img_file.rsplit('.', 1)[0] + '.txt')

            if not os.path.exists(label_path):
                print(f"No label file found for {img_file}. Skipping.")
                continue

            # Read image
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            # Read annotations
            with open(label_path, 'r') as f:
                annotations = [line.strip().split() for line in f]

            # Draw bounding boxes
            for ann in annotations:
                class_id, x_center, y_center, w, h = map(float, ann)
                class_id = int(class_id)

                # Calculate pixel coordinates
                x1 = int((x_center - w/2) * width)
                y1 = int((y_center - h/2) * height)
                x2 = int((x_center + w/2) * width)
                y2 = int((y_center + h/2) * height)

                # Draw rectangle
                color = color_map.get(class_id, (0, 0, 0))  # Default to black if class_id not in color_map
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Add label
                label = f'{class_names[class_id]}: {class_id}'
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - label_height - 5), (x1 + label_width, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save the image with bounding boxes
            output_path = os.path.join(output_dir, f'boxed_{img_file}')
            cv2.imwrite(output_path, img)

    print("Bounding boxes drawn and images saved.")

def main():
    parser = argparse.ArgumentParser(description="Draw bounding boxes on images using YOLO format labels.")
    parser.add_argument('images_dir', type=str, help="Path to the directory containing images.")
    parser.add_argument('labels_dir', type=str, help="Path to the directory containing YOLO format label files.")
    parser.add_argument('output_dir', type=str, help="Path to the directory to save images with bounding boxes.")
    args = parser.parse_args()
    if not args.images_dir or not args.labels_dir or not args.output_dir:
        print("Error: All arguments are required.")
        print("Usage: python script_name.py <images_dir> <labels_dir> <output_dir>")
        sys.exit(1)
    
    draw_boxes(args.images_dir, args.labels_dir, args.output_dir)

if __name__ == "__main__":
    main()
