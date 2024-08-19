import os
from collections import Counter
import pandas as pd
import argparse
import sys

def count_class_instances(labels_dir):
    class_counter = Counter()
    image_counter = Counter()
    total_images = 0

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            total_images += 1
            image_classes = set()
            
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counter[class_id] += 1
                    image_classes.add(class_id)
            
            for class_id in image_classes:
                image_counter[class_id] += 1

    # Create a DataFrame for better visualization
    df = pd.DataFrame({
        'Class ID': class_counter.keys(),
        'Total Instances': class_counter.values(),
        'Images with Class': image_counter.values(),
        'Avg Instances per Image': [count / image_counter[class_id] if image_counter[class_id] > 0 else 0 
                                    for class_id, count in class_counter.items()]
    })
    
    df = df.sort_values('Total Instances', ascending=False).reset_index(drop=True)
    df['Percentage of Images'] = df['Images with Class'] / total_images * 100

    print(f"Total number of images: {total_images}")
    print("\nClass Distribution:")
    print(df.to_string(index=False))

    print("\nImbalance Analysis:")
    min_instances = df['Total Instances'].min()
    max_instances = df['Total Instances'].max()
    imbalance_ratio = max_instances / min_instances if min_instances > 0 else float('inf')
    print(f"Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Count instances of each class in YOLO format labels.")
    parser.add_argument('labels_dir', type=str, nargs='?', help="Path to the directory containing YOLO format label files.")
    args = parser.parse_args()

    if not args.labels_dir:
        print("Error: No directory path provided.")
        print("Usage: python script_name.py <labels_dir>")
        sys.exit(1)
    
    count_class_instances(args.labels_dir)

if __name__ == "__main__":
    main()