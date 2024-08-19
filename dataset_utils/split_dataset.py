# script to filter out labels except person label and split a dataset into train, validation, and test sets
# Usage: python split_dataset.py <src_dir> <dest_dir> [--split_ratios SPLIT_RATIOS]

import os, shutil, random, argparse, sys
from pathlib import Path

def filter_labels_and_copy(src_dir, dest_dir, split_ratios=(0.8, 0.1, 0.1)):
    images_dir = os.path.join(src_dir, 'images')
    labels_dir = os.path.join(src_dir, 'labels')
    
    # Ensure destination directory structure
    for split in ['train', 'valid', 'test']:
        Path(os.path.join(dest_dir, split, 'images')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dest_dir, split, 'labels')).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle the images for random splitting
    random.shuffle(image_files)
    
    # Calculate split indices
    total_images = len(image_files)
    train_end = int(total_images * split_ratios[0])
    valid_end = train_end + int(total_images * split_ratios[1])
    
    train_files = image_files[:train_end]
    valid_files = image_files[train_end:valid_end]
    test_files = image_files[valid_end:]
    
    # Process each split
    for split, split_files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        for image_file in split_files:
            # Copy the image file
            src_image_path = os.path.join(images_dir, image_file)
            dest_image_path = os.path.join(dest_dir, split, 'images', image_file)
            shutil.copyfile(src_image_path, dest_image_path)
            
            # Process the corresponding label file
            label_file = os.path.splitext(image_file)[0] + '.txt'
            src_label_path = os.path.join(labels_dir, label_file)
            dest_label_path = os.path.join(dest_dir, split, 'labels', label_file)
            
            if os.path.exists(src_label_path):
                with open(src_label_path, 'r') as f:
                    lines = f.readlines()
                
                # Filter for class_id 0
                filtered_lines = [line for line in lines if line.startswith('0')]
                
                # Write the filtered or empty label file
                with open(dest_label_path, 'w') as f:
                    f.writelines(filtered_lines)
            else:
                # Create an empty label file if the label doesn't exist
                open(dest_label_path, 'w').close()

def main():
    parser = argparse.ArgumentParser(description="Filter person labels from YOLO format label files and split the dataset.")
    parser.add_argument('src_dir', type=str, help="Path to the source dataset directory.")
    parser.add_argument('dest_dir', type=str, help="Path to the destination directory to save the filtered dataset.")
    parser.add_argument('--split_ratios', nargs='+', type=float, default=[0.8, 0.1, 0.1], help="Train, validation, and test split ratios.")
    args = parser.parse_args()
    if not args.src_dir or not args.dest_dir:
        print("Error: Both src_dir and dest_dir arguments are required.")
        print("Usage: python script_name.py <src_dir> <dest_dir> [--split_ratios SPLIT_RATIOS]")
        sys.exit(1)
    
    filter_labels_and_copy(args.src_dir, args.dest_dir, args.split_ratios)

if __name__ == "__main__":
    main()
