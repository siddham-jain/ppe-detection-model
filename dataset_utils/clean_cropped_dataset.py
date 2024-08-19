import os
import argparse, sys

def clear_empty_labels(label_dir, image_dir):
    # Iterate through all label files in the label directory
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)
            
            # Check if the label file is empty
            if os.path.getsize(label_path) == 0:
                # Construct the corresponding image file path
                image_file = label_file.replace(".txt", ".jpg")
                image_path = os.path.join(image_dir, image_file)
                
                # Delete the empty label file and the corresponding image
                print(f"Deleting {label_path} and {image_path}")
                os.remove(label_path)
                
                if os.path.exists(image_path):
                    os.remove(image_path)


def main():
    parser = argparse.ArgumentParser(description="Clear empty label files and corresponding images.")

    parser.add_argument('label_dir', type=str, help="Path to the directory containing label files.")
    parser.add_argument('image_dir', type=str, help="Path to the directory containing image files.")

    args = parser.parse_args()

    if not args.label_dir or not args.image_dir:
        print("Error: Both label_dir and image_dir arguments are required.")
        print("Usage: python script_name.py <label_dir> <image_dir>")
        sys.exit(1)
    
    clear_empty_labels(args.label_dir, args.image_dir)

if __name__ == "__main__":
    main()
