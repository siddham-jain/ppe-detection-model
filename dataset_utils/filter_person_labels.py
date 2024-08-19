# script to filter person labels from YOLO format label files
# Usage: python filter_person_labels.py <input_labels_dir> <output_labels_dir>

import os, argparse, sys

def filter_labels(input_labels_dir, output_labels_dir):
    os.makedirs(output_labels_dir, exist_ok=True)
    
    for label_file in os.listdir(input_labels_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(input_labels_dir, label_file), 'r') as infile:
                lines = infile.readlines()
            
            filtered_lines = [line for line in lines if line.startswith('0 ')]
            
            if filtered_lines:
                with open(os.path.join(output_labels_dir, label_file), 'w') as outfile:
                    outfile.writelines(filtered_lines)

def main():
    parser = argparse.ArgumentParser(description="Filter person labels from YOLO format label files.")
    parser.add_argument('input_labels_dir', type=str, help="Path to the directory containing input label files.")
    parser.add_argument('output_labels_dir', type=str, help="Path to the directory to save filtered label files.")
    args = parser.parse_args()
    if not args.input_labels_dir or args.output_labels_dir:
        print("Error: Both input_labels_dir and output_labels_dir arguments are required.")
        print("Usage: python script_name.py <input_labels_dir> <output_labels_dir>")
        sys.exit(1)
    
    filter_labels(args.input_labels_dir, args.output_labels_dir)

if __name__ == "__main__":
    main()

