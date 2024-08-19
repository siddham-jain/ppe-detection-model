import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description="Object detection for persons and PPE")
    parser.add_argument("--input_dir", required=True, help="Input directory containing images")
    parser.add_argument("--output_dir", required=True, help="Output directory to save results")
    parser.add_argument("--person_model", required=True, help="Path to the person detection model")
    parser.add_argument("--ppe_models_dir", required=True, help="Directory containing PPE detection models")
    return parser.parse_args()

def load_models(person_model_path, ppe_models_dir):
    person_model = YOLO(person_model_path)
    ppe_models = []
    for file in os.listdir(ppe_models_dir):
        if file.startswith("ppe_fold_") and file.endswith(".pt"):
            model_path = os.path.join(ppe_models_dir, file)
            ppe_models.append(YOLO(model_path))
    return person_model, ppe_models

def detect_persons(image, model):
    results = model(image)[0]
    return results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()

def detect_ppe_ensemble(cropped_image, models):
    all_predictions = []
    for model in models:
        results = model(cropped_image)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        if len(boxes) > 0:
            all_predictions.append(np.column_stack((boxes, scores, classes)))
    
    if not all_predictions:
        return np.array([]), np.array([]), np.array([])
    
    # Combine predictions from all models
    all_predictions = np.vstack(all_predictions)
    
    # Perform non-maximum suppression
    final_boxes, final_scores, final_classes = non_max_suppression(all_predictions)
    
    return final_boxes, final_scores, final_classes

def non_max_suppression(predictions, iou_threshold=0.5, score_threshold=0.5):
    # Sort by score
    indices = np.argsort(predictions[:, 4])[::-1]
    predictions = predictions[indices]
    
    keep = []
    while predictions.shape[0] > 0:
        keep.append(predictions[0])
        if predictions.shape[0] == 1:
            break
        ious = calculate_iou(predictions[0, :4], predictions[1:, :4])
        predictions = predictions[1:][ious < iou_threshold]
    
    keep = np.array(keep)
    
    # Handle the case where keep is 1D (only one prediction kept)
    if keep.ndim == 1:
        keep = keep.reshape(1, -1)
    
    final_boxes = keep[:, :4]
    final_scores = keep[:, 4]
    final_classes = keep[:, 5]
    
    # Filter by score threshold
    mask = final_scores >= score_threshold
    final_boxes = final_boxes[mask]
    final_scores = final_scores[mask]
    final_classes = final_classes[mask]
    
    return final_boxes, final_scores, final_classes

def calculate_iou(box, boxes):
    # Calculate IoU between a box and an array of boxes
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    
    return intersection / union

def draw_boxes(image, boxes, classes, scores, colors):
    for box, cls, score in zip(boxes, classes, scores):
        if cls in colors:  # Only draw boxes for PPE items
            x1, y1, x2, y2 = map(int, box[:4])
            color = colors[cls]
            label = f"{cls}: {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def process_image(image_path, person_model, ppe_models, output_dir):
    image = cv2.imread(image_path)
    person_boxes, person_scores = detect_persons(image, person_model)
    
    ppe_results = []
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped_image = image[y1:y2, x1:x2]
        ppe_boxes, ppe_scores, ppe_classes = detect_ppe_ensemble(cropped_image, ppe_models)
        
        for ppe_box, ppe_score, ppe_class in zip(ppe_boxes, ppe_scores, ppe_classes):
            ppe_x1, ppe_y1, ppe_x2, ppe_y2 = map(int, ppe_box)
            ppe_results.append({
                'box': [x1 + ppe_x1, y1 + ppe_y1, x1 + ppe_x2, y1 + ppe_y2],
                'class': int(ppe_class),  # Convert to int
                'score': ppe_score
            })
    
    # Only use PPE detections
    all_boxes = np.array([r['box'] for r in ppe_results])
    all_classes = np.array([r['class'] for r in ppe_results])
    all_scores = np.array([r['score'] for r in ppe_results])
    
    # Define colors for different classes (using integer keys)
    colors = {
        0: (255, 0, 0),    # hard-hat
        1: (0, 255, 0),    # gloves
        2: (0, 0, 255),    # mask
        3: (255, 255, 0),  # glasses
        4: (255, 0, 255),  # boots
        5: (0, 255, 255),  # vest
        6: (128, 0, 128)   # ppe-suit
    }
    
    # Draw boxes on the image
    annotated_image = draw_boxes(image.copy(), all_boxes, all_classes, all_scores, colors)
    
    # Save the annotated image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_image)

def main():
    args = parse_arguments()
    
    person_model, ppe_models = load_models(args.person_model, args.ppe_models_dir)
    
    if not ppe_models:
        raise ValueError("No PPE models found in the specified directory")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for image_file in os.listdir(args.input_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.input_dir, image_file)
            process_image(image_path, person_model, ppe_models, args.output_dir)

if __name__ == "__main__":
    main()