# ENTIRELY CHAT GPT GENERATED

import os
import json
from tqdm import tqdm

def convert_bbox(size, bbox):
    """Convert COCO bbox (x_min, y_min, width, height) to YOLO (x_center, y_center, width, height) normalized."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0
    return (x_center * dw, y_center * dh, width * dw, height * dh)

def coco_to_yolo(coco_json_path, images_dir, output_labels_dir, class_name_to_id):
    """Convert COCO annotations to YOLO format."""

    with open(coco_json_path) as f:
        coco_data = json.load(f)

    # Create mapping from image_id to filename and size
    image_id_to_info = {}
    for img in coco_data['images']:
        image_id_to_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        annotations_by_image.setdefault(img_id, []).append(ann)

    os.makedirs(output_labels_dir, exist_ok=True)

    for img_id, anns in tqdm(annotations_by_image.items(), desc="Converting COCO to YOLO"):
        img_info = image_id_to_info[img_id]
        label_path = os.path.join(output_labels_dir, os.path.splitext(img_info['file_name'])[0] + ".txt")

        with open(label_path, 'w') as f:
            for ann in anns:
                category_id = ann['category_id']
                class_id = class_name_to_id.get(category_id, None)
                if class_id is None:
                    continue  # Skip classes we don't care about

                bbox = ann['bbox']
                bbox_yolo = convert_bbox((img_info['width'], img_info['height']), bbox)

                line = f"{class_id} {' '.join(f'{x:.6f}' for x in bbox_yolo)}\n"
                f.write(line)

    print(f"Conversion completed! Labels are saved at {output_labels_dir}")

if __name__ == "__main__":
    coco_json_path = "../DATA/car_bee_detection/test/_annotations.coco.json" 
    images_dir = "../DATA/car_bee_detection/test" 
    output_labels_dir = "../DATA/car_bee_detection/test_labels_yolo" 

    class_name_to_id = {
        1: 1, # bee
        2: 0, # car  
        3: 1, # hornet
    }

    coco_to_yolo(coco_json_path, images_dir, output_labels_dir, class_name_to_id)
