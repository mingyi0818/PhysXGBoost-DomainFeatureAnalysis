"""
Prepare CropAndWeed dataset for few-shot learning.
Crop bounding boxes from images and organize by class.
"""
import os
import csv
import sys
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, r'D:\datasets\CropAndWeed\cropandweed-dataset\cnw')
from utilities.datasets import DATASETS

CROPANDWEED_ROOT = r'D:\datasets\CropAndWeed\CropAndWeed'
OUTPUT_ROOT = r'D:\datasets\CropAndWeed\CropAndWeed_cropped'
MIN_SIZE = 32
PADDING = 10


def crop_and_save(min_images_per_class=20):
    dataset_info = DATASETS['CropAndWeed']
    bbox_dir = os.path.join(CROPANDWEED_ROOT, 'bboxes', 'CropAndWeed')
    img_dir = os.path.join(CROPANDWEED_ROOT, 'images')

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    class_counts = {}
    bbox_files = [f for f in os.listdir(bbox_dir) if f.endswith('.csv')]
    print(f"Processing {len(bbox_files)} images...")

    for bbox_file in tqdm(bbox_files):
        img_name = bbox_file.replace('.csv', '.jpg')
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except:
            continue

        csv_path = os.path.join(bbox_dir, bbox_file)
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5:
                    continue
                try:
                    left = max(0, int(float(row[0])) - PADDING)
                    top = max(0, int(float(row[1])) - PADDING)
                    right = min(img_w, int(float(row[2])) + PADDING)
                    bottom = min(img_h, int(float(row[3])) + PADDING)
                    label_id = int(row[4])
                except:
                    continue

                if right - left < MIN_SIZE or bottom - top < MIN_SIZE:
                    continue

                label_name = dataset_info.get_label_name(label_id)
                if label_name is None:
                    continue
                if label_name in ('Soil', 'Vegetation'):
                    continue

                class_dir = os.path.join(OUTPUT_ROOT, label_name.replace(' ', '_'))
                os.makedirs(class_dir, exist_ok=True)

                crop = img.crop((left, top, right, bottom))
                crop_name = f"{img_name.replace('.jpg', '')}_{label_id}_{len(os.listdir(class_dir)):04d}.jpg"
                crop_path = os.path.join(class_dir, crop_name)
                crop.save(crop_path, quality=90)

                class_counts[label_name] = class_counts.get(label_name, 0) + 1

    print(f"\nTotal classes: {len(class_counts)}")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_classes[:30]:
        print(f"  {name}: {count}")

    valid_classes = {k: v for k, v in class_counts.items() if v >= min_images_per_class}
    print(f"\nClasses with >= {min_images_per_class} images: {len(valid_classes)}")

    return class_counts


if __name__ == "__main__":
    print("=" * 60)
    print("CropAndWeed Dataset Preparation - Cropping Bounding Boxes")
    print("=" * 60)
    counts = crop_and_save(min_images_per_class=20)
    print("\nDone!")
