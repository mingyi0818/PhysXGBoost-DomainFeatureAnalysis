"""
Organize DeepWeeds dataset into class folders for few-shot learning.

DeepWeeds dataset: 17,509 images, 9 classes (8 weeds + 1 negative)
Filename format: YYYYMMDD-HHMMSS-L.jpg where L is class label (0-8)

Classes:
0: Chinee apple
1: Lantana
2: Parkinsonia
3: Parthenium
4: Prickly acacia
5: Rubber vine
6: Siam weed
7: Snake weed
8: Negative (no weed)
"""
import os
import shutil
from PIL import Image

DEEPWEEDS_ROOT = "D:/datasets/DeepWeeds/DeepWeeds"
OUTPUT_ROOT = "D:/datasets/DeepWeeds/DeepWeeds_organized"

CLASS_NAMES = [
    "Chinee_apple",
    "Lantana",
    "Parkinsonia",
    "Parthenium",
    "Prickly_acacia",
    "Rubber_vine",
    "Siam_weed",
    "Snake_weed",
    "Negative"
]


def organize_by_class():
    """Organize images into class folders"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(OUTPUT_ROOT, class_name)
        os.makedirs(class_dir, exist_ok=True)

    image_files = [f for f in os.listdir(DEEPWEEDS_ROOT) if f.lower().endswith('.jpg')]
    print(f"Found {len(image_files)} images")

    class_counts = [0] * 9
    errors = 0

    for filename in image_files:
        try:
            class_idx = int(filename.split('-')[-1].replace('.jpg', ''))
            if class_idx < 0 or class_idx > 8:
                errors += 1
                continue

            src_path = os.path.join(DEEPWEEDS_ROOT, filename)
            dst_path = os.path.join(OUTPUT_ROOT, CLASS_NAMES[class_idx], filename)

            shutil.copy(src_path, dst_path)
            class_counts[class_idx] += 1
        except Exception as e:
            errors += 1
            continue

    print(f"\nClass distribution:")
    total = 0
    for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
        print(f"  {i}: {name}: {count} images")
        total += count
    print(f"\nTotal copied: {total}")
    print(f"Errors: {errors}")


def verify_images():
    """Verify all images are valid"""
    invalid = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(OUTPUT_ROOT, class_name)
        if not os.path.exists(class_dir):
            continue
        for f in os.listdir(class_dir):
            if f.lower().endswith('.jpg'):
                try:
                    img = Image.open(os.path.join(class_dir, f))
                    img.verify()
                except:
                    invalid.append(os.path.join(class_dir, f))

    if invalid:
        print(f"Found {len(invalid)} invalid images")
    else:
        print("All images verified successfully")


def create_fewshot_split():
    """Create base/val/novel splits for few-shot learning"""
    split_dir = os.path.join(OUTPUT_ROOT, 'fewshot_split')
    os.makedirs(split_dir, exist_ok=True)

    for split in ['base', 'val', 'novel']:
        os.makedirs(os.path.join(split_dir, split), exist_ok=True)

    n_classes = len(CLASS_NAMES)
    n_base = max(1, int(n_classes * 0.6))
    n_val = max(1, int(n_classes * 0.15))

    base_classes = CLASS_NAMES[:n_base]
    val_classes = CLASS_NAMES[n_base:n_base + n_val]
    novel_classes = CLASS_NAMES[n_base + n_val:]

    print(f"\nFew-shot split:")
    print(f"  Base: {base_classes}")
    print(f"  Val: {val_classes}")
    print(f"  Novel: {novel_classes}")

    for class_name in base_classes:
        src_dir = os.path.join(OUTPUT_ROOT, class_name)
        dst_dir = os.path.join(split_dir, 'base', class_name)
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if f.lower().endswith('.jpg'):
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    for class_name in val_classes:
        src_dir = os.path.join(OUTPUT_ROOT, class_name)
        dst_dir = os.path.join(split_dir, 'val', class_name)
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if f.lower().endswith('.jpg'):
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    for class_name in novel_classes:
        src_dir = os.path.join(OUTPUT_ROOT, class_name)
        dst_dir = os.path.join(split_dir, 'novel', class_name)
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if f.lower().endswith('.jpg'):
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    print(f"\nSplit created in: {split_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("DeepWeeds Dataset Organizer")
    print("=" * 60)

    organize_by_class()
    verify_images()
    create_fewshot_split()

    print("\nDone! DeepWeeds is ready for few-shot learning.")
    print(f"Organized path: {OUTPUT_ROOT}")
