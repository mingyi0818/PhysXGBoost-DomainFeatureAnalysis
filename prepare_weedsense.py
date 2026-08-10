"""
Download WeedSense dataset images from HuggingFace and prepare for few-shot learning.
"""
import os
import zipfile
import shutil
import pandas as pd
from huggingface_hub import hf_hub_download

LOCAL_DIR = "D:/datasets/WeedSense"
SPECIES_DIR = os.path.join(LOCAL_DIR, "all_species")


def download_split_images(split='train'):
    """Download images for a split"""
    zip_path = os.path.join(LOCAL_DIR, split, "images.zip")

    if os.path.exists(zip_path):
        print(f"{split}/images.zip already exists: {zip_path}")
        return zip_path

    print(f"Downloading {split} images (this may take a while)...")
    try:
        path = hf_hub_download(
            repo_id='baselab/weedsense',
            repo_type='dataset',
            filename=f'{split}/images.zip',
            local_dir=LOCAL_DIR,
        )
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Downloaded: {path} ({size_mb:.1f} MB)")
        return path
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def build_species_folders(split='train'):
    """Extract images and organize by species folders"""
    csv_path = os.path.join(LOCAL_DIR, split, f"{split}_data.csv")
    zip_path = os.path.join(LOCAL_DIR, split, "images.zip")

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return None
    if not os.path.exists(zip_path):
        print(f"Images zip not found: {zip_path}")
        return None

    # Check if already built
    split_species_dir = os.path.join(LOCAL_DIR, f"{split}_species")
    if os.path.exists(split_species_dir):
        n_species = len(os.listdir(split_species_dir))
        if n_species >= 16:
            print(f"{split} species folders already exist: {n_species} classes")
            return split_species_dir

    os.makedirs(split_species_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Extract all images to temp
    temp_dir = os.path.join(LOCAL_DIR, f"{split}_temp")
    if not os.path.exists(temp_dir):
        print(f"Extracting {split} images...")
        os.makedirs(temp_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        print(f"  Extracted to: {temp_dir}")

    # Organize by species
    print(f"Organizing {split} images by species...")
    species_list = df['species'].unique()
    for species in species_list:
        species_dir = os.path.join(split_species_dir, species)
        os.makedirs(species_dir, exist_ok=True)
        species_imgs = df[df['species'] == species]['img'].tolist()
        moved = 0
        for img_name in species_imgs:
            src = os.path.join(temp_dir, img_name)
            dst = os.path.join(species_dir, img_name)
            if os.path.exists(src):
                shutil.move(src, dst)
                moved += 1
        print(f"  {species}: {moved} images")

    # Clean up temp
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Done! {len(species_list)} species in {split_species_dir}")

    return split_species_dir


def build_all_species():
    """Combine train/val/test species into one folder"""
    if os.path.exists(SPECIES_DIR) and len(os.listdir(SPECIES_DIR)) >= 16:
        print(f"All species folder already exists: {SPECIES_DIR}")
        print(f"  Classes: {len(os.listdir(SPECIES_DIR))}")
        return SPECIES_DIR

    os.makedirs(SPECIES_DIR, exist_ok=True)

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(LOCAL_DIR, f"{split}_species")
        if not os.path.exists(split_dir):
            print(f"Skipping {split}: not found")
            continue
        for species in os.listdir(split_dir):
            src = os.path.join(split_dir, species)
            dst = os.path.join(SPECIES_DIR, species)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
            else:
                # Copy additional images
                for f in os.listdir(src):
                    src_f = os.path.join(src, f)
                    dst_f = os.path.join(dst, f)
                    if not os.path.exists(dst_f):
                        shutil.copy2(src_f, dst_f)

    n_species = len(os.listdir(SPECIES_DIR))
    total_imgs = sum(len(os.listdir(os.path.join(SPECIES_DIR, s)))
                     for s in os.listdir(SPECIES_DIR))
    print(f"\nCombined dataset: {n_species} species, {total_imgs} images")
    for s in sorted(os.listdir(SPECIES_DIR)):
        n = len(os.listdir(os.path.join(SPECIES_DIR, s)))
        print(f"  {s}: {n}")

    return SPECIES_DIR


if __name__ == "__main__":
    print("=" * 60)
    print("WeedSense Dataset Preparation")
    print("=" * 60)

    # Download all splits
    for split in ['train', 'val', 'test']:
        download_split_images(split)
        build_species_folders(split)

    # Combine all
    build_all_species()
