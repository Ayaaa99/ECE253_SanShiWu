import csv
import random
import shutil
from pathlib import Path

# CONFIGURATION
INPUT_DIR = "images/angle_dataset" 
OUTPUT_DIR = "images/finetune_dataset"
VAL_SPLIT = 0.2
SEED = 42

random.seed(SEED)

# Collect and shuffle images
extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
images = [f for f in Path(INPUT_DIR).iterdir() if f.suffix.lower() in extensions]
random.shuffle(images)

# Split into train/val
split_idx = int(len(images) * VAL_SPLIT)
splits = {"val": images[:split_idx], "train": images[split_idx:]}

# Process each split
for split_name, files in splits.items():
    img_dir = Path(OUTPUT_DIR) / split_name / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = []
    for src in files:
        shutil.copy2(src, img_dir / src.name)
        annotations.append({"image_path": f"images/{src.name}", "plate_text": src.stem.upper()})
    
    csv_path = Path(OUTPUT_DIR) / split_name / "annotations.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "plate_text"])
        writer.writeheader()
        writer.writerows(annotations)
    
    print(f"{split_name}: {len(files)} images -> {csv_path}")