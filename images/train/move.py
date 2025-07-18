import os
import shutil
from pathlib import Path
import sys

# Get destination directory from user input
if len(sys.argv) < 2:
    print("Usage: python move_images.py /path/to/target/folder")
    sys.exit(1)

target_dir = Path(sys.argv[1])
source_dir = Path('.')  # Current directory
num_images_to_move = 9000

# Ensure target directory exists
target_dir.mkdir(parents=True, exist_ok=True)

# Gather list of image files (adjust extensions if needed)
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
image_files = [f for f in source_dir.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]

# Limit to first N
images_to_move = image_files[:num_images_to_move]

# Move files
for img_path in images_to_move:
    shutil.move(str(img_path), str(target_dir / img_path.name))

print(f"Moved {len(images_to_move)} images from {source_dir.resolve()} to {target_dir.resolve()}")
