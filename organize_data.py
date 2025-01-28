"""Script to organize welding images into labeled directories."""

import os
import shutil
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
LABELED_DIR = BASE_DIR / "data" / "labeled"


def get_base_name(filename):
    """Get base name without the random suffix."""
    # Split by rf. and take the first part
    parts = filename.split("rf.")
    if len(parts) > 1:
        return parts[0].rstrip("_")
    return filename


def check_label_file_exists(directory, img_path):
    """Check if either .txt or .json label file exists."""
    # Try exact match first
    txt_path = directory / f"{img_path.stem}.txt"
    json_path = directory / f"{img_path.stem}.json"
    if txt_path.exists() or json_path.exists():
        return True

    # Try matching by base name
    base_name = get_base_name(img_path.stem)
    for label_file in directory.glob("*.*"):
        if label_file.suffix in [".txt", ".json"]:
            label_base = get_base_name(label_file.stem)
            if base_name == label_base:
                return True
    return False


def organize_data():
    """Organize images into defect and no_defect directories."""
    print("Starting data organization...")

    # Create directories if they don't exist
    defect_dir = LABELED_DIR / "defect"
    no_defect_dir = LABELED_DIR / "no_defect"
    defect_dir.mkdir(parents=True, exist_ok=True)
    no_defect_dir.mkdir(parents=True, exist_ok=True)

    # Get list of images in raw directory
    raw_images = list(RAW_DIR.glob("*.jpg"))
    print(f"Found {len(raw_images)} images in raw directory")

    defect_count = 0
    no_defect_count = 0
    unmatched_count = 0

    # Process each image
    for img_path in raw_images:
        print(f"\nProcessing: {img_path.name}")
        print(f"Base name: {get_base_name(img_path.stem)}")

        # Check if there's a corresponding label file in defect directory
        if check_label_file_exists(defect_dir, img_path):
            # Copy to defect directory
            target_path = defect_dir / img_path.name
            if not target_path.exists():  # Only copy if not already there
                shutil.copy2(str(img_path), str(target_path))
                defect_count += 1
                print("-> Copied to defect directory")
        # Check in no_defect directory
        elif check_label_file_exists(no_defect_dir, img_path):
            target_path = no_defect_dir / img_path.name
            if not target_path.exists():  # Only copy if not already there
                shutil.copy2(str(img_path), str(target_path))
                no_defect_count += 1
                print("-> Copied to no_defect directory")
        else:
            unmatched_count += 1
            print("-> No label found")

    print(f"\nSummary:")
    print(f"- Organized {defect_count} defect images")
    print(f"- Organized {no_defect_count} no-defect images")
    print(f"- Found {unmatched_count} images without labels")


if __name__ == "__main__":
    organize_data()
