"""Script to organize welding images into labeled directories."""

import shutil
from pathlib import Path
from typing import Optional

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
LABELED_DIR = BASE_DIR / "data" / "labeled"


def get_base_name(filename: str) -> str:
    """Get base name without the random suffix."""
    # Split by rf. and take the first part
    parts = filename.split("rf.")
    if len(parts) > 1:
        return parts[0].rstrip("_")
    return filename


def check_label_file_exists(directory: Path, img_path: Path) -> bool:
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


def organize_data() -> None:
    """Organize images into defect and no_defect directories."""
    print("Starting data organization...")

    # Create directories if they don't exist
    defect_dir = LABELED_DIR / "defect"
    no_defect_dir = LABELED_DIR / "no_defect"
    defect_dir.mkdir(parents=True, exist_ok=True)
    no_defect_dir.mkdir(parents=True, exist_ok=True)

    # Get list of images in raw directory
    raw_images = list(RAW_DIR.glob("*.jpg"))
    print(f"Found {len(raw_images)} images")

    # Process each image
    for img_path in raw_images:
        # Check if label file exists
        has_label = check_label_file_exists(RAW_DIR, img_path)

        # Move to appropriate directory
        if has_label:
            dest_dir = defect_dir
        else:
            dest_dir = no_defect_dir

        # Copy image to destination
        shutil.copy2(img_path, dest_dir / img_path.name)

    # Print summary
    defect_count = len(list(defect_dir.glob("*.jpg")))
    no_defect_count = len(list(no_defect_dir.glob("*.jpg")))
    print(f"Organized {defect_count} defect images")
    print(f"Organized {no_defect_count} no-defect images")


if __name__ == "__main__":
    organize_data()
