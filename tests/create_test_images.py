"""Create test images for unit tests."""

from pathlib import Path

import numpy as np
from PIL import Image


def create_test_images():
    """Create sample test images."""
    # Create test directories
    test_dir = Path("../data/test")
    defect_dir = test_dir / "defect"
    no_defect_dir = test_dir / "no_defect"

    defect_dir.mkdir(parents=True, exist_ok=True)
    no_defect_dir.mkdir(parents=True, exist_ok=True)

    # Create a sample defect image (red circle on white background)
    img_size = (224, 224)
    defect_img = Image.new("RGB", img_size, "white")
    pixels = defect_img.load()

    # Draw a red circle
    center = (img_size[0] // 2, img_size[1] // 2)
    radius = 50
    for x in range(img_size[0]):
        for y in range(img_size[1]):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius**2:
                pixels[x, y] = (255, 0, 0)  # Red

    defect_img.save(defect_dir / "defect_0.jpg")

    # Create a sample no-defect image (plain white)
    no_defect_img = Image.new("RGB", img_size, "white")
    no_defect_img.save(no_defect_dir / "no_defect_0.jpg")


if __name__ == "__main__":
    create_test_images()
