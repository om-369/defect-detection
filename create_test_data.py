"""Script to create synthetic test data for defect detection."""

from pathlib import Path

import cv2
import numpy as np


def create_synthetic_image(size=(224, 224), has_defect=False):
    """Create a synthetic image with or without defects."""
    # Create base image
    image = (
        np.ones((size[0], size[1], 3), dtype=np.uint8) * 200
    )  # Light gray background

    # Add some random texture
    noise = np.random.normal(0, 25, size).astype(np.uint8)
    image[:, :, 0] += noise
    image[:, :, 1] += noise
    image[:, :, 2] += noise

    if has_defect:
        # Add random defects
        num_defects = np.random.randint(1, 4)
        for _ in range(num_defects):
            # Random position and size for defect
            x = np.random.randint(0, size[0])
            y = np.random.randint(0, size[1])
            radius = np.random.randint(5, 20)
            color = np.random.randint(0, 100, 3).tolist()  # Dark color for defect

            # Draw defect
            cv2.circle(image, (x, y), radius, color, -1)

    return image


def main():
    """Create synthetic dataset."""
    # Paths
    defect_dir = Path("data/labeled/defect")
    no_defect_dir = Path("data/labeled/no_defect")

    # Create 10 images for each class
    for i in range(10):
        # Create defect image
        defect_img = create_synthetic_image(has_defect=True)
        cv2.imwrite(str(defect_dir / f"defect_{i}.jpg"), defect_img)

        # Create non-defect image
        no_defect_img = create_synthetic_image(has_defect=False)
        cv2.imwrite(str(no_defect_dir / f"no_defect_{i}.jpg"), no_defect_img)

    print("Created synthetic dataset:")
    print(f"- Defect images: {len(list(defect_dir.glob('*.jpg')))} images")
    print(f"- No-defect images: {len(list(no_defect_dir.glob('*.jpg')))} images")


if __name__ == "__main__":
    main()
