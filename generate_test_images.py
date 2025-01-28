"""Generate test images for CI/CD pipeline testing."""

import numpy as np
import cv2
from pathlib import Path

def create_test_image(size=(224, 224), has_defect=False):
    """Create a test image with optional defect."""
    # Create base image
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 200
    
    if has_defect:
        # Add a defect (black rectangle)
        cv2.rectangle(image, (50, 50), (100, 100), (0, 0, 0), -1)
    
    return image

def main():
    """Generate test images."""
    test_dir = Path("data/test")
    defect_dir = test_dir / "defect"
    no_defect_dir = test_dir / "no_defect"
    
    # Create 5 images for each class
    for i in range(5):
        # Create and save defect image
        defect_img = create_test_image(has_defect=True)
        cv2.imwrite(str(defect_dir / f"defect_{i}.jpg"), defect_img)
        
        # Create and save non-defect image
        no_defect_img = create_test_image(has_defect=False)
        cv2.imwrite(str(no_defect_dir / f"no_defect_{i}.jpg"), no_defect_img)
    
    print("Created test images:")
    print(f"- Defect images: {len(list(defect_dir.glob('*.jpg')))} images")
    print(f"- No-defect images: {len(list(no_defect_dir.glob('*.jpg')))} images")

if __name__ == "__main__":
    main()
