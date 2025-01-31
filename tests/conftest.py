import pytest
import os
import shutil

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary test data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create train/val/test directories
    for split in ["train", "valid", "test"]:
        (data_dir / split / "images").mkdir(parents=True)
        (data_dir / split / "labels").mkdir(parents=True)
    
    return data_dir

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up any files created during tests."""
    yield
    # Clean up any temporary files created during tests
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")
    if os.path.exists("models/temp"):
        shutil.rmtree("models/temp")
