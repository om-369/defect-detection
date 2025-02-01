"""Source code for the defect detection project."""

from importlib.metadata import PackageNotFoundError, version

from .preprocessing import load_dataset, preprocess, preprocess_batch

try:
    __version__ = version("defect_detection")
except PackageNotFoundError:
    __version__ = "unknown"

__package__ = "defect_detection"

__all__ = ["load_dataset", "preprocess", "preprocess_batch"]
