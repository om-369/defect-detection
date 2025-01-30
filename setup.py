"""Setup script for the defect-detection package."""

from setuptools import setup, find_packages

setup(
    name="defect-detection",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "flask>=2.0.0",
        "numpy>=1.19.2",
        "opencv-python>=4.4.0",
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "scikit-learn>=0.23.2",
        "tensorflow>=2.0.0",
    ],
    python_requires=">=3.10",
)
