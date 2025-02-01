"""Setup script for the defect_detection package."""

from setuptools import setup, find_namespace_packages

setup(
    name="defect_detection",
    version="0.1.0",
    packages=find_namespace_packages(include=["src.*"]),
    package_dir={"": "."},
    include_package_data=True,
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.12",  # Updated Python version requirement
)
