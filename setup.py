from setuptools import setup, find_packages

setup(
    name="defect-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy>=1.19.2",
        "opencv-python>=4.4.0",
        "scikit-learn>=0.23.2",
        "flask>=2.0.0",
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
    ],
    python_requires=">=3.10",
)
