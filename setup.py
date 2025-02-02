from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("-")]

setup(
    name="defect_detection",
    version="1.0.0",
    author="Om-369",
    author_email="author@example.com",
    description="A defect detection system using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/om-369/defect-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "bandit>=1.7.0",
            "safety>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "defect-detect=defect_detection.app:main",
        ],
    },
)
