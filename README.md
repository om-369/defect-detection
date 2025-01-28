# Manufacturing Defect Detection Project

This project implements a deep learning-based solution for detecting defects in manufacturing images using TensorFlow.

## Project Structure
```
comp_vision/
├── data/
│   ├── raw/                # Raw images
│   ├── processed/          # Preprocessed images
│   └── labeled/            # Labeled dataset
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── model.py
│   │   └── training.py
│   └── utils/
│       ├── visualization.py
│       └── metrics.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
├── config.py
├── train.py
└── predict.py
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
```bash
python src/data/preprocessing.py
```

2. Train Model:
```bash
python train.py
```

3. Make Predictions:
```bash
python predict.py
```

## Model Architecture

The project uses a ResNet50-based architecture fine-tuned for defect detection. The model includes custom top layers optimized for our specific use case.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- IoU (Intersection over Union)

## License

MIT License
