# Defect Detection System

[![CI/CD Pipeline](https://github.com/om-369/defect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/om-369/defect-detection/actions/workflows/ci.yml)

An AI-powered system for detecting manufacturing defects using deep learning and computer vision. The system provides real-time defect detection through a web interface and REST API.

## Features

- Real-time defect detection using deep learning
- Web interface for single and batch image processing
- REST API for easy integration
- Authentication and user management
- Monitoring dashboard with Prometheus metrics
- Automatic model updates from Google Cloud Storage
- Containerized deployment with health checks
- CI/CD pipeline with GitHub Actions
- Cloud deployment on Google Cloud Run

## Tech Stack

- Python 3.12
- TensorFlow 2.16+
- Flask + Flask-Login
- Docker with multi-stage builds
- GitHub Actions
- Google Cloud Run
- Google Cloud Storage
- Prometheus monitoring

## Project Structure

```
defect-detection/
├── .github/
│   └── workflows/        # CI/CD pipeline configurations
├── data/
│   ├── labeled/         # Training data
│   ├── uploads/         # Temporary storage for uploaded images
│   └── test/           # Test data
├── models/             # Model files
├── monitoring/         # Monitoring data and logs
├── scripts/           # Utility scripts
├── templates/         # HTML templates
├── tests/             # Unit tests
├── app.py            # Flask application
├── train.py          # Model training script
├── evaluate.py       # Model evaluation script
├── Dockerfile        # Container configuration
└── requirements.txt  # Python dependencies
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/om-369/defect-detection.git
cd defect-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Required
export SECRET_KEY=your-secret-key
export MODEL_BUCKET=your-gcs-bucket-name

# Optional
export PORT=8080  # Default: 8080
```

## Local Development

1. Generate test data:
```bash
python tests/create_test_images.py
```

2. Run tests:
```bash
pytest tests/
```

3. Start Flask application:
```bash
python app.py
```

4. Build and run with Docker:
```bash
docker build -t defect-detection .
docker run -p 8080:8080 \
    -e SECRET_KEY=your-secret-key \
    -e MODEL_BUCKET=your-gcs-bucket-name \
    defect-detection
```

## API Endpoints

### Authentication
```
POST /login
Content-Type: application/x-www-form-urlencoded
Body: username=admin&password=admin
```

### Health Check
```
GET /health
Response: {"status": "healthy", "timestamp": "..."}
```

### Single Image Prediction
```
POST /predict
Content-Type: multipart/form-data
Body: file=@image.jpg
Response: {
    "defect_probability": 0.95,
    "has_defect": true
}
```

### Batch Prediction
```
POST /batch_predict
Content-Type: multipart/form-data
Body: files[]=@image1.jpg&files[]=@image2.jpg
Response: [
    {
        "filename": "image1.jpg",
        "defect_probability": 0.95,
        "has_defect": true
    },
    ...
]
```

### Model Management
```
POST /reload_model
Response: {"status": "success", "message": "Model reloaded successfully"}
```

### Monitoring
```
GET /metrics  # Prometheus metrics
GET /status   # Detailed system status
```

## Deployment

The application is automatically deployed to Google Cloud Run when changes are pushed to the main branch.

### Prerequisites for deployment:
1. DockerHub account
2. Google Cloud project with Cloud Run and Cloud Storage enabled
3. GitHub repository secrets configured:
   - DOCKER_USERNAME
   - DOCKER_PASSWORD
   - GCP_PROJECT_ID
   - GCP_SA_KEY
   - APP_SECRET_KEY
   - GCP_MODEL_BUCKET

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
