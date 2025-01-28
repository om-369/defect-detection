# Defect Detection System

[![CI/CD Pipeline](https://github.com/om-369/defect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/om-369/defect-detection/actions/workflows/ci.yml)

An AI-powered system for detecting manufacturing defects using deep learning and computer vision.

## Features

- Real-time defect detection using deep learning
- REST API for easy integration
- Containerized deployment
- CI/CD pipeline with GitHub Actions
- Cloud deployment on Google Cloud Run

## Tech Stack

- Python 3.9
- TensorFlow 2.x
- Flask
- Docker
- GitHub Actions
- Google Cloud Run

## Project Structure

```
defect-detection/
├── .github/
│   └── workflows/       # CI/CD pipeline configurations
├── src/
│   ├── data/           # Data preprocessing utilities
│   ├── models/         # Model architecture and training
│   └── utils/          # Helper functions
├── tests/              # Unit tests
├── app.py             # Flask application
├── train.py           # Model training script
├── Dockerfile         # Container configuration
└── requirements.txt   # Python dependencies
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/defect-detection.git
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

## Local Development

1. Run tests:
```bash
pytest tests/
```

2. Start Flask application:
```bash
python app.py
```

3. Build Docker image:
```bash
docker build -t defect-detection .
```

4. Run Docker container:
```bash
docker run -p 8080:8080 defect-detection
```

## API Endpoints

### Health Check
```
GET /health
```

### Predict
```
POST /predict
Content-Type: multipart/form-data
Body: image=@image.jpg
```

## Deployment

The application is automatically deployed to Google Cloud Run when changes are pushed to the main branch.

### Prerequisites for deployment:
1. DockerHub account
2. Google Cloud project with Cloud Run enabled
3. GitHub repository secrets configured:
   - DOCKER_USERNAME
   - DOCKER_PASSWORD
   - GCP_SA_KEY

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
