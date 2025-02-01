"""Flask application for defect detection."""
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_login import LoginManager, login_required
from prometheus_client import start_http_server, Counter, Histogram
import time
import logging

from .predict import predict_image
from .train import train_model
from .evaluate import evaluate_model


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = os.environ.get('MODEL_PATH', 'models/model.pth')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
PREDICTION_TIME = Histogram('prediction_time_seconds',
                          'Time spent processing prediction')
PREDICTION_COUNT = Counter('prediction_total', 'Total number of predictions')


# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)


def main():
    """Run the Flask application."""
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '127.0.0.1')

    # Start metrics server
    metrics_port = int(os.environ.get('METRICS_PORT', 8000))
    start_http_server(metrics_port)

    app.run(host=host, port=port)


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Predict endpoint for single image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    try:
        start_time = time.time()
        result = predict_image(file)
        prediction_time = time.time() - start_time

        # Record metrics
        PREDICTION_TIME.observe(prediction_time)
        PREDICTION_COUNT.inc()

        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
@login_required
def batch_predict():
    """Predict endpoint for multiple images."""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    results = []
    for file in files:
        try:
            result = predict_image(file)
            results.append({
                'filename': file.filename,
                **result
            })
        except Exception as e:
            logger.error(f"Prediction error for {file.filename}: {str(e)}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify(results)


@app.route('/train', methods=['POST'])
@login_required
def train():
    """Train endpoint to retrain the model."""
    try:
        train_model()
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully'
        })
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/evaluate', methods=['POST'])
@login_required
def evaluate():
    """Evaluate endpoint to assess model performance."""
    try:
        metrics = evaluate_model()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    main()
