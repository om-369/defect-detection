"""Flask application for defect detection service."""

import os
import time
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.models.model import create_model, compile_model
from src.data.preprocessing import load_image, resize_image, normalize_image
from config import IMG_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics
PREDICTION_REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total number of prediction requests'
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction requests'
)
ERROR_COUNT = Counter(
    'error_count_total',
    'Total number of errors',
    ['error_type']
)

# Load model
try:
    model = create_model()
    model = compile_model(model)
    model.load_weights('models/latest.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    ERROR_COUNT.labels(error_type='model_load').inc()

@app.route('/metrics', methods=['GET'])
def metrics() -> Tuple[bytes, int, Dict[str, str]]:
    """Endpoint for Prometheus metrics."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/health', methods=['GET'])
def health_check() -> Tuple[Dict[str, Any], int]:
    """Health check endpoint."""
    health_status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.utcnow().isoformat(),
        'version': os.environ.get('VERSION', 'dev')
    }
    return jsonify(health_status), 200

@app.route('/predict', methods=['POST'])
def predict() -> Tuple[Dict[str, Any], int]:
    """Endpoint for defect detection predictions."""
    PREDICTION_REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        if model is None:
            ERROR_COUNT.labels(error_type='model_not_loaded').inc()
            return jsonify({'error': 'Model not loaded'}), 503

        if 'image' not in request.files:
            ERROR_COUNT.labels(error_type='no_image').inc()
            return jsonify({'error': 'No image provided'}), 400

        # Load and preprocess image
        image_file = request.files['image']
        image = load_image(image_file)
        image = resize_image(image, IMG_SIZE)
        image = normalize_image(image)
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)
        probability = float(prediction[0, 0])
        
        # Format response
        response = {
            'defect_probability': probability,
            'defect_detected': probability > 0.5,
            'processing_time': time.time() - start_time,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Prediction made successfully: {response}")
        PREDICTION_LATENCY.observe(time.time() - start_time)
        return jsonify(response), 200

    except Exception as e:
        ERROR_COUNT.labels(error_type='prediction_error').inc()
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/status', methods=['GET'])
def status() -> Tuple[Dict[str, Any], int]:
    """Detailed status endpoint for monitoring."""
    status_info = {
        'status': 'operational',
        'model_loaded': model is not None,
        'model_info': {
            'input_shape': model.input_shape if model else None,
            'output_shape': model.output_shape if model else None
        },
        'system_info': {
            'timestamp': datetime.utcnow().isoformat(),
            'version': os.environ.get('VERSION', 'dev'),
            'environment': os.environ.get('ENVIRONMENT', 'development')
        }
    }
    return jsonify(status_info), 200

@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Endpoint for making predictions on multiple images."""
    # Check if images were uploaded
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    files = request.files.getlist("images")
    results = []

    for file in files:
        # Check file type
        if not allowed_file(file.filename):
            continue

        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join("/tmp", filename)
            file.save(temp_path)

            # Preprocess image
            img = load_image(temp_path)
            img = resize_image(img)
            img = normalize_image(img)
            img = np.expand_dims(img, axis=0)

            # Make prediction
            prediction = float(model.predict(img)[0, 0])

            # Clean up
            os.remove(temp_path)

            # Add result
            results.append(
                {
                    "filename": file.filename,
                    "prediction": "defect" if prediction > 0.5 else "no_defect",
                    "confidence": float(
                        prediction if prediction > 0.5 else 1 - prediction
                    ),
                }
            )

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return jsonify({"results": results}), 200

@app.errorhandler(404)
def not_found(error: Any) -> Tuple[Dict[str, str], int]:
    """Handle 404 errors."""
    ERROR_COUNT.labels(error_type='not_found').inc()
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error: Any) -> Tuple[Dict[str, str], int]:
    """Handle 500 errors."""
    ERROR_COUNT.labels(error_type='internal_error').inc()
    return jsonify({'error': 'Internal server error'}), 500

def allowed_file(filename):
    """Check if file has an allowed extension."""
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_filename(filename):
    """Secure filename by removing special characters."""
    return filename.replace(" ", "_").replace("/", "_").replace("\\", "_")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
