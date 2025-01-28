"""Flask application for defect detection service."""

import os
import logging
from typing import Dict, Any, Tuple

import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf

from src.models.model import create_model, compile_model
from src.data.preprocessing import load_image, resize_image, normalize_image
from config import IMG_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model
try:
    model = create_model()
    model = compile_model(model)
    model.load_weights('models/latest.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.route('/health', methods=['GET'])
def health_check() -> Tuple[Dict[str, Any], int]:
    """Health check endpoint."""
    health_status = {
        'status': 'healthy',
        'model_loaded': model is not None
    }
    return jsonify(health_status), 200

@app.route('/predict', methods=['POST'])
def predict() -> Tuple[Dict[str, Any], int]:
    """Endpoint for defect detection predictions."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
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
            'defect_detected': probability > 0.5
        }
        
        logger.info(f"Prediction made successfully: {response}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error: Any) -> Tuple[Dict[str, str], int]:
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

def allowed_file(filename):
    """Check if file has an allowed extension."""
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
