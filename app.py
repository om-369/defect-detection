"""Flask application for defect detection service."""

import os
import time
import logging
from typing import Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from prometheus_client import Counter, Histogram, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from pythonjsonlogger import jsonlogger

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key')  # Change this in production

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Prometheus metrics
PREDICTION_REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
PREDICTION_ERROR_COUNT = Counter('prediction_errors_total', 'Total prediction errors')
ERROR_COUNT = Counter('error_count_total', 'Total number of errors', ['error_type'])

# Global variables
model = None
model_lock = threading.Lock()
prediction_queue = queue.Queue()

# User class for authentication
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

# Mock user database (replace with real database in production)
users = {
    'admin': User('1', 'admin', generate_password_hash('admin'))
}

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == user_id:
            return user
    return None

def load_model_safe():
    """Safely load the model with error handling."""
    global model
    try:
        with model_lock:
            model = tf.keras.models.load_model('models/model.h5')
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def process_image(image_data):
    """Process image data for prediction."""
    try:
        img = tf.image.decode_image(image_data)
        img = tf.image.resize(img, (224, 224))
        img = tf.expand_dims(img, 0)
        return img
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def save_prediction_history(image_path, result):
    """Save prediction results to history."""
    history_dir = Path('monitoring/predictions')
    history_dir.mkdir(parents=True, exist_ok=True)
    
    history_file = history_dir / 'history.json'
    history = []
    
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    
    history.append({
        'timestamp': datetime.utcnow().isoformat(),
        'image_path': str(image_path),
        'result': result
    })
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = users.get(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        
        return 'Invalid username or password'
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle prediction requests."""
    start_time = time.time()
    PREDICTION_REQUEST_COUNT.inc()
    
    try:
        if 'image' not in request.files:
            raise ValueError("No image file provided")
        
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Process image
        img = process_image(image_data)
        
        # Make prediction
        with model_lock:
            if model is None:
                if not load_model_safe():
                    raise ValueError("Model not available")
            prediction = model.predict(img)
        
        # Process results
        defect_probability = float(prediction[0][0])
        defect_detected = defect_probability > 0.5
        processing_time = time.time() - start_time
        
        result = {
            'defect_detected': defect_detected,
            'defect_probability': defect_probability,
            'processing_time': processing_time
        }
        
        # Save to history
        save_prediction_history(image_file.filename, result)
        
        PREDICTION_LATENCY.observe(processing_time)
        return jsonify(result)
    
    except Exception as e:
        PREDICTION_ERROR_COUNT.inc()
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['GET'])
@login_required
def batch():
    """Render batch processing page."""
    return render_template('batch.html')

@app.route('/history')
@login_required
def history():
    """Render history page."""
    return render_template('history.html')

@app.route('/api/history')
@login_required
def get_history():
    """Get prediction history."""
    try:
        history_file = Path('monitoring/predictions/history.json')
        if not history_file.exists():
            return jsonify([])
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error loading history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
@login_required
def dashboard():
    """Render monitoring dashboard."""
    return render_template('dashboard.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/reload-model', methods=['POST'])
@login_required
def reload_model():
    """Endpoint to reload the model."""
    success = load_model_safe()
    return jsonify({
        'success': success,
        'message': 'Model reloaded successfully' if success else 'Failed to reload model'
    })

@app.route('/metrics', methods=['GET'])
def metrics() -> Tuple[bytes, int, Dict[str, str]]:
    """Endpoint for Prometheus metrics."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

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
@login_required
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

def allowed_file(filename):
    """Check if file has an allowed extension."""
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_filename(filename):
    """Secure filename by removing special characters."""
    return filename.replace(" ", "_").replace("/", "_").replace("\\", "_")

if __name__ == '__main__':
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Load the model
    load_model_safe()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
