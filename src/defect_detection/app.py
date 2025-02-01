"""Flask application for defect detection service."""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    login_required,
    login_user,
    logout_user,
)
from google.cloud import storage
from google.oauth2 import service_account
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
    start_http_server,
)
from pythonjsonlogger import jsonlogger
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

class Config:
    """Configuration management singleton class."""

    _instance: Optional["Config"] = None

    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration."""
        if not hasattr(self, "config"):
            config_path = Path("config/config.yaml")
            with open(config_path) as f:
                self.config = yaml.safe_load(f)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get configuration value."""
        return self.config.get(key, default)


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default-secret-key")

# Initialize config
config = Config()

# Initialize defect detection app
class DefectDetectionApp:
    def __init__(self):
        self.model_path = config.get("model_path", "models/latest")
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        self.upload_folder = config.get("upload_folder", "uploads")
        self.allowed_extensions = set(
            config.get("allowed_extensions", ["png", "jpg", "jpeg"])
        )
        Path(self.upload_folder).mkdir(parents=True, exist_ok=True)
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

    def allowed_file(self, filename: str) -> bool:
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.allowed_extensions
        )

defect_app = DefectDetectionApp()

# Download the latest model from GCS
def download_model_from_gcs():
    """Download the latest model from GCS."""
    try:
        bucket_name = os.environ.get("MODEL_BUCKET")
        if not bucket_name:
            logging.error("MODEL_BUCKET environment variable not set")
            return False

        bucket = storage.Client().bucket(bucket_name)
        model_blob = None
        latest_model = None
        latest_time = 0

        # Find the latest model
        for blob in bucket.list_blobs(prefix="models/"):
            if blob.name.endswith(".pth"):
                if blob.updated.timestamp() > latest_time:
                    latest_time = blob.updated.timestamp()
                    latest_model = blob.name
                    model_blob = blob

        if not model_blob:
            logging.error("No model found in GCS bucket")
            return False

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "model.pth")

        # Download the model
        model_blob.download_to_filename(model_path)
        logging.info(f"Downloaded model from GCS: {latest_model}")
        return True

    except Exception as e:
        logging.error(f"Error downloading model from GCS: {str(e)}")
        return False

# Safely load the model with error handling
def load_model_safe():
    """Safely load the model with error handling."""
    global model
    try:
        model = torch.load("models/model.pth")
        logging.info("Model loaded successfully")
        return True
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return False

# Load model
def load_model(model_path):
    """Load model from file."""
    try:
        model = torch.load(model_path)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

# Make prediction on image
def predict(model, filepath):
    """Make prediction on image."""
    try:
        img = torch.load(filepath)
        img = torch.unsqueeze(img, 0)
        prediction = model(img)
        result = {
            "class": torch.argmax(prediction[0]),
            "confidence": float(torch.max(prediction[0])) * 100,
            "all_probabilities": prediction[0].tolist(),
        }
        return result
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        raise

# User class for authentication
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

# Mock user database (replace with real database in production)
users = {"admin": User("1", "admin", generate_password_hash("admin"))}

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login."""
    return users.get(user_id)

# Process image data for prediction
def process_image(image_data):
    """Process image data for prediction."""
    try:
        img = torch.load(image_data)
        img = torch.unsqueeze(img, 0)
        return img
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise

# Save prediction results to history
def save_prediction_history(image_path, result):
    """Save prediction results to history."""
    history_file = Path("history.json")
    history = []
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)

    history.append({
        "timestamp": datetime.now().isoformat(),
        "image": str(image_path),
        "result": result
    })

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

# File upload configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Check if file has an allowed extension
def allowed_file(filename):
    """Check if file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# Initialize Prometheus metrics
PREDICTION_REQUEST_COUNT = Counter(
    "prediction_requests_total", "Total prediction requests"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds"
)
PREDICTION_ERROR_COUNT = Counter("prediction_errors_total", "Total prediction errors")

# Start time for uptime tracking
start_time = time.time()

# Get host configuration based on environment
def get_host_config():
    """Get host configuration based on environment."""
    if os.environ.get("ENVIRONMENT") == "production":
        return "0.0.0.0", int(os.environ.get("PORT", 8080))
    return "localhost", 5000

# Render the main page
@app.route("/")
@login_required
def index():
    """Render the main page."""
    return render_template("index.html")

# Handle user login
@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = users.get(username)

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("index"))

        return "Invalid username or password"

    return render_template("login.html")

# Handle user logout
@app.route("/logout")
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    return redirect(url_for("login"))

# Handle defect detection prediction
@app.route("/predict", methods=["POST"])
@login_required
def predict_defect():
    """Handle defect detection prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = Path(UPLOAD_FOLDER) / filename
        file.save(filepath)

        result = predict(model, str(filepath))
        save_prediction_history(filepath, result)

        return jsonify({
            "filename": filename,
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Render batch processing page
@app.route("/batch")
@login_required
def batch():
    """Render batch processing page."""
    return render_template("batch.html")

# Render history page
@app.route("/history")
@login_required
def history():
    """Render history page."""
    return render_template("history.html")

# Get prediction history
@app.route("/api/history")
@login_required
def get_history():
    """Get prediction history."""
    history_file = Path("history.json")
    if not history_file.exists():
        return jsonify([])

    with open(history_file) as f:
        history = json.load(f)

    return jsonify(history)

# Render monitoring dashboard
@app.route("/dashboard")
@login_required
def dashboard():
    """Render monitoring dashboard."""
    return render_template("dashboard.html")

# Health check endpoint
@app.route("/health")
def health():
    """Health check endpoint."""
    try:
        uptime = time.time() - start_time
        model_loaded = model is not None

        return jsonify({
            "status": "healthy",
            "uptime": uptime,
            "model_loaded": model_loaded
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# Endpoint to reload the model
@app.route("/reload", methods=["POST"])
@login_required
def reload_model():
    """Endpoint to reload the model."""
    try:
        if download_model_from_gcs():
            load_model_safe()
            return jsonify({
                "status": "success",
                "message": "Model reloaded successfully"
            })
        return jsonify({
            "status": "unchanged",
            "message": "No new model available"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Endpoint for Prometheus metrics
@app.route("/metrics")
def metrics():
    """Endpoint for Prometheus metrics."""
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# Detailed status endpoint for monitoring
@app.route("/status")
def status():
    """Detailed status endpoint for monitoring."""
    status_info = {
        "uptime": time.time() - start_time,
        "model_info": {
            "loaded": model is not None,
            "path": str(defect_app.model_path),
        },
    }
    return jsonify(status_info)

# Endpoint for making predictions on multiple images
@app.route("/batch/predict", methods=["POST"])
@login_required
def batch_predict():
    """Endpoint for making predictions on multiple images."""
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected"}), 400

    results = []
    errors = []

    for file in files:
        if file.filename == "":
            continue

        if not allowed_file(file.filename):
            errors.append(f"File type not allowed: {file.filename}")
            continue

        try:
            filename = secure_filename(file.filename)
            filepath = Path(UPLOAD_FOLDER) / filename
            file.save(filepath)

            result = predict(model, str(filepath))
            save_prediction_history(filepath, result)

            results.append({
                "filename": filename,
                "prediction": result
            })
        except Exception as e:
            errors.append(f"Error processing {file.filename}: {str(e)}")

    return jsonify({
        "results": results,
        "errors": errors
    })

if __name__ == "__main__":
    start_http_server(8000)
    host, port = get_host_config()
    app.run(host=host, port=port)
