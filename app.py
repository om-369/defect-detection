"""Flask application for defect detection service."""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from google.cloud import storage
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


# Configuration Management
class Config:
    """Configuration management singleton class."""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from environment.yml file and environment variables."""
        try:
            with open("config/environment.yml", "r") as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning("No environment.yml file found, using defaults")
            self._config = {}

        # Override with environment variables
        for key in self._config:
            env_value = os.environ.get(key.upper())
            if env_value:
                self._config[key] = env_value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY", "your-secret-key"
)  # Change this in production

# Initialize config
config = Config()

# Initialize Flask app with dynamic configuration
app.config["SECRET_KEY"] = config.get("secret_key", os.urandom(24).hex())
app.config["MAX_CONTENT_LENGTH"] = int(
    config.get("max_content_length", 16 * 1024 * 1024)
)
app.config["UPLOAD_FOLDER"] = config.get("upload_folder", "uploads")


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


# Initialize defect detection app
defect_app = DefectDetectionApp()

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Configure logging
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Prometheus metrics
PREDICTION_REQUEST_COUNT = Counter(
    "prediction_requests_total", "Total prediction requests"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds"
)
PREDICTION_ERROR_COUNT = Counter("prediction_errors_total", "Total prediction errors")
ERROR_COUNT = Counter("error_count_total", "Total number of errors", ["error_type"])
MODEL_DOWNLOAD_COUNT = Counter("model_downloads_total", "Total model downloads")
MODEL_DOWNLOAD_ERROR_COUNT = Counter(
    "model_download_errors_total", "Total model download errors"
)

# Global variables
model = None
model_lock = threading.Lock()
prediction_queue = queue.Queue()
storage_client = storage.Client()


def download_model_from_gcs():
    """Download the latest model from GCS."""
    try:
        bucket_name = os.environ.get("MODEL_BUCKET")
        if not bucket_name:
            logger.error("MODEL_BUCKET environment variable not set")
            return False

        bucket = storage_client.bucket(bucket_name)
        model_blob = None
        latest_model = None
        latest_time = 0

        # Find the latest model
        for blob in bucket.list_blobs(prefix="models/"):
            if blob.name.endswith(".h5"):
                if blob.updated.timestamp() > latest_time:
                    latest_time = blob.updated.timestamp()
                    latest_model = blob.name
                    model_blob = blob

        if not model_blob:
            logger.error("No model found in GCS bucket")
            return False

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "model.h5")

        # Download the model
        model_blob.download_to_filename(model_path)
        MODEL_DOWNLOAD_COUNT.inc()
        logger.info(f"Downloaded model from GCS: {latest_model}")
        return True

    except Exception as e:
        MODEL_DOWNLOAD_ERROR_COUNT.inc()
        logger.error(f"Error downloading model from GCS: {str(e)}")
        return False


def load_model_safe():
    """Safely load the model with error handling."""
    global model
    try:
        with model_lock:
            if not os.path.exists("models/model.h5"):
                if not download_model_from_gcs():
                    return False
            model = tf.keras.models.load_model("models/model.h5")
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


# User class for authentication
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash


# Mock user database (replace with real database in production)
users = {"admin": User("1", "admin", generate_password_hash("admin"))}


@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == user_id:
            return user
    return None


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
    history_dir = Path("monitoring/predictions")
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / "history.json"
    history = []
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
    history.append(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "image_path": str(image_path),
            "result": result,
        }
    )
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


# File upload configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = users.get(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("index"))
        return "Invalid username or password"
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    return redirect(url_for("index"))


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """Handle single image prediction."""
    try:
        if "file" not in request.files:
            logger.error("No file part in request")
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        if not defect_app.allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return (
                jsonify({"error": "Invalid file type. Allowed types: png, jpg, jpeg"}),
                400,
            )

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Process image and make prediction
        with PREDICTION_LATENCY.time():
            PREDICTION_REQUEST_COUNT.inc()
            try:
                image_data = tf.io.read_file(filepath)
                image_data = process_image(image_data)
                with model_lock:
                    if model is None:
                        load_model_safe()
                    prediction = model.predict(image_data)

                result = {
                    "defect_probability": float(prediction[0][0]),
                    "has_defect": bool(prediction[0][0] > 0.5),
                }

                # Save prediction to history
                save_prediction_history(filepath, result)

                return jsonify(result)

            except Exception as e:
                PREDICTION_ERROR_COUNT.inc()
                logger.error(f"Error making prediction: {str(e)}")
                return jsonify({"error": str(e)}), 500

    except Exception as e:
        ERROR_COUNT.labels(error_type="prediction").inc()
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/batch")
@login_required
def batch():
    """Render batch processing page."""
    return render_template("batch.html")


@app.route("/history")
@login_required
def history():
    """Render history page."""
    return render_template("history.html")


@app.route("/get_history")
@login_required
def get_history():
    """Get prediction history."""
    history_file = Path("monitoring/predictions/history.json")
    if not history_file.exists():
        return jsonify([])

    with open(history_file, "r") as f:
        history = json.load(f)

    return jsonify(history)


@app.route("/dashboard")
@login_required
def dashboard():
    """Render monitoring dashboard."""
    return render_template("dashboard.html")


@app.route("/health")
def health():
    """Health check endpoint."""
    model_status = "healthy" if model is not None else "degraded"
    if model_status == "degraded":
        # Try to load model if it's not loaded
        if load_model_safe():
            model_status = "healthy"

    return jsonify(
        {
            "status": model_status,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.route("/reload_model", methods=["POST"])
@login_required
def reload_model():
    """Endpoint to reload the model."""
    try:
        if download_model_from_gcs() and load_model_safe():
            return jsonify(
                {"status": "success", "message": "Model reloaded successfully"}
            )
        else:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Failed to reload model. Check logs for details.",
                    }
                ),
                500,
            )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/metrics")
def metrics():
    """Endpoint for Prometheus metrics."""
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/status")
@login_required
def status():
    """Detailed status endpoint for monitoring."""
    return jsonify(
        {
            "status": "healthy" if model is not None else "degraded",
            "model_loaded": model is not None,
            "prediction_queue_size": prediction_queue.qsize(),
            "uptime": time.time() - start_time,
            "memory_usage": {
                "total": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
                "available": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.route("/batch_predict", methods=["POST"])
@login_required
def batch_predict():
    """Endpoint for making predictions on multiple images."""
    try:
        if "files[]" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist("files[]")
        if not files:
            return jsonify({"error": "No files selected"}), 400

        results = []
        for file in files:
            if file.filename == "":
                continue

            if not defect_app.allowed_file(file.filename):
                results.append(
                    {
                        "filename": file.filename,
                        "error": "Invalid file type",
                    }
                )
                continue

            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                with PREDICTION_LATENCY.time():
                    PREDICTION_REQUEST_COUNT.inc()
                    image_data = tf.io.read_file(filepath)
                    image_data = process_image(image_data)
                    with model_lock:
                        if model is None:
                            load_model_safe()
                        prediction = model.predict(image_data)

                result = {
                    "filename": filename,
                    "defect_probability": float(prediction[0][0]),
                    "has_defect": bool(prediction[0][0] > 0.5),
                }

                save_prediction_history(filepath, result)
                results.append(result)

            except Exception as e:
                PREDICTION_ERROR_COUNT.inc()
                results.append(
                    {
                        "filename": file.filename,
                        "error": str(e),
                    }
                )

        return jsonify(results)

    except Exception as e:
        ERROR_COUNT.labels(error_type="batch_prediction").inc()
        return jsonify({"error": str(e)}), 500


def allowed_file(filename):
    """Check if file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Start time for uptime tracking
start_time = time.time()


def get_host_config():
    """Get host configuration based on environment.
    
    Returns:
        tuple: (host, port) configuration for the server
        
    Note:
        In production environments like Cloud Run, we need to bind to all interfaces.
        This is safe because:
        1. Cloud Run provides its own security layer
        2. Only internal traffic within the Cloud Run container is allowed
        3. External traffic is handled by Cloud Run's load balancer
    """
    port = int(os.environ.get("PORT", 8080))
    
    # Default to localhost for security
    host = os.environ.get("HOST", "127.0.0.1")
    
    # Only bind to all interfaces in Cloud Run environment
    if os.environ.get("CLOUD_RUN") == "1":
        host = "0.0.0.0"  # nosec B104 - Binding to all interfaces is required and safe in Cloud Run
    
    return host, port


if __name__ == "__main__":
    start_http_server(8000)
    
    # Get host configuration
    host, port = get_host_config()
    
    # Run the Flask app with proper host binding
    app.run(host=host, port=port)
