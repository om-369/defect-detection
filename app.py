"""Flask application for defect detection service."""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
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
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
    start_http_server,
)
from pythonjsonlogger import jsonlogger
from werkzeug.security import check_password_hash, generate_password_hash
import yaml

# Configuration Management
class Config:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from environment.yml file and environment variables"""
        config_path = Path("config/environment.yml")
        
        # Load from YAML if exists
        if config_path.exists():
            with open(config_path) as f:
                self._config = yaml.safe_load(f)
        
        # Override with environment variables
        env_prefix = "APP_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                self._config[config_key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)

# Initialize config
config = Config()

# Initialize Flask app with dynamic configuration
app = Flask(__name__)
app.config["SECRET_KEY"] = config.get("secret_key", os.urandom(24).hex())
app.config["MAX_CONTENT_LENGTH"] = int(config.get("max_content_length", 16 * 1024 * 1024))
app.config["UPLOAD_FOLDER"] = config.get("upload_folder", "uploads")

class DefectDetectionApp:
    def __init__(self):
        self.model_path = config.get("model_path", "models/latest")
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        self.upload_folder = config.get("upload_folder", "uploads")
        self.allowed_extensions = set(config.get("allowed_extensions", ["png", "jpg", "jpeg"]))
        
        # Ensure directories exist
        Path(self.upload_folder).mkdir(parents=True, exist_ok=True)
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

    def allowed_file(self, filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in self.allowed_extensions

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
users = {"admin": User("1", "admin", generate_password_hash("admin"))}


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
            model = tf.keras.models.load_model(defect_app.model_path)
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


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if error_message:
        logger.error(f"Error message: {error_message}")
        return jsonify({"error": error_message}), 500

    try:
        image_file = request.files['image']
        logger.debug(f"Received image file: {image_file.filename}")

        processed_image, image_error = process_image(image_file)

        if image_error:
            logger.error(f"Image processing error: {image_error}")
            return jsonify({"error": image_error}), 400

        # Assuming you have a function to make a prediction
        prediction = model.predict(processed_image)  # Ensure this returns a valid result
        logger.debug(f"Prediction result: {prediction}")

        result = prediction[0]  # Adjust based on your model output

        # Ensure result is a number before calling toFixed
        if isinstance(result, (int, float)):
            return jsonify({"prediction": result}), 200
        else:
            logger.error("Invalid prediction format")
            return jsonify({"error": "Invalid prediction format"}), 400

    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400


@app.route("/batch", methods=["GET"])
@login_required
def batch():
    """Render batch processing page."""
    return render_template("batch.html")


@app.route("/history")
@login_required
def history():
    """Render history page."""
    return render_template("history.html")


@app.route("/api/history")
@login_required
def get_history():
    """Get prediction history."""
    try:
        history_file = Path("monitoring/predictions/history.json")
        if not history_file.exists():
            return jsonify([])

        with open(history_file, "r") as f:
            history = json.load(f)
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error loading history: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard")
@login_required
def dashboard():
    """Render monitoring dashboard."""
    return render_template("dashboard.html")


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy" if model is not None else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.route("/reload-model", methods=["POST"])
@login_required
def reload_model():
    """Endpoint to reload the model."""
    success = load_model_safe()
    return jsonify(
        {
            "success": success,
            "message": (
                "Model reloaded successfully" if success else "Failed to reload model"
            ),
        }
    )


@app.route("/metrics", methods=["GET"])
def metrics() -> Tuple[bytes, int, Dict[str, str]]:
    """Endpoint for Prometheus metrics."""
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/status", methods=["GET"])
def status() -> Tuple[Dict[str, Any], int]:
    """Detailed status endpoint for monitoring."""
    status_info = {
        "status": "operational",
        "model_loaded": model is not None,
        "model_info": {
            "input_shape": model.input_shape if model else None,
            "output_shape": model.output_shape if model else None,
        },
        "system_info": {
            "timestamp": datetime.utcnow().isoformat(),
            "version": os.environ.get("VERSION", "dev"),
            "environment": os.environ.get("ENVIRONMENT", "development"),
        },
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
        if not defect_app.allowed_file(file.filename):
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


def secure_filename(filename):
    """Secure filename by removing special characters."""
    return filename.replace(" ", "_").replace("/", "_").replace("\\", "_")


if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)

    # Load the model
    load_model_safe()

    # Run the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
