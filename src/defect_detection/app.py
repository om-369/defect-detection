"""Flask application for defect detection service."""

# Standard library imports
import os
from pathlib import Path
from typing import Dict, Union

# Third-party imports
from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Local imports
from .config import Config
from .predict import PredictionDict, predict_single_image

config = Config()
app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "default-secret-key")
app.config["UPLOAD_FOLDER"] = str(config.upload_dir)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)


@app.route("/", methods=["GET"])
def index() -> str:
    """Serve the index page.

    Returns:
        HTML content of the index page
    """
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    """Handle prediction requests.

    Returns:
        Prediction results or error message
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"})

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        return jsonify({"error": "Invalid file type. Please upload an image file."})

    try:
        filename = secure_filename(file.filename)
        filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
        file.save(filepath)

        result = predict_single_image(filepath)
        
        # Clean up the uploaded file
        filepath.unlink(missing_ok=True)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


def main():
    """Run the Flask application."""
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
