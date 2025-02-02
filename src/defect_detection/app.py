"""Flask application for defect detection service."""

# Standard library imports
import os
from pathlib import Path
from typing import Dict, Union

# Third-party imports
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# Local imports
from .config import Config
from .predict import predict_single_image

config = Config()
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "default-secret-key")
app.config["UPLOAD_FOLDER"] = "uploads"

# Ensure upload directory exists
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)


@app.route("/", methods=["GET"])
def index() -> str:
    """Serve the index page.

    Returns:
        HTML content of the index page
    """
    return """
    <html>
        <head>
            <title>Defect Detection</title>
        </head>
        <body>
            <h1>Defect Detection Service</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <input type="submit" value="Upload and Predict">
            </form>
        </body>
    </html>
    """


@app.route("/predict", methods=["POST"])
def predict() -> Union[Dict[str, str], str]:
    """Handle prediction requests.

    Returns:
        Prediction results or error message
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"})

    filename = secure_filename(file.filename)
    filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
    file.save(filepath)

    try:
        result = predict_single_image(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
