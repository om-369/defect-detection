"""Flask application for defect detection service."""

# Standard library imports
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
import yaml
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# Local imports
from .predict import predict_image


class Config:
    """Configuration management singleton class."""

    _instance = None
    config: Dict[str, Any] = {}

    def __new__(cls) -> "Config":
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file
        """
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)


config = Config()
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default-secret-key")


@app.route("/predict", methods=["POST"])
def predict() -> Dict[str, Any]:
    """Handle prediction request.

    Returns:
        Prediction results
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    temp_path = Path("temp") / filename
    temp_path.parent.mkdir(exist_ok=True)
    file.save(temp_path)

    try:
        result = predict_image(temp_path)
        return jsonify(result)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(debug=True)
