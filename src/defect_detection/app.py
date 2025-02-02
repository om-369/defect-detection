"""Flask application for defect detection service."""

import os
from pathlib import Path
from typing import Optional

import yaml
from flask import Flask
from flask_login import LoginManager


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


config = Config()
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default-secret-key")
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
