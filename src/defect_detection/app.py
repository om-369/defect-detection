"""Flask application for defect detection service."""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user
from google.cloud import storage
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, start_http_server
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from defect_detection.models.model import DefectDetectionModel
from defect_detection.preprocessing import preprocess


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
