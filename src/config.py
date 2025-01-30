"""Configuration management for the defect detection project."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from yaml.loader import SafeLoader


class Config:
    """Configuration manager for the defect detection project."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "Config":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration."""
        if not self._config:
            self.load_config()

    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / "config" / "base_config.yml"

        # Load base configuration
        with open(config_path, "r") as f:
            self._config = yaml.load(f, Loader=SafeLoader)

        # Override with environment-specific configuration if it exists
        env = os.getenv("APP_ENV", "development")
        env_config_path = base_dir / "config" / f"environment.{env}.yml"
        if env_config_path.exists():
            with open(env_config_path, "r") as f:
                env_config = yaml.load(f, Loader=SafeLoader)
                self._deep_update(self._config, env_config)

        # Create necessary directories
        self._setup_directories()

    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update a dictionary."""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        base_dir = Path(__file__).resolve().parent.parent
        directories = [
            self._config["data"]["raw_dir"],
            self._config["data"]["processed_dir"],
            self._config["data"]["labeled_dir"],
            self._config["data"]["model_dir"],
            self._config["data"]["upload_dir"],
            self._config["data"]["logs_dir"],
            self._config["monitoring"]["log_dir"],
        ]

        for directory in directories:
            dir_path = base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            value = self._config
            for k in key.split("."):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax."""
        return self.get(key)

    @property
    def config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self._config.copy()


# Create a global instance
config = Config()
