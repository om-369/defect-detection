"""Automated backup system for prediction history and model data."""

import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import schedule

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages automated backups of system data."""

    def __init__(self, base_dir: str = "backups"):
        """Initialize backup manager.

        Args:
            base_dir: Base directory for storing backups
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.predictions_dir = self.base_dir / "predictions"
        self.models_dir = self.base_dir / "models"
        self.configs_dir = self.base_dir / "configs"

        for directory in [self.predictions_dir, self.models_dir, self.configs_dir]:
            directory.mkdir(exist_ok=True)

    def backup_predictions(self) -> bool:
        """Backup prediction history."""
        try:
            source = Path("monitoring/predictions/history.json")
            if not source.exists():
                logger.warning("No prediction history to backup")
                return False

            # Create timestamped backup file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = self.predictions_dir / f"predictions_{timestamp}.json"

            # Copy and compress the file
            with open(source, "r") as f:
                data = json.load(f)

            with open(backup_file, "w") as f:
                json.dump(data, f, indent=2)

            # Create zip archive
            zip_file = backup_file.with_suffix(".zip")
            with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(backup_file, backup_file.name)

            # Remove uncompressed file
            backup_file.unlink()

            # Cleanup old backups (keep last 30 days)
            self._cleanup_old_backups(self.predictions_dir, days=30)

            logger.info(f"Prediction history backed up to {zip_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup predictions: {str(e)}")
            return False

    def backup_model(self, model_path: Path) -> bool:
        """Backup model files."""
        try:
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return False

            # Create timestamped backup file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = self.models_dir / f"model_{timestamp}.h5"

            # Copy model file
            shutil.copy2(model_path, backup_file)

            # Create zip archive
            zip_file = backup_file.with_suffix(".zip")
            with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(backup_file, backup_file.name)

            # Remove uncompressed file
            backup_file.unlink()

            # Cleanup old backups (keep last 5 versions)
            self._cleanup_old_backups(self.models_dir, keep_versions=5)

            logger.info(f"Model backed up to {zip_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup model: {str(e)}")
            return False

    def backup_configs(self) -> bool:
        """Backup configuration files."""
        try:
            # List of config files to backup
            config_files = [
                ".env",
                "config.py",
                "requirements.txt",
                ".github/workflows/ci.yml",
            ]

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.configs_dir / f"configs_{timestamp}"
            backup_dir.mkdir(exist_ok=True)

            # Copy each config file
            for config_file in config_files:
                source = Path(config_file)
                if source.exists():
                    dest = backup_dir / source.name
                    shutil.copy2(source, dest)

            # Create zip archive
            zip_file = backup_dir.with_suffix(".zip")
            with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file in backup_dir.glob("*"):
                    zipf.write(file, file.name)

            # Remove uncompressed directory
            shutil.rmtree(backup_dir)

            # Cleanup old backups (keep last 10 versions)
            self._cleanup_old_backups(self.configs_dir, keep_versions=10)

            logger.info(f"Configurations backed up to {zip_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup configs: {str(e)}")
            return False

    def _cleanup_old_backups(
        self, directory: Path, days: int = None, keep_versions: int = None
    ):
        """Clean up old backup files.

        Args:
            directory: Directory containing backups
            days: Number of days to keep backups
            keep_versions: Number of versions to keep
        """
        files = sorted(directory.glob("*.zip"), key=lambda x: x.stat().st_mtime)

        if days is not None:
            # Remove files older than specified days
            cutoff = datetime.now().timestamp() - (days * 24 * 3600)
            old_files = [f for f in files if f.stat().st_mtime < cutoff]
            for file in old_files:
                file.unlink()
                logger.info(f"Removed old backup: {file}")

        elif keep_versions is not None and len(files) > keep_versions:
            # Remove excess versions
            files_to_remove = files[:-keep_versions]
            for file in files_to_remove:
                file.unlink()
                logger.info(f"Removed old version: {file}")


class BackupScheduler:
    """Scheduler for automated backups."""

    def __init__(self):
        """Initialize the backup scheduler."""
        self.backup_manager = BackupManager()
        self.running = False
        self.thread = None

    def start(self):
        """Start the backup scheduler."""
        if self.running:
            return

        # Schedule daily backups
        schedule.every().day.at("00:00").do(self.backup_manager.backup_predictions)
        schedule.every().day.at("00:00").do(self.backup_manager.backup_configs)

        # Schedule model backups after each training
        schedule.every().hour.do(
            self.backup_manager.backup_model, model_path=Path("models/model.h5")
        )

        self.running = True
        self.thread = threading.Thread(target=self._run_schedule, daemon=True)
        self.thread.start()

        logger.info("Backup scheduler started")

    def stop(self):
        """Stop the backup scheduler."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Backup scheduler stopped")

    def _run_schedule(self):
        """Run the scheduler loop."""
        while self.running:
            schedule.run_pending()
            time.sleep(60)


# Global scheduler instance
backup_scheduler = BackupScheduler()
