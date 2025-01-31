"""Backup utilities for model and data files."""

import logging
import shutil
from datetime import datetime
from pathlib import Path


def create_backup(source_dir, backup_dir):
    """Create a backup of the source directory.

    Args:
        source_dir (str): Directory to backup
        backup_dir (str): Directory to store backups

    Returns:
        Path: Path to the created backup
    """
    source_path = Path(source_dir)
    backup_path = Path(backup_dir)

    # Create backup directory if it doesn't exist
    backup_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source_path.name}_{timestamp}"
    backup_dest = backup_path / backup_name

    try:
        # Copy directory
        if source_path.exists():
            shutil.copytree(source_path, backup_dest)
            logging.info(f"Created backup at {backup_dest}")
        else:
            logging.warning(f"Source directory {source_path} does not exist")

        return backup_dest

    except Exception as e:
        logging.error(f"Backup failed: {str(e)}")
        raise


def cleanup_old_backups(backup_dir, max_backups=5):
    """Remove old backups, keeping only the most recent ones.

    Args:
        backup_dir (str): Directory containing backups
        max_backups (int): Maximum number of backups to keep

    Returns:
        int: Number of backups removed
    """
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return 0

    # List all backup directories
    backups = sorted(
        [d for d in backup_path.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    # Remove old backups
    removed = 0
    if len(backups) > max_backups:
        for backup in backups[max_backups:]:
            try:
                shutil.rmtree(backup)
                removed += 1
                logging.info(f"Removed old backup: {backup}")
            except Exception as e:
                logging.error(f"Failed to remove backup {backup}: {str(e)}")

    return removed
