"""Script to handle monitoring and backups."""

import time
import schedule
from pathlib import Path
import yaml
from prometheus_client import start_http_server, Gauge

from src.utils.backup import create_backup, cleanup_old_backups
from src.utils.notifications import setup_logging


# Prometheus metrics
BACKUP_GAUGE = Gauge(
    'last_backup_timestamp',
    'Timestamp of last successful backup'
)
DISK_USAGE_GAUGE = Gauge(
    'model_disk_usage_bytes',
    'Disk space used by model files'
)


def load_config():
    """Load configuration from yaml file."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def calculate_model_size(model_dir):
    """Calculate total size of model files.

    Args:
        model_dir (Path): Path to model directory

    Returns:
        int: Total size in bytes
    """
    total_size = 0
    for file_path in model_dir.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def update_metrics(model_dir):
    """Update Prometheus metrics.

    Args:
        model_dir (Path): Path to model directory
    """
    # Update disk usage metric
    disk_usage = calculate_model_size(model_dir)
    DISK_USAGE_GAUGE.set(disk_usage)


def perform_backup(config, logger):
    """Perform backup operation.

    Args:
        config (dict): Configuration dictionary
        logger (logging.Logger): Logger instance
    """
    try:
        # Create backup
        create_backup('models', config['backup']['backup_dir'])
        logger.info("Created backup successfully")

        # Update backup timestamp metric
        BACKUP_GAUGE.set_to_current_time()

        # Cleanup old backups
        removed = cleanup_old_backups(
            config['backup']['backup_dir'],
            config['backup']['max_backups']
        )
        if removed > 0:
            logger.info(f"Removed {removed} old backup(s)")

    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")


def main():
    """Main monitoring function."""
    # Load configuration
    config = load_config()

    # Setup logging
    logger = setup_logging(config['logging']['log_dir'])
    logger.info("Starting monitoring service...")

    # Start Prometheus metrics server
    start_http_server(config['monitoring']['prometheus_port'])
    logger.info(
        f"Started Prometheus metrics server on port "
        f"{config['monitoring']['prometheus_port']}"
    )

    # Schedule backup job
    schedule.every(config['backup']['backup_interval']).hours.do(
        perform_backup, config, logger
    )

    # Schedule metrics update
    model_dir = Path(config['model']['checkpoint_dir'])
    schedule.every(1).minutes.do(update_metrics, model_dir)

    # Run initial backup and metrics update
    perform_backup(config, logger)
    update_metrics(model_dir)

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
