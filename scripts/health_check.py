"""Automated health check script for the defect detection service."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("health_checks.log"),
    ],
)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker for the defect detection service."""

    def __init__(self, base_url: str, check_interval: int = 60):
        """Initialize health checker.

        Args:
            base_url: Base URL of the service
            check_interval: Interval between checks in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.check_interval = check_interval
        self.endpoints = {
            "health": "/health",
            "status": "/status",
            "metrics": "/metrics",
        }

    def check_endpoint(self, endpoint: str) -> tuple:
        """Check a specific endpoint."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return (
                True,
                response.json() if endpoint != "/metrics" else {"status": "ok"}
            )
        except Exception as e:
            logger.error(f"Error checking {endpoint}: {str(e)}")
            return False, {"error": str(e)}

    def check_prediction(self, test_image_path: str) -> tuple:
        """Test prediction endpoint with a sample image.

        Args:
            test_image_path: Path to test image

        Returns:
            Tuple of (success, response)
        """
        try:
            url = f"{self.base_url}/predict"
            with open(test_image_path, "rb") as f:
                files = {"file": f}
                response = requests.post(url, files=files, timeout=30)
            response.raise_for_status()
            return True, response.json()
        except Exception as e:
            logger.error(f"Error testing prediction: {str(e)}")
            return False, {"error": str(e)}

    def run_checks(self, test_image_path: str = None) -> dict:
        """Run all health checks.

        Args:
            test_image_path: Optional path to test image

        Returns:
            Dictionary with check results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "endpoints": {},
        }

        # Check basic endpoints
        for name, endpoint in self.endpoints.items():
            success, response = self.check_endpoint(endpoint)
            results["endpoints"][name] = {
                "status": "healthy" if success else "unhealthy",
                "response": response,
            }

        # Test prediction if image provided
        if test_image_path:
            success, response = self.check_prediction(test_image_path)
            results["prediction_test"] = {
                "status": "healthy" if success else "unhealthy",
                "response": response,
            }

        return results

    def monitor(self, test_image_path: str = None):
        """Run continuous monitoring.

        Args:
            test_image_path: Optional path to test image
        """
        logger.info("Starting health monitoring...")
        while True:
            try:
                results = self.run_checks(test_image_path)
                self._save_results(results)
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.check_interval)

    def _save_results(self, results: dict):
        """Save check results to file.

        Args:
            results: Check results to save
        """
        output_dir = Path("monitoring")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "health_checks.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)


def main():
    """Main entry point."""
    base_url = "http://localhost:5000"
    checker = HealthChecker(base_url)
    checker.monitor()


if __name__ == "__main__":
    main()
