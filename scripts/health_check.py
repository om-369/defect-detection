"""Automated health check script for the defect detection service."""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("health_checks.log")],
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

    def check_endpoint(self, endpoint: str) -> Tuple[bool, Dict[str, Any]]:
        """Check a specific endpoint."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return True, response.json() if endpoint != "/metrics" else {"status": "ok"}
        except Exception as e:
            logger.error(f"Error checking {endpoint}: {str(e)}")
            return False, {"error": str(e)}

    def check_prediction(self, test_image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Test prediction endpoint with a sample image."""
        try:
            url = f"{self.base_url}/predict"
            with open(test_image_path, "rb") as f:
                response = requests.post(url, files={"image": f}, timeout=30)
            response.raise_for_status()
            return True, response.json()
        except Exception as e:
            logger.error(f"Error testing prediction: {str(e)}")
            return False, {"error": str(e)}

    def run_health_check(self) -> Dict[str, Any]:
        """Run a complete health check."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "service_url": self.base_url,
            "checks": {},
        }

        # Check basic endpoints
        for name, endpoint in self.endpoints.items():
            success, response = self.check_endpoint(endpoint)
            results["checks"][name] = {
                "status": "healthy" if success else "unhealthy",
                "response": response,
            }

        # Check prediction if test image exists
        test_image = Path("data/test/sample_image.jpg")
        if test_image.exists():
            success, response = self.check_prediction(str(test_image))
            results["checks"]["prediction"] = {
                "status": "healthy" if success else "unhealthy",
                "response": response,
            }

        # Calculate overall health
        healthy_checks = sum(
            1 for check in results["checks"].values() if check["status"] == "healthy"
        )
        total_checks = len(results["checks"])
        results["overall_health"] = {
            "status": "healthy" if healthy_checks == total_checks else "degraded",
            "healthy_checks": healthy_checks,
            "total_checks": total_checks,
        }

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save health check results to file."""
        output_dir = Path("monitoring/health_checks")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save latest results
        latest_file = output_dir / "latest.json"
        with open(latest_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save historical results
        timestamp = datetime.fromisoformat(results["timestamp"])
        historical_file = (
            output_dir / f"health_check_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(historical_file, "w") as f:
            json.dump(results, f, indent=2)

    def run_continuous(self) -> None:
        """Run health checks continuously."""
        logger.info(
            f"Starting continuous health checks every {self.check_interval} seconds"
        )
        while True:
            try:
                results = self.run_health_check()
                self.save_results(results)

                # Log results
                status = results["overall_health"]["status"]
                healthy_checks = results["overall_health"]["healthy_checks"]
                total_checks = results["overall_health"]["total_checks"]
                logger.info(
                    f"Health check completed - Status: {status} "
                    f"({healthy_checks}/{total_checks} checks healthy)"
                )

                # If service is unhealthy, log detailed information
                if status != "healthy":
                    unhealthy_checks = [
                        name
                        for name, check in results["checks"].items()
                        if check["status"] != "healthy"
                    ]
                    logger.warning(f"Unhealthy checks: {', '.join(unhealthy_checks)}")

            except Exception as e:
                logger.error(f"Error running health check: {str(e)}")

            time.sleep(self.check_interval)


def main():
    """Main entry point."""
    service_url = os.environ.get("SERVICE_URL", "http://localhost:8080")
    check_interval = int(os.environ.get("CHECK_INTERVAL", "60"))

    checker = HealthChecker(service_url, check_interval)
    checker.run_continuous()


if __name__ == "__main__":
    main()
