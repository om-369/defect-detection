"""Test script to verify deployment."""
import os
import time
import requests
from pathlib import Path

def test_deployment(base_url: str) -> None:
    """Test all endpoints of the deployed service."""
    print("\n=== Testing Deployment ===")
    
    # Test health endpoint
    print("\nTesting /health endpoint...")
    health_response = requests.get(f"{base_url}/health")
    print(f"Health Status: {health_response.json()}")
    
    # Test status endpoint
    print("\nTesting /status endpoint...")
    status_response = requests.get(f"{base_url}/status")
    print(f"Status: {status_response.json()}")
    
    # Test metrics endpoint
    print("\nTesting /metrics endpoint...")
    metrics_response = requests.get(f"{base_url}/metrics")
    print("Metrics available:", metrics_response.status_code == 200)
    
    # Test prediction endpoint
    print("\nTesting /predict endpoint...")
    test_image_path = Path("data/test/sample_image.jpg")
    if test_image_path.exists():
        with open(test_image_path, "rb") as f:
            files = {"image": f}
            predict_response = requests.post(f"{base_url}/predict", files=files)
            print(f"Prediction: {predict_response.json()}")
    else:
        print(f"Test image not found at {test_image_path}")

if __name__ == "__main__":
    # Replace with your deployed service URL
    service_url = os.environ.get("SERVICE_URL", "http://localhost:8080")
    test_deployment(service_url)
