"""Script to run automated tests with coverage."""

import subprocess
import sys
from pathlib import Path
import yaml


def load_config():
    """Load configuration from yaml file."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_tests():
    """Run pytest with coverage."""
    # Run tests with coverage
    result = subprocess.run([
        "pytest",
        "--cov=src",
        "--cov-report=xml",
        "--cov-report=term-missing",
        "tests/"
    ], capture_output=True, text=True)

    # Print test output
    print(result.stdout)
    if result.stderr:
        print("Errors:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)

    return result.returncode == 0


def run_linting():
    """Run flake8 linting."""
    result = subprocess.run([
        "flake8",
        "src/",
        "tests/"
    ], capture_output=True, text=True)

    if result.stdout:
        print("Linting issues found:")
        print(result.stdout)
    if result.stderr:
        print("Linting errors:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)

    return result.returncode == 0


def main():
    """Main function to run tests and linting."""
    print("Running tests with coverage...")
    tests_passed = run_tests()
    
    print("\nRunning linting checks...")
    linting_passed = run_linting()

    if not tests_passed:
        print("❌ Tests failed!")
        sys.exit(1)
    if not linting_passed:
        print("❌ Linting checks failed!")
        sys.exit(1)
    
    print("✅ All checks passed!")


if __name__ == "__main__":
    main()
