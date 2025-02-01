"""Script to run all tests and generate coverage report."""
import argparse
import subprocess
import sys
from pathlib import Path

def run_tests(test_dir: str, coverage_dir: str) -> bool:
    """Run pytest with coverage reporting.

    Args:
        test_dir: Directory containing tests
        coverage_dir: Directory to save coverage reports

    Returns:
        True if all tests pass, False otherwise
    """
    # Ensure coverage directory exists
    Path(coverage_dir).mkdir(parents=True, exist_ok=True)

    # Run tests with coverage
    cmd = [
        "pytest",
        test_dir,
        "-v",
        "--cov=src",
        "--cov-report=term-missing",
        f"--cov-report=html:{coverage_dir}/htmlcov",
        f"--cov-report=xml:{coverage_dir}/coverage.xml",
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Run test suite."""
    parser = argparse.ArgumentParser(description="Run test suite with coverage")
    parser.add_argument(
        "--test-dir",
        default="tests",
        help="Directory containing tests",
    )
    parser.add_argument(
        "--coverage-dir",
        default="coverage",
        help="Directory to save coverage reports",
    )
    args = parser.parse_args()

    success = run_tests(args.test_dir, args.coverage_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
