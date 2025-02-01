"""Source code for the defect detection project."""

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("defect_detection").version
except pkg_resources.DistributionNotFound:
    __version__ = "unknown"

__package__ = "defect_detection"
