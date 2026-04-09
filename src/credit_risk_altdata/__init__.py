"""Top-level package for explainable credit risk modeling with alternative data."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("credit-risk-altdata")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
