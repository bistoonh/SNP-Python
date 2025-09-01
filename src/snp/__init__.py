"""
SNP: Stepwise Noise Peeling for Nadaraya-Watson Regression

A scalable alternative to Direct Generalized Cross-Validation (DGCV) for
bandwidth selection in Nadaraya-Watson regression with Gaussian kernels.
"""

from .core import SNP
from .dgcv import DGCV
from .utils import construct_W

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "SNP",
    "DGCV", 
    "construct_W",
]