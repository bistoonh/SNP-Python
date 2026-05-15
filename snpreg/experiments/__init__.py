# experiments/__init__.py
"""
Experimental functions for SNP package.

This module provides ready-to-run experiments that reproduce the results
from the paper "Stepwise Noise Peeling: A Spectral Iterative Smoothing 
Framework for Kernel Regression".

Available experiments:
- mixture_experiment: Synthetic mixture regression
- realdata_1d: California housing (1D)
- realdata_2d: California housing (2D)
- runtime_benchmark: Runtime scaling analysis
"""

from .mixture import mixture_experiment
from .realdata_1d import run_realdata_1d
from .realdata_2d import run_realdata_2d
from .runtime import runtime_benchmark

__all__ = [
    'mixture_experiment',
    'run_realdata_1d',
    'run_realdata_2d',
    'runtime_benchmark',
]
