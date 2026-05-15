# snpreg/__init__.py
"""
SNP: Stepwise Noise Peeling for Kernel Regression

A Python package implementing the Stepwise Noise Peeling (SNP) algorithm
for nonparametric kernel regression with automatic bandwidth selection.

Main functions:
- nw_snp: SNP estimator with iterative smoothing
- nw_direct_gcv: Direct GCV bandwidth selection for Nadaraya-Watson

Helper functions:
- construct_W: Build Nadaraya-Watson weight matrix
- generate_slices: Generate local slices for bandwidth initialization
- rmse: Root mean squared error
- mape_shift: Shifted mean absolute percentage error

Experiments:
- mixture_experiment: Reproduce mixture regression experiment
- run_realdata_1d: California housing 1D analysis
- run_realdata_2d: California housing 2D analysis
- runtime_benchmark: Runtime scaling benchmark

Example:
    >>> import snpreg
    >>> import numpy as np
    >>> X = np.random.randn(100, 2)
    >>> y = X[:, 0]**2 + np.random.randn(100) * 0.1
    >>> result = snpreg.nw_snp(X, y)
    >>> print(result['k_opt'])
"""

from .kernels import construct_W
from .dgcv import nw_direct_gcv
from .snp import nw_snp, generate_slices
from .metrics import rmse, mape_shift

# Import experiments
from .experiments import (
    mixture_experiment,
    run_realdata_1d,
    run_realdata_2d,
    runtime_benchmark,
)

# Import dataset utilities
from .datasets import get_housing_data

__version__ = "0.1.0"
__author__ = "Bistoon Hosseini"
__email__ = "your.email@example.com"  # Update with actual email
__license__ = "MIT"

__all__ = [
    # Core functions
    'nw_snp',
    'nw_direct_gcv',
    
    # Helper functions
    'construct_W',
    'generate_slices',
    
    # Metrics
    'rmse',
    'mape_shift',
    
    # Datasets
    'get_housing_data',
    
    # Experiments
    'mixture_experiment',
    'run_realdata_1d',
    'run_realdata_2d',
    'runtime_benchmark',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__',
]

