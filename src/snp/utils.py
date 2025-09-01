"""
Utility functions for SNP package.
"""

import numpy as np
from typing import Union


def construct_W(x: np.ndarray, h: float) -> np.ndarray:
    """
    Construct normalized Gaussian kernel weight matrix.
    
    Constructs a row-stochastic weight matrix for Nadaraya-Watson regression
    using Gaussian kernels with specified bandwidth.
    
    Parameters
    ----------
    x : np.ndarray
        Predictor values, shape (n,).
    h : float
        Bandwidth parameter for the Gaussian kernel, must be positive.
        
    Returns
    -------
    np.ndarray
        Weight matrix W where W[i,j] represents the weight given to observation j
        when predicting at point x[i]. Each row sums to 1 (row-stochastic property).
        Shape (n, n).
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> h = 0.5
    >>> W = construct_W(x, h)
    >>> # Check row sums (should all be close to 1)
    >>> np.allclose(W.sum(axis=1), 1.0)
    True
    
    Notes
    -----
    The function computes a Gaussian kernel weight matrix where:
    K(x_i, x_j) = (1/√(2π)) * exp(-(x_i - x_j)²/(2h²))
    
    Each row is then normalized so that the weights sum to 1, making the matrix
    row-stochastic. This ensures that the Nadaraya-Watson estimator is a proper
    weighted average.
    """
    # Input validation
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
        
    if np.any(np.isnan(x)):
        raise ValueError("x cannot contain NaN values")
        
    if not isinstance(h, (int, float)) or h <= 0:
        raise ValueError("h must be a positive number")
    
    n = len(x)
    
    # Compute pairwise differences
    # Using broadcasting: x[:, None] - x[None, :] creates (n, n) distance matrix
    dist_mat = x[:, None] - x[None, :]
    
    # Apply Gaussian kernel
    # Using 1/sqrt(2π) ≈ 0.3989423 for computational efficiency
    K_mat = 0.3989423 * np.exp(-0.5 * (dist_mat / h) ** 2)
    
    # Normalize each row so rows sum to 1 (row-stochastic property)
    row_sums = K_mat.sum(axis=1)
    
    # Avoid division by zero (though this should rarely happen with Gaussian kernels)
    if np.any(row_sums == 0):
        import warnings
        warnings.warn("Some rows have zero sum. This may indicate bandwidth is too small.")
        row_sums = np.where(row_sums == 0, 1, row_sums)
    
    # Return normalized matrix
    return K_mat / row_sums[:, None]


def silverman_bandwidth(x: np.ndarray) -> float:
    """
    Calculate Silverman's rule of thumb bandwidth.
    
    Parameters
    ----------
    x : np.ndarray
        Input data points.
        
    Returns
    -------
    float
        Silverman's bandwidth estimate.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
        
    n = len(x)
    return 1.06 * np.std(x) * (n ** (-1/5))


def validate_inputs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert input arrays.
    
    Parameters
    ----------
    x : array-like
        Predictor values.
    y : array-like
        Response values.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated x and y arrays.
        
    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    # Convert to numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Check dimensions
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
        
    # Check lengths
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")
    
    # Check for NaN/inf values
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values (NaN or inf)")
    if np.any(~np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or inf)")
        
    # Check minimum length
    if len(x) < 3:
        raise ValueError("Need at least 3 data points")
        
    return x, y


def gcv_score(y_true: np.ndarray, y_pred: np.ndarray, trace_smoother: float) -> float:
    """
    Calculate Generalized Cross-Validation score.
    
    Parameters
    ----------
    y_true : np.ndarray
        True response values.
    y_pred : np.ndarray
        Predicted response values.
    trace_smoother : float
        Trace of the smoother matrix.
        
    Returns
    -------
    float
        GCV score.
    """
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return rss / ((1 - trace_smoother / n) ** 2)