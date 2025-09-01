"""
Direct Generalized Cross-Validation (DGCV) implementation.
"""

import time
import numpy as np
from typing import Dict, Any

from .utils import construct_W, validate_inputs, silverman_bandwidth, gcv_score


def DGCV(x: np.ndarray, y: np.ndarray, num_h_points: int = 40, 
         verbose: bool = True) -> Dict[str, Any]:
    """
    Direct Generalized Cross-Validation for Nadaraya-Watson Regression.
    
    Implements Direct Generalized Cross-Validation (DGCV) for bandwidth selection
    in Nadaraya-Watson regression with Gaussian kernels. This is the traditional
    reference method that SNP aims to approximate efficiently.
    
    Parameters
    ----------
    x : array-like
        Predictor values (should be sorted), shape (n,).
    y : array-like
        Response values corresponding to x, shape (n,).
    num_h_points : int, default=50
        Number of bandwidth candidates to evaluate across the continuous 
        bandwidth space.
    verbose : bool, default=True
        Whether to print progress information.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'y_h_opt': Final smoothed output using optimal bandwidth
        - 'h_opt_gcv': Optimal bandwidth selected by GCV
        - 'gcv_h': GCV scores for all bandwidth candidates
        - 'time_elapsed': Execution time in seconds
        
    Examples
    --------
    >>> import numpy as np
    >>> from snp import DGCV
    >>> 
    >>> # Generate sample data
    >>> np.random.seed(123)
    >>> n = 100
    >>> x = np.sort(np.random.uniform(0, 1, n))
    >>> y = np.sin(2*np.pi*x) + np.random.normal(0, 0.1, n)
    >>> 
    >>> # Apply DGCV smoothing
    >>> result = DGCV(x, y)
    >>> smoothed_y = result['y_h_opt']
    >>> optimal_bandwidth = result['h_opt_gcv']
        
    Notes
    -----
    DGCV performs an exhaustive search over a continuous bandwidth space,
    evaluating the GCV criterion for each candidate bandwidth. While statistically
    rigorous, this approach becomes computationally prohibitive for large datasets
    due to the need to construct and evaluate the full smoothing matrix for each
    bandwidth candidate.
    
    The bandwidth search range is determined using Silverman's rule of thumb,
    with candidates uniformly distributed between 0.1% and 100% of the
    Silverman bandwidth.
    """
    start_time = time.time()
    
    # Input validation
    x, y = validate_inputs(x, y)
    n = len(x)
    
    if num_h_points <= 0:
        raise ValueError("num_h_points must be positive")
    
    # Bandwidth range based on Silverman's rule
    h_s = silverman_bandwidth(x)
    h_min = 0.001 * h_s
    h_max = 1.0 * h_s
    h_candidates = np.linspace(h_min, h_max, num_h_points)
    
    if verbose:
        print(f"h_candidates: [{h_candidates.min():.4f}, {h_candidates.max():.4f}]")
    
    # Initialize containers for results
    yk_list = []        # Store smoothed outputs for each h
    gcv_h = []          # Store GCV scores for each h
    
    # Loop over all bandwidth candidates
    for i, h in enumerate(h_candidates):
        if verbose and num_h_points > 20 and (i + 1) % 10 == 0:
            print(f"Processing bandwidth {i+1}/{num_h_points}...")
            
        W = construct_W(x, h)                    # Construct Gaussian kernel weight matrix
        yhat = W @ y                             # Apply smoothing
        trace_W = np.trace(W)                    # Compute trace of smoother matrix
        gcv = gcv_score(y, yhat, trace_W)        # GCV score
        
        gcv_h.append(gcv)
        yk_list.append(yhat.copy())
    
    # Convert to numpy arrays
    gcv_h = np.array(gcv_h)
    
    # Select h that minimizes GCV
    inds_min = np.argmin(gcv_h)
    h_opt_gcv = h_candidates[inds_min]
    
    elapsed = time.time() - start_time
    
    # Print summary for the user
    if verbose:
        print("\n--- Original GCV Smoothing Summary ---")
        print(f"h_opt_gcv: {h_opt_gcv}")
        print(f"time_elapsed: {elapsed:.4f}")
    
    # Return the best result and associated values
    return {
        'y_h_opt': yk_list[inds_min],         # Final smoothed output
        'h_opt_gcv': h_opt_gcv,               # Optimal bandwidth
        'gcv_h': gcv_h,                       # All GCV scores
        'h_candidates': h_candidates,         # All bandwidth candidates
        'time_elapsed': elapsed               # Elapsed time in seconds
    }


def compare_methods(x: np.ndarray, y: np.ndarray, num_h_points: int = 50,
                   verbose: bool = True) -> Dict[str, Any]:
    """
    Compare SNP and DGCV performance on the same dataset.
    
    Parameters
    ----------
    x : array-like
        Predictor values.
    y : array-like  
        Response values.
    num_h_points : int, default=50
        Number of bandwidth candidates for both methods.
    verbose : bool, default=True
        Whether to print comparison results.
        
    Returns
    -------
    dict
        Dictionary containing results from both methods and comparison metrics.
    """
    from .core import SNP  # Import here to avoid circular imports
    
    # Run both methods
    if verbose:
        print("Running DGCV...")
    dgcv_result = DGCV(x, y, num_h_points, verbose=verbose)
    
    if verbose:
        print("\nRunning SNP...")
    snp_result = SNP(x, y, num_h_points, verbose=verbose)
    
    # Calculate comparison metrics
    mse = np.mean((dgcv_result['y_h_opt'] - snp_result['y_k_opt']) ** 2)
    speedup = dgcv_result['time_elapsed'] / snp_result['time_elapsed']
    
    if verbose:
        print(f"\n--- Performance Comparison ---")
        print(f"DGCV time: {dgcv_result['time_elapsed']:.4f}s")
        print(f"SNP time:  {snp_result['time_elapsed']:.4f}s")
        print(f"Speedup:   {speedup:.2f}x")
        print(f"MSE between methods: {mse:.6f}")
    
    return {
        'dgcv': dgcv_result,
        'snp': snp_result,
        'speedup': speedup,
        'mse_difference': mse
    }