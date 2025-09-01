"""
Core SNP (Stepwise Noise Peeling) algorithm implementation.
"""

import time
import numpy as np
from typing import Dict, Any, Optional
import warnings

from .utils import construct_W, validate_inputs, silverman_bandwidth, gcv_score


def SNP(x: np.ndarray, y: np.ndarray, num_h_points: int = 40, 
        verbose: bool = True) -> Dict[str, Any]:
    """
    Stepwise Noise Peeling for Nadaraya-Watson Regression.
    
    Implements the Stepwise Noise Peeling (SNP) algorithm for efficient bandwidth
    selection in Nadaraya-Watson regression with Gaussian kernels. SNP provides
    a scalable alternative to Direct Generalized Cross-Validation (DGCV).
    
    Parameters
    ----------
    x : array-like
        Predictor values (should be sorted), shape (n,).
    y : array-like
        Response values corresponding to x, shape (n,).
    num_h_points : int, default=40
        Number of bandwidth candidates to evaluate in Phase I.
    verbose : bool, default=True
        Whether to print progress information.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'y_k_opt': Final smoothed output vector
        - 'h_start': Final chosen initial bandwidth  
        - 'k_opt': Optimal number of iterations
        - 'gcv_approx_k': GCV values for each iteration
        - 'time_elapsed': Execution time in seconds
        
    Examples
    --------
    >>> import numpy as np
    >>> from snp import SNP
    >>> 
    >>> # Generate sample data
    >>> np.random.seed(123)
    >>> n = 100
    >>> x = np.sort(np.random.uniform(0, 1, n))
    >>> y = np.sin(2*np.pi*x) + np.random.normal(0, 0.1, n)
    >>> 
    >>> # Apply SNP smoothing
    >>> result = SNP(x, y)
    >>> smoothed_y = result['y_k_opt']
    
    Notes
    -----
    The SNP algorithm operates in two phases:
    
    Phase I: Constructs a conservative initial bandwidth using random slices of
    data and lightweight GCV within each slice.
    
    Phase II: Fixes the smoothing operator and repeatedly applies it, selecting
    optimal iterations via discrete GCV.
    
    This reformulation preserves the adaptivity of GCV while converting costly
    continuous bandwidth search into lightweight discrete selection.
    """
    start_time = time.time()
    
    # Input validation
    x, y = validate_inputs(x, y)
    n = len(x)
    
    if num_h_points <= 0:
        raise ValueError("num_h_points must be positive")
    
    # Fixed parameters (as in R implementation)
    num_slices = 60
    k_max = 10
    
    # Initial bandwidth range based on Silverman's rule of thumb
    h_s = silverman_bandwidth(x)
    h_min = 0.001 * h_s  # Lower bound: ensures W is not too sparse
    h_max = 1.0 * h_s    # Upper bound: standard Silverman bandwidth
    
    if verbose:
        print(f"h_candidates: [{h_min:.4f}, {h_max:.4f}]")
    
    # Determine slice size based on sample size
    min_slice = 50
    
    if n < min_slice:
        slice_size = n
    else:
        slice_size = int(max(min_slice, np.sqrt(n * np.log(n))))
    
    # Randomly select starting indices for each slice
    np.random.seed(None)  # Use current random state
    max_start_idx = max(1, n - slice_size + 1)
    start_indices = np.random.choice(max_start_idx, size=num_slices, replace=True)
    
    # Create slice indices
    slice_indices = [
        np.arange(start_idx, min(start_idx + slice_size, n)) 
        for start_idx in start_indices
    ]
    
    def compute_h_opt(idx: np.ndarray) -> float:
        """Compute optimal h for a given slice using GCV."""
        x_slice = x[idx]
        y_slice = y[idx]
        h_candidates = np.random.uniform(h_min, h_max, num_h_points)
        
        # Compute GCV for each h candidate
        gcv_scores = []
        for h in h_candidates:
            W_slice = construct_W(x_slice, h)
            y_hat = W_slice @ y_slice
            gcv = np.mean((y_slice - y_hat) ** 2) / ((1 - np.mean(np.diag(W_slice))) ** 2)
            gcv_scores.append(gcv)
        
        # Return the h with minimum GCV
        return h_candidates[np.argmin(gcv_scores)]
    
    # Apply to all slices
    h_opts = [compute_h_opt(idx) for idx in slice_indices]
    
    elapsed_h = time.time() - start_time
    start_time = time.time()
    
    # Use median of h_opt estimates as starting point
    h_start = 0.5 * np.median(h_opts)
    
    if verbose:
        print(f"h_start: {h_start}")
        print(f"summary h_opts: min={np.min(h_opts):.4f}, "
              f"median={np.median(h_opts):.4f}, max={np.max(h_opts):.4f}")
    
    def trace_Wk(trWh: float, k: int, cap_one: bool = True) -> float:
        """Trace approximation function."""
        val = 1 + (trWh - 1) / np.sqrt(k)
        return max(1, val) if cap_one else val
    
    # Adaptive h_start adjustment
    i0 = 1
    while i0 <= 10:
        if i0 == 10 and verbose:
            print("Last chance for change h_start")
        
        # Initial weight matrix with h_start
        W = construct_W(x, h_start)
        trWh = np.trace(W)
        
        # Apply initial smoothing
        yk = W @ y
        yk_list = []
        gcv_approx_k = []
        
        for k in range(1, k_max + 1):
            trace_k = trace_Wk(trWh, k)
            gcv_k = gcv_score(y, yk, trace_k)
            gcv_approx_k.append(gcv_k)
            yk_list.append(yk.copy())
            yk = W @ yk  # Apply one more smoothing iteration
        
        # Choose k (number of iterations) with lowest approximate GCV
        k_opt = np.argmin(gcv_approx_k) + 1  # +1 because k starts from 1
        
        if len(np.unique(gcv_approx_k)) > 1:  # Check if we have variation in GCV scores
            if k_opt == 1:
                h_start = 0.5 * h_start
                i0 += 1
                if verbose:
                    print(f"new smaller h_start: {h_start}")
            elif k_opt == k_max:
                h_start = 0.5 * np.sqrt(k_max) * h_start
                i0 += 1
                if verbose:
                    print(f"new bigger h_start: {h_start}")
            else:
                break  # Found good h_start
        else:
            if verbose:
                print(f"new bigger h_start: {h_start} because trace = 0")
            h_start = 1.5 * h_start
            i0 = max(1, i0 - 2)
    
    elapsed_k = time.time() - start_time
    elapsed_total = elapsed_h + elapsed_k
    
    # Print summary
    if verbose:
        print("\n--- Adaptive Smoothing Summary ---")
        print(f"time_elapsed_h: {elapsed_h:.4f}, time_elapsed_k: {elapsed_k:.4f}")
        print(f"h_start (final): {h_start}")
        print(f"k_opt (final): {k_opt}")
        print(f"k_max: {k_max}")
        print(f"time_elapsed: {elapsed_total:.4f}")
    
    # Return results
    return {
        'y_k_opt': yk_list[k_opt - 1],  # -1 because list is 0-indexed
        'h_start': h_start,
        'k_opt': k_opt,
        'gcv_approx_k': np.array(gcv_approx_k),
        'time_elapsed': elapsed_total
    }