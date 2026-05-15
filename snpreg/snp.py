import numpy as np
import time
import gc

import numpy as np
from .kernels import construct_W

def generate_slices(X, s, T):
    """
    Generate local slices using nearest neighbor sampling.
    
    This function creates T local neighborhoods (slices) by randomly selecting
    anchor points and finding their s nearest neighbors. Each slice represents
    a local region of the covariate space for local bandwidth selection.
    
    Parameters
    ----------
    X : ndarray of shape (n, d)
        Covariate data points from which to generate slices.
    s : int
        Slice size, i.e., number of nearest neighbors to include in each slice.
        Must satisfy 1 <= s <= n.
    T : int
        Number of slices to generate. Larger T provides more stable bandwidth
        selection at the cost of computation.
    
    Returns
    -------
    slices : list of ndarray
        List of length T, where each element is an array of indices 
        (shape (s,)) representing the observations in that slice.
    
    Notes
    -----
    - Anchor points are selected uniformly at random with replacement.
    - Euclidean distance is used for nearest neighbor computation.
    - The anchor point itself is always included in its slice (distance = 0).
    
    Examples
    --------
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    >>> slices = generate_slices(X, s=3, T=2)
    >>> len(slices)
    2
    >>> len(slices[0])
    3
    """
    # Generate T local slices of size s using nearest neighbors
    
    n = X.shape[0]
    slices = []

    for _ in range(T):
        
        # Random anchor point
        center_idx = np.random.randint(n)
        center = X[center_idx]

        # Vectorized Euclidean distances
        distances = np.linalg.norm(X - center, axis=1)

        # s nearest points
        slice_indices = np.argsort(distances)[:s]

        slices.append(slice_indices)

    return slices


def nw_snp(X, y, num_h_points=30, num_slices=50, X_new=None):
    """
    Stepwise Noise Peeling (SNP) for Nadaraya-Watson regression.
    
    This function implements the SNP algorithm, which performs iterative smoothing
    using a Nadaraya-Watson kernel smoother. The method adaptively selects the
    optimal number of smoothing iterations (k_opt) and the initial bandwidth (h_start)
    through a two-phase procedure with adaptive restarts.
    
    Parameters
    ----------
    X : array-like, shape (n, d) or (n,)
        Training predictor matrix. If 1D, will be reshaped to (n, 1).
    y : array-like, shape (n,) or (n, 1)
        Training response vector.
    num_h_points : int, optional (default=30)
        Number of random bandwidth candidates to evaluate per slice in Phase I.
    num_slices : int, optional (default=50)
        Number of local slices to use for bandwidth initialization in Phase I.
    X_new : array-like, shape (m, d) or (m,) or None, optional (default=None)
        New predictor points for prediction. If None, no prediction is performed.
    
    Returns
    -------
    y_k_opt : ndarray, shape (n,)
        Fitted values on training data after k_opt iterations.
    y_k_minus_1_opt : ndarray, shape (n,)
        Fitted values on training data after (k_opt - 1) iterations.
    y_new_pred : ndarray, shape (m, 1) or None
        Predicted values for X_new. None if X_new is not provided.
    h_start : ndarray, shape (d,)
        Final initial bandwidth vector used in Phase II.
    k_opt : int
        Optimal number of smoothing iterations selected by GCV.
    gcv_approx_k : ndarray, shape (k_max,)
        Approximate GCV scores for k = 1, 2, ..., k_max.
    traces : ndarray, shape (k_max,)
        Approximate trace values for each iteration.
    time_elapsed : float
        Total computation time in seconds.
    B : int
        Total number of bandwidth initialization attempts (including restarts).
    
    Notes
    -----
    The SNP algorithm consists of two phases:
    
    **Phase I: Bandwidth Initialization**
    - Generates `num_slices` local slices of the data using `generate_slices()`.
    - For each slice, evaluates `num_h_points` random bandwidth candidates using GCV.
    - Computes the median of optimal bandwidths across slices.
    - Sets h_start = rho * median(h_opts), where rho = 0.5.
    
    **Phase II: Iterative Smoothing with Adaptive Restarts**
    - Constructs the smoothing matrix W using h_start.
    - Iteratively applies smoothing: y^(k) = W @ y^(k-1), starting from y^(0) = y.
    - Evaluates approximate GCV for k = 1, 2, ..., k_max (default k_max = 10).
    - Selects k_opt = argmin(GCV_k).
    - **Adaptive Restart Logic:**
        - If k_opt == 1: bandwidth too large → reduce h_start by factor rho and restart.
        - If k_opt == k_max: bandwidth too small → increase h_start by factor rho * sqrt(k_max) and restart.
        - Otherwise: accept k_opt and h_start.
    - Maximum of 10 restarts allowed.
    
    **Prediction:**
    - If X_new is provided, constructs cross-smoothing matrix W_new = construct_W(X, h_start, X_new).
    - Predicts using y_new_pred = W_new @ y^(k_opt - 1).
    
    **Bandwidth Range:**
    - Uses Silverman-like rule: h_s = 1.06 * sd(X_j) * n^(-1/(4+d)) for each dimension j.
    - Search range: [0.01 * h_s, 2.0 * h_s].
    
    **Subsample Size:**
    - slice_size = max(50, floor(sqrt(n * log(n)))) if n >= 50, else n.
    
    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 200
    >>> X = np.random.uniform(0, 1, n)
    >>> y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, n)
    >>> result = nw_snp(X, y)
    >>> print(f"Optimal k: {result['k_opt']}")
    >>> print(f"Final bandwidth: {result['h_start']}")
    
    >>> # With prediction on new data
    >>> X_new = np.linspace(0, 1, 50)
    >>> result = nw_snp(X, y, X_new=X_new)
    >>> y_pred = result['y_new_pred']
    
    References
    ----------
    [Add your paper reference here]
    """

    # Convert to proper shapes
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if X_new is not None:

        X_new = np.asarray(X_new, dtype=float)

        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, X.shape[1])
    
    start_time = time.time()
    
    k_max = 10
    rho = 0.5
    
    n, d = X.shape
    
    # Silverman-like bandwidth for each dimension
    sd_vec = np.std(X, axis=0, ddof=1)

    h_s = 1.06 * sd_vec * n**(-1/(4 + d))
    
    h_min = 0.01 * h_s

    h_max = 2.0 * h_s
    
    # Subsample size
    min_slice = 50

    slice_size = (
        n if n < min_slice
        else int(np.floor(max(min_slice, np.sqrt(n * np.log(n)))))
    )
    
    slice_indices = generate_slices(X, slice_size, num_slices)
    
    # Phase I: Bandwidth initialization
    def compute_h_opt(idx):

        X_slice = X[idx, :]

        y_slice = y[idx]
        
        h_candidates = np.zeros((num_h_points, d))

        for j in range(d):

            h_candidates[:, j] = np.random.uniform(
                h_min[j],
                h_max[j],
                num_h_points
            )
        
        gcv_scores = np.zeros(num_h_points)

        for i in range(num_h_points):

            h_vec = h_candidates[i, :]

            W_slice = construct_W(X_slice, h_vec)

            y_hat = W_slice @ y_slice

            denom = (1 - np.mean(np.diag(W_slice)))**2

            # Numerical safeguard
            if denom < 1e-12:
                denom = 1e-12

            gcv_scores[i] = np.mean((y_slice - y_hat)**2) / denom
        
        return h_candidates[np.argmin(gcv_scores), :]
    
    h_opts = np.array([compute_h_opt(idx) for idx in slice_indices])
    
    elapsed_h = time.time() - start_time
    
    h_start = rho * np.median(h_opts, axis=0)
    
    # Trace approximation
    def trace_Wk(trWh, k, d):

        return np.maximum(
            1,
            1 + (trWh - 1) / (k**(d/2))
        )
    
    # Phase II: Iterative smoothing with adaptive restarts
    start_time_k = time.time()
    
    max_restarts = 10

    restart_count = 0

    B = 0
    
    while restart_count < max_restarts:

        B += 1

        if restart_count == max_restarts - 1:
            print("Last chance for h_start adjustment")
        
        W = construct_W(X, h_start)

        trWh = np.trace(W)
        
        yk = W @ y

        yk_list = []

        gcv_approx_k = np.zeros(k_max)

        traces = np.zeros(k_max)
        
        for k in range(1, k_max + 1):

            traces[k-1] = trace_Wk(trWh, k, d)

            denom = (1 - traces[k-1] / n)**2

            # Numerical safeguard
            if denom < 1e-12:
                denom = 1e-12

            gcv_approx_k[k-1] = np.sum((y - yk)**2) / denom

            # Keep original shape (important for multivariate responses)
            yk_list.append(yk.copy())
            
            if k < k_max:
                yk = W @ yk
        
        k_opt = np.argmin(gcv_approx_k) + 1
        
        # Adaptive restart logic
        if k_opt == 1:

            h_start = rho * h_start

            restart_count += 1

            print("Restart: smaller h_start")

            print(h_start)

        elif k_opt == k_max:

            h_start = rho * h_start * np.sqrt(k_max)

            restart_count += 1

            print("Restart: larger h_start")

            print(h_start)

        else:
            break
    
    elapsed_k = time.time() - start_time_k

    elapsed = elapsed_h + elapsed_k

    # -------- prediction for new data --------
    y_new_pred = None

    if X_new is not None:

        W_new = construct_W(X, h_start, X_new)

        # Safe handling when k_opt == 1
        if k_opt == 1:
            y_prev = y
        else:
            y_prev = yk_list[k_opt - 2]

        y_new_pred = W_new @ y_prev
    # -----------------------------------------
    
    gc.collect()
    
    return {
        'y_k_opt': yk_list[k_opt - 1],
        'y_k_minus_1_opt': y if k_opt == 1 else yk_list[k_opt - 2],
        'y_new_pred': y_new_pred,
        'h_start': h_start,
        'k_opt': k_opt,
        'gcv_approx_k': gcv_approx_k,
        'traces': traces,
        'time_elapsed': elapsed,
        'B': B
    }

