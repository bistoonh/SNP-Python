import numpy as np
import time
import gc

import numpy as np
from .kernels import construct_W


def nw_direct_gcv(X, y, num_h_points=30, mode="random", X_new=None):
    """
    Nadaraya-Watson regression with direct GCV bandwidth selection.
    
    This function performs Nadaraya-Watson kernel regression by selecting
    the optimal bandwidth via direct Generalized Cross-Validation (GCV).
    It evaluates GCV over a grid or random sample of bandwidth candidates
    and returns the fitted values along with the optimal bandwidth.
    
    Parameters
    ----------
    X : ndarray of shape (n, d)
        Training covariate matrix.
        
    y : ndarray of shape (n,) or (n, 1)
        Training response vector.
        
    num_h_points : int, default=30
        Number of bandwidth candidates to evaluate.
        - If mode="grid" and d > 1, total candidates = num_h_points^d.
        - If mode="random", exactly num_h_points candidates are sampled.
        
    mode : {"random", "grid"}, default="random"
        Bandwidth sampling strategy:
        - "random": uniformly sample bandwidths in [h_min, h_max] per dimension.
        - "grid": create a regular grid (forced for d=1).
        
    X_new : ndarray of shape (m, d), optional
        New covariate points for out-of-sample prediction. If provided,
        predictions at X_new are returned.
    
    Returns
    -------
    dict with keys:
    
        y_train_opt : ndarray of shape (n, 1)
            Fitted values at training points using optimal bandwidth.
            
        y_new_pred : ndarray of shape (m, 1) or None
            Predictions at X_new if provided, else None.
            
        h_opt_gcv : ndarray of shape (d,)
            Optimal bandwidth vector selected by GCV.
            
        gcv_h : ndarray of shape (num_combinations,)
            GCV scores for all evaluated bandwidths.
            
        h_grid : ndarray of shape (num_combinations, d)
            All evaluated bandwidth vectors.
            
        time_elapsed : float
            Computation time in seconds.
    
    Notes
    -----
    - Bandwidth search range is based on Silverman's rule of thumb:
    
          h_s = 1.06 * sd * n^(-1/(4+d))
    
      with:
    
          h_min = 0.01*h_s
          h_max = 2.0*h_s
    
    - GCV criterion:
    
          GCV(h) = RSS / (1 - tr(W)/n)^2
    
      where W is the smoother matrix.
    
    - For d=1, mode is automatically set to "grid".
    
    Examples
    --------
    >>> X = np.random.randn(100, 2)
    >>> y = X[:, 0]**2 + np.random.randn(100) * 0.1
    >>> result = nw_direct_gcv(X, y, num_h_points=20, mode="random")
    >>> result['h_opt_gcv']
    array([0.25, 0.30])
    """

    start_time = time.time()
    
    # Convert to proper shapes
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
        mode = "grid"

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    if X_new is not None:

        X_new = np.asarray(X_new, dtype=float)

        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, X.shape[1])
    
    n, d = X.shape
    
    if n != y.shape[0]:
        raise ValueError("Mismatch between X rows and y length")
    
    # Silverman bandwidth range
    sd_vec = np.std(X, axis=0, ddof=1)

    h_s = 1.06 * sd_vec * n**(-1/(4 + d))

    h_min = 0.01 * h_s

    h_max = 2.0 * h_s
    
    # Generate bandwidth candidates
    if mode == "random":

        H_candidates = np.zeros((num_h_points, d))

        for j in range(d):

            H_candidates[:, j] = np.random.uniform(
                h_min[j],
                h_max[j],
                num_h_points
            )

    else:

        grid_per_dim = [
            np.linspace(h_min[j], h_max[j], num_h_points)
            for j in range(d)
        ]
        
        if d == 1:

            H_candidates = grid_per_dim[0].reshape(-1, 1)

        else:

            grids = np.meshgrid(*grid_per_dim, indexing='ij')

            H_candidates = np.stack(
                [g.ravel() for g in grids],
                axis=1
            )
    
    h_grid = H_candidates

    num_combinations = h_grid.shape[0]
    
    gcv_h = np.zeros(num_combinations)

    yhat_list = []
    
    for i in range(num_combinations):

        h_current = h_grid[i, :]
        
        W = construct_W(X, h_current)

        yhat = W @ y
        
        trW = np.trace(W)

        rss = np.sum((y - yhat)**2)

        # Numerical safeguard
        denom = (1 - trW / n)**2

        if denom < 1e-12:
            denom = 1e-12
        
        gcv_h[i] = rss / denom
        
        yhat_list.append(yhat)
    
    # Optimal bandwidth
    inds_min = np.argmin(gcv_h)

    h_opt_gcv = h_grid[inds_min, :]

    y_train_opt = yhat_list[inds_min]
    
    elapsed = time.time() - start_time
    
    # Out-of-sample prediction
    y_new = None

    if X_new is not None:

        W_new = construct_W(X, h_opt_gcv, X_new)

        y_new = W_new @ y
    
    gc.collect()

    return {
        "y_train_opt": y_train_opt,
        "y_new_pred": y_new,
        "h_opt_gcv": h_opt_gcv,
        "gcv_h": gcv_h,
        "h_grid": h_grid,
        "time_elapsed": elapsed
    }

