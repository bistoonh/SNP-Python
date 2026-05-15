import numpy as np
import time
import gc

def construct_W(X_train, h, X_new=None):
    """
    Construct Nadaraya-Watson smoother matrix using Gaussian kernel.
    
    This function computes the weight matrix W where each row contains the 
    normalized kernel weights for smoothing. When X_new is provided, it 
    constructs a cross-smoother matrix for prediction.
    
    Parameters
    ----------
    X_train : ndarray of shape (n_train, d)
        Training data points used as kernel centers.
        
    h : float or ndarray of shape (d,)
        Bandwidth parameter controlling the kernel width. Larger values 
        produce smoother estimates.
        
    X_new : ndarray of shape (n_new, d), optional
        New points at which to evaluate the smoother. If None, uses X_train
        (default behavior for fitting).
    
    Returns
    -------
    W : ndarray of shape (n_new, n_train) or (n_train, n_train)
        Smoother weight matrix. Each row sums to 1 and contains the kernel
        weights for the corresponding observation in X_new.
    
    Notes
    -----
    The Gaussian kernel is defined as:
        K(u) = exp(-0.5 * ||u||^2)
    
    The weight matrix elements are:
        W[i,j] = K((X_new[i] - X_train[j]) / h) / sum_k K((X_new[i] - X_train[k]) / h)
    
    Examples
    --------
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> W = construct_W(X, h=1.0)
    >>> W.shape
    (3, 3)
    >>> np.allclose(W.sum(axis=1), 1.0)
    True
    """
    
    if X_new is None:
        X_new = X_train

    X_train_scaled = X_train / h
    X_new_scaled = X_new / h

    dist_sq = ((X_new_scaled[:, None, :] - X_train_scaled[None, :, :])**2).sum(axis=2)

    K = np.exp(-0.5 * dist_sq)

    return K / K.sum(axis=1, keepdims=True)
