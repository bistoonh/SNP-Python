"""
Example: Synthetic mixture regression

This script reproduces the mixture regression experiment used in the paper.
It compares:

- NW with Direct GCV bandwidth selection
- SNP-NW estimator

Evaluation metrics:
- RMSE
- shifted MAPE

Run:
    python mixture.py
"""

import numpy as np
import matplotlib.pyplot as plt

from snpreg import (
    construct_W,
    nw_direct_gcv,
    nw_snp,
    rmse,
    mape_shift,
)


# --------------------------------------------------------
# Surface definition
# --------------------------------------------------------

def mixture_local_surface(x1, x2):

    ridge = 1.2 * np.exp(-100 * (x2 - 0.4)**2)

    bump = 1.0 * np.exp(-80 * ((x1 - 0.75)**2 + (x2 - 0.75)**2))

    hill = 0.8 * np.exp(-6 * ((x1 - 0.30)**2 + (x2 - 0.25)**2))

    trough = -0.7 * np.exp(-60 * ((x1 - 0.55)**2 + (x2 - 0.40)**2))

    return ridge + bump + hill + trough


# --------------------------------------------------------
# Data generator
# --------------------------------------------------------

def generate_data(n, noise_sd=0.5):
    X1 = np.random.rand(n)
    X2 = np.random.rand(n)
    Ytrue = mixture_local_surface(X1, X2)
    Y = Ytrue + np.random.normal(0, noise_sd, n)
    X = np.column_stack((X1, X2))
    return X, Y, Ytrue


# --------------------------------------------------------
# Main experiment
# --------------------------------------------------------

def mixture_experiment(
    n=5000,
    noise_sd=0.5,
    seed=111,
    grid_n=70
):
    """
    Reproduce the Mixture-of-Local-Features experiment.
    """

    #rng = np.random.default_rng(seed)

    # ---------------------------
    # Generate data
    # ---------------------------
    np.random.seed(seed)
    X, Y, Ytrue = generate_data(n, noise_sd=noise_sd)

    # ---------------------------
    # Fit models
    # ---------------------------
    np.random.seed(seed)
    res_dgcv = nw_direct_gcv(X, Y, num_h_points=30)
    
    np.random.seed(seed)
    res_snp = nw_snp(X, Y, num_h_points=30, num_slices=50)

    yhat_dgcv = res_dgcv["y_train_opt"].ravel()
    yhat_snp  = res_snp["y_k_opt"].ravel()

    # ---------------------------
    # Build evaluation grid
    # ---------------------------

    g = np.linspace(0, 1, grid_n)

    X1g, X2g = np.meshgrid(g, g)

    Xg = np.column_stack((X1g.flatten(), X2g.flatten()))

    Ztrue = mixture_local_surface(Xg[:,0], Xg[:,1])

    # DGCV surface
    h_opt = res_dgcv["h_opt_gcv"]
    W_grid_dg = construct_W(X, h_opt, Xg)
    Zdg = (W_grid_dg @ Y).reshape(X1g.shape)

    # SNP surface
    h_start = res_snp["h_start"]
    Yk = res_snp["y_k_minus_1_opt"]
    W_grid_snp = construct_W(X, h_start, Xg)
    Zsnp = (W_grid_snp @ Yk).reshape(X1g.shape)

    # ---------------------------
    # Metrics
    # ---------------------------

    rmse_dg = rmse(yhat_dgcv, Ytrue)
    rmse_sn = rmse(yhat_snp, Ytrue)

    mape_dg = mape_shift(Ytrue, yhat_dgcv)
    mape_sn = mape_shift(Ytrue, yhat_snp)

    print("DGCV time:", res_dgcv["time_elapsed"])
    print("DGCV RMSE:", rmse_dg)
    print("DGCV MAPE:", mape_dg)
    print("DGCV h_opt_gcv:", res_dgcv["h_opt_gcv"])
    
    print("SNP time:", res_snp["time_elapsed"])
    print("SNP RMSE:", rmse_sn)
    print("SNP MAPE:", mape_sn)
    print("SNP h_start:", res_snp['h_start'])
    print("SNP k_opt:", res_snp['k_opt'])
    print("SNP B:", res_snp['B'])

    # --------------------------------------------------------
    # Plot surfaces
    # --------------------------------------------------------

    fig = plt.figure(figsize=(12,12))

    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(X[:,0], X[:,1], Y, s=5, c=Y, cmap='viridis')
    ax1.set_title("Simulated Data")

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_surface(X1g, X2g, Ztrue.reshape(grid_n,grid_n), cmap='viridis')
    ax2.set_title("True Surface")

    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot_surface(X1g, X2g, Zdg, cmap='viridis')
    ax3.set_title(f"DGCV Surface (RMSE={rmse_dg:.4f})")

    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot_surface(X1g, X2g, Zsnp, cmap='viridis')
    ax4.set_title(f"SNP Surface (RMSE={rmse_sn:.4f})")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # GCV profiles
    # --------------------------------------------------------

    gcv_k = np.array(res_snp["gcv_approx_k"])

    k_vals = np.arange(1, len(gcv_k)+1)

    h_raw = np.array(res_dgcv["h_grid"])
    gcv_raw = np.array(res_dgcv["gcv_h"])

    h_scalar = np.sqrt(h_raw[:,0] * h_raw[:,1])

    idx = np.argsort(h_scalar)

    h_sorted = h_scalar[idx]
    gcv_sorted = gcv_raw[idx]

    fig, ax = plt.subplots(1,2, figsize=(12,5))

    ax[0].plot(k_vals, gcv_k, '-o')
    ax[0].set_title("GCV(k) for SNP")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("GCV(k)")
    ax[0].grid(True)

    ax[1].plot(h_sorted, gcv_sorted, '-o')
    ax[1].set_xscale("log")
    ax[1].set_title("GCV(h) for DGCV")
    ax[1].set_xlabel("Bandwidth")
    ax[1].set_ylabel("GCV(h)")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()