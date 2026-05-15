"""
Example: California Housing (2D)

Fits SNP-NW and Direct-GCV NW using

median_income
housing_median_age
-> median_house_value

Run:
    python realdata_2d.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

from snpreg import nw_direct_gcv, nw_snp, construct_W, rmse


def load_data():

    data_path = Path(__file__).resolve().parents[1] / "datasets" / "housing.csv"

    df = pd.read_csv(data_path).dropna()

    x1 = df["median_income"].values
    x2 = df["housing_median_age"].values
    y  = df["median_house_value"].values

    X = np.column_stack((x1, x2))

    return X, x1, x2, y


def run_realdata_2d(seed=111):

    X, x1, x2, y = load_data()

    np.random.seed(seed)

    res_dgcv = nw_direct_gcv(
        X,
        y,
        num_h_points=30,
        mode="random"
    )

    np.random.seed(seed)

    res_snp = nw_snp(
        X,
        y,
        num_h_points=30,
        num_slices=50
    )

    yhat_dgcv = np.asarray(res_dgcv["y_train_opt"]).ravel()
    yhat_snp = np.asarray(res_snp["y_k_opt"]).ravel()

    rmse_dg = rmse(y, yhat_dgcv)
    rmse_sn = rmse(y, yhat_snp)
    
    print("DGCV time:", res_dgcv["time_elapsed"])
    print("DGCV RMSE:", rmse_dg)
    print("DGCV h_opt_gcv:", res_dgcv["h_opt_gcv"])
    print("SNP time:", res_snp["time_elapsed"])
    print("SNP RMSE:", rmse_sn)
    print("SNP h_start:", res_snp['h_start'])
    print("SNP k_opt:", res_snp['k_opt'])
    print("SNP B:", res_snp['B'])

    # -------------------------
    # grid
    # -------------------------

    x1_grid = np.linspace(x1.min(), x1.max(), 50)
    x2_grid = np.linspace(x2.min(), x2.max(), 50)

    X1g, X2g = np.meshgrid(x1_grid, x2_grid)

    X_grid = np.column_stack((X1g.ravel(), X2g.ravel()))

    # DGCV surface

    h_opt = res_dgcv["h_opt_gcv"]

    W_grid = construct_W(X, h_opt, X_grid)

    z_dgcv = (W_grid @ y).reshape(X1g.shape)

    # SNP surface

    h_start = res_snp["h_start"]
    Yk = res_snp["y_k_minus_1_opt"]

    W_grid_snp = construct_W(X, h_start, X_grid)

    z_snp = (W_grid_snp @ Yk).reshape(X1g.shape)

    # -------------------------
    # plot
    # -------------------------

    y_plot = y / 1000
    z_snp_plot = z_snp / 1000
    z_dgcv_plot = z_dgcv / 1000

    fig = plt.figure(figsize=(14,6))

    ax1 = fig.add_subplot(1,2,1,projection="3d")

    ax1.scatter(x1, x2, y_plot, s=5, c="gray", alpha=0.2)

    ax1.plot_surface(
        X1g,
        X2g,
        z_snp_plot,
        cmap=cm.Blues,
        edgecolor="none",
        alpha=0.9
    )

    ax1.set_title("SNP Surface")

    ax1.set_xlabel("Median Income")
    ax1.set_ylabel("Housing Median Age")
    ax1.set_zlabel("House Value (×10³)")

    ax1.zaxis.set_major_locator(MaxNLocator(6))

    ax2 = fig.add_subplot(1,2,2,projection="3d")

    ax2.scatter(x1, x2, y_plot, s=5, c="gray", alpha=0.2)

    ax2.plot_surface(
        X1g,
        X2g,
        z_dgcv_plot,
        cmap=cm.Reds,
        edgecolor="none",
        alpha=0.9
    )

    ax2.set_title("DGCV Surface")

    ax2.set_xlabel("Median Income")
    ax2.set_ylabel("Housing Median Age")
    ax2.set_zlabel("House Value (×10³)")

    ax2.zaxis.set_major_locator(MaxNLocator(6))

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_realdata_2d()
