"""
Example: California Housing (1D)

Fits SNP-NW and Direct-GCV NW using
median_income -> median_house_value

Run:
    python realdata_1d.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from snpreg import nw_direct_gcv, nw_snp, rmse
from pathlib import Path


def load_data():

    data_path = Path(__file__).resolve().parents[1] / "datasets" / "housing.csv"

    df = pd.read_csv(data_path).dropna()

    x = df["median_income"].values
    y = df["median_house_value"].values

    idx = np.argsort(x)

    return x[idx], y[idx]


def run_realdata_1d(seed=111):

    x, y = load_data()

    np.random.seed(seed)

    res_dgcv = nw_direct_gcv(
        x,
        y,
        num_h_points=30,
        mode="random"
    )

    yhat_dgcv = np.asarray(res_dgcv["y_train_opt"]).ravel()

    np.random.seed(seed)

    res_snp = nw_snp(
        x,
        y,
        num_h_points=30,
        num_slices=50
    )

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

    plt.figure(figsize=(7,5))

    plt.scatter(
        x,
        y,
        s=2,
        alpha=0.35,
        color="black",
        edgecolors="none",
        label="Data"
    )

    plt.plot(
        x,
        yhat_snp,
        color="#1f77b4",
        linewidth=2.6,
        label="SNP"
    )

    plt.plot(
        x,
        yhat_dgcv,
        color="#d62728",
        linewidth=2.2,
        label="DGCV"
    )

    plt.xlabel("Median Income")
    plt.ylabel("Median House Value")

    plt.legend(frameon=False)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_realdata_1d()
