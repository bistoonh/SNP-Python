"""
Example: Runtime scaling benchmark

Compares runtime of:

- Direct GCV bandwidth selection
- SNP-NW estimator

for increasing sample sizes and dimensions.

Outputs:
    runtime_results_raw.csv
    runtime_results_mean.csv
    runtime_comparison.png
    runtime_comparison.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.gridspec import GridSpec

from snpreg import nw_direct_gcv, nw_snp, rmse, mape_shift


# --------------------------------------------------
# Synthetic data generator
# --------------------------------------------------

def generate_data(n, d, noise_scale=0.2, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    if d == 1:

        X1 = rng.uniform(0, 1, n)
        X = X1[:, None]

        y_true = 1.5 * np.sin(2 * np.pi * X1)

    elif d == 2:

        X1 = rng.uniform(0, 1, n)
        X2 = rng.uniform(-3, 3, n)

        X = np.column_stack((X1, X2))

        y_true = (
            1.5 * np.sin(2 * np.pi * X1)
            + 0.8 * np.cos(3 * X2)
        )

    elif d == 3:

        X1 = rng.uniform(0, 1, n)
        X2 = rng.uniform(-3, 3, n)
        X3 = rng.uniform(0, 10, n)

        X = np.column_stack((X1, X2, X3))

        y_true = (
            1.5 * np.sin(2 * np.pi * X1)
            + 0.8 * np.cos(3 * X2)
            + 0.3 * (X3 - 5) ** 2 * np.exp(-0.2 * X3)
        )

    else:
        raise ValueError("d must be 1, 2, or 3")

    noise = rng.normal(0, noise_scale, n)
    y = y_true + noise

    return X, y, y_true


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _save_progress(records, output_dir):
    """
    Save raw and aggregated runtime results to CSV.
    """
    df_raw = pd.DataFrame(records)

    raw_path = output_dir / "runtime_results_raw.csv"
    mean_path = output_dir / "runtime_results_mean.csv"

    df_raw.to_csv(raw_path, index=False)

    if len(df_raw) == 0:
        df_mean = pd.DataFrame(
            columns=["dim", "n", "method", "rmse", "mape_shift", "time_elapsed", "n_rep"]
        )
    else:
        df_mean = (
            df_raw
            .groupby(["dim", "n", "method"], as_index=False)
            .agg(
                rmse=("rmse", "mean"),
                mape_shift=("mape_shift", "mean"),
                time_elapsed=("time_elapsed", "mean"),
                n_rep=("rep", "count")
            )
        )

    df_mean.to_csv(mean_path, index=False)

    return df_raw, df_mean


def _make_plot(df_mean, output_dir):
    """
    Build and save summary plot.
    """
    if df_mean.empty:
        print("[runtime_benchmark] No results available for plotting.")
        return

    methods = list(df_mean["method"].unique())
    dims_sorted = sorted(df_mean["dim"].unique())

    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(2, 2, figure=fig)

    ax_time = fig.add_subplot(gs[0, :])
    ax_rmse = fig.add_subplot(gs[1, 0])
    ax_mape = fig.add_subplot(gs[1, 1])

    for d in dims_sorted:
        df_d = df_mean[df_mean["dim"] == d]

        for method in methods:
            df_dm = df_d[df_d["method"] == method].sort_values("n")

            if df_dm.empty:
                continue

            label = f"{method} (d={d})"

            ax_time.plot(
                df_dm["n"],
                df_dm["time_elapsed"],
                marker="o",
                linewidth=2,
                label=label
            )

            ax_rmse.plot(
                df_dm["n"],
                df_dm["rmse"],
                marker="o",
                linewidth=2,
                label=label
            )

            ax_mape.plot(
                df_dm["n"],
                df_dm["mape_shift"],
                marker="o",
                linewidth=2,
                label=label
            )

    ax_time.set_title("Runtime vs Sample Size")
    ax_time.set_xlabel("n")
    ax_time.set_ylabel("Time (seconds)")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(ncol=3, fontsize=9)

    ax_rmse.set_title("RMSE vs Sample Size")
    ax_rmse.set_xlabel("n")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.grid(True, alpha=0.3)

    ax_mape.set_title("MAPE vs Sample Size")
    ax_mape.set_xlabel("n")
    ax_mape.set_ylabel("MAPE (%)")
    ax_mape.grid(True, alpha=0.3)

    fig.tight_layout()

    png_path = output_dir / "runtime_comparison.png"
    pdf_path = output_dir / "runtime_comparison.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"[runtime_benchmark] Plot saved to: {png_path.resolve()}")
    print(f"[runtime_benchmark] Plot saved to: {pdf_path.resolve()}")

    plt.show()


# --------------------------------------------------
# Runtime experiment
# --------------------------------------------------

def runtime_benchmark(
    n_list=(500, 1500, 3000, 8000, 13000, 20000, 30000),
    dims=(1, 2, 3),
    n_rep=10,
    seed=111,
    output_dir="runtime_results"
):
    # مسیر ذخیره‌سازی
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[runtime_benchmark] Saving outputs to: {output_dir.resolve()}")

    records = []

    for n in n_list:
        print(f"\n========== Sample size n = {n} ==========")
        for d in dims:
            print(f"[runtime_benchmark] Starting dimension d = {d}")

            for rep in range(1, n_rep + 1):
                # ------------------------------------------------------
                # 1. ایجاد Seed منحصربه‌فرد برای این تکرار خاص
                # این کار باعث می‌شود rep=1 همیشه نتیجه یکسانی بدهد، 
                # چه تنها اجرا شود چه در یک لیست بزرگ.
                # ------------------------------------------------------
                current_seed = seed + (n * 10000) + (d * 1000) + rep
                
                # تنظیم رندوم سراسری برای توابع داخلی (مثل DGCV random mode)
                np.random.seed(current_seed)
                
                # ایجاد RNG محلی برای تولید دیتا
                local_rng = np.random.default_rng(current_seed)

                # تولید دیتا با استفاده از RNG محلی
                X, y, y_true = generate_data(
                    n=n,
                    d=d,
                    noise_scale=0.2,
                    rng=local_rng
                )

                print(f"  rep {rep}/{n_rep} (Seed: {current_seed})")

                # -----------------------------
                # Direct GCV
                # -----------------------------
                # چون قبل از این خط np.random.seed زدیم، 
                # حالت random این تابع هم فیکس می‌شود.
                out_dg = nw_direct_gcv(
                    X,
                    y,
                    num_h_points=30,
                    mode="random"
                )

                yhat_dg = np.asarray(out_dg["y_train_opt"]).ravel()

                records.append({
                    "dim": d,
                    "n": n,
                    "rep": rep,
                    "method": "DGCV",
                    "rmse": rmse(y_true, yhat_dg),
                    "mape_shift": mape_shift(y_true, yhat_dg),
                    "time_elapsed": out_dg["time_elapsed"]
                })

                # -----------------------------
                # SNP
                # -----------------------------
                out_snp = nw_snp(
                    X,
                    y,
                    num_h_points=30,
                    num_slices=50
                )

                yhat_snp = np.asarray(out_snp["y_k_opt"]).ravel()

                records.append({
                    "dim": d,
                    "n": n,
                    "rep": rep,
                    "method": "SNP",
                    "rmse": rmse(y_true, yhat_snp),
                    "mape_shift": mape_shift(y_true, yhat_snp),
                    "time_elapsed": out_snp["time_elapsed"]
                })

            # ذخیره تدریجی
            df_raw, df_mean = _save_progress(records, output_dir)

    # رسم پلات نهایی
    _make_plot(df_mean, output_dir)

    return df_raw, df_mean


if __name__ == "__main__":
    runtime_benchmark()
