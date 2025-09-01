"""
Paper Examples for SNP Package

This file contains the examples used in the research paper demonstrating
the performance of SNP vs DGCV on different types of functions and real data.
"""

import numpy as np
import time
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from SNP import SNP, DGCV


def example_stepwise():
    """Generate stepwise function with heteroscedastic noise"""
    
    def stepwise_function(x):
        """Define stepwise function"""
        y = np.zeros(len(x))
        
        # Define intervals (steps)
        idx1 = x <= 20
        idx2 = (x > 20) & (x <= 35)
        idx3 = (x > 35) & (x <= 45)
        idx4 = x > 45
        
        # Function values in each interval
        y[idx1] = 2
        y[idx2] = -1
        y[idx3] = 3
        y[idx4] = 0.5
        
        return y
    
    # Generate data with heteroscedastic noise
    np.random.seed(2025)
    n = 5000
    x = np.random.uniform(0, 60, n)
    x = np.random.permutation(x)  # shuffle
    x = np.sort(x)
    y_true = stepwise_function(x)
    
    # Different noise variance in each interval
    noise_sd = np.zeros(n)
    noise_sd[x <= 20] = 0.2
    noise_sd[(x > 20) & (x <= 35)] = 0.8
    noise_sd[(x > 35) & (x <= 45)] = 0.1
    noise_sd[x > 45] = 1.5
    
    noise = np.random.normal(0, noise_sd)
    y = y_true + noise
    
    print("=== Stepwise Function Example ===")
    print("Dataset size:", n)
    
    # Apply SNP
    snp_result = SNP(x, y)
    
    # Apply DGCV
    dgcv_result = DGCV(x, y, num_h_points=50)
    
    # Calculate RMSE against true function
    rmse_snp = np.sqrt(np.mean((snp_result['y_k_opt'] - y_true)**2))
    rmse_dgcv = np.sqrt(np.mean((dgcv_result['y_h_opt'] - y_true)**2))
    speedup = dgcv_result['time_elapsed'] / snp_result['time_elapsed']
    
    print(f"\n--- Results Summary ---")
    print(f"SNP:  h_start={snp_result['h_start']:.4f}, k_opt={snp_result['k_opt']}, RMSE={rmse_snp:.4f}, Time={snp_result['time_elapsed']:.4f}s")
    print(f"DGCV: h_opt={dgcv_result['h_opt_gcv']:.4f}, RMSE={rmse_dgcv:.4f}, Time={dgcv_result['time_elapsed']:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Plot results (sample 2000 points for clarity)
    if HAS_MATPLOTLIB:
        sample_idx = np.random.choice(n, 2000, replace=False)
        x_sample = x[sample_idx]
        y_sample = y[sample_idx]
        y_true_sample = y_true[sample_idx]
        snp_sample = snp_result['y_k_opt'][sample_idx]
        dgcv_sample = dgcv_result['y_h_opt'][sample_idx]
        
        # Sort by x for proper line plotting
        sort_idx = np.argsort(x_sample)
        x_sample = x_sample[sort_idx]
        y_sample = y_sample[sort_idx]
        y_true_sample = y_true_sample[sort_idx]
        snp_sample = snp_sample[sort_idx]
        dgcv_sample = dgcv_sample[sort_idx]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_sample, y_sample, s=5, c="gray", alpha=0.7, label="Data")
        plt.plot(x_sample, y_true_sample, 'k-', linewidth=2, label="True Function")
        plt.plot(x_sample, snp_sample, 'r-', linewidth=2, label="SNP")
        plt.plot(x_sample, dgcv_sample, 'b--', linewidth=2, label="DGCV")
        plt.title("Stepwise Function: SNP vs DGCV")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return {
        'performance': {
            'rmse_snp': rmse_snp,
            'rmse_dgcv': rmse_dgcv
        }
    }


def example_wavy():
    """Generate complex wavy function with mixed noise"""
    
    np.random.seed(2025)
    n = 10000
    x = np.random.uniform(0, 40, n)
    x = np.random.permutation(x)
    x = np.sort(x)
    
    # Complex function with multiple components
    def f_true(x):
        y = np.zeros(len(x))
        
        # Part 1: Sinusoidal with variable frequency and amplitude
        idx1 = x <= 10
        y[idx1] = np.sin(0.3 * x[idx1]) * (1 + 0.3 * np.cos(0.5 * x[idx1]))
        
        # Part 2: Negative parabolic trend with steep slope
        idx2 = (x > 10) & (x <= 20)
        y[idx2] = -0.05 * (x[idx2] - 15)**2 + 0.5 * np.sin(0.7 * x[idx2])
        
        # Part 3: Linear with positive slope and severe noise
        idx3 = (x > 20) & (x <= 30)
        y[idx3] = 0.2 * (x[idx3] - 20) + 0.1 * np.sin(2 * x[idx3])
        
        # Part 4: High frequency oscillation with gradually decreasing amplitude
        idx4 = x > 30
        y[idx4] = 0.5 * np.sin(5 * x[idx4]) * np.exp(-0.1 * (x[idx4] - 30))
        
        return y
    
    y_true = f_true(x)
    
    # Heterogeneous noise: mixed Gaussian and sparse noise
    noise_sd = 0.2 + 0.8 * (x > 25) + 0.5 * np.sin(0.2 * x)**2
    noise_gauss = np.random.normal(0, noise_sd)
    noise_sparse = np.random.normal(0, 3, n) * (np.random.uniform(0, 1, n) < 0.02)  # 2% strong jumps
    noise = noise_gauss + noise_sparse
    
    y = y_true + noise
    
    print("=== Complex Wavy Function Example ===")
    print("Dataset size:", n)
    
    # Apply SNP
    snp_result = SNP(x, y)
    
    # Apply DGCV
    dgcv_result = DGCV(x, y, num_h_points=50)
    
    # Calculate RMSE against true function
    rmse_snp = np.sqrt(np.mean((snp_result['y_k_opt'] - y_true)**2))
    rmse_dgcv = np.sqrt(np.mean((dgcv_result['y_h_opt'] - y_true)**2))
    speedup = dgcv_result['time_elapsed'] / snp_result['time_elapsed']
    
    print(f"\n--- Results Summary ---")
    print(f"SNP:  h_start={snp_result['h_start']:.4f}, k_opt={snp_result['k_opt']}, RMSE={rmse_snp:.4f}, Time={snp_result['time_elapsed']:.4f}s")
    print(f"DGCV: h_opt={dgcv_result['h_opt_gcv']:.4f}, RMSE={rmse_dgcv:.4f}, Time={dgcv_result['time_elapsed']:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Plot results (sample 2000 points for clarity)
    if HAS_MATPLOTLIB:
        sample_idx = np.random.choice(n, 2000, replace=False)
        x_sample = x[sample_idx]
        y_sample = y[sample_idx]
        y_true_sample = y_true[sample_idx]
        snp_sample = snp_result['y_k_opt'][sample_idx]
        dgcv_sample = dgcv_result['y_h_opt'][sample_idx]
        
        # Sort by x for proper line plotting
        sort_idx = np.argsort(x_sample)
        x_sample = x_sample[sort_idx]
        y_sample = y_sample[sort_idx]
        y_true_sample = y_true_sample[sort_idx]
        snp_sample = snp_sample[sort_idx]
        dgcv_sample = dgcv_sample[sort_idx]
        
        plt.figure(figsize=(12,