"""
Paper Examples for SNP Package

This file contains the examples used in the research paper demonstrating
the performance of SNP vs DGCV on different types of functions and real data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional
import warnings

from snp import SNP, DGCV


def stepwise_function(x: np.ndarray) -> np.ndarray:
    """Define stepwise function."""
    y = np.zeros_like(x)
    
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


def example_stepwise() -> Dict[str, Any]:
    """
    Generate stepwise function with heteroscedastic noise.
    
    Returns
    -------
    dict
        Results from SNP and DGCV analysis
    """
    print("=== Stepwise Function Example ===")
    
    # Generate data with heteroscedastic noise
    np.random.seed(2025)
    n = 5000
    x = np.random.uniform(0, 60, n)
    np.random.shuffle(x)  # shuffle
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
    
    print(f"Dataset size: {n}")
    
    # Apply SNP
    print("\nRunning SNP...")
    snp_result = SNP(x, y, verbose=True)
    
    # Apply DGCV (with fewer points for speed)
    print("\nRunning DGCV...")
    dgcv_result = DGCV(x, y, num_h_points=30, verbose=True)
    
    # Performance comparison
    speedup = dgcv_result['time_elapsed'] / snp_result['time_elapsed']
    mse_diff = np.mean((snp_result['y_k_opt'] - dgcv_result['y_h_opt']) ** 2)
    
    print(f"\nPerformance Summary:")
    print(f"SNP time:    {snp_result['time_elapsed']:.4f} seconds")
    print(f"DGCV time:   {dgcv_result['time_elapsed']:.4f} seconds")
    print(f"Speedup:     {speedup:.2f}x")
    print(f"MSE diff:    {mse_diff:.6f}")
    
    # Plot results (sample for better visualization)
    if n > 3000:
        sample_idx = np.random.choice(n, 3000, replace=False)
        x_plot = x[sample_idx]
        y_plot = y[sample_idx] / 1000  # Convert to thousands
        snp_plot = snp_result['y_k_opt'][sample_idx] / 1000
        dgcv_plot = dgcv_result['y_h_opt'][sample_idx] / 1000
    else:
        x_plot = x
        y_plot = y / 1000
        snp_plot = snp_result['y_k_opt'] / 1000
        dgcv_plot = dgcv_result['y_h_opt'] / 1000
    
    # Sort for plotting
    sort_idx = np.argsort(x_plot)
    x_plot = x_plot[sort_idx]
    y_plot = y_plot[sort_idx]
    snp_plot = snp_plot[sort_idx]
    dgcv_plot = dgcv_plot[sort_idx]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(x_plot, y_plot, s=10, alpha=0.4, color='gray', label='Data')
    plt.plot(x_plot, snp_plot, 'r-', linewidth=2, label='SNP')
    plt.plot(x_plot, dgcv_plot, 'b--', linewidth=2, label='DGCV')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value (thousands $)')
    plt.title('California Housing: Median Income vs House Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'snp_result': snp_result,
        'dgcv_result': dgcv_result,
        'data': {'x': x, 'y': y},
        'performance': {'speedup': speedup, 'mse_diff': mse_diff}
    }


def run_all_examples() -> Dict[str, Any]:
    """
    Run all paper examples.
    
    Returns
    -------
    dict
        Results from all examples
    """
    print("Running all paper examples...\n")
    
    # Example 1: Stepwise
    result1 = example_stepwise()
    input("Press [Enter] to continue to next example...")
    
    # Example 2: Wavy
    result2 = example_wavy()
    input("Press [Enter] to continue to next example...")
    
    # Example 3: Real data
    result3 = example_california_housing()
    
    print("\n=== All Examples Complete ===")
    
    # Summary table
    if result1 is not None and result2 is not None and result3 is not None:
        print("\nPerformance Summary:")
        print(f"{'Example':<15} {'SNP Time':<10} {'DGCV Time':<10} {'Speedup':<8} {'MSE Diff':<10}")
        print("-" * 55)
        print(f"{'Stepwise':<15} {result1['snp_result']['time_elapsed']:<10.4f} "
              f"{result1['dgcv_result']['time_elapsed']:<10.4f} "
              f"{result1['performance']['speedup']:<8.2f}x "
              f"{result1['performance']['mse_diff']:<10.2e}")
        print(f"{'Wavy':<15} {result2['snp_result']['time_elapsed']:<10.4f} "
              f"{result2['dgcv_result']['time_elapsed']:<10.4f} "
              f"{result2['performance']['speedup']:<8.2f}x "
              f"{result2['performance']['mse_diff']:<10.2e}")
        print(f"{'Housing':<15} {result3['snp_result']['time_elapsed']:<10.4f} "
              f"{result3['dgcv_result']['time_elapsed']:<10.4f} "
              f"{result3['performance']['speedup']:<8.2f}x "
              f"{result3['performance']['mse_diff']:<10.2e}")
    
    return {
        'stepwise': result1,
        'wavy': result2,
        'housing': result3
    }


def benchmark_performance(sizes: list = [100, 500, 1000, 2000, 5000]) -> Dict[str, Any]:
    """
    Benchmark SNP vs DGCV performance across different dataset sizes.
    
    Parameters
    ----------
    sizes : list
        List of dataset sizes to test
        
    Returns
    -------
    dict
        Benchmark results
    """
    print("=== Performance Benchmark ===")
    
    results = {
        'sizes': sizes,
        'snp_times': [],
        'dgcv_times': [],
        'speedups': [],
        'mse_diffs': []
    }
    
    for n in sizes:
        print(f"\nTesting size n={n}...")
        
        # Generate test data
        np.random.seed(42)
        x = np.sort(np.random.uniform(0, 10, n))
        y = np.sin(x) + np.random.normal(0, 0.1, n)
        
        # Run SNP
        snp_result = SNP(x, y, num_h_points=20, verbose=False)
        
        # Run DGCV
        dgcv_result = DGCV(x, y, num_h_points=20, verbose=False)
        
        # Calculate metrics
        speedup = dgcv_result['time_elapsed'] / snp_result['time_elapsed']
        mse_diff = np.mean((snp_result['y_k_opt'] - dgcv_result['y_h_opt']) ** 2)
        
        results['snp_times'].append(snp_result['time_elapsed'])
        results['dgcv_times'].append(dgcv_result['time_elapsed'])
        results['speedups'].append(speedup)
        results['mse_diffs'].append(mse_diff)
        
        print(f"  SNP: {snp_result['time_elapsed']:.4f}s")
        print(f"  DGCV: {dgcv_result['time_elapsed']:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Timing comparison
    ax1.loglog(sizes, results['snp_times'], 'r-o', label='SNP', linewidth=2)
    ax1.loglog(sizes, results['dgcv_times'], 'b-s', label='DGCV', linewidth=2)
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Computation Time vs Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speedup
    ax2.semilogx(sizes, results['speedups'], 'g-^', linewidth=2)
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Speedup (DGCV time / SNP time)')
    ax2.set_title('Performance Speedup')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print(f"\n{'Size':<8} {'SNP Time':<10} {'DGCV Time':<10} {'Speedup':<8} {'MSE Diff':<12}")
    print("-" * 52)
    for i, n in enumerate(sizes):
        print(f"{n:<8} {results['snp_times'][i]:<10.4f} {results['dgcv_times'][i]:<10.4f} "
              f"{results['speedups'][i]:<8.2f}x {results['mse_diffs'][i]:<12.2e}")
    
    return results


if __name__ == "__main__":
    # Run all examples
    print("SNP Package - Paper Examples")
    print("=" * 40)
    
    # Choice menu
    print("\nSelect example to run:")
    print("1. Stepwise function")
    print("2. Complex wavy function") 
    print("3. California housing data")
    print("4. Run all examples")
    print("5. Performance benchmark")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            example_stepwise()
        elif choice == "2":
            example_wavy()
        elif choice == "3":
            example_california_housing()
        elif choice == "4":
            run_all_examples()
        elif choice == "5":
            benchmark_performance()
        else:
            print("Invalid choice. Running all examples...")
            run_all_examples()
            
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your environment and dependencies.")_elapsed']:.4f} seconds")
    print(f"DGCV time:   {dgcv_result['time_elapsed']:.4f} seconds")
    print(f"Speedup:     {speedup:.2f}x")
    print(f"MSE diff:    {mse_diff:.6f}")
    
    # Plot results (sample 2000 points for clarity)
    sample_idx = np.random.choice(n, 2000, replace=False)
    x_sample = x[sample_idx]
    y_sample = y[sample_idx]
    y_true_sample = y_true[sample_idx]
    snp_sample = snp_result['y_k_opt'][sample_idx]
    dgcv_sample = dgcv_result['y_h_opt'][sample_idx]
    
    # Sort for plotting
    sort_idx = np.argsort(x_sample)
    x_sample = x_sample[sort_idx]
    y_sample = y_sample[sort_idx]
    y_true_sample = y_true_sample[sort_idx]
    snp_sample = snp_sample[sort_idx]
    dgcv_sample = dgcv_sample[sort_idx]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(x_sample, y_sample, s=8, alpha=0.5, color='gray', label='Data')
    plt.plot(x_sample, y_true_sample, 'k-', linewidth=2, label='True Function')
    plt.plot(x_sample, snp_sample, 'r-', linewidth=2, label='SNP')
    plt.plot(x_sample, dgcv_sample, 'b--', linewidth=2, label='DGCV')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Stepwise Function: SNP vs DGCV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'snp_result': snp_result,
        'dgcv_result': dgcv_result,
        'data': {'x': x, 'y': y, 'y_true': y_true},
        'performance': {'speedup': speedup, 'mse_diff': mse_diff}
    }


def complex_wavy_function(x: np.ndarray) -> np.ndarray:
    """Complex function with multiple components."""
    y = np.zeros_like(x)
    
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


def example_wavy() -> Dict[str, Any]:
    """
    Generate complex wavy function with mixed noise.
    
    Returns
    -------
    dict
        Results from SNP and DGCV analysis
    """
    print("=== Complex Wavy Function Example ===")
    
    np.random.seed(2025)
    n = 10000
    x = np.random.uniform(0, 40, n)
    np.random.shuffle(x)
    x = np.sort(x)
    
    y_true = complex_wavy_function(x)
    
    # Heterogeneous noise: mixed Gaussian and sparse noise
    noise_sd = 0.2 + 0.8 * (x > 25) + 0.5 * np.sin(0.2 * x)**2
    noise_gauss = np.random.normal(0, noise_sd)
    noise_sparse = np.random.normal(0, 3) * (np.random.uniform(0, 1, n) < 0.02)  # 2% strong jumps
    noise = noise_gauss + noise_sparse
    
    y = y_true + noise
    
    print(f"Dataset size: {n}")
    
    # Apply SNP
    print("\nRunning SNP...")
    snp_result = SNP(x, y, verbose=True)
    
    # Apply DGCV (with fewer points for speed on large data)
    print("\nRunning DGCV...")
    dgcv_result = DGCV(x, y, num_h_points=25, verbose=True)
    
    # Performance comparison
    speedup = dgcv_result['time_elapsed'] / snp_result['time_elapsed']
    mse_diff = np.mean((snp_result['y_k_opt'] - dgcv_result['y_h_opt']) ** 2)
    
    print(f"\nPerformance Summary:")
    print(f"SNP time:    {snp_result['time_elapsed']:.4f} seconds")
    print(f"DGCV time:   {dgcv_result['time_elapsed']:.4f} seconds")
    print(f"Speedup:     {speedup:.2f}x")
    print(f"MSE diff:    {mse_diff:.6f}")
    
    # Plot sample of 2000 points for clarity
    sample_idx = np.random.choice(n, 2000, replace=False)
    x_sample = x[sample_idx]
    y_sample = y[sample_idx]
    y_true_sample = y_true[sample_idx]
    snp_sample = snp_result['y_k_opt'][sample_idx]
    dgcv_sample = dgcv_result['y_h_opt'][sample_idx]
    
    # Sort for plotting
    sort_idx = np.argsort(x_sample)
    x_sample = x_sample[sort_idx]
    y_sample = y_sample[sort_idx]
    y_true_sample = y_true_sample[sort_idx]
    snp_sample = snp_sample[sort_idx]
    dgcv_sample = dgcv_sample[sort_idx]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(x_sample, y_sample, s=8, alpha=0.5, color='gray', label='Data')
    plt.plot(x_sample, y_true_sample, 'k-', linewidth=2, label='True Function')
    plt.plot(x_sample, snp_sample, 'r-', linewidth=2, label='SNP')
    plt.plot(x_sample, dgcv_sample, 'b--', linewidth=2, label='DGCV')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Complex Wavy Function: SNP vs DGCV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'snp_result': snp_result,
        'dgcv_result': dgcv_result,
        'data': {'x': x, 'y': y, 'y_true': y_true},
        'performance': {'speedup': speedup, 'mse_diff': mse_diff}
    }


def example_california_housing() -> Optional[Dict[str, Any]]:
    """
    California Housing Dataset Example.
    
    Returns
    -------
    dict or None
        Results from SNP and DGCV analysis, or None if data loading fails
    """
    print("=== California Housing Dataset Example ===")
    
    # Load California Housing data
    try:
        url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
        california_housing = pd.read_csv(url)
        
    except Exception as e:
        print("Error loading California Housing dataset:")
        print("Please check internet connection")
        print(f"Error message: {e}")
        return None
    
    # Extract variables
    x = california_housing['median_income'].values
    y = california_housing['median_house_value'].values
    
    # Remove missing values
    complete_cases = ~(pd.isna(x) | pd.isna(y))
    x = x[complete_cases]
    y = y[complete_cases]
    
    # Sort by x for better visualization
    sorted_idx = np.argsort(x)
    x = x[sorted_idx]
    y = y[sorted_idx]
    
    n = len(x)
    
    print(f"Dataset size: {n}")
    print(f"Median income range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"House value range: [{y.min()/1000:.0f}K, {y.max()/1000:.0f}K]")
    
    # Apply SNP
    print("\nRunning SNP...")
    snp_result = SNP(x, y, verbose=True)
    
    # Apply DGCV
    print("\nRunning DGCV...")
    dgcv_result = DGCV(x, y, num_h_points=30, verbose=True)
    
    # Performance comparison
    speedup = dgcv_result['time_elapsed'] / snp_result['time_elapsed']
    mse_diff = np.mean((snp_result['y_k_opt'] - dgcv_result['y_h_opt']) ** 2)
    
    print(f"\nPerformance Summary:")
    print(f"SNP time:    {snp_result['time