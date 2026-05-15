# SNP: Stepwise Noise Peeling for Nonparametric Regression

A Python implementation of the Stepwise Noise Peeling (SNP) method for efficient bandwidth selection in Nadaraya-Watson kernel regression.

## Overview

SNP is a computationally efficient alternative to Generalized Cross-Validation (GCV) for bandwidth selection in nonparametric regression. It transforms the continuous $d$-dimensional bandwidth optimization problem into a discrete one-dimensional path, achieving significant speedups (over $30\times$) while maintaining or improving prediction accuracy.

### Key Features

- **Fast bandwidth selection**: Over $30\times$ faster than traditional GCV
- **High accuracy**: Comparable or better prediction performance than GCV
- **Scalability**: Efficient for high-dimensional problems
- **Stability**: Avoids local minima in bandwidth optimization
- **Simple API**: Easy-to-use interface for both 1D and multidimensional regression

### How It Works

SNP operates in two phases:

**Phase I (Coarse Search):**
- Evaluates bandwidths on a logarithmic grid using data slices
- Uses fast spectral approximation for effective degrees of freedom
- Identifies promising bandwidth regions efficiently

**Phase II (Refinement):**
- Performs adaptive restarts from Phase I candidates
- Uses approximate GCV on full data for fine-tuning
- Selects optimal bandwidth with minimal computational cost

## Installation

Currently, install directly from GitHub:
```bash
pip install https://github.com/bistoonh/SNP-Python/archive/refs/heads/main.zip
```
### Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib


## Quick Start

This example demonstrates basic usage of SNP for 1D regression:

```python
import numpy as np
import snpreg

# Generate synthetic data
np.random.seed(111)
n = 10000
X = np.random.uniform(0, 10, n)
Y = np.sin(2*X) + np.random.normal(0, 0.25, n)

# Fit using SNP (fast)
result = snpreg.nw_snp(X, Y)
hstart = result['h_start']
kopt = result['k_opt']
B = result['B']
timeElapsed = result['time_elapsed']

print(f"h_start: {hstart}, k_opt: {kopt}, B: {B}, time_elapsed: {timeElapsed:.2f}(Sec)")
```

## Examples

### Example 1: 1D Smoothing with Comparison

This example compares SNP with Direct GCV on a noisy sine wave, demonstrating SNP's speed advantage while maintaining accuracy:

```python
import numpy as np
import matplotlib.pyplot as plt
import snpreg

# Generate noisy sine wave
np.random.seed(111)
n = 2000

X = np.random.uniform(0, 4*np.pi, n)
Y_true = np.sin(2*X)
Y = Y_true + np.random.normal(0, 0.3, n)

# Fit with SNP
result_snp = snpreg.nw_snp(X, Y, num_h_points=30, num_slices=50)

# Fit with Direct GCV (for comparison)
result_gcv = snpreg.nw_direct_gcv(X, Y, num_h_points=30)

# Evaluate
rmse_snp = snpreg.rmse(Y_true, result_snp['y_k_opt'])
rmse_gcv = snpreg.rmse(Y_true, result_gcv['y_train_opt'])
mape_snp = snpreg.mape_shift(Y_true, result_snp['y_k_opt'])
mape_gcv = snpreg.mape_shift(Y_true, result_gcv['y_train_opt'])

print(f"SNP: Elapsed Time: {result_snp['time_elapsed']:.1f}(Sec), RMSE: {rmse_snp:.3f}, MAPE: {mape_snp:.2f}")
print(f"GCV: Elapsed Time: {result_gcv['time_elapsed']:.1f}(Sec), RMSE: {rmse_gcv:.3f}, MAPE: {mape_gcv:.2f}")

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, Y, alpha=0.3, s=20, label='Training data', color='lightgray')
plt.plot(X, Y_true, 'k-', linewidth=2, label='True function')
plt.plot(X, result_snp['y_k_opt'], 'r-', linewidth=2, label='SNP')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'SNP: RMSE={rmse_snp:.3f}, MAPE={mape_snp:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X, Y, alpha=0.3, s=20, label='Training data', color='lightgray')
plt.plot(X, Y_true, 'k-', linewidth=2, label='True function')
plt.plot(X, result_gcv['y_train_opt'], 'g-', linewidth=2, label='GCV')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Direct GCV: RMSE={rmse_gcv:.3f}, MAPE={mape_gcv:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Experiments Section

The package includes reproducible experiments from the paper. These functions run comprehensive benchmarks and generate publication-quality figures:

```python
import snpreg

# Run runtime benchmark
snpreg.runtime_benchmark()

# Run mixture experiment (synthetic data)
snpreg.mixture_experiment()

# Run 1D real data experiment
snpreg.realdata_1d()

# Run 2D real data experiment
snpreg.realdata_2d()
```

## Parameter Tuning

SNP provides two key parameters for balancing speed and accuracy:

```python
# Faster computation (fewer bandwidth candidates and slices)
result_fast = snpreg.nw_snp(X, Y, num_h_points=20, num_slices=30)

# More thorough search (more bandwidth candidates and slices)
result_thorough = snpreg.nw_snp(X, Y, num_h_points=60, num_slices=100)

# Performance comparison
print(f"Fast SNP time: {result_fast['time_elapsed']:.2f} seconds")
print(f"Thorough SNP time: {result_thorough['time_elapsed']:.2f} seconds")
```
## Project Structure


SNP-Python/
├── snpreg/
│   ├── __init__.py
│   ├── kernels.py          # Kernel functions and weight matrix
│   ├── dgcv.py             # Direct GCV implementation
│   ├── snp.py              # SNP algorithm
│   ├── metrics.py          # Evaluation metrics
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── housing.csv     # California housing data
│   └── experiments/
│       ├── __init__.py
│       ├── mixture.py      # Mixture experiment
│       ├── realdata_1d.py  # 1D real data experiment
│       ├── realdata_2d.py  # 2D real data experiment
│       └── runtime.py      # Runtime benchmark
├── pyproject.toml
└── README.md

## License

MIT License - see LICENSE file for details.

## Author

**Bistoon Hosseini**  
Email: bistoon.hosseini@gmail.com  
GitHub: [github.com/bistoonh/SNP-Python](https://github.com/bistoonh/SNP-Python)
