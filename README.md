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
git clone https://github.com/bistoonh/SNP-Python.git
cd SNP-Python
pip install -e .

### Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib

## Quick Start

python
import numpy as np
import snpreg

# Generate synthetic data
np.random.seed(42)
X_train = np.random.uniform(0, 10, 100)
y_train = np.sin(X_train) + np.random.normal(0, 0.2, 100)

# Fit using SNP (fast)
result = snpreg.nw_snp(X_train, y_train)
y_pred = result['y_fit']
h_opt = result['h_opt']

print(f"Optimal bandwidth: {h_opt}")
print(f"Training RMSE: {snpreg.rmse(y_train, y_pred):.4f}")

## Examples

### Example 1: 1D Smoothing with Comparison

python
import numpy as np
import matplotlib.pyplot as plt
import snpreg

# Generate noisy sine wave
np.random.seed(123)
n_train = 200
n_test = 100

X_train = np.random.uniform(0, 4*np.pi, n_train)
X_test = np.linspace(0, 4*np.pi, n_test)
y_train = np.sin(X_train) + np.random.normal(0, 0.3, n_train)
y_test = np.sin(X_test)

# Fit with SNP
result_snp = snpreg.nw_snp(X_train, y_train, X_test=X_test)

# Fit with Direct GCV (for comparison)
result_gcv = snpreg.nw_direct_gcv(X_train, y_train, X_test=X_test)

# Evaluate
rmse_snp = snpreg.rmse(y_test, result_snp['y_pred'])
rmse_gcv = snpreg.rmse(y_test, result_gcv['y_pred'])
mape_snp = snpreg.mape_shift(y_test, result_snp['y_pred'])
mape_gcv = snpreg.mape_shift(y_test, result_gcv['y_pred'])

print(f"SNP  - Bandwidth: {result_snp['h_opt']:.4f}, RMSE: {rmse_snp:.4f}, MAPE: {mape_snp:.4f}")
print(f"GCV  - Bandwidth: {result_gcv['h_opt']:.4f}, RMSE: {rmse_gcv:.4f}, MAPE: {mape_gcv:.4f}")

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.3, s=20, label='Training data')
plt.plot(X_test, y_test, 'k-', linewidth=2, label='True function')
plt.plot(X_test, result_snp['y_pred'], 'r-', linewidth=2, label=f'SNP (h={result_snp["h_opt"]:.3f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'SNP: RMSE={rmse_snp:.4f}, MAPE={mape_snp:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, alpha=0.3, s=20, label='Training data')
plt.plot(X_test, y_test, 'k-', linewidth=2, label='True function')
plt.plot(X_test, result_gcv['y_pred'], 'b-', linewidth=2, label=f'GCV (h={result_gcv["h_opt"]:.3f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Direct GCV: RMSE={rmse_gcv:.4f}, MAPE={mape_gcv:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('1d_comparison.png', dpi=150)
plt.show()

### Example 2: 2D Surface Regression

python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import snpreg

# Generate 2D data
np.random.seed(456)
n_train = 500
n_grid = 30

# Training data
X1_train = np.random.uniform(-3, 3, n_train)
X2_train = np.random.uniform(-3, 3, n_train)
X_train = np.column_stack([X1_train, X2_train])

# True function: mixture of Gaussians
y_train = (np.exp(-0.5 * (X1_train**2 + X2_train**2)) + 
0.5 * np.exp(-0.5 * ((X1_train-2)**2 + (X2_train-2)**2)) +
np.random.normal(0, 0.1, n_train))

# Test grid
x1_grid = np.linspace(-3, 3, n_grid)
x2_grid = np.linspace(-3, 3, n_grid)
X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
X_test = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])

# True surface
y_test = (np.exp(-0.5 * (X_test[:, 0]**2 + X_test[:, 1]**2)) + 
0.5 * np.exp(-0.5 * ((X_test[:, 0]-2)**2 + (X_test[:, 1]-2)**2)))

# Fit with SNP
result_snp = snpreg.nw_snp(X_train, y_train, X_test=X_test)

# Evaluate
rmse_snp = snpreg.rmse(y_test, result_snp['y_pred'])
mape_snp = snpreg.mape_shift(y_test, result_snp['y_pred'])

print(f"SNP 2D - Bandwidth: {result_snp['h_opt']}, RMSE: {rmse_snp:.4f}, MAPE: {mape_snp:.4f}")

# Plot
fig = plt.figure(figsize=(15, 5))

# Training data
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X1_train, X2_train, y_train, c=y_train, cmap='viridis', alpha=0.6, s=20)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('y')
ax1.set_title('Training Data')

# True surface
ax2 = fig.add_subplot(132, projection='3d')
Y_true = y_test.reshape(n_grid, n_grid)
surf = ax2.plot_surface(X1_mesh, X2_mesh, Y_true, cmap='viridis', alpha=0.8)
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('y')
ax2.set_title('True Surface')
fig.colorbar(surf, ax=ax2, shrink=0.5)

# SNP prediction
ax3 = fig.add_subplot(133, projection='3d')
Y_pred = result_snp['y_pred'].reshape(n_grid, n_grid)
surf = ax3.plot_surface(X1_mesh, X2_mesh, Y_pred, cmap='viridis', alpha=0.8)
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.set_zlabel('y')
ax3.set_title(f'SNP Prediction\nRMSE={rmse_snp:.4f}')
fig.colorbar(surf, ax=ax3, shrink=0.5)

plt.tight_layout()
plt.savefig('2d_surface.png', dpi=150)
plt.show()

### Example 3: Real Data - Housing Dataset

python
import snpreg
import numpy as np
import matplotlib.pyplot as plt

# Load housing data
data = snpreg.get_housing_data()
X = data[['MedInc', 'Latitude', 'Longitude']].values
y = data['MedHouseVal'].values

# Train-test split
np.random.seed(789)
n = len(y)
train_idx = np.random.choice(n, int(0.8*n), replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Fit with SNP
result_snp = snpreg.nw_snp(X_train, y_train, X_test=X_test)

# Fit with Direct GCV
result_gcv = snpreg.nw_direct_gcv(X_train, y_train, X_test=X_test)

# Evaluate
rmse_snp = snpreg.rmse(y_test, result_snp['y_pred'])
rmse_gcv = snpreg.rmse(y_test, result_gcv['y_pred'])
mape_snp = snpreg.mape_shift(y_test, result_snp['y_pred'])
mape_gcv = snpreg.mape_shift(y_test, result_gcv['y_pred'])

print(f"Housing Data Results:")
print(f"SNP  - RMSE: {rmse_snp:.4f}, MAPE: {mape_snp:.4f}")
print(f"GCV  - RMSE: {rmse_gcv:.4f}, MAPE: {mape_gcv:.4f}")

# Plot predictions vs actual
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, result_snp['y_pred'], alpha=0.5, s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title(f'SNP: RMSE={rmse_snp:.4f}')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test, result_gcv['y_pred'], alpha=0.5, s=10)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title(f'GCV: RMSE={rmse_gcv:.4f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('housing_predictions.png', dpi=150)
plt.show()

## API Reference

### Main Functions

#### `nw_snp(X_train, y_train, X_test=None, **kwargs)`

Nadaraya-Watson regression with SNP bandwidth selection.

**Parameters:**
- `X_train`: Training features (n_train, d) or (n_train,) for 1D
- `y_train`: Training targets (n_train,)
- `X_test`: Test features (n_test, d) or (n_test,) for 1D (optional)
- `h_grid`: Bandwidth grid for Phase I (default: 20 log-spaced values)
- `n_slices`: Number of data slices for Phase I (default: 5)
- `n_restarts`: Number of adaptive restarts in Phase II (default: 3)
- `tol`: Convergence tolerance (default: 1e-4)

**Returns:**
Dictionary containing:
- `h_opt`: Optimal bandwidth (scalar or array)
- `y_fit`: Fitted values on training data
- `y_pred`: Predictions on test data (if X_test provided)
- `gcv_score`: Final GCV score
- `phase1_results`: Phase I search results
- `phase2_results`: Phase II refinement results

#### `nw_direct_gcv(X_train, y_train, X_test=None, **kwargs)`

Nadaraya-Watson regression with direct GCV bandwidth selection.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training targets
- `X_test`: Test features (optional)
- `h_grid`: Bandwidth grid for search (default: 30 log-spaced values)

**Returns:**
Dictionary containing:
- `h_opt`: Optimal bandwidth
- `y_fit`: Fitted values on training data
- `y_pred`: Predictions on test data (if X_test provided)
- `gcv_score`: Final GCV score
- `gcv_curve`: GCV scores for all bandwidths

### Helper Functions

#### `construct_W(X, h, kernel='gaussian')`

Construct Nadaraya-Watson weight matrix.

**Parameters:**
- `X`: Data matrix (n, d)
- `h`: Bandwidth (scalar or array of length d)
- `kernel`: Kernel type (default: 'gaussian')

**Returns:**
- Weight matrix (n, n)

#### `generate_slices(n, n_slices)`

Generate data slices for Phase I bandwidth selection.

**Parameters:**
- `n`: Total number of samples
- `n_slices`: Number of slices

**Returns:**
- List of index arrays

### Metrics

#### `rmse(y_true, y_pred)`

Root Mean Squared Error.

#### `mape_shift(y_true, y_pred, shift=1.0)`

Mean Absolute Percentage Error with shift to avoid division by zero.

## Experiments

The package includes reproducible experiments from the paper:

python
import snpreg

# Run mixture experiment (synthetic data)
snpreg.mixture_experiment()

# Run 1D real data experiment
snpreg.realdata_1d()

# Run 2D real data experiment
snpreg.realdata_2d()

# Run runtime benchmark
snpreg.runtime_benchmark()

## Performance

SNP provides significant computational advantages over traditional GCV:

| Dataset | Dimension | SNP Time | GCV Time | Speedup |
|---------|-----------|----------|----------|---------|
| Synthetic 1D | 1 | 0.12s | 3.8s | 31.7× |
| Housing | 3 | 0.45s | 15.2s | 33.8× |
| Synthetic 2D | 2 | 0.28s | 9.1s | 32.5× |

*Times measured on Intel i7-10700K, 16GB RAM*

## Algorithm Details

### Phase I: Coarse Search

1. Generate $K$ data slices via random sampling
2. Evaluate bandwidths on logarithmic grid: $h_j = h_{\min} \cdot (h_{\max}/h_{\min})^{j/G}$
3. For each bandwidth and slice:
   - Compute weight matrix $W(h)$
   - Use spectral approximation for effective degrees of freedom
   - Calculate approximate GCV score
4. Select top $R$ bandwidths with lowest average GCV

### Phase II: Refinement

1. For each of $R$ candidate bandwidths:
   - Perform gradient-based refinement on full data
   - Use approximate GCV (no matrix inversion)
   - Apply adaptive step size with convergence check
2. Select bandwidth with minimum GCV score

### GCV Criterion

$$\text{GCV}(h) = \frac{n \cdot \text{RSS}(h)}{(n - \text{df}(h))^2}$$

where:
- $\text{RSS}(h) = \|y - \hat{y}(h)\|^2$ is residual sum of squares
- $\text{df}(h) = \text{tr}(W(h))$ is effective degrees of freedom
- $W(h)$ is the Nadaraya-Watson smoother matrix

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

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details.

## Author

**Bistoon Hosseini**  
Email: bistoon.hosseini@gmail.com  
GitHub: [github.com/bistoonh/SNP-Python](https://github.com/bistoonh/SNP-Python)

## Acknowledgments

This implementation is based on the Stepwise Noise Peeling method for efficient bandwidth selection in nonparametric regression. The method demonstrates significant computational advantages while maintaining high prediction accuracy across various datasets and dimensions.


