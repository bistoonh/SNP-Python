# SNP: Stepwise Noise Peeling for Nadaraya-Watson Regression

<!-- badges: start -->
[![Python package](https://github.com/bistoonh/SNP-Python/workflows/Python%20package/badge.svg)](https://github.com/bistoonh/SNP-Python/actions)
<!-- badges: end -->

The **SNP** package implements the Stepwise Noise Peeling algorithm that bypasses bandwidth selection in Nadaraya-Watson regression by using iterative smoothing. SNP provides a scalable alternative to Direct Generalized Cross-Validation (DGCV) by converting continuous bandwidth optimization into discrete iteration selection, dramatically reducing computational cost while maintaining statistical equivalence.

## Installation

You can install the development version of SNP from GitHub with:

```bash
# Install from GitHub
pip install git+https://github.com/bistoonh/SNP-Python.git
```

## Quick Start

```python
import numpy as np
from SNP import SNP, DGCV
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(123)
n = 1500
x = np.sort(np.random.uniform(0, 1, n))
y = np.sin(2*np.pi*x) + np.random.normal(0, 0.1, n)

# Apply SNP smoothing with default parameters
snp_result = SNP(x, y)

# Compare with traditional DGCV
dgcv_result = DGCV(x, y)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=5, c="gray", alpha=0.7, label="Data")
plt.plot(x, snp_result['y_k_opt'], 'r-', linewidth=2, label="SNP")
plt.plot(x, dgcv_result['y_h_opt'], 'b--', linewidth=2, label="DGCV")
plt.title("SNP vs DGCV Comparison")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Performance comparison
print(f"SNP time: {snp_result['time_elapsed']:.4f} seconds")
print(f"DGCV time: {dgcv_result['time_elapsed']:.4f} seconds")
```

## Parameter Tuning

SNP provides two key parameters for balancing speed and accuracy:

```python
# Faster computation (fewer bandwidth candidates and slices)
snp_fast = SNP(x, y, num_h_points=20, num_slices=30)

# More thorough search (more bandwidth candidates and slices)
snp_thorough = SNP(x, y, num_h_points=60, num_slices=100)

# Performance comparison
print(f"Fast SNP time: {snp_fast['time_elapsed']:.4f} seconds")
print(f"Thorough SNP time: {snp_thorough['time_elapsed']:.4f} seconds")
```

## Key Features

- **Fast**: Orders of magnitude faster than DGCV for large datasets
- **Accurate**: Statistically equivalent results to DGCV
- **Adaptive**: Automatically adjusts bandwidth through iterative process
- **Configurable**: Tunable parameters for speed vs accuracy trade-offs
- **Robust**: Handles edge cases and various data sizes
- **Well-documented**: Comprehensive help files and examples

## Algorithm Overview

SNP operates in two phases:

1. **Phase I**: Constructs a conservative initial bandwidth using random slices of data and lightweight GCV within each slice
   - `num_slices`: Controls number of random data slices (default: 60)
   - `num_h_points`: Controls bandwidth candidates per slice (default: 40)
   
2. **Phase II**: Fixes the smoothing operator and repeatedly applies it, selecting optimal iterations via discrete GCV

This reformulation preserves the adaptivity of GCV while converting costly continuous bandwidth search into lightweight discrete selection.

## Main Functions

- `SNP(x, y, num_h_points=40, num_slices=60)`: Main Stepwise Noise Peeling algorithm
- `DGCV(x, y, num_h_points=50)`: Direct Generalized Cross-Validation (reference method)  
- `construct_W(x, h)`: Construct Gaussian kernel weight matrix

## Performance

For datasets with n > 1000, SNP typically shows:
- **Speed**: Orders of magnitude faster than DGCV
- **Accuracy**: < 1% difference in RMSE compared to DGCV
- **Memory**: More efficient memory usage due to iterative approach
- **Scalability**: Parameter tuning allows adaptation to computational constraints

## Citation

If you use this package in your research, please cite:

```

```

## Contributing



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Issues

Found a bug? Have a feature request? Please [open an issue](https://github.com/bistoonh/SNP-Python/issues) on GitHub.