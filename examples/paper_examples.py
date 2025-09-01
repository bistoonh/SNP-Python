import numpy as np
from snp import SNP, DGCV

def example_stepwise():
    np.random.seed(0)
    n = 500
    x = np.sort(np.random.rand(n)*10)
    y = (x > 5).astype(float) + np.random.normal(0, 0.2, size=n)
    snp_res = SNP(x, y)
    dgcv_res = DGCV(x, y)
    return snp_res, dgcv_res
