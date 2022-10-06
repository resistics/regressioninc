"""
Generate regression testing data
"""
from typing import Optional
import numpy as np


def linear_real(coef: Optional[np.ndarray] = None, intercept: int = 0):
    """
    Produce real data for testing without any noise
    

    coefficients = [3 -7]
    intercept if added = 12
    """
    np.random.seed(42)    
    if coef is None:
        coef = np.array([3, -7])
    n_samples = coef.size * 2
    n_features = coef.size
    X = np.random.randint(-20, 20, size=(n_samples, n_features))
    y = (X * coef) + intercept
    return X, y

