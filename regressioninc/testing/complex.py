"""
Functions for generating and visualising complex-valued testing data
"""
from typing import Optional
import numpy as np


def generate_linear(coef: np.ndarray, intercept: Optional[complex] = None):
    """Produce complex data for testing without any noise"""
    n_samples = coef.size * 2
    n_features = coef.size
    shape = (n_samples, n_features)
    # generate the data
    X = np.random.uniform(-20, 20, size=shape)
    X = X.astype(complex) + 1.0j * np.random.uniform(-20, 20, size=shape)
    y = np.matmul(X, coef) + intercept
    return X, y
