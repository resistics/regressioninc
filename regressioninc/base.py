"""
Base class for regression in C
"""
from loguru import logger
import numpy as np


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add an intercept to the regressors

    Parameters
    ----------
    X : np.ndarray
        The regressors

    Returns
    -------
    np.ndarray
        The regressors with an extra column of ones
    """
    n_samples = X.shape[0]
    return np.hstack((np.ones(shape=(n_samples, 1), dtype=X.dtype), X))


class Regressor:

    pass


class LinearRegressor(Regressor):
    def fit(self):
        raise NotImplementedError("fit is not implemented in the base class")

    def predict(self):
        return NotImplementedError("predict is not implemented in the base class")
