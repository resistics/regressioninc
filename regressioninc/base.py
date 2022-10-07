"""
Base class for regression in C

The base class implements some basic functions and the abstract classes for
regressors.
"""
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
        The regressors with an extra column of ones to represent to allow
        solving for the intercept term

    Examples
    --------
    >>> import numpy as np
    >>> from regressioninc.base import add_intercept
    >>> X = np.array([[2,3], [4,5]])
    >>> X
    array([[2, 3],
           [4, 5]])

    Now add the intercept

    >>> X = add_intercept(X)
    >>> X
    array([[2, 3, 1],
           [4, 5, 1]])
    """
    n_samples = X.shape[0]
    return np.hstack((X, np.ones(shape=(n_samples, 1), dtype=X.dtype)))


class Regressor:
    """Base class for any regressor"""

    pass


class LinearRegressor(Regressor):
    """Base class for a linear regressor"""

    def fit(self):
        raise NotImplementedError("fit is not implemented in the base class")

    def predict(self):
        return NotImplementedError("predict is not implemented in the base class")
