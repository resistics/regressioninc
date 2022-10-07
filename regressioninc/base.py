"""
Base class for regression in C

The base class implements some basic functions and the abstract classes for
regressors.
"""
from typing import Optional
from pydantic import BaseModel
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


class Regressor(BaseModel):
    """Base class for any regressor"""

    class Config:
        arbitrary_types_allowed: bool = True

    pass


class LinearRegressor(Regressor):
    """Base class for LinearRegression"""

    coef: Optional[np.ndarray] = None
    """The coefficients"""
    residuals: Optional[np.ndarray] = None
    """The square residuals"""
    rank: Optional[int] = None
    """The rank of the predictors"""
    singular: Optional[np.ndarray] = None
    """The singular values of the predictors"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the linear model"""
        raise NotImplementedError("fit is not implemented in the base class")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model

        Parameters
        ----------
        X : np.ndarray
            The predictors

        Returns
        -------
        np.ndarray
            The predicted observations

        Raises
        ------
        ValueError
            If the coefficients are None. This is most likely the case if the
            model has not been fitted first.
        """
        if self.coef is None:
            raise ValueError(f"{self.coef=}, fit the model first")
        return np.matmul(X, self.coef)

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit and predict on predictors X and observations y

        Parameters
        ----------
        X : np.ndarray
            The predictors to use for the fit and predict
        y : np.ndarray
            The observations for the fit

        Returns
        -------
        np.ndarray
            The predicted observations
        """
        self.fit(X, y)
        return self.predict(X)
