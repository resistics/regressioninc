"""
This module performs ordinary least squares regression on complex-valued 
variables. 

It is essentially a lightweight wrapper around the numpy least squares 
implementation as this supports complex-valued data.
"""
from loguru import logger
import numpy as np
import numpy.linalg as linalg

from regressioninc.base import LinearRegressor


class LeastSquares(LinearRegressor):
    """Standard linear regression"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the linear problem using least squares regression

        Parameters
        ----------
        X : np.ndarray
            The predictors
        y : np.ndarray
            The observations

        Returns
        -------
        np.ndarray
            The coefficients
        """
        result = linalg.lstsq(X, y, rcond=None)
        self.coef, self.residuals, self.rank, self.singular = result
        return self.coef


class WeightedLeastSquares(LinearRegressor):
    r"""
    Transform X and y using the weights to perform a weighted least squares

    .. math::
        \sqrt{weights} y = \sqrt{weights} X coef ,

    is equivalent to,

    .. math::
        X^H weights y = X^H weights X coef ,

    where :math:`X^H` is the hermitian transpose of X.

    In this method, both the observations y and the predictors X are multipled
    by the square root of the weights and then returned.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Apply weights to observations and predictors and find the coefficients
        of the linear model

        Parameters
        ----------
        X : np.ndarray
            The predictors
        y : np.ndarray
            The observations
        weights : np.ndarray
            The weights to apply to the samples

        Returns
        -------
        np.ndarray
            The coefficients for the model

        Raises
        ------
        ValueError
            If the size of weights does not match the size of the observations
            y
        """
        if weights.size != y.size:
            raise ValueError(f"{weights.size=} != {y.size=}")
        weights_sqrt = np.sqrt(weights)
        y = weights_sqrt * y
        X = X * weights_sqrt[..., np.newaxis]
        result = linalg.lstsq(X, y, rcond=None)
        self.coef, self.residuals, self.rank, self.singular = result
        return self.coef
