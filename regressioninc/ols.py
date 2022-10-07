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

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.coef, resids2, rank, s = linalg.lstsq(X, y, rcond=None)
        return self.coef
        

    def predict(self, X: np.ndarray):
        if self.coef is None:
            raise ValueError("Model has not been fitted")
        return np.matmul(X, self.coef)


class WeightedLeastSquares(LinearRegressor):
    """
    Perform weighted least-squares
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        return

    def predict(self, X: np.ndarray):
        return
