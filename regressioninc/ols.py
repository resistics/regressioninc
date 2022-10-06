"""
This module performs ordinary least squares regression on complex-valued 
variables. 

It is essentially a lightweight wrapper around the numpy least squares 
implementation as this supports complex-valued data.
"""
from loguru import logger
import numpy as np
import numpy.linalg as linalg

from base import LinearRegressor


class LeastSquares(LinearRegressor):
    """
    Standard linear regression
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.coef, resids2, rank, s = linalg.lstsq(X, y)
        

    def predict(self, X: np.ndarray):
        return 


class WeightedLeastSquares(LinearRegressor):
    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        return

    def predict(self, X: np.ndarray):
        return
