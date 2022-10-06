"""
An implementation of mm estimates for complex-valued variables. The 
implementation API is designed to match that of scikit-learn.
"""
from loguru import logger
import numpy as np

from base import LinearRegressor


class MM_estimate(LinearRegressor):
    def fit(self, X: np.ndarray, y: np.ndarray):
        return

    def predict(self):
        return
