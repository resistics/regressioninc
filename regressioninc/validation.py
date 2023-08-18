"""
This module implements a range of validation checks for input data
Unfortunately, the validation checks in scikit learn do not support complex
data, so implementing validation checks here.
"""
from loguru import logger
from typing import Tuple
import numpy as np
from sklearn.utils.validation import check_consistent_length


def check_X_y(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Some checks for arrays"""
    check_consistent_length(X, y)
    return X, y
