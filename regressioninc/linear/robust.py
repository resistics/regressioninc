"""
Functions to enable robust linear regression
"""
from loguru import logger
from typing import Optional
import numpy as np
from scipy.stats import norm as Gaussian

from regressioninc.math import geometric_median


def mad(arr: np.ndarray, c: float = Gaussian.ppf(3 / 4.0), center: Optional[float] = None):
    """
    The Median Absolute Deviation along given axis of an array

    Notes
    -----
    There is no immediately obvious median for complex data. Therefore, the
    absolute value is taken before calculating our the median.

    Returns
    -------
    mad : float
        `mad` = median(abs(`a` - center))/`c`
    """
    if center is None:
        center = geometric_median(arr)
    err = (np.abs(arr - center)) / c
    return np.median(err)
