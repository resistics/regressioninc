"""
Functions to enable robust linear regression
"""
from typing import Optional
import numpy as np
from scipy.stats import norm as Gaussian

from regressioninc.math import geometric_median


def mad(
    arr: np.ndarray, c: float = Gaussian.ppf(3 / 4.0), center: Optional[float] = None
) -> float:
    """
    The Median Absolute Deviation from the center

    There is no immediately obvious median for complex data. Therefore, the
    absolute value is taken before calculating out the median.

    Parameters
    ----------
    arr : np.ndarray
        The data of which to take the
    c : float, optional
        A scaling factor, by default Gaussian.ppf(3 / 4.0)
    center : Optional[float], optional
        The center from which to calculate the median deviation, by default
        None. If it is None, the geometric median of the data will be used.

    Returns
    -------
    float
        `mad` = median(abs(`a` - center))/`c`
    """
    if center is None:
        center = geometric_median(arr)
    err = (np.abs(arr - center)) / c
    return np.median(err)
