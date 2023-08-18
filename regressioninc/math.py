"""
Helpful arithmetic functions for complex numbers
"""
import numpy as np
from geom_median.numpy import compute_geometric_median

from regressioninc.transform import complex_to_real_2d


def transpose(arr: np.ndarray) -> np.ndarray:
    """
    Hermitian transpose of an array (transpose and complex conjugation)

    Parameters
    ----------
    arr : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Hermitian transpose
    """
    return np.conjugate(np.transpose(arr))


def sum_r2(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> float:
    """
    Calculate sum of square residuals

    Parameters
    ----------
    X : np.ndarray
        The regressors
    y : np.ndarray
        The observations
    coef : np.ndarray
        The estimated coefficients

    Returns
    -------
    float
        The sum of square residuals
    """
    residuals = y - np.matmul(X, coef)
    residuals2 = residuals * transpose(residuals)
    return np.sum(residuals2.real)


def geometric_median(arr: np.ndarray) -> np.complex_:
    """
    Calculate the geometric median for a 1-D array of complex numbers

    https://en.wikipedia.org/wiki/Geometric_median

    Parameters
    ----------
    arr : np.ndarray
        A 1-D complex array

    Returns
    -------
    np.complex_
        The geometric median

    Examples
    --------

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from regressioninc.math import geometric_median
        >>> arr = np.array([1 - 1j, -1 - 1j, -1 + 1j, 1 + 1j])
        >>> med = geometric_median(arr)
        >>> print(np.round(med, 6))
        0j
        >>> plt.figure() # doctest: +SKIP
        >>> plt.scatter(arr.real, arr.imag, c="b", marker="x") # doctest: +SKIP
        >>> plt.scatter(med.real, med.imag, c="r", marker="o") # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
    
    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from regressioninc.math import geometric_median    
        >>> arr = np.array([3 - 2j, -4 - 9j, 2 + 1j, -3 + 6j])
        >>> med = geometric_median(arr)
        >>> print(np.round(med, 6))
        (1.444444+0.074074j)
        >>> plt.figure() # doctest: +SKIP
        >>> plt.scatter(arr.real, arr.imag, c="b", marker="x") # doctest: +SKIP
        >>> plt.scatter(med.real, med.imag, c="r", marker="o") # doctest: +SKIP
        >>> plt.tight_layout() # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
    """
    arr = complex_to_real_2d(arr)
    out = compute_geometric_median(arr)
    return out.median[0] + 1j * out.median[1]
