"""
A module with functions for checking and validating arrays
"""
from typing import Tuple
import numpy as np


def num_samples(x: np.ndarray) -> int:
    """
    Get the numbers of samples in an array

    Parameters
    ----------
    x : np.ndarray
        The array for which to calculate the number of samples

    Returns
    -------
    int
        The number of samples
    """
    return x.shape[0]


def validate_non_negative(x: np.ndarray):
    """
    Validate that an array has no negative values

    Parameters
    ----------
    x : np.ndarray
        The array to check

    Raises
    ------
    ValueError
        If the array has negative values

    Examples
    --------
    >>> import numpy as np
    >>> from regressioninc.validation import validate_non_negative
    >>> test = np.array([1,2,3,-4])
    >>> validate_non_negative(test)
    Traceback (most recent call last):
    ...
    ValueError: Negative values are not allowed

    >>> test[3] = 4
    >>> validate_non_negative(test)
    """
    if np.min(x) < 0:
        raise ValueError("Negative values are not allowed")


def validate_array(x: np.ndarray):
    """Perform various checks on array"""
    pass


def validate_lengths(*arrays: np.ndarray) -> None:
    """
    Check whether arrays have consistent length

    Parameters
    ----------
    *arrays : np.ndarray
        The arguments which should all be numpy arrays whose lengths will be
        checked for consistency

    Raises
    ------
    ValueError
        If the arrays have inconsistent lengths

    Examples
    --------

    >>> import numpy as np
    >>> from regressioninc.validation import validate_lengths
    >>> test1 = np.array([1,2,3,4,5])
    >>> test2 = np.array([6,7,8,9,10])
    >>> test3 = np.array([11,12,13])
    >>> validate_lengths(test1, test2)
    >>> validate_lengths(test1, test3)
    Traceback (most recent call last):
    ...
    ValueError: Row counts [5, 3] are not consistent
    """
    row_counts = [num_samples(x) for x in arrays]
    if len(set(row_counts)) > 1:
        raise ValueError(f"Row counts {row_counts} are not consistent")


def validate_X_y(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run a series of checks for X and y"""
    validate_lengths(X, y)
    return X, y


def validate_weights(weights, X, non_negative=True):
    """Validate weights"""
    n_samples = num_samples(X)

    if weights.ndim != 1:
        raise ValueError("Sample weights must be 1D array or scalar")
    if weights.shape != (n_samples,):
        raise ValueError(f"{weights.shape=} is inconsistent with {n_samples=}")
    if non_negative:
        validate_non_negative(weights)
    return weights
