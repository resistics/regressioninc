"""
Generate regression testing data
"""
from loguru import logger
from typing import Optional
import numpy as np


def linear_real(
    coef: np.ndarray, intercept: float = 0, n_samples: Optional[int] = None
):
    """Produce real data for testing without any noise"""
    n_features = coef.size
    if n_samples is None:
        n_samples = coef.size * 2
    if n_samples < n_features:
        raise ValueError(f"{n_samples=} must be >= {n_features=}")

    # generate the data
    shape = (n_samples, n_features)
    X = np.random.uniform(-20, 20, size=shape)
    y = np.matmul(X, coef) + intercept
    return X, y


def linear_real_with_noise(
    coef: np.ndarray,
    intercept: float = 0,
    scale: float = 3,
    n_samples: Optional[int] = None,
):
    """Produce real data with normally distributed noise on the observations"""
    X, y = linear_real(coef, intercept, n_samples)
    y = y + np.random.normal(loc=0, scale=scale, size=y.shape)
    return X, y


def linear_real_with_outliers(
    coef: np.ndarray,
    intercept: float = 0,
    outlier_percent: float = 5,
    n_samples: Optional[int] = None,
):
    """Produce real data with large outliers in the observations"""
    X, y = linear_real(coef, intercept, n_samples)
    n_samples = y.size

    if outlier_percent == 0:
        return X, y

    # make outlier array
    n_outliers = int((outlier_percent / 100) * n_samples)
    max_y = np.max(y)
    outliers = np.random.uniform(max_y, max_y * 3, size=n_outliers)

    # add to observations
    logger.debug(f"Adding {n_outliers=} to observations")
    outlier_indices = np.random.randint(0, n_samples, size=n_outliers)
    y[outlier_indices] = y[outlier_indices] + outliers
    return X, y


def linear_real_with_leverage():
    pass


def linear_real_with_outliers_and_leverage():
    pass


def linear_complex(coef: np.ndarray, intercept: Optional[complex] = None):
    """Produce complex data for testing without any noise"""
    n_samples = coef.size * 2
    n_features = coef.size
    shape = (n_samples, n_features)
    # generate the data
    X = np.random.uniform(-20, 20, size=shape)
    X = X.astype(complex) + 1.0j * np.random.uniform(-20, 20, size=shape)
    y = np.matmul(X, coef) + intercept
    return X, y
