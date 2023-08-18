"""
Test functions in the base class
"""
import numpy as np
import numpy.linalg as linalg


np.random.seed(42)


def test_sum_r2():
    """Test calculating the sum of square residuals"""
    from regressioninc.testing.real import generate_linear, add_gaussian_noise
    from regressioninc.math import sum_r2
    from regressioninc.linear.models import add_intercept

    coef = np.array([3, 4])
    X, y = generate_linear(coef, intercept=10, n_samples=20)
    y = add_gaussian_noise(y)
    X = add_intercept(X)
    np_coef, np_residuals, _np_rank, _np_singular = linalg.lstsq(X, y, rcond=None)
    residuals = sum_r2(X, y, np_coef)
    np.testing.assert_almost_equal(np_residuals, residuals, 12)
