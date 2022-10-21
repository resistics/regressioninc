"""
Test linear regressors
"""
from typing import List
import pytest
import numpy as np
import statsmodels.api as sm


np.random.seed(42)


@pytest.mark.parametrize(
    "coef, n_samples, intercept, expected",
    [
        ([3.0, -7.0], 10, 0, [3.0, -7.0]),
        ([3.0, -7.0], 20, 10, [3.0, -7.0, 10]),
        ([17.2, 23.6, -50.2], 25, 0, [17.2, 23.6, -50.2]),
        ([17.2, 23.6, -50.2], 32, -80.5, [17.2, 23.6, -50.2, -80.5]),
    ],
)
def test_LeastSquares_real(
    coef: List[float], n_samples: int, intercept: float, expected: List[float]
):
    """Test ordinary least squares with real data"""
    from regressioninc.testing.real import generate_linear
    from regressioninc.linear import add_intercept, LeastSquares

    X, y = generate_linear(np.array(coef), n_samples, intercept=intercept)
    if intercept != 0:
        X = add_intercept(X)
    model = LeastSquares()
    model.fit(X, y)
    predictions = model.predict(X)
    np.testing.assert_almost_equal(expected, model.coef, decimal=8)
    np.testing.assert_almost_equal(y, predictions, decimal=8)


@pytest.mark.parametrize(
    "coef, n_samples, intercept, expected",
    [
        (
            [3.0 + 6j, -7.0 + 3j],
            5,
            0,
            [3.0 + 6j, -7.0 + 3j],
        ),
        (
            [3.0 + 6j, -7.0 + 3j],
            10,
            -12.5 - 13.7j,
            [3.0 + 6j, -7.0 + 3j, -12.5 - 13.7j],
        ),
        (
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
            27,
            0,
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
        ),
        (
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
            14,
            9.9 - 13.8j,
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j, 9.9 - 13.8j],
        ),
    ],
)
def test_LeastSquares_complex(
    coef: List[complex], n_samples: int, intercept: complex, expected: List[complex]
):
    """Test ordinary least squares with complex data"""
    from regressioninc.testing.complex import generate_linear_random
    from regressioninc.linear import add_intercept, LeastSquares

    X, y = generate_linear_random(np.array(coef), n_samples, intercept=intercept)
    if intercept != 0:
        X = add_intercept(X)
    model = LeastSquares()
    model.fit(X, y)
    predictions = model.predict(X)
    np.testing.assert_almost_equal(expected, model.coef, decimal=8)
    np.testing.assert_almost_equal(y, predictions, decimal=8)


@pytest.mark.parametrize(
    "coef, n_samples, intercept",
    [
        ([3.0, -7.0], 10, 0),
        ([17.2, 23.6, -50.2], 15, 0),
        ([17.2, 23.6, -50.2], 101, 10),
    ],
)
def test_WeightedLeastSquares_real(coef: List[float], n_samples: int, intercept: float):
    """Test weighted least squares using scikit-learn as a comparison"""
    from regressioninc.testing.real import generate_linear
    from regressioninc.linear import add_intercept, WeightedLeastSquares
    from sklearn.linear_model import LinearRegression

    X, y = generate_linear(np.array(coef), n_samples, intercept=intercept)
    # regressioninc
    Xrinc = X if intercept == 0 else add_intercept(X)
    weights = np.random.uniform(0, 1, size=y.shape)
    model = WeightedLeastSquares()
    result = model.fit(Xrinc, y, np.array(weights))
    # solve with scikit learn
    fit_intercept = intercept != 0
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y, sample_weight=weights)
    compare = model.coef_
    if fit_intercept:
        compare = compare.tolist()
        compare.append(model.intercept_)
    # check they match
    np.testing.assert_almost_equal(compare, result)


@pytest.mark.parametrize(
    "coef, n_samples, intercept, scale",
    [
        ([3.0, -7.0], 12, 0, 3),
        ([3.0, -7.0], 6, 0, 10),
        ([17.2, 23.6, -50.2], 21, 0, 3),
        ([17.2, 23.6, -50.2], 54, 0, 7),
        ([17.2, 23.6, -50.2], 42, 10, 4),
        ([17.2, 23.6, -50.2], 18, 10, 8),
    ],
)
def test_WeightedLeastSquares_real_with_noise(
    coef: List[float], n_samples: int, intercept: float, scale: float
):
    """Test weighted least squares using scikit-learn as a comparison"""
    from regressioninc.testing.real import generate_linear, add_gaussian_noise
    from regressioninc.linear import add_intercept, WeightedLeastSquares
    from sklearn.linear_model import LinearRegression

    X, y = generate_linear(np.array(coef), n_samples, intercept=intercept)
    y = add_gaussian_noise(y, scale=scale)
    # regressioninc
    Xrinc = X if intercept == 0 else add_intercept(X)
    weights = np.random.uniform(0, 1, size=y.shape)
    model = WeightedLeastSquares()
    result = model.fit(Xrinc, y, np.array(weights))
    # solve with scikit learn
    fit_intercept = intercept != 0
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y, sample_weight=weights)
    compare = model.coef_
    if fit_intercept:
        compare = compare.tolist()
        compare.append(model.intercept_)
    # check they match
    np.testing.assert_almost_equal(compare, result)


@pytest.mark.parametrize(
    "coef, n_samples, intercept, expected",
    [
        (
            [3.0 + 6j, -7.0 + 3j],
            87,
            0,
            [3.0 + 6j, -7.0 + 3j],
        ),
        (
            [3.0 + 6j, -7.0 + 3j],
            45,
            -12.5 - 13.7j,
            [3.0 + 6j, -7.0 + 3j, -12.5 - 13.7j],
        ),
        (
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
            32,
            0,
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
        ),
        (
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
            201,
            9.9 - 13.8j,
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j, 9.9 - 13.8j],
        ),
    ],
)
def test_WeightedLeastSquares_complex(
    coef: List[complex], n_samples: int, intercept: complex, expected: List[complex]
):
    """Test weighted least squares for complex data without any noise"""
    from regressioninc.testing.complex import generate_linear_random
    from regressioninc.linear import add_intercept, WeightedLeastSquares

    np.random.seed(42)

    X, y = generate_linear_random(np.array(coef), n_samples, intercept=intercept)
    if intercept != 0:
        X = add_intercept(X)
    weights = np.random.uniform(0, 1, size=y.shape)
    model = WeightedLeastSquares()
    model.fit(X, y, weights=weights)
    predictions = model.predict(X)
    np.testing.assert_almost_equal(expected, model.coef, decimal=8)
    np.testing.assert_almost_equal(y, predictions, decimal=8)


def test_M_estimate_real():
    """Test m estimates on real data vs. statsmodels"""
    from regressioninc.testing.real import generate_linear, add_gaussian_noise
    from regressioninc.testing.real import add_outliers_to_observations
    from regressioninc.linear import add_intercept, M_estimate

    coef = np.array([5, 7])
    intercept = 10
    X, y = generate_linear(coef, intercept=intercept, n_samples=30)
    y = add_gaussian_noise(y)
    y = add_outliers_to_observations(y, outlier_percent=20)
    X_wint = add_intercept(X)
    m_coef = M_estimate().fit(X_wint, y)
    print(m_coef)
    rlm_model = sm.RLM(y, X_wint, M=sm.robust.norms.TukeyBiweight())
    rlm_results = rlm_model.fit(conv="sresid")
    print(rlm_results.summary())
    print(rlm_results.params)
    assert False
    np.testing.assert_almost_equal(m_coef, rlm_results.params)
