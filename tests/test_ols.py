from typing import List
import pytest
import numpy as np

np.random.seed(42)


@pytest.mark.parametrize(
    "coef, intercept, expected",
    [
        ([3.0, -7.0], 0, [3.0, -7.0]),
        ([3.0, -7.0], 10, [3.0, -7.0, 10]),
        ([17.2, 23.6, -50.2], 0, [17.2, 23.6, -50.2]),
        ([17.2, 23.6, -50.2], -80.5, [17.2, 23.6, -50.2, -80.5]),
    ],
)
def test_LeastSquares_real(coef: List[float], intercept: float, expected: List[float]):
    """Test ordinary least squares with real data"""
    from regressioninc.testing.real import generate_linear
    from regressioninc.base import add_intercept
    from regressioninc.ols import LeastSquares

    X, y = generate_linear(np.array(coef), intercept=intercept)
    if intercept != 0:
        X = add_intercept(X)
    model = LeastSquares()
    result = model.fit(X, y)
    predictions = model.predict(X)
    np.testing.assert_almost_equal(expected, result, decimal=8)
    np.testing.assert_almost_equal(y, predictions, decimal=8)


@pytest.mark.parametrize(
    "coef, intercept, expected",
    [
        ([3.0 + 6j, -7.0 + 3j], 0, [3.0 + 6j, -7.0 + 3j]),
        ([3.0 + 6j, -7.0 + 3j], -12.5 - 13.7j, [3.0 + 6j, -7.0 + 3j, -12.5 - 13.7j]),
        (
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
            0,
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
        ),
        (
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
            9.9 - 13.8j,
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j, 9.9 - 13.8j],
        ),
    ],
)
def test_LeastSquares_complex(
    coef: List[complex], intercept: complex, expected: List[complex]
):
    """Test ordinary least squares with complex data"""
    from regressioninc.testing.complex import generate_linear
    from regressioninc.base import add_intercept
    from regressioninc.ols import LeastSquares

    X, y = generate_linear(np.array(coef), intercept=intercept)
    if intercept != 0:
        X = add_intercept(X)
    model = LeastSquares()
    result = model.fit(X, y)
    predictions = model.predict(X)
    np.testing.assert_almost_equal(expected, result, decimal=8)
    np.testing.assert_almost_equal(y, predictions, decimal=8)


@pytest.mark.parametrize(
    "coef, intercept",
    [
        ([3.0, -7.0], 0),
        ([17.2, 23.6, -50.2], 0),
        ([17.2, 23.6, -50.2], 10),
    ],
)
def test_WeightedLeastSquares_real(coef: List[float], intercept: float):
    """Test weighted least squares using scikit-learn as a comparison"""
    from regressioninc.testing.real import generate_linear
    from regressioninc.base import add_intercept
    from regressioninc.ols import WeightedLeastSquares
    from sklearn.linear_model import LinearRegression

    np.random.seed(42)

    X, y = generate_linear(np.array(coef), intercept=intercept)
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
    "coef, intercept, scale",
    [
        ([3.0, -7.0], 0, 3),
        ([3.0, -7.0], 0, 10),
        ([17.2, 23.6, -50.2], 0, 3),
        ([17.2, 23.6, -50.2], 0, 7),
        ([17.2, 23.6, -50.2], 10, 4),
        ([17.2, 23.6, -50.2], 10, 8),
    ],
)
def test_WeightedLeastSquares_real_with_noise(
    coef: List[float], intercept: float, scale: float
):
    """Test weighted least squares using scikit-learn as a comparison"""
    from regressioninc.testing.real import generate_linear, add_gaussian_noise
    from regressioninc.base import add_intercept
    from regressioninc.ols import WeightedLeastSquares
    from sklearn.linear_model import LinearRegression

    np.random.seed(42)

    X, y = generate_linear(np.array(coef), intercept=intercept)
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
    "coef, intercept, expected",
    [
        ([3.0 + 6j, -7.0 + 3j], 0, [3.0 + 6j, -7.0 + 3j]),
        ([3.0 + 6j, -7.0 + 3j], -12.5 - 13.7j, [3.0 + 6j, -7.0 + 3j, -12.5 - 13.7j]),
        (
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
            0,
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
        ),
        (
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j],
            9.9 - 13.8j,
            [17.2 - 31.7j, 23.6 + 5.4j, -50.2 + 16.3j, 9.9 - 13.8j],
        ),
    ],
)
def test_WeightedLeastSquares_complex(
    coef: List[complex], intercept: complex, expected: List[complex]
):
    """Test weighted least squares for complex data without any noise"""
    from regressioninc.testing.complex import generate_linear
    from regressioninc.base import add_intercept
    from regressioninc.ols import WeightedLeastSquares

    np.random.seed(42)

    X, y = generate_linear(np.array(coef), intercept=intercept)
    if intercept != 0:
        X = add_intercept(X)
    weights = np.random.uniform(0, 1, size=y.shape)
    model = WeightedLeastSquares()
    result = model.fit(X, y, weights=weights)
    predictions = model.predict(X)
    np.testing.assert_almost_equal(expected, result, decimal=8)
    np.testing.assert_almost_equal(y, predictions, decimal=8)
