"""
This module implements estimators for linear complex problems

These are built on scipy as scipy.linalg.lstsq does allow and support
complex-valued inputs. However, it does not output residuals for complex-valued
variables, therefore these are calculated out explicitly.
"""
from loguru import logger
from typing import Optional, Union
import numpy as np
import scipy.linalg as linalg
import statsmodels.api as sm

from regressioninc.base import Estimate, Estimator
from regressioninc.validation import validate_X_y, validate_weights
from regressioninc.linear.robust import mad
from regressioninc.errors import EstimatorError


RobustNorm = sm.robust.norms.RobustNorm


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add an intercept to the regressors

    Parameters
    ----------
    X : np.ndarray
        The regressors

    Returns
    -------
    np.ndarray
        The regressors with an extra column of ones to represent to allow
        solving for the intercept term

    Examples
    --------
    >>> import numpy as np
    >>> from regressioninc.linear.models import add_intercept
    >>> X = np.array([[2,3], [4,5]])
    >>> X
    array([[2, 3],
           [4, 5]])

    Now add the intercept

    >>> X = add_intercept(X)
    >>> X
    array([[2, 3, 1],
           [4, 5, 1]])
    """
    n_samples = X.shape[0]
    return np.hstack((X, np.ones(shape=(n_samples, 1), dtype=X.dtype)))


def complex_to_real(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert complex-valued linear problem to a real-valued linear regression

    Parameters
    ----------
    X : np.ndarray
        The regressors
    y : np.ndarray
        The observations

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X, y converted to real-valued generalised linear regression

    Examples
    --------
    >>> from regressioninc.linear.models import add_intercept, complex_to_real
    >>> X = np.array([3 + 4j, 2 + 8j, 6 + 4j]).reshape(3,1)
    >>> X = add_intercept(X)
    >>> y = np.array([-2 - 4j, -1 - 3j, 5 + 6j])
    >>> X_real, y_real = complex_to_real(X, y)
    >>> X_real
    array([[ 3., -4.,  1., -0.],
           [ 4.,  3.,  0.,  1.],
           [ 2., -8.,  1., -0.],
           [ 8.,  2.,  0.,  1.],
           [ 6., -4.,  1., -0.],
           [ 4.,  6.,  0.,  1.]])
    >>> y_real
    array([-2., -4., -1., -3.,  5.,  6.])
    """
    n_samples, n_regressors = X.shape
    # the obesrvations
    y_real = np.empty(shape=(n_samples * 2), dtype=float)
    y_real[0::2] = y.real
    y_real[1::2] = y.imag
    # the regressors
    X_real = np.empty(shape=(n_samples * 2, n_regressors * 2), dtype=float)
    X_real[0::2, 0::2] = X.real
    X_real[0::2, 1::2] = -X.imag
    X_real[1::2, 0::2] = X.imag
    X_real[1::2, 1::2] = X.real
    return X_real, y_real


def real_params_to_complex(params: np.ndarray) -> np.ndarray:
    """
    Convert real-valued parameters to complex-valued ones for problems that
    were transformed from complex-valued to real-valued.

    Parameters
    ----------
    params : np.ndarray
        Real-valued parameters array

    Returns
    -------
    np.ndarray
        The complex-valued parameters

    Examples
    --------
    Let's generate a complex-valued linear problem, pose it as a real-valued
    linear problem and then convert the returned parameters back to complex.

    Generate the linear problem and add an intercept column to the regressors

    >>> import numpy as np
    >>> np.set_printoptions(precision=3, suppress=True)
    >>> from regressioninc.testing.complex import ComplexGrid, generate_linear_grid
    >>> from regressioninc.linear.models import add_intercept, OLS
    >>> from regressioninc.linear.models import complex_to_real, real_params_to_complex
    >>> params = np.array([3 + 2j])
    >>> grid = ComplexGrid(r1=-1, r2=1, nr=3, i1=4, i2=6, ni=3)
    >>> X, y = generate_linear_grid(params, [grid], intercept=10)
    >>> X = add_intercept(X)

    Convert the complex-valued problem to a real-valued problem

    >>> X_real, y_real = complex_to_real(X, y)

    Solve the real-valued linear problem

    >>> model = OLS()
    >>> model.fit(X_real, y_real)
    OLS...

    Look at the real-valued coefficients

    >>> model.estimate.params
    array([ 3.,  2., 10.,  0.])

    Convert the coefficients back to the complex domain

    >>> real_params_to_complex(model.estimate.params)
    array([ 3.+2.j, 10.+0.j])
    """
    return params[0::2] + 1j * params[1::2]


def apply_weights(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Transform X and y using the weights to perform a weighted least squares

    .. math::
        \sqrt{weights} y = \sqrt{weights} X coef ,

    is equivalent to,

    .. math::
        X^H weights y = X^H weights X coef ,

    where :math:`X^H` is the hermitian transpose of X.

    In this method, both the observations y and the predictors X are multipled
    by the square root of the weights and then returned.

    Parameters
    ----------
    X : np.ndarray
        The regressors
    y : np.ndarray
        The regressands
    weights : np.ndarray
        The weights

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The weighted regressors and the weights regrassands

    Examples
    --------
    >>> import numpy as np
    >>> X = [5, 3]
    >>> y = [2, 8]
    >>> weights = []
    """
    weights = validate_weights(weights, X, non_negative=True)
    sqrt = np.sqrt(weights)
    y = sqrt * y
    X = X * sqrt[..., np.newaxis]
    return X, y


class LinearEstimate(Estimate):
    """A linear estimate"""

    residues: Union[float, np.ndarray]
    rank: int
    singular_values: Union[None, np.ndarray]


class LinearEstimator(Estimator):
    """Base class for a linear estimator"""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regrassands given regressors using the estimated parameters

        Parameters
        ----------
        X : np.ndarray
            The regressors

        Returns
        -------
        np.ndarray
            The regressands

        Raises
        ------
        EstimatorError
            If no estimate exists
        """
        if self.estimate is None:
            raise EstimatorError("No estimated parameters have been calculated")
        return np.dot(X, self.estimate.params.T)


def lstsq(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    cond=None,
    overwrite_a=False,
    overwrite_b=False,
    check_finite=True,
    lapack_driver=None,
):
    """OLS from scipy wrapped here in case support for complex is dropped"""
    if weights is not None:
        X, y = apply_weights(X, y, weights)
    params, residues, rank, singular_values = linalg.lstsq(
        X, y, cond, overwrite_a, overwrite_b, check_finite, lapack_driver
    )
    return LinearEstimate(
        params=params, residues=residues, rank=rank, singular_values=singular_values
    )


class OLS(LinearEstimator):
    """Ordinary least square regression"""

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the linear problem using least squares regression

        Parameters
        ----------
        X : np.ndarray
            The predictors
        y : np.ndarray
            The observations

        Returns
        -------
        np.ndarray
            The coefficients
        """
        X, y = validate_X_y(X, y)
        self.estimate = lstsq(X, y)
        return self


class WLS(LinearEstimator):
    """Weighted least squares"""

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Apply weights to observations and predictors and find the coefficients
        of the linear model

        Parameters
        ----------
        X : np.ndarray
            The predictors
        y : np.ndarray
            The observations
        weights : np.ndarray
            The weights to apply to the samples

        Returns
        -------
        np.ndarray
            The coefficients for the model
        """
        X, y = validate_X_y(X, y)
        self.estimate = lstsq(X, y, weights=weights)
        return self


class MEstimate(LinearEstimate):
    scale: float
    """An estimate of scale to be used weighting the rows of the linear problem"""


class MEstimator(LinearEstimator):
    max_iter: int = 50
    """The maximum number of iterations"""
    early_stopping: float = 1e-8
    """Minimum change required in residues between iterations to continue"""
    warm_start: bool = False
    """Use existing solution to initialise a new call to fit"""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        M: Optional[RobustNorm] = None,
    ):
        if M is None:
            M = sm.robust.norms.TrimmedMean()

        if not self.warm_start or self.estimate is None:
            # do an initial least squares solution
            ols_estimate = lstsq(X, y)
            self.estimate = MEstimate(**ols_estimate.model_dump(), scale=0)

        resids = y - np.dot(X, self.estimate.params.T)
        self.estimate.scale = mad(resids, center=0)
        prev_resids = np.absolute(resids).real
        iter = 0
        while iter < self.max_iter:
            # calculate scale and next iteration of weights
            if np.sum(prev_resids) < self.early_stopping:
                logger.debug(f"{prev_resids=} residuals small, quitting")
                break
            if self.estimate.scale == 0.0:
                logger.debug(
                    f"{self.estimate.scale=}, suggests near perfect fit last iteration"
                )
                break

            # perform another solving iteration
            weights = M.weights(prev_resids / self.estimate.scale)
            new_estimate = lstsq(X, y, weights=weights)
            resids = y - np.dot(X, new_estimate.params.T)
            scale = mad(resids, center=0)
            self.estimate = MEstimate(**new_estimate.model_dump(), scale=scale)
            abs_resids = np.absolute(resids).real

            # check the change in residuals and quit if insignificant
            if linalg.norm(abs_resids - prev_resids) < self.early_stopping:
                logger.debug("Early stopping criteria met, breaking")
                break
            prev_resids = abs_resids

        return self
