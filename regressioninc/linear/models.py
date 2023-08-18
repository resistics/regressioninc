"""
This module implements estimators for linear complex problems

These are built on scipy as scipy.linalg.lstsq does allow and support
complex-valued inputs. However, it does not output residuals for complex-valued
variables, therefore these are calculated out explicitly.
"""
from loguru import logger
from typing import Optional
import numpy as np
import scipy.linalg as linalg
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
import statsmodels.api as sm

from regressioninc.base import ComplexRegressorMixin
from regressioninc.validation import check_X_y
from regressioninc.linear.robust import mad


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


def real_coef_to_complex(coef: np.ndarray) -> np.ndarray:
    """
    Convert real-valued coefficients to complex-valued one for problems that
    were transformed from complex-valued to real-valued.

    Parameters
    ----------
    coef : np.ndarray
        Real-valued coefficients array

    Returns
    -------
    np.ndarray
        The complex-valued coefficients

    Examples
    --------
    Let's generate a complex-valued linear problem, pose it as a real-valued
    linear problem and then convert the returned coefficients back to complex.

    Generate the linear problem and add an intercept column to the regressors

    >>> import numpy as np
    >>> np.set_printoptions(precision=3, suppress=True)
    >>> from regressioninc.testing.complex import ComplexGrid, generate_linear_grid
    >>> from regressioninc.linear.models import add_intercept, OLS
    >>> from regressioninc.linear.models import complex_to_real, real_coef_to_complex
    >>> coef = np.array([3 + 2j])
    >>> grid = ComplexGrid(r1=-1, r2=1, nr=3, i1=4, i2=6, ni=3)
    >>> X, y = generate_linear_grid(coef, [grid], intercept=10)
    >>> X = add_intercept(X)

    Convert the complex-valued problem to a real-valued problem

    >>> X_real, y_real = complex_to_real(X, y)

    Solve the real-valued linear problem

    >>> model = OLS()
    >>> model.fit(X_real, y_real)
    OLS...

    Look at the real-valued coefficients

    >>> model.coef_
    array([ 3.,  2., 10.,  0.])

    Convert the coefficients back to the complex domain

    >>> real_coef_to_complex(model.coef_)
    array([ 3.+2.j, 10.+0.j])
    """
    return coef[0::2] + 1j * coef[1::2]


def _apply_sample_weights(X, y, sample_weight):
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
    """
    sample_weight = _check_sample_weight(
        sample_weight, X, dtype=X.dtype, only_non_negative=True
    )
    sqrt = np.sqrt(sample_weight)
    y = sqrt * y
    X = X * sqrt[..., np.newaxis]
    return X, y


def _lstsq(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    cond=None,
    overwrite_a=False,
    overwrite_b=False,
    check_finite=True,
    lapack_driver=None,
):
    """
    A function to implement ordinary least squares

    This function wraps the scipy.linalg.lstsq least squares implementation.
    Rather than use that directly, have implemented this in case support for
    complex-values is later removed from scipy and another package needs to be
    used.
    """
    if sample_weight is not None:
        X, y = _apply_sample_weights(X, y, sample_weight)
    return linalg.lstsq(
        X, y, cond, overwrite_a, overwrite_b, check_finite, lapack_driver
    )


class ComplexLinearModel(LinearModel):
    """A LinearModel base class the supports complex data"""

    def _decision_function(self, X):
        check_is_fitted(self)

        # X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
        return safe_sparse_dot(X, self.coef_.T, dense_output=True)

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)
    

class OLS(ComplexRegressorMixin, ComplexLinearModel):
    """Standard linear regression"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        X, y = check_X_y(X, y)
        self.coef_, self.residuals_, self.rank_, self.singular_ = _lstsq(X, y)
        return self


class WLS(ComplexRegressorMixin, ComplexLinearModel):
    """Weighted least squares"""

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
    ) -> np.ndarray:
        """
        Apply weights to observations and predictors and find the coefficients
        of the linear model

        Parameters
        ----------
        X : np.ndarray
            The predictors
        y : np.ndarray
            The observations
        sample_weight : np.ndarray
            The weights to apply to the samples

        Returns
        -------
        np.ndarray
            The coefficients for the model
        """
        X, y = check_X_y(X, y)
        self.coef_, self.residuals_, self.rank_, self.singular_ = _lstsq(
            X, y, sample_weight=sample_weight
        )
        return self


class M_estimate(ComplexRegressorMixin, ComplexLinearModel):
    def __init__(
        self,
        max_iter: int = 50,
        early_stopping: float = 1e-8,
        warm_start: bool = False,
    ):
        """Something here"""
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.warm_start = warm_start

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        M: Optional[RobustNorm] = None,
    ):
        if M is None:
            M = sm.robust.norms.TrimmedMean()

        if not self.warm_start or not hasattr(self, "coef_"):
            # do an initial least squares solution
            logger.info("Here")
            self.coef_, self.residuals_, self.rank_, self.singular_ = _lstsq(X, y)

        resids = y - np.dot(X, self.coef_.T)
        self.scale_ = mad(resids, center=0)
        prev_resids = np.absolute(resids).real
        iter = 0
        while iter < self.max_iter:
            # calculate scale and next iteration of weights
            if np.sum(prev_resids) < self.early_stopping:
                logger.debug(f"{prev_resids=} residuals small, quitting")
                break
            if self.scale_ == 0.0:
                logger.debug(f"{self.scale_=}, suggests near perfect fit last iteration")
                break

            # perform another solving iteration
            sample_weight = M.weights(prev_resids / self.scale_)
            self.coef_, self.residuals_, self.rank_, self.singular_ = _lstsq(
                X, y, sample_weight=sample_weight
            )
            resids = y - np.dot(X, self.coef_.T)
            self.scale_ = mad(resids, center=0)
            abs_resids = np.absolute(resids).real

            # check the change in residuals and quit if insignificant
            if linalg.norm(abs_resids - prev_resids) < self.early_stopping:
                logger.debug("Early stopping criteria met, breaking")
                break
            prev_resids = abs_resids

        return self
