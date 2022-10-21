"""
Functions for generating and visualising complex-valued testing data
"""
from typing import Optional
from pydantic import BaseModel
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from regressioninc.base import Regressor


class ComplexGrid(BaseModel):
    """A class to generate a grid for complex data"""

    r1: float
    """Starting real pt"""
    r2: float
    """End real pt"""
    nr: int
    """Number of real pts"""
    i1: float
    """Start imaginary pt"""
    i2: float
    """End imaginary pt"""
    ni: int
    """Number of imaginary pts"""

    @property
    def n_pts(self) -> int:
        """
        Get the number of points grid

        Returns
        -------
        int
            Number of points in grid

        Examples
        --------
        >>> from regressioninc.testing.complex import ComplexGrid
        >>> grid = ComplexGrid(r1=0, r2=5, nr=6, i1=0, i2=5, ni=6)
        >>> grid.n_pts
        36
        """
        return self.nr * self.ni

    def grid(self) -> np.ndarray:
        """
        Get the grid points as an 2-D array

        Returns
        -------
        np.ndarray
            The grid points as a 2-D array

        Examples
        --------
        >>> from regressioninc.testing.complex import ComplexGrid
        >>> grid = ComplexGrid(r1=-1, r2=1, nr=3, i1=-1, i2=1, ni=3)
        >>> grid.grid()
        [[-1.-1.j -1.+0.j -1.+1.j]
         [ 0.-1.j  0.+0.j  0.+1.j]
         [ 1.-1.j  1.+0.j  1.+1.j]]
        """
        r_pts = np.linspace(self.r1, self.r2, num=self.nr)
        i_pts = np.linspace(self.i1, self.i2, num=self.ni)
        reals, imags = np.meshgrid(r_pts, i_pts, indexing="ij")
        return reals + 1j * imags

    def flat_grid(self) -> np.ndarray:
        """
        Get the grid as a flat array

        Returns
        -------
        np.ndarray
            The grid of points as a 1-D flattened array

        Examples
        --------
        >>> from regressioninc.testing.complex import ComplexGrid
        >>> grid = ComplexGrid(r1=-1, r2=1, nr=3, i1=-1, i2=1, ni=3)
        >>> grid.grid()
        [-1.-1.j -1.+0.j -1.+1.j  0.-1.j  0.+0.j  0.+1.j  1.-1.j  1.+0.j  1.+1.j]
        """
        return self.grid().flatten()


def generate_linear_grid(
    coef: np.ndarray, grids: list[ComplexGrid], intercept: complex = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate complex regression data from coefficients and grids of regressors

    Parameters
    ----------
    coef : np.ndarray
        The coefficients for the regressors
    grids : list[ComplexGrid]
        The grid of points for each regressor
    intercept : complex, optional
        The intercept, by default 0

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X, y for the regression problem

    Raises
    ------
    ValueError
        If the number of grids provided does not equal the number of
        coefficients
    ValueError
        If the grids have different numbers of points
    """
    n_coef = coef.size

    if (n_grids := len(grids)) != n_coef:
        raise ValueError(f"{n_grids=} != {n_coef=}")
    pts = [g.n_pts for g in grids]
    if len(set(pts)) != 1:
        raise ValueError(f"The regressor grids have different numbers of points {pts}")
    n_pts = pts[0]

    X = np.empty(shape=(n_pts, n_coef), dtype=complex)
    for icoef in range(n_coef):
        X[:, icoef] = np.squeeze(grids[icoef].flat_grid())
    y = np.matmul(X, coef) + intercept
    return X, y


def generate_linear_random(
    coef: np.ndarray, intercept: complex = 0, n_samples: Optional[int] = 0
):
    """Produce complex data for testing without any noise"""
    n_features = coef.size
    if n_samples is None:
        n_samples = coef.size * 2
    if n_samples < n_features:
        raise ValueError(f"{n_samples=} must be >= {n_features=}")
    shape = (n_samples, n_features)
    # generate the data
    X = np.random.uniform(-20, 20, size=shape)
    X = X.astype(complex) + 1.0j * np.random.uniform(-20, 20, size=shape)
    y = np.matmul(X, coef) + intercept
    return X, y


def add_gaussian_noise(
    data: np.ndarray,
    loc: Optional[tuple[float]] = None,
    scale: Optional[tuple[float]] = None,
) -> np.ndarray:
    """Add Gaussian noise to data"""
    n_samples = data.shape[0]
    scale = 0.5 * np.array([[scale[0], 0], [0, scale[1]]])
    z = np.random.multivariate_normal(loc, scale, size=n_samples).view(complex)
    return data + np.squeeze(z)


def plot_observations(y: np.ndarray, size: int = 10, alpha: float = 1.0) -> None:
    """Plot observation data"""
    plt.scatter(
        y.real,
        y.imag,
        s=size,
        color="red",
        edgecolor="firebrick",
        marker="d",
        alpha=alpha,
        label="Observations",
    )
    plt.title("Observations")


def plot_observations_original(
    y_orig: np.ndarray, size: int = 10, alpha: float = 1.0
) -> None:
    """Plot original observations, meant for data without noise"""
    plt.scatter(
        y_orig.real,
        y_orig.imag,
        s=size,
        color="teal",
        edgecolor="darkslategrey",
        marker="*",
        alpha=alpha,
        label="Observations original",
    )
    plt.title("Observations original")
    plt.legend()


def plot_regressor(
    reg: np.ndarray, color: np.ndarray, size: int = 10, alpha: float = 1.0
) -> None:
    """Plot regression data"""
    plt.scatter(
        reg.real,
        reg.imag,
        s=size,
        color=color,
        edgecolor="k",
        alpha=alpha,
    )


def plot_estimate(
    est: np.ndarray,
    color: np.ndarray,
    size: int = 10,
    alpha: float = 1.0,
    label: str = "estimate",
) -> None:
    """Plot model estimates"""
    plt.scatter(est.real, est.imag, s=size, color=color, alpha=alpha, label=label)


def plot_complex(
    X,
    y,
    models: dict[str, Regressor],
    y_orig: Optional[np.ndarray] = None,
    size_obs: int = 10,
    size_reg: int = 10,
    size_est: int = 10,
):
    """Plot the complex data"""
    n_observations = 1 if y_orig is None else 2
    n_regressors = X.shape[1]
    n_models = len(models)
    n_rows = 2 if n_models == 0 else 3
    n_cols = max([n_observations, n_regressors, n_models])

    fig, axs = plt.subplots(n_rows, n_cols)
    # plot the observations
    plt.sca(axs[0, 0])
    plot_observations(y, size=size_obs)
    if y_orig is not None:
        plt.sca(axs[0, 1])
        plot_observations(y, size=size_obs, alpha=0.1)
        plot_observations_original(y_orig, size=size_obs)

    # plot the regressors
    colors = matplotlib.cm.get_cmap("Pastel1", n_regressors).colors
    for ireg, color in enumerate(colors):
        plt.sca(axs[1, ireg])
        plot_regressor(X[:, ireg], color=color, size=size_reg)
        plt.title(f"Regressor {ireg + 1} of {n_regressors}")

    # plot the model predictions
    model_names = list(models.keys())
    colors = matplotlib.cm.get_cmap("Set2", n_models).colors
    for imodel in range(n_models):
        plt.sca(axs[2, imodel])
        name = model_names[imodel]
        color = colors[imodel]
        model = models[name]
        y_est = model.predict(X)
        plot_estimate(y_est, color=color, size=size_est, label=name)
        plot_observations(y, size=size_obs, alpha=0.1)
        if y_orig is not None:
            plot_observations_original(y_orig, size=size_obs, alpha=0.4)
        plt.title(name)

    plt.tight_layout()
    return fig
