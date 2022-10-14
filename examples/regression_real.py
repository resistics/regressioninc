import numpy as np
import matplotlib.pyplot as plt
from regressioninc.testing.real import generate_linear, plot_2d
from regressioninc.testing.real import add_gaussian_noise, add_outliers_to_observations
from regressioninc.base import add_intercept
from regressioninc.ols import LeastSquares

np.random.seed(42)

# coef = np.array([5])
# intercept = 10

# X, y = generate_linear(coef, intercept=intercept, n_samples=30)
# y = add_gaussian_noise(y)
# y = add_outliers_to_observations(y, outlier_percent=20)
# fig = plot_1d(X, y, coefs={"actual": (coef, intercept)})
# plt.show()

coef = np.array([5, 7])
intercept = 10

X, y = generate_linear(coef, intercept=intercept, n_samples=30)
y = add_gaussian_noise(y)
y = add_outliers_to_observations(y, outlier_percent=20)
X_wint = add_intercept(X)
ols_coef = LeastSquares().fit(X_wint, y)
fig = plot_2d(
    X,
    y,
    coefs={
        "actual": (coef, intercept),
        "least_squares": (ols_coef[:-1], ols_coef[-1]),
    },
)
plt.show()
