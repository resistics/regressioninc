import numpy as np
import matplotlib.pyplot as plt
from regressioninc.linear import add_intercept, LeastSquares, M_estimate
from regressioninc.testing.complex import ComplexGrid, generate_linear_grid
from regressioninc.testing.complex import add_gaussian_noise, plot_complex
from regressioninc.testing.complex import add_outliers_to_observations

# coef = np.array([3 + 0j])
# grid = ComplexGrid(r1=0, r2=5, nr=6, i1=0, i2=5, ni=6)
# X, y_orig = generate_linear_grid(coef, [grid], intercept=100)
# y = add_gaussian_noise(y_orig, (0, 0), (5, 5))
# X = add_intercept(X)
# ols = LeastSquares()
# ols.fit(X, y)
# plot_complex(X, y, {"ols": ols}, y_orig=y_orig, size_obs=50, size_reg=30, size_est=30)
# plt.show()


# coef = np.array([3 + 0j, 0 + 6j])
# grid1 = ComplexGrid(r1=0, r2=5, nr=6, i1=0, i2=5, ni=6)
# grid2 = ComplexGrid(r1=-2, r2=3, nr=6, i1=-8, i2=-3, ni=6)
# X, y_orig = generate_linear_grid(coef, [grid1, grid2], intercept=100)
# y = add_gaussian_noise(y_orig, (0, 0), (5, 5))
# X = add_intercept(X)
# ols = LeastSquares()
# ols.fit(X, y)
# plot_complex(X, y, {"ols": ols}, y_orig=y_orig, size_obs=50, size_reg=30, size_est=30)
# plt.show()

# np.random.seed(42)
np.random.seed(43)

coef = np.array([3 + 0j, 0 + 6j])
grid1 = ComplexGrid(r1=0, r2=5, nr=6, i1=0, i2=5, ni=6)
grid2 = ComplexGrid(r1=-2, r2=3, nr=6, i1=-8, i2=-3, ni=6)
X, y_orig = generate_linear_grid(coef, [grid1, grid2], intercept=100)
y = add_gaussian_noise(y_orig, (0, 0), (5, 5))
y = add_outliers_to_observations(y, outlier_percent=20, mult_min=5, mult_max=10)
X = add_intercept(X)
ols = LeastSquares()
ols.fit(X, y)
mest = M_estimate()
mest.fit(X, y)
# glr = ComplexAsGLR()
# glr.fit(X, y, LeastSquares())
plot_complex(
    X,
    y,
    {"ols": ols, "mest": mest},
    # {"ols": ols, "mest": mest, "glr": glr},
    y_orig=y_orig,
    size_obs=50,
    size_reg=30,
    size_est=30,
)
plt.show()

print(mest.coef)
# print(glr.coef)
