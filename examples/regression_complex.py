import numpy as np
import matplotlib.pyplot as plt
from regressioninc.base import add_intercept
from regressioninc.testing.complex import ComplexGrid, generate_linear_grid
from regressioninc.testing.complex import add_gaussian_noise, plot_complex
from regressioninc.ols import LeastSquares

coef = np.array([3 + 0j])
grid = ComplexGrid(r1=0, r2=5, nr=6, i1=0, i2=5, ni=6)
X, y_orig = generate_linear_grid(coef, [grid], intercept=100)
y = add_gaussian_noise(y_orig, (0, 0), (5, 5))
X = add_intercept(X)
ols = LeastSquares()
ols.fit(X, y)
plot_complex(X, y, {"ols": ols}, y_orig=y_orig, size_obs=50, size_reg=30, size_est=30)
plt.show()


coef = np.array([3 + 0j, 0 + 6j])
grid1 = ComplexGrid(r1=0, r2=5, nr=6, i1=0, i2=5, ni=6)
grid2 = ComplexGrid(r1=-2, r2=3, nr=6, i1=-8, i2=-3, ni=6)
X, y_orig = generate_linear_grid(coef, [grid1, grid2], intercept=100)
y = add_gaussian_noise(y_orig, (0, 0), (5, 5))
X = add_intercept(X)
ols = LeastSquares()
ols.fit(X, y)
plot_complex(X, y, {"ols": ols}, y_orig=y_orig, size_obs=50, size_reg=30, size_est=30)
plt.show()
