r"""
Noise in C
^^^^^^^^^^

Unlike the prior examples, most real world data have noise. Noise can occur in
both the observations and the depending on how data is acquired.

Typical types of noise include:

- Random measurement error, which can occur on both the observations and the
  regressors (for instance if both come from measurements)
- Outliers that can skew results

Unexpected observations due to unknown regressors. These observations are
sometimes treated as noise when they are really a limitation of the modelling.

In this example, we'll explore adding noise to the data and seeing the impact
this has on the estimated coefficients.
"""
import numpy as np
import matplotlib.pyplot as plt
from regressioninc.linear import add_intercept, LeastSquares
from regressioninc.testing.complex import ComplexGrid
from regressioninc.testing.complex import generate_linear_random, plot_complex

np.random.seed(42)

# %%
# Let's begin where the previous example ended and create our data which has two
# regressors and an intercept.
coef_grid = np.array([0.5 + 2j])
grid = ComplexGrid(r1=0, r2=10, nr=11, i1=-5, i2=5, ni=11)
X_grid = grid.flat_grid()
coef_random = np.array([2.7 - 1.8j])
X_random, _ = generate_linear_random(coef_random, grid.n_pts)
# put our regressors together
coef = np.concatenate((coef_grid, coef_random))
X = np.concatenate((X_grid, X_random), axis=1)
intercept = 20 + 20j
y = np.matmul(X, coef) + intercept

fig = plot_complex(X, y, {})
plt.show()

# %%
# Our data has coefficients 0.5 + 2j, 2.7 -1.8j and an intercept of 20 + 20j.
# These coefficients can be estimated accurately from the regressors and
# observations using least squares regression.
X = add_intercept(X)
model = LeastSquares()
model.fit(X, y)
for idx, coef in enumerate(model.coef):
    print(f"Coefficient {idx}: {coef:.6f}")
