r"""
Outliers in C
^^^^^^^^^^^^^

Outliers are another type of noise in measurements. There are many causes of
outliers such as malfunctioning instrumentation or unknown regressors. The
existence of outliers can have a significant effect on estimates of
coefficients.
"""
import numpy as np
import matplotlib.pyplot as plt
from regressioninc.linear import add_intercept, LeastSquares
from regressioninc.testing.complex import ComplexGrid, generate_linear_grid
from regressioninc.testing.complex import add_outliers_to_observations, plot_complex

np.random.seed(42)

# %%
# Let's setup another linear regression problem with complex values
coef = np.array([0.5 + 2j, -3 - 1j])
grid_r1 = ComplexGrid(r1=0, r2=10, nr=11, i1=-5, i2=5, ni=11)
grid_r2 = ComplexGrid(r1=-25, r2=-5, nr=11, i1=-5, i2=5, ni=11)
X, y = generate_linear_grid(coef, [grid_r1, grid_r2], intercept=20 + 20j)

fig = plot_complex(X, y, {})
fig.set_size_inches(7, 6)
plt.tight_layout()
plt.show()

# %%
# Estimating the coefficients using least squares gives the expected values.
X = add_intercept(X)
model = LeastSquares()
model.fit(X, y)
for idx, coef in enumerate(model.coef):
    print(f"Coefficient {idx}: {coef:.6f}")

fig = plot_complex(X, y, {"least squares": model})
fig.set_size_inches(7, 9)
plt.tight_layout()
plt.show()

# %%
# Add some noise to the observations and let's see what they look like now
y_noise = add_outliers_to_observations(y, outlier_percent=20, mult_min=5, mult_max=7)

fig = plot_complex(X, y_noise, {}, y_orig=y)
fig.set_size_inches(7, 6)
plt.tight_layout()
plt.show()

# %%
# Now let's try and estimate the coefficients again but with the noisy
# observations. In this case, the coefficients estimates are slightly off the
# actual value due to the existence of the noise.
model = LeastSquares()
model.fit(X, y_noise)
for idx, coef in enumerate(model.coef):
    print(f"Coefficient {idx}: {coef:.6f}")

fig = plot_complex(X, y_noise, {"least squares": model}, y_orig=y)
fig.set_size_inches(7, 9)
plt.tight_layout()
plt.show()
