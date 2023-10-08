r"""
Robust regression
^^^^^^^^^^^^^^^^^

Robust regression
"""
import numpy as np
import matplotlib.pyplot as plt
from regressioninc.linear.models import add_intercept, OLS, MEstimator
from regressioninc.testing.complex import ComplexGrid, generate_linear_grid
from regressioninc.testing.complex import add_gaussian_noise, add_outliers, plot_complex

np.random.seed(42)

# %%
# Let's setup another linear regression problem with complex values
params = np.array([0.5 + 2j, -3 - 1j])
grid_r1 = ComplexGrid(r1=0, r2=10, nr=11, i1=-5, i2=5, ni=11)
grid_r2 = ComplexGrid(r1=-25, r2=-5, nr=11, i1=-5, i2=5, ni=11)
X, y = generate_linear_grid(params, [grid_r1, grid_r2], intercept=20 + 20j)

fig = plot_complex(X, y, {})
fig.set_size_inches(7, 6)
plt.tight_layout()
plt.show()


# %%
# Add high leverage points to our regressors
seeds = [22, 36]
for ireg in range(X.shape[1]):
    np.random.seed(seeds[ireg])
    X[:, ireg] = add_outliers(
        X[:, ireg],
        outlier_percent=20,
        mult_min=7,
        mult_max=10,
        random_signs_real=True,
        random_signs_imag=True,
    )
np.random.seed(42)

intercept = 20 + 20j
y = np.matmul(X, params) + intercept

fig = plot_complex(X, y, {})
fig.set_size_inches(7, 6)
plt.tight_layout()
plt.show()


# %%
# Solve
X = add_intercept(X)
model = OLS()
model.fit(X, y)
for idx, params in enumerate(model.estimate.params):
    print(f"parameter {idx}: {params:.6f}")

# %%
# Add some outliers
y_noise = add_gaussian_noise(y, loc=(0, 0), scale=(21, 21))
model_ls = OLS()
model_ls.fit(X, y_noise)
for idx, params in enumerate(model_ls.estimate.params):
    print(f"parameter {idx}: {params:.6f}")

fig = plot_complex(X, y_noise, {"least squares": model_ls}, y_orig=y)
fig.set_size_inches(7, 9)
plt.tight_layout()
plt.show()

# %%
# Add some outliers
y_noise = add_gaussian_noise(y, loc=(0, 0), scale=(21, 21))
model_mest = MEstimator(warm_start=True)
model_mest.fit(X, y_noise)
for idx, params in enumerate(model_mest.estimate.params):
    print(f"parameter {idx}: {params:.6f}")

fig = plot_complex(
    X, y_noise, {"least squares": model_ls, "M estimate": model_mest}, y_orig=y
)
fig.set_size_inches(7, 9)
plt.tight_layout()
plt.show()
