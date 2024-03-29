r"""
Complex to real
^^^^^^^^^^^^^^^

Complex-valued linear problems can be reformulated as real-valued problems by
splitting out the real and imaginary parts of the equations.

.. math::

    a + ib = C_1 (x + iy) .

Remember that we are solving for :math:`C_1 = (c_{1r} + c_{1i})`, which is also
complex-valued, therefore,

.. math::

    a + ib = (c_{1r} + c_{1i}) (x + iy) .

This can be expanded out

.. math::

    a + ib &= (c_{1r} + ic_{1i}) (x + iy) \\
           &= c_{1r} x - c_{1i} y + i c_{1r} y + i c_{1i} x \\
           &= (c_{1r} x - c_{1i} y) + i (c_{1r} y + i c_{1i} x) ,

which gives,

.. math::

    a &= c_{1r} x - c_{1i} y \\
    b &= c_{1r} y + i c_{1i} x .

For the complex-valued problem, the aim is to solve for :math:`C_1`. Making this
real-valued means we are solving for :math:`c_{1r}` and :math:`c_{1i}`.

Moving from complex-valued to real-valued results in the following

- Doubling the number of observations as the real and imaginary parts of the
  observations are split up
- Doubling the number of regressors as we are now solving for the real and
  imaginary component of each regressor explicitly
"""
import numpy as np
import matplotlib.pyplot as plt
from regressioninc.linear import add_intercept, LeastSquares, M_estimate
from regressioninc.testing.complex import ComplexGrid, generate_linear_grid
from regressioninc.testing.complex import add_gaussian_noise
from regressioninc.testing.complex import add_outliers, plot_complex
from regressioninc.linear import complex_to_glr, glr_coef_to_complex

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
y = np.matmul(X, coef) + intercept

fig = plot_complex(X, y, {})
fig.set_size_inches(7, 6)
plt.tight_layout()
plt.show()


# %%
# Solve
X = add_intercept(X)
model = LeastSquares()
model.fit(X, y)
for idx, coef in enumerate(model.coef):
    print(f"Coefficient {idx}: {coef:.6f}")

# %%
# Add some outliers
y_noise = add_gaussian_noise(y, loc=(0, 0), scale=(21, 21))
model_ls = LeastSquares()
model_ls.fit(X, y_noise)
for idx, coef in enumerate(model_ls.coef):
    print(f"Coefficient {idx}: {coef:.6f}")

fig = plot_complex(X, y_noise, {"least squares": model_ls}, y_orig=y)
fig.set_size_inches(7, 9)
plt.tight_layout()
plt.show()

# %%
# Add some outliers
y_noise = add_gaussian_noise(y, loc=(0, 0), scale=(21, 21))
model_mest = M_estimate()
model_mest.fit(X, y_noise)
for idx, coef in enumerate(model_mest.coef):
    print(f"Coefficient {idx}: {coef:.6f}")

fig = plot_complex(
    X, y_noise, {"least squares": model_ls, "M_estimate": model_mest}, y_orig=y
)
fig.set_size_inches(7, 9)
plt.tight_layout()
plt.show()


# %%
# Try running as a real-valued problem
X_real, y_real = complex_to_glr(X, y_noise)
model_ls = LeastSquares()
model_ls.fit(X_real, y_real)
coef = glr_coef_to_complex(model_ls.coef)
for idx, coef in enumerate(coef):
    print(f"Coefficient {idx}: {coef:.6f}")


# %%
# Try running using real-valued M_estimates
model_mest = M_estimate()
model_mest.fit(X_real, y_real)
coef = glr_coef_to_complex(model_mest.coef)
for idx, coef in enumerate(coef):
    print(f"Coefficient {idx}: {coef:.6f}")
