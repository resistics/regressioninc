r"""
Multiple regressors
^^^^^^^^^^^^^^^^^^^

The previous example showed complex-valued regression with a single regressor.
In practice, it is common to have multiple regressors. The following example
will generate a complex-valued linear problem with multiple regressors and try
and visualise it.
"""
# sphinx_gallery_thumbnail_number = 5
import numpy as np
import matplotlib.pyplot as plt
from regressioninc.linear.models import add_intercept, OLS
from regressioninc.testing.complex import generate_linear_random, plot_complex
from regressioninc.testing.complex import ComplexGrid

np.random.seed(42)

# %%
# Let's begin where the previous example ended.
grid = ComplexGrid(r1=0, r2=10, nr=11, i1=-5, i2=5, ni=11)
X = grid.flat_grid()
params = np.array([0.5 + 2j])
intercept = 20 + 20j
y = np.matmul(X, params) + intercept

fig = plt.figure()
for iobs in range(y.size):
    plt.plot(
        [y[iobs].real, X[iobs, 0].real],
        [y[iobs].imag, X[iobs, 0].imag],
        color="k",
        lw=0.3,
    )
plt.scatter(X.real, X.imag, c="tab:blue", label="Regressors")
plt.grid()
plt.title("Regressors X")
plt.scatter(y.real, y.imag, c="tab:red", label="Observations")
plt.grid()
plt.legend()
plt.title("Complex regression")
plt.show()

# %%
# Now plot this in a different way that will make it easier to visualise more
# than a single regressor.
fig = plot_complex(X, y, {})
plt.show()


# %%
# Let's add in a second regressor but rather than having a grid of input points
# use some random input points.
params_random = np.array([2.7 - 1.8j])
X_random, _ = generate_linear_random(params_random, y.size)
params = np.concatenate((params, params_random))
X = np.concatenate((X, X_random), axis=1)
intercept = 20 + 20j
y = np.matmul(X, params) + intercept

fig = plot_complex(X, y, {})
fig.set_size_inches(7, 6)
plt.tight_layout()
plt.show()

# %%
# These examples have been adding an intercept to y. To solve for an intecerpt,
# an intercept column needs to be explicitly added to the regressors X before
# passing X through to the model. Adding an intercept simply adds a column of 1s
# to the regressors.
X = add_intercept(X)
fig = plot_complex(X, y, {})
fig.set_size_inches(7, 6)
plt.tight_layout()
plt.show()

# %%
# Now the parameters can be estimated using the regrassands y and the regressors
# X.
model = OLS()
model.fit(X, y)
for idx, params in enumerate(model.estimate.params):
    print(f"parameter {idx}: {params:.6f}")

# %%
# Finally, the predicted regressands calculated from the estimated parameters
# can be added to the visualisation.
fig = plot_complex(X, y, {"least squares": model})
fig.set_size_inches(7, 9)
plt.tight_layout()
plt.show()
