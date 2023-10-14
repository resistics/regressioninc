r"""
Linear regression
^^^^^^^^^^^^^^^^^

This example begins with linear regression in the real domain and then builds up
to show how linear problems can be thought of in the complex domain.

Useful references:

- https://stats.stackexchange.com/questions/66088/analysis-with-complex-data-anything-different
- https://www.chrishenson.net/article/complex_regression
"""
# sphinx_gallery_thumbnail_number = 5
import numpy as np
import matplotlib.pyplot as plt
from regressioninc.linear.models import add_intercept, OLS
from regressioninc.testing.complex import ComplexGrid

# %%
# One of the most straightforward linear problems to understand is the equation
# of a line. Let's look at a line with gradient 3 and intercept -2.
params = np.array([3])
intercept = -2
X = np.arange(-5, 5).reshape(10, 1)
y = np.matmul(X, params) + intercept

fig = plt.figure()
plt.scatter(y, X)
plt.xlabel("Independent variable")
plt.ylabel("Dependent variable")
plt.tight_layout()
fig.show()

# %%
# When performing linear regressions, the aim is to:
#
# - calculate the parameters (also called coefficients)
# - given the regressors X (values of the independent variable)
# - and values of the regrassands y (values of the dependent variable)
#
# This can be done with linear regression, and the most common method of linear
# regression is least squares, which aims to estimate the parameters whilst
# minimising the squared misfit between the regrassands and predicted
# regrassands calculated using the estimated parameters.
X = add_intercept(X)
model = OLS()
model.fit(X, y)
print(model.estimate.params)

# %%
# Least squares was able to correctly calculate the slope and intercept for the
# real-valued regression problem. Let's look at the predicted regrassands using
# the estimated parameters.
preds = model.predict(X)
print(preds)

# %%
# It is also possible to have linear problems in the complex domain. These
# commonly occur in signal processing problems. Let's define parameters and
# regressors X and calculate out the regrassands y for an example problem.
params = np.array([2 + 3j])
X = np.array([1 + 1j, 2 + 1j, 3 + 1j, 1 + 2j, 2 + 2j, 3 + 2j]).reshape(6, 1)
y = np.matmul(X, params)

# %%
# It is a bit harder to visualise the complex-valued version, but let's try and
# visualise the regressors X and regrassands y.
fig, axs = plt.subplots(nrows=1, ncols=2)
plt.sca(axs[0])
plt.scatter(X.real, X.imag, c="tab:blue")
plt.xlim(X.real.min() - 3, X.real.max() + 3)
plt.ylim(X.imag.min() - 3, X.imag.max() + 3)
plt.title("Regressors X")
plt.sca(axs[1])
plt.scatter(y.real, y.imag, c="tab:red")
plt.xlim(y.real.min() - 3, y.real.max() + 3)
plt.ylim(y.imag.min() - 3, y.imag.max() + 3)
plt.title("Regrassands y")
plt.show()

# %%
# Visualsing the regressors X and the regrassands y this way gives a geometric
# indication of the linear problem in the complex domain. Multiplying the
# regressors by the parameters can be considered like a scaling and a rotation
# of the independent variables to give the dependent variables y.
#
# With more samples, this can be a bit easier to visualise. In the below
# example, regressors and regrassands are generated again, this time with more
# samples. To start off with, the parameter is a real number to demonstrate the
# scaling without any rotation. Both the regressors and regrassands are plotted
# on the same axis with lines to show the mapping between independent and
# dependent values.
grid = ComplexGrid(r1=0, r2=10, nr=11, i1=-5, i2=5, ni=11)
X = grid.flat_grid()
params = np.array([0.5])
y = np.matmul(X, params)

fig = plt.figure()
for iobs in range(y.size):
    plt.plot(
        [y[iobs].real, X[iobs, 0].real],
        [y[iobs].imag, X[iobs, 0].imag],
        color="k",
        lw=0.5,
    )
plt.scatter(X.real, X.imag, c="tab:blue", label="Regressors")
plt.grid()
plt.title("Regressors X")
plt.scatter(y.real, y.imag, c="tab:red", label="Regrassands")
plt.grid()
plt.legend()
plt.title("Complex regression")
plt.show()

# %%
# Now let's add a complex component to the coefficient to demonstrate the
# rotational aspect.
params = np.array([0.5 + 2j])
y = np.matmul(X, params)

fig = plt.figure()
for iobs in range(y.size):
    plt.plot(
        [y[iobs].real, X[iobs, 0].real],
        [y[iobs].imag, X[iobs, 0].imag],
        color="k",
        lw=0.5,
    )
plt.scatter(X.real, X.imag, c="tab:blue", label="Regressors")
plt.grid()
plt.title("Regressors X")
plt.scatter(y.real, y.imag, c="tab:red", label="Regrassands")
plt.grid()
plt.legend()
plt.title("Complex regression")
plt.show()

# %%
# Finally, adding an intercept gives a translation.
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
plt.scatter(y.real, y.imag, c="tab:red", label="Regrassands")
plt.grid()
plt.legend()
plt.title("Complex regression")
plt.show()


# %%
# Similar to the real-valued problem, linear regression can be used to estimate
# the values of the parameters for the complex-valued problem. Again, least
# squares is one of the most common methods of linear regression. However, not
# all least squares algorithms support complex data, though some do such as the
# least squares in Scipy. The focus of regression in C is to provide regression
# methods for complex-valued data.
#
# Note that adding an intercept column to X allows for solving of the intercept.
# Regression in C does not automatically solve for the intercept and if desired,
# an intercept column needs to be added to the regressors.
X = add_intercept(X)
model = OLS()
model.fit(X, y)
print(model.estimate.params)
