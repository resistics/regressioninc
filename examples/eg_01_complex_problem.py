r"""
Regression in C
^^^^^^^^^^^^^^^

This package focuses on regression in the complex domain.

y = X \beta


https://stats.stackexchange.com/questions/66088/analysis-with-complex-data-anything-different

"""
import numpy as np
import matplotlib.pyplot as plt
from regressioninc.linear import add_intercept, LeastSquares

# %%
# One of the most straightforward linear problems to understand is the equation
# of line. Let's look at a line with gradient 3 and intercept -2
coef = np.array([3])
intercept = -2
X = np.arange(-5, 5).reshape(10, 1)
y = X * coef + intercept

fig = plt.figure()
plt.scatter(y, X)
plt.xlabel("Independent variable")
plt.ylabel("Dependent variable")
plt.tight_layout()
fig.show()

# %%
# When performing linear regressions, the aim is to:
#
# - calculate the coefficients (coef, also called parameters)
# - given the regressors (X, values of the independent variable)
# - and values of the observations (y, values of the dependent variable)
#
# This can be done with linear regression, and the most common method of linear
# regression is least squares, which aims to estimate the coefficients whilst
# minimising the misfit between the observations and estimated observations
# calculated using the estimated coefficients.
X = add_intercept(X)
model = LeastSquares()
model.fit(X, y)
print(model.coef)

# %%
# Least squares was able to correctly calculate the slope and intercept for the
# real-valued regression problem.
#
# It is also possible to have linear problems in the complex domain. These
# commonly occur in signal processing problems. Let's define linear coefficients
# and regressors X and calculate out the observations y.
coef = np.array([2 + 3j])
X = np.array([1 + 1j, 2 + 1j, 3 + 1j, 1 + 2j, 2 + 2j, 3 + 2j]).reshape(6, 1)
y = np.matmul(X, coef)

# %%
# It is a bit harder to visualise the complex-valued version, but let's try and
# visualise the regressors X and observations y.
fig, axs = plt.subplots(nrows=1, ncols=2)
plt.sca(axs[0])
plt.scatter(X.real, X.imag, c="tab:blue")
plt.xlim(X.real.min() - 3, X.real.max() + 3)
plt.ylim(X.imag.min() - 3, X.imag.max() + 3)
plt.grid()
plt.title("Regressors X")
plt.sca(axs[1])
plt.scatter(y.real, y.imag, c="tab:red")
plt.xlim(y.real.min() - 3, y.real.max() + 3)
plt.ylim(y.imag.min() - 3, y.imag.max() + 3)
plt.grid()
plt.title("Observations y")
plt.show()

# %%
# Visualsing the regressors X and the observations y this way gives a geometric
# indication of the linear problem in the complex domain. Multiplying the
# regressors by the coefficients can be considered like a scaling and a rotation
# of the independent variables to give the observations y, or the dependent
# variables.
#
# Similar to the real-valued problem, linear regression can be used to estimate
# the values of the coefficients for the complex-valued problem. Again, least
# squares is one of the most common methods of linear regression. However, not
# all least squares algorithms support complex data, though some do such as the
# least squares in numpy. The focus of regressioninc is to provide regression
# methods for complex-valued data.
model = LeastSquares()
model.fit(X, y)
print(model.coef)
