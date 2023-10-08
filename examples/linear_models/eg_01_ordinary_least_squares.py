r"""
Ordinary least-squares regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of ordinary least squares regression with complex values. The results
well be benchmarked against scipy linear regression which supports complex
numbers.
"""
import numpy as np
import matplotlib.pyplot as plt
from regressioninc.linear.models import add_intercept, OLS

# %%
# Create a linear complex-valued problem.
params = np.array([-13 + 17j])
intercept = 5 - 7j
X = np.arange(-5, 5).reshape(10, 1)
y = X * params + intercept

fig = plt.figure()
plt.scatter(y, X)
plt.xlabel("Independent variable")
plt.ylabel("Dependent variable")
plt.tight_layout()
fig.show()

# %%
# Run
X = add_intercept(X)
model = OLS()
model.fit(X, y)
print(model.estimate.params)
