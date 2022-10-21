import numpy as np
from regressioninc.linear import add_intercept, complex_to_glr, glr_coef_to_complex
from regressioninc.linear import LeastSquares
from regressioninc.testing.complex import ComplexGrid, generate_linear_grid


coef = np.array([3 + 2j])
grid = ComplexGrid(r1=-1, r2=1, nr=3, i1=4, i2=6, ni=3)
X, y = generate_linear_grid(coef, [grid], intercept=10)
X = add_intercept(X)
X_glr, y_glr = complex_to_glr(X, y)
model = LeastSquares()
model.fit(X_glr, y_glr)
glr_coef_to_complex(model.coef)


# X = np.array([3 + 4j, 2+8j, 6+4j]).reshape(3,1)
# X = add_intercept(X)
# y = np.array([-2-4j, -1-3j, 5+6j])
# X_glr, y_glr = complex_to_glr(X, y)
# print(X_glr)
# print(y_glr)
