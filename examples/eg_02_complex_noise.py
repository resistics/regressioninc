r"""
Visualising noise in data
^^^^^^^^^^^^^^^^^^^^^^^^^

This package focuses on regression in the complex domain.

y = X \beta

"""
import numpy as np
import matplotlib.pyplot as plt
from regressioninc.testing.complex import ComplexGrid, add_gaussian_noise
from regressioninc.testing.complex import add_outliers_to_observations

np.random.seed(42)


grid = ComplexGrid(r1=-10, r2=10, nr=5, i1=-10, i2=10, ni=5)
grid_pts = grid.flat_grid()
grid_noisy = add_gaussian_noise(grid_pts, loc=(0, 0), scale=(1, 1))
noise = grid_noisy - grid_pts
# print(noise)

# plt.figure()
# plt.scatter(grid_pts.real, grid_pts.imag, c="tab:blue", marker="*", label="original")
# plt.scatter(grid_noisy.real, grid_noisy.imag, c="tab:red", marker="d", label="noisy")
# plt.tight_layout()
# plt.show()


grid_outliers = add_outliers_to_observations(
    grid_pts, outlier_percent=20, mult_min=5, mult_max=10
)

plt.figure()
plt.scatter(
    grid_pts.real, grid_pts.imag, c="tab:blue", marker="*", label="original", alpha=0.3
)
plt.scatter(
    grid_outliers.real,
    grid_outliers.imag,
    c="tab:red",
    marker="d",
    label="outliers",
    alpha=0.5,
)
plt.tight_layout()
plt.show()
