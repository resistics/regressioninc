"""
Module for regressioninc base classes
"""
from sklearn.base import MultiOutputMixin, RegressorMixin


class ComplexRegressorMixin(MultiOutputMixin, RegressorMixin):
    pass
