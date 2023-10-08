"""
Regression in C specific errors
"""
from regressioninc.base import Estimator


###
# Estimator errors
###
class EstimatorError(Exception):
    """For errors with estimator usage"""

    def __init__(self, estimator: Estimator, msg: str):
        self.name = type(estimator).__name__
        self.msg = msg

    def __str__(self) -> str:
        return self.msg
