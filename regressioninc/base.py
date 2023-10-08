"""
Common parent classes
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
import numpy as np


class Estimate(BaseModel):
    """Base class for the results for an Estimator"""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    params: np.ndarray
    """The estimated parameters"""


class Estimator(BaseModel, ABC):
    """
    Base class for estimators. Estimators fit parameters to the given data,
    typically using a set of assumptions about the data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    estimate: Estimate = None
    """Attribute with variables estimated from the data"""

    @abstractmethod
    def fit(self, X, y):
        """Method to estimate the parameters"""
        pass
