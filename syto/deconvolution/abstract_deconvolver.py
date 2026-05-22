from typing import Optional

from sklearn.base import BaseEstimator

from syto.deconvolution.history import DeconvolutionHistory
from ..cross_validation_engine import CrossValidationCompatibleModel


class AbstractDeconvolver(CrossValidationCompatibleModel, BaseEstimator):
    """
    Abstract base class for deconvolution methods.
    All deconvolvers share a ``history`` attribute that stores training
    metrics as a :class:`DeconvolutionHistory` (or ``None`` before fitting).
    """

    def __init__(self):
        self.history: Optional[DeconvolutionHistory] = None
