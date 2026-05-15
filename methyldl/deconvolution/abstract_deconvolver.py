from sklearn.base import BaseEstimator

from ..cross_validation_engine import CrossValidationCompatibleModel


class AbstractDeconvolver(CrossValidationCompatibleModel, BaseEstimator):
    """
    Abstract base class for deconvolution methods.
    For now only a pass-through, but we can add shared utilities between deconvolvers
    that are not shared to other CrossValidationCompatibleModels (e.g. linear calibrators)
    here in the future.
    """
