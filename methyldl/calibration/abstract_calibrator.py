from sklearn.base import BaseEstimator
from ..cross_validation_engine import CrossValidationCompatibleModel


class AbstractCalibrator(CrossValidationCompatibleModel, BaseEstimator):
    """
    Abstract base class for post-processing deconvolution predictions.
    For now only a pass-through, but we can add utilities common to all calibrators
    here in the future.
    """
