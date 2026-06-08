from abc import ABC, abstractmethod

import pandas as pd


class AbstractOmicsSignatureHandler(ABC):
    """
    Abstract base class for omics signature handlers
    """

    VALID_DISTANCE_NAMES = set()

    @abstractmethod
    @classmethod
    def extract_signature(cls, read_data: pd.Series):
        """
        Extract an omics signature from the given read data.
        """

    @abstractmethod
    @classmethod
    def compute_distance_dispatcher(cls, signature1, signature2, distance_name: str):
        """
        Compute the distance between two omics signatures using the specified distance metric.
        """
