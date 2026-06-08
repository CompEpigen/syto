from abc import ABC, abstractmethod
from typing import Callable
import pandas as pd


class AbstractOmicsSignatureHandler(ABC):
    """
    Abstract base class for omics signature handlers
    """

    VALID_DISTANCES = set()

    @abstractmethod
    def extract_signature(self, read_data: pd.Series):
        """
        Extract an omics signature from the given read data.
        """

    @abstractmethod
    def compute_distance_dispatcher(
        self, signature1, signature2, distance_name: str
    ) -> Callable:
        """
        Compute the distance between two omics signatures using the specified distance metric.
        """
