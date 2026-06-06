""""""

from abc import ABC, abstractmethod


class AbstractSignatureDistance(ABC):
    """
    Abstract base class for signature distance metrics
    """

    @abstractmethod
    def compute_distance(self, signature1, signature2) -> float:
        """
        Compute the distance between two signatures.
        """
