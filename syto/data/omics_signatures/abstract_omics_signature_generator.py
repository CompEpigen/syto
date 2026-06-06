from abc import ABC, abstractmethod


class AbstractOmicsSignatureGenerator(ABC):
    """
    Abstract base class for omics signature generators
    """

    @abstractmethod
    @classmethod
    def extract_signature(cls, *args):
        """
        Extract an omics signature from the given read data.
        """
