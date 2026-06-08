from abc import ABC, abstractmethod

import pandas as pd


class AbstractLabeler(ABC):
    """
    Abstract base class for labelers
    """

    @abstractmethod
    def compute_labels(self, reads_df: pd.DataFrame) -> pd.DataFrame:
        """
        Label the given read dataframe
        """
