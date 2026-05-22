"""
Provides a unified prediction interface for the classifiers of DNA Reads.
"""

import logging
from collections import OrderedDict
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


class AbstractReadClassifier(ABC):
    """Abstract base class for read-level classifiers."""

    @abstractmethod
    def predict_split(self, split_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run the classifier on a prepared split DataFrame.

        Args:
            split_df (pd.DataFrame): Input DataFrame with read-level data.
            **kwargs: Additional parameters for prediction.

        Returns:
            pd.DataFrame: The input DataFrame enriched with prediction columns.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "AbstractReadClassifier":
        """Load a ReadClassifier from a checkpoint."""
