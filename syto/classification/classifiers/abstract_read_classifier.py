"""
Provides a unified prediction interface for the classifiers of DNA Reads.
"""

import logging
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

    @abstractmethod
    def fit_split(
        self,
        train_df: pd.DataFrame,
        val_df: Union[pd.DataFrame, None] = None,
        output_dir: Union[str, Path, None] = None,
        **kwargs,
    ) -> "AbstractReadClassifier":
        """Fit the classifier on training data.

        Args:
            train_df: Training DataFrame with read-level data.
            val_df: Optional validation DataFrame for early stopping / metrics.
            output_dir: Where to save checkpoints and artifacts.
            **kwargs: Classifier-specific hyperparameters.

        Returns:
            self (the fitted classifier instance).
        """

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted classifier to disk."""

