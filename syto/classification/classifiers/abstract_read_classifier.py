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
    def fit_classificaton(
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

    def required_fit_columns(self, config: dict) -> list[str]:
        """Feature/auxiliary columns this classifier reads at fit time.

        Excludes the training label column, which the fitting pipeline supplies
        separately from its ``label_column`` config. ``config`` is the full
        parsed fit config (``model``, ``training``, ...). Concrete classifiers
        override this; the base raises so gaps are loud.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not declare required_fit_columns"
        )

    def mlflow_fit_params(self) -> dict:
        """Architecture-relevant hyperparameters to record on an MLflow run.

        Mirrors :meth:`required_fit_columns`: each classifier surfaces only the
        init/hyperparameters meaningful to *its* architecture, so runs are not
        polluted with the defaults other architectures ignore (e.g.
        ``dismir_flavor`` on a non-Dismir run). The fitting pipeline logs these
        under a ``model.*`` namespace. The base returns an empty dict;
        classifiers whose fit-time hyperparameters already flow through the
        ``training`` config need not override it.
        """
        return {}

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted classifier to disk."""
