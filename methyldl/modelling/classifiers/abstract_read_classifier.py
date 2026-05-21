"""
ReadClassifier

Provides a unified prediction interface for MethylBERT, Dismir,
and CancerDetector (stub).  Each adapter loads a pre-trained checkpoint
and produces a DataFrame enriched with per-label prediction columns.
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


class ClassifierAdapter:
    """Unified prediction interface across classifiers.

    Parameters
    ----------
    classifier_type : str
        One of ``"methylbert"``, ``"dismir"``, ``"cancer_detector"``.
    checkpoint_path : str
        Path to the pre-trained classifier checkpoint.
    labels_dict : dict
        ``{int_label: cell_type_name}`` mapping.
    num_labels : int
        Number of output labels.  Default: 39.
    seq_length : int
        Maximum sequence length for the classifier.  Default: 150.
    foundation_model_path : str
        Path or HuggingFace id for MethylBERT foundation model.
    classifier_head_implementation : str
        Classifier head variant — ``"vanilla"`` or ``"dmr_attention_based"``.
    dmr_label_column : str
        Column name for the DMR label, used by both MethylBERT and Dismir.
    dismir_flavor : str
        Dismir backbone flavour — ``"lstm"`` or ``"mingru"``.
    batch_size : int
        Prediction batch size.  Default: 2200.
    """

    SUPPORTED_CLASSIFIERS = ("methylbert",)

    def __init__(
        self,
        classifier_type: str,
        checkpoint_path: str,
        labels_dict: dict,
        num_labels: int = 39,
        num_dmr_labels: int = 39,
        seq_length: int = 150,
        # MethylBERT-specific
        foundation_model_path: str = "hanyangii/methylbert_hg19_12l",
        classifier_head_implementation: str = "dmr_attention_based",
        # DMR label column (relevant for both MethylBERT and Dismir and CancerDetector)
        dmr_label_column: str = "dmr_ctype_label",
        # Dismir-specific
        dismir_flavor: str = "lstm",
        # CancerDetector-specific parameters
        cancer_detector_prior_type: str = "uniform",
        # Shared
        batch_size: int = 2200,
        soft_labels: bool = True,
    ):
        if classifier_type not in self.SUPPORTED_CLASSIFIERS:
            raise ValueError(
                f"classifier_type must be one of {self.SUPPORTED_CLASSIFIERS}, "
                f"got '{classifier_type}'"
            )

        self.classifier_type = classifier_type
        self.checkpoint_path = checkpoint_path
        self.labels_dict = labels_dict
        self.num_labels = num_labels
        self.seq_length = seq_length
        self.foundation_model_path = foundation_model_path
        self.classifier_head_implementation = classifier_head_implementation
        self.dmr_label_column = dmr_label_column
        self.cancer_detector_prior_type = cancer_detector_prior_type
        self.dismir_flavor = dismir_flavor
        self.batch_size = batch_size
        self.soft_labels = soft_labels
        self._model = None
        self.num_dmr_labels = num_dmr_labels

    # ─── MethylBERT ─────────────────────────────────────────────────

    def _load_methylbert(self):
        from methyldl.modelling.classifiers.methylbert import MethylBert

        if self.soft_labels:
            loss = "cwce"
        elif self.num_labels == 2:
            loss = "bce"
        else:
            loss = "ce"
        rrms_config = OrderedDict(
            [
                ("lr", 0.0004),
                ("beta", (0.9, 0.98)),
                ("weight_decay", 0.1),
                ("warmup_step", 100),
                ("eps", 1e-6),
                ("with_cuda", True),
                ("log_freq", 200),
                ("eval_freq", 200),
                ("n_hidden", None),
                ("decrease_steps", 200),
                ("eval", False),
                ("amp", True),
                ("gradient_accumulation_steps", 1),
                ("max_grad_norm", 1.0),
                ("save_freq", None),
                ("loss", loss),
                ("adam_beta1", 0.9),
                ("adam_beta2", 0.98),
                ("seed", 950410),
            ]
        )

        self._model = MethylBert(
            custom_config=rrms_config,
            foundation_model_path=self.foundation_model_path,
            num_labels=self.num_labels,
            num_dmr_labels=self.num_dmr_labels,
            fine_tuned_model_path=self.checkpoint_path,
            classifier_implementation=self.classifier_head_implementation,
            soft_labels=self.soft_labels,
            seq_len=self.seq_length,
            batch_size=self.batch_size,
        )

    def _predict_methylbert(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """Run MethylBERT prediction on a single split."""
        from methyldl.modelling.classifiers.methylbert import (
            MethylBertFinetuneDataset,
            MethylVocab,
        )
        from methyldl.modelling.data_preprocessing_for_inference import (
            prepare_methylbert_list_inference,
        )
        from methyldl.modelling.prediction_aggregation import (
            aggregate_chuncked_predictions_weighted,
        )

        # Prepare chunked input data
        input_df = split_df.rename(
            columns={"seq": "input_ids", "pattern": "methylation_ids"}
        )

        data_list = prepare_methylbert_list_inference(
            input_df,
            dmr_label_column=self.dmr_label_column,
            seq_length=self.seq_length,
            stride=int(self.seq_length / 2),
            soft_labels=self.soft_labels,
            is_binary=True if self.num_labels == 2 else False,
        )

        dataset = MethylBertFinetuneDataset(
            data_source=data_list,
            vocab=MethylVocab(k=3),
            seq_len=self.seq_length,
            lazy_tokenization=True,
            soft_labels=self.soft_labels,
        )

        # Run predictions
        predictions = self._model.predict(dataset, batch_size=self.batch_size)

        # Build predictions DataFrame
        pred_cols = [f"prediction_{i}" for i in range(self.num_labels)]
        pred_df = pd.DataFrame(predictions[0], columns=pred_cols)
        pred_df["read_name"] = [data_list[i][-3] for i in range(1, len(data_list))]
        pred_df["ncpgs_marked"] = [data_list[i][-2] for i in range(1, len(data_list))]

        # Aggregate chunked predictions by read
        pred_df = aggregate_chuncked_predictions_weighted(pred_df)

        # Merge back with original split
        result = pd.merge(split_df, pred_df, on="read_name")
        return result

    # ─── Public API ─────────────────────────────────────────────────

    def predict_split(
        self,
        split_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run the classifier on a prepared split DataFrame.

        Parameters
        ----------
        split_df : pd.DataFrame
            Input DataFrame with read-level data.

        Returns
        -------
        pd.DataFrame
            The input DataFrame enriched with ``prediction_0 .. prediction_{num_labels-1}``
            columns.
        """
        self._lazy_load_model()

        if self.classifier_type == "methylbert":
            return self._predict_methylbert(split_df)
