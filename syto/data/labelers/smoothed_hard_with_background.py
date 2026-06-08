""" """

import pandas as pd
import numpy as np

from syto.data.labelers.abstract_labeler import AbstractLabeler


class SmoothedHardWithBackgroundLabeler(AbstractLabeler):
    """
    Labeler that assigns smoothed hard labels by applying label smoothing to the hard
    labels with background.
    """

    def compute_labels(
        self,
        reads_df: pd.DataFrame,
        num_original_classes: int,
        epsilon: float = 0.1,
        class_label_column="original_label",
        grg_class_label_column="dmr_ctype_label",
    ) -> pd.DataFrame:
        """
        Label the given read dataframe with hard labels, assigning reads to the
        background class if they are off-target.

        Args:
            reads_df: DataFrame containing the reads to label. Must have columns
                for the read's original class label and the GRG class label.
            num_original_classes: The number of original classes
                (used to assign the background class).
            class_label_column: Name of the column in reads_df that contains the
                read's original class label (e.g., cell type).
            grg_class_label_column: Name of the column in reads_df that contains
                the GRG class label (e.g., the cell type associated with the GR group).

        Returns:
            A copy of reads_df with an additional 'label' column containing the
            labels (original class label for on-target reads, background class
            for off-target reads).
        """
        labeled_df = reads_df.copy()
        original_labels = labeled_df[class_label_column].values.astype(int)
        grg_labels = labeled_df[grg_class_label_column].values.astype(int)
        is_on_target = original_labels == grg_labels
        hard_labels = (
            original_labels * is_on_target
            + num_original_classes * np.logical_not(is_on_target)
        )
        one_hot_labels = np.eye(num_original_classes + 1, dtype=int)[hard_labels]
        smoothed_labels = one_hot_labels * (1 - epsilon) + epsilon / (
            num_original_classes + 1
        )
        labeled_df["smoothed_label"] = list(smoothed_labels)
        return labeled_df
