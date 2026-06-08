"""
Module containing the HardWithBackgroundLabeler class, which assigns hard labels based
on whether reads are on-target or off-target (in which case they are assigned to the background class).
"""

import pandas as pd

from syto.data.labelers.abstract_labeler import AbstractLabeler


class HardWithBackgroundLabeler(AbstractLabeler):
    """
    Labeler that assigns hard labels based on the read being on-target
    or off-target (in which case they are assigned to the background class).

    A read is on-target if it is in a region specific for its cell type,
    and off-target otherwise.
    """

    def compute_labels(
        self,
        reads_df: pd.DataFrame,
        num_original_classes: int,
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
        labeled_df["label"] = labeled_df.apply(
            lambda row: (
                row[class_label_column]
                if row[class_label_column] == row[grg_class_label_column]
                else num_original_classes
            ),  # assign to background class
            axis=1,
        )
        return labeled_df
