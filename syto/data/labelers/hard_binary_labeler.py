"""Module containing the HardBinaryLabeler class, which assigns hard binary labels based
on whether reads are on-target or off-target."""

import pandas as pd

from syto.data.labelers.abstract_labeler import AbstractLabeler


class HardBinaryLabeler(AbstractLabeler):
    """
    Labeler that assigns hard binary labels based on the read being on-target
    or off-target.

    A read is on-target if it is in a region specific for its cell type,
    and off-target otherwise.
    """

    def compute_labels(
        self,
        reads_df: pd.DataFrame,
        class_label_column="original_label",
        grg_class_label_column="dmr_ctype_label",
    ) -> pd.DataFrame:
        """
        Label the given read dataframe with hard binary labels.

        Args:
            reads_df: DataFrame containing the reads to label. Must have columns
                for the read's original class label and the GRG class label.
            class_label_column: Name of the column in reads_df that contains the
                read's original class label (e.g., cell type).
            grg_class_label_column: Name of the column in reads_df that contains
                the GRG class label (e.g., the cell type associated with the GR group).

        Returns:
            A copy of reads_df with an additional 'label' column containing the
            binary labels (1 for on-target, 0 for off-target).
        """
        labeled_df = reads_df.copy()
        labeled_df["label"] = (
            labeled_df[class_label_column] == labeled_df[grg_class_label_column]
        ).astype(int)
        return labeled_df
