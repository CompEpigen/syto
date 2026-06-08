from tqdm import tqdm
import pandas as pd
import numpy as np

from syto.data.labelers.abstract_labeler import AbstractLabeler
from syto.data.omics_signatures_handlers.abstract_omics_signature import (
    AbstractOmicsSignatureHandler,
)


class DataDrivenSoftLabeler(AbstractLabeler):
    """
    Perform end to end computation of data driven soft labels for a given dataframe of reads.
    """

    def __init__(
        self, distance_name: str, signature_handler: AbstractOmicsSignatureHandler
    ):
        """
        Args:
            distance_name: the name of the distance metric to use for pooling (e.g. "jaccard")
            signature_handler: an instance of AbstractOmicsSignatureHandler to extract signatures
                and compute distances (instantiated with the approriate column names)
        """
        self.signature_handler = signature_handler
        self.distance_name = distance_name
        self.distance_fct = signature_handler.compute_distance_dispatcher(distance_name)

    def compute_labels(
        self,
        reads_df,
        perform_pooling: bool = True,
        min_reads=30,
        max_distance=0.5,
        num_classes=39,
        keep_intermediate_values=True,
        precomputed_signature_column=None,
    ):
        """
        Compute the data driven soft labels for the given dataframe of reads.
        The DataFrame should have the following columns:
        - "name": the name of the genomic region (e.g. "chr1:1000-2000" or "colon-fibro-specific")
        - "original_label": the original class label for this read
            (expects integer from 0 to num_classes-1)

        Args:
            reads_df: DataFrame containing the reads and their original labels
            perform_pooling: whether to perform KNN pooling of signatures within each region
            min_reads: minimum number of reads to pool together
                (including those of the signature itself)
            max_distance: maximum distance between signatures to be considered neighbors for pooling
            num_classes: total number of classes in the original labels
            keep_intermediate_values: whether to keep intermediate values
                (raw counts, weighted counts) in the final output
            precomputed_signature_column: if None, the signatures are computed from the reads
                using the signature handler. If not None, the column with this name is used as
                the precomputed signature for each read (and signatures are not recomputed).

        Returns:
            DataFrame with original columns plus:
            - "signature": the extracted omics signature for each read
            - "soft_label": the computed soft label (probability distribution over classes) for each read
            If keep_intermediate_values is True, also includes:
            - "raw_counts_c": the raw count of reads with class c for the same signature
            - "weighted_counts_c": the weighted count of reads with class c for the same signature
            - "raw_counts_pooled": the raw count of reads pooled together for this signature (if pooling is performed)
            - "num_signatures_pooled": the number of signatures pooled together for this signature (if pooling is performed)
        """
        assert all(
            col in reads_df.columns for col in ["name", "original_label"]
        ), "Input DataFrame must contain 'name' and 'original_label' columns."
        assert (
            reads_df["original_label"].between(0, num_classes - 1).all()
        ), f"All 'original_label' values must be between 0 and {num_classes - 1}."
        reads_df = reads_df.copy()

        ## Compute the signatures for each read (or reuse precomputed ones)
        if precomputed_signature_column is None:
            tqdm.pandas(desc="Extracting signatures")
            reads_df["signature"] = reads_df.progress_apply(
                self.signature_handler.extract_signature, axis=1
            )
        else:
            assert precomputed_signature_column in reads_df.columns, (
                f"Column '{precomputed_signature_column}' not found in the input DataFrame."
            )
            reads_df["signature"] = reads_df[precomputed_signature_column]

        ## Aggregate the counts of classes by signatures
        class_counts = (
            reads_df.groupby(["name", "signature", "original_label"])
            .size()
            .unstack(fill_value=0)
        )  # df with index (name, signature) and columns original_label with counts as values
        # reset index to have name and signature as columns
        class_counts.reset_index(inplace=True)
        class_counts["total_reads"] = class_counts[class_counts.columns[2:]].sum(axis=1)
        # for missing classes, add columns with 0 counts
        for c in range(num_classes):
            if c not in class_counts.columns:
                class_counts[c] = 0

        class_counts.rename(
            columns={c: "raw_counts_" + str(c) for c in range(num_classes)},
            inplace=True,
        )
        class_count_columns = ["raw_counts_" + str(c) for c in range(num_classes)]

        ## prepare the weights for normalization of counts
        global_class_counts = class_counts[class_count_columns].sum(axis=0)

        # we multiply all weights by the median
        # (although this has no mathematical effect on the final probabilities)
        nonzero_counts = global_class_counts[global_class_counts > 0]
        median_count = nonzero_counts.median()
        epsilon = 1e-9
        class_weights = (median_count / (global_class_counts + epsilon)).values

        ## If perform_pooling is True, pool the counts
        if perform_pooling:
            pooled_counts_df = self.nn_count_pooling(
                class_counts,
                num_classes=num_classes,
                min_reads=min_reads,
                max_distance=max_distance,
            )
            class_counts = class_counts.merge(
                pooled_counts_df,
                on=["name", "signature"],
            )

        ## Perform weighting of counts by global class frequencies
        for c in range(num_classes):
            count_col = (
                "pooled_counts_" + str(c) if perform_pooling else "raw_counts_" + str(c)
            )
            class_counts["weighted_counts_" + str(c)] = (
                class_counts[count_col] * class_weights[c]
            )

        ## Compute the probabilities
        weighted_count_columns = [
            "weighted_counts_" + str(c) for c in range(num_classes)
        ]
        weighted_count_matrix = class_counts[weighted_count_columns].values
        total_weighted_counts = weighted_count_matrix.sum(axis=1, keepdims=True)
        probabilities = np.divide(
            weighted_count_matrix,
            total_weighted_counts,
            out=np.zeros_like(weighted_count_matrix),
            where=total_weighted_counts != 0,
        )
        class_counts["soft_label"] = probabilities.tolist()

        ## merge the soft labels back to the original reads dataframe
        if not keep_intermediate_values:
            class_counts = class_counts[["name", "signature", "soft_label"]]
        reads_df = reads_df.merge(class_counts, on=["name", "signature"])

        return reads_df

    def nn_count_pooling(
        self,
        class_counts: pd.DataFrame,
        num_classes: int = 39,
        min_reads=30,
        max_distance=0.5,
    ) -> pd.DataFrame:
        """
        Pool counts within genomic regions using nearest neighbor aggregation.

        Args:
            class_counts: DataFrame with columns "name", "signature", "total_reads",
                and "raw_counts_c" for each class c
            num_classes: total number of classes in the original labels
            min_reads: minimum number of reads to pool together
                (including those of the signature itself)
            max_distance: maximum distance between signatures to be considered neighbors for pooling
        """
        class_counts_columns = ["raw_counts_" + str(c) for c in range(num_classes)]
        pooled_signature_counts = []

        grouped = class_counts.groupby("name")
        for region, group in tqdm(
            grouped, desc="Pooling counts within genomic regions"
        ):
            group = group.reset_index(drop=True)
            n_sigs = len(group)

            sigs_array = group["signature"].values
            class_counts_matrix = group[class_counts_columns].values.astype(float)
            total_reads_array = group["total_reads"].values

            # Compute the distance matrix between pairs of signatures for this region
            dist_mat = np.zeros((n_sigs, n_sigs))
            for i in range(n_sigs):
                for j in range(i + 1, n_sigs):
                    d = self.distance_fct(sigs_array[i], sigs_array[j])
                    dist_mat[i, j] = d
                    dist_mat[j, i] = d

            # Apply pooling
            for i in range(n_sigs):
                accumulated_counts = np.zeros(num_classes)
                accumulated_total = 0
                sigs_pooled = 0

                neighbor_indices = np.argsort(dist_mat[i])

                # Aggregate counts from neighbors (including self) until we reach
                # the minimum read threshold or exceed the distance threshold
                for idx in neighbor_indices:
                    if dist_mat[i, idx] > max_distance:
                        break

                    accumulated_counts += class_counts_matrix[idx]
                    accumulated_total += total_reads_array[idx]
                    sigs_pooled += 1

                    if accumulated_total >= min_reads:
                        break

                pooled_signature_counts.append(
                    {
                        "name": region,
                        "signature": sigs_array[i],
                        "raw_counts_pooled": int(accumulated_total),
                        "num_signatures_pooled": sigs_pooled,
                    }
                    | {
                        "pooled_counts_" + str(c): int(accumulated_counts[c])
                        for c in range(num_classes)
                    }
                )

        return pd.DataFrame(pooled_signature_counts)
