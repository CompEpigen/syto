from typing import List, Optional

import numpy as np
import pandas as pd

# Valid substitution strategies for missing DMR labels.
VALID_SUBSTITUTION_STRATEGIES = (
    "prior_blending",
    "prior_imputation",
    "uniform_number",
)


def _fill_in_missing_labels(
    df: pd.DataFrame,
    group_cols: List[str],
    labels_dict: dict,
    substitution_strategy: str = "uniform_number",
    uniform_prior: Optional[pd.DataFrame] = None,
    prior_weight: float = 1.0,
) -> pd.DataFrame:
    """Fill in missing DMR labels and optionally substitute predictions.

    After inserting synthetic zero rows for any labels in *labels_dict*
    that are absent from *df*, an optional post-processing step is applied
    depending on *substitution_strategy*:

    * ``"prior_blending"`` – blend **every** row with the uniform prior
      using the per-row ``n_reads`` count:
      ``blended = (n / (n + w)) * observed + (w / (n + w)) * prior``
      where *w* = *prior_weight*.  Rows with ``n_reads == 0`` collapse
      entirely to the prior.
    * ``"prior_imputation"`` – replace only rows with ``n_reads == 0``
      with the corresponding prior row; all other rows are untouched.
    * ``"uniform_number"`` - assigns each cell a probability of 1/n_classes (default)

    Parameters
    ----------
    df : pd.DataFrame
        GR-aggregated predictions (output of ``aggregate_predictions_by_grg``).
    group_cols : list[str]
        Columns used for grouping during aggregation.
    labels_dict : dict
        ``{int: str}`` mapping from label id to cell-type name.
    substitution_strategy : str
        One of ``"prior_blending"``, ``"prior_imputation"``, ``"uniform_number"``.
    uniform_prior : pd.DataFrame or None
        Pre-computed uniform prior matrix (required for ``prior_blending``
        and ``prior_imputation``).
    prior_weight : float
        Weight of the prior in the blending formula.  Default ``1.0``.
    """
    if substitution_strategy not in VALID_SUBSTITUTION_STRATEGIES:
        raise ValueError(
            f"Unknown substitution_strategy '{substitution_strategy}'. "
            f"Must be one of {VALID_SUBSTITUTION_STRATEGIES}."
        )
    if substitution_strategy not in ["uniform_number"] and uniform_prior is None:
        raise ValueError(
            f"uniform_prior must be provided when substitution_strategy="
            f"'{substitution_strategy}'."
        )

    labels_dict_pd = pd.DataFrame(labels_dict, index=["dmr_ctype"]).T.reset_index()
    labels_dict_pd.columns = ["dmr_ctype_label", "dmr_ctype"]
    if set(labels_dict_pd["dmr_ctype_label"]).difference(set(df["dmr_ctype_label"])):
        df = pd.merge(
            df,
            labels_dict_pd,
            on=["dmr_ctype_label", "dmr_ctype"],
            how="outer",
            indicator=True,
        )
        synthetic_rows = df["_merge"] == "right_only"
        df.drop(columns=["_merge"], inplace=True)
        fill_columns = [col for col in df.columns if col not in group_cols]
        string_fill_columns = [
            col for col in fill_columns if pd.api.types.is_string_dtype(df[col].dtype)
        ]
        if string_fill_columns:
            df = df.astype({col: object for col in string_fill_columns}, copy=False)
        df.loc[synthetic_rows, "label"] = -1
        df["total_weight"] = df["total_weight"].apply(lambda x: max(x, 1))
        if substitution_strategy == "uniform_number":
            prediction_columns = [x for x in fill_columns if "prediction" in x]
            df.loc[:, prediction_columns] = df.loc[:, prediction_columns].fillna(
                1 / len(labels_dict)
            )  # In this case, each observation has a probability of 1/n_classes
        df.loc[:, fill_columns] = df.loc[:, fill_columns].fillna(
            0
        )  # Filling with zero otherwise + filling n_reads with 0

    # ── Apply substitution strategy ─────────────────────────────────────
    if substitution_strategy in ("prior_blending", "prior_imputation"):
        df = _apply_prior_substitution(
            df, uniform_prior, substitution_strategy, prior_weight
        )

    return df


def fill_in_missing_gr_groups(
    df: pd.DataFrame,
    expected_grg_ids: List[int],
    grg_label_column: str = "dmr_ctype_label",
    n_classes: int = 39,
    substitution_strategy: str = "uniform_number",
    uniform_prior: Optional[pd.DataFrame] = None,
    prior_weight: float = 1.0,
) -> pd.DataFrame:
    """Fill in missing GR groups and optionally substitute predictions.

    This is a simplified v2 of ``_fill_in_missing_labels`` that takes a list
    of expected GR group IDs instead of a labels dict.

    After inserting synthetic zero rows for any GR IDs in *expected_grg_ids*
    that are absent from *df*, a post-processing step is applied depending
    on *substitution_strategy*:

    * ``"prior_blending"`` - blend **every** row with the uniform prior
      using the per-row ``n_reads`` count:
      ``blended = (n / (n + w)) * observed + (w / (n + w)) * prior``
      where *w* = *prior_weight*.  Rows with ``n_reads == 0`` collapse
      entirely to the prior.
    * ``"prior_imputation"`` - replace only rows with ``n_reads == 0``
      with the corresponding prior row; all other rows are untouched.
    * ``"uniform_number"`` - assigns each cell a probability of 1/n_classes.

    Parameters
    ----------
    df : pd.DataFrame
        GR-aggregated predictions (output of ``aggregate_predictions_by_grg``).
    expected_grg_ids : List[int]
        List of GR group IDs that should be present in the output.
    grg_label_column : str
        Column name containing the GR group labels. Default: "dmr_ctype_label".
    n_classes : int
        Number of classes for uniform_number substitution. Default: 39.
    substitution_strategy : str
        One of ``"prior_blending"``, ``"prior_imputation"``, ``"uniform_number"``.
    uniform_prior : pd.DataFrame or None
        Pre-computed uniform prior matrix (required for ``prior_blending``
        and ``prior_imputation``).
    prior_weight : float
        Weight of the prior in the blending formula. Default: 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with all expected GR groups present.
    """
    if substitution_strategy not in VALID_SUBSTITUTION_STRATEGIES:
        raise ValueError(
            f"Unknown substitution_strategy '{substitution_strategy}'. "
            f"Must be one of {VALID_SUBSTITUTION_STRATEGIES}."
        )
    if substitution_strategy not in ["uniform_number"] and uniform_prior is None:
        raise ValueError(
            f"uniform_prior must be provided when substitution_strategy="
            f"'{substitution_strategy}'."
        )

    df = df.copy()

    # Find missing GR IDs
    present_grg_ids = set(df[grg_label_column].unique())
    expected_grg_ids_set = set(expected_grg_ids)
    missing_grg_ids = expected_grg_ids_set - present_grg_ids

    if missing_grg_ids:
        # Create synthetic rows for missing GR IDs
        # Identify columns to fill
        non_group_cols = [col for col in df.columns if col != grg_label_column]
        prediction_cols = [col for col in non_group_cols if "prediction" in col]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        string_cols = df.select_dtypes(include=[object]).columns.tolist()

        synthetic_rows = []
        for grg_id in missing_grg_ids:
            row = {grg_label_column: grg_id}
            # Fill numeric columns with appropriate defaults
            for col in numeric_cols:
                if col == grg_label_column:
                    continue
                if col in prediction_cols:
                    if substitution_strategy == "uniform_number":
                        row[col] = 1.0 / n_classes
                    else:
                        row[col] = 0.0
                elif col == "total_weight":
                    row[col] = 1  # Avoid division by zero
                elif col == "label":
                    row[col] = -1  # Synthetic label
                else:
                    row[col] = 0
            # Fill string columns with empty string
            for col in string_cols:
                row[col] = ""
            synthetic_rows.append(row)

        synthetic_df = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, synthetic_df], ignore_index=True)

    # Sort by GR label for consistent ordering
    df = df.sort_values(grg_label_column).reset_index(drop=True)

    # ── Apply substitution strategy ─────────────────────────────────────
    if substitution_strategy in ("prior_blending", "prior_imputation"):
        df = _apply_prior_substitution(
            df, uniform_prior, substitution_strategy, prior_weight
        )

    return df


def _apply_prior_substitution(
    df: pd.DataFrame,
    uniform_prior: pd.DataFrame,
    strategy: str,
    prior_weight: float,
) -> pd.DataFrame:
    """Apply prior-based substitution to prediction columns.

    Parameters
    ----------
    df : pd.DataFrame
        GR-aggregated DataFrame (with synthetic rows already inserted).
    uniform_prior : pd.DataFrame
        Uniform prior matrix keyed by ``dmr_ctype_label``.
    strategy : str
        ``"prior_blending"`` or ``"prior_imputation"``.
    prior_weight : float
        Weight of the prior in the blending formula.
    """
    # Identify prediction columns present in both df and prior
    pred_cols = [
        c
        for c in df.columns
        if (c.startswith("prediction_") and (c.endswith("_wavg") or c.endswith("_avg")))
        or c == "methylation_level_wavg"
        or c == "methylation_level_avg"
    ]
    prior_cols = [c for c in pred_cols if c in uniform_prior.columns]

    if not prior_cols:
        raise ValueError(
            "No matching prediction columns found between the aggregated "
            "DataFrame and the uniform prior."
        )

    # Build a lookup from dmr_ctype_label -> prior row values
    prior_indexed = uniform_prior.set_index("dmr_ctype_label")

    if strategy == "prior_blending":
        # Blend every row:  (n / (n + w)) * observed + (w / (n + w)) * prior
        n_reads = df["n_reads"].values.astype(float)
        alpha = n_reads / (n_reads + prior_weight)  # weight for observed

        for col in prior_cols:
            prior_values = (
                df["dmr_ctype_label"]
                .map(
                    prior_indexed[col]
                    if col in prior_indexed.columns
                    else pd.Series(dtype=float)
                )
                .values.astype(float)
            )
            observed = df[col].values.astype(float)
            df[col] = alpha * observed + (1.0 - alpha) * prior_values

    elif strategy == "prior_imputation":
        # Replace only rows where n_reads == 0
        zero_mask = df["n_reads"] == 0
        if zero_mask.any():
            for col in prior_cols:
                prior_values = df.loc[zero_mask, "dmr_ctype_label"].map(
                    prior_indexed[col]
                    if col in prior_indexed.columns
                    else pd.Series(dtype=float)
                )
                df.loc[zero_mask, col] = prior_values.values

    return df


def aggregate_predictions_by_grg(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    prediction_cols: Optional[List[str]] = None,
    weight_col: str = "total_marked_cpgs",
    create_weight_from_cpgs: bool = True,
    fill_in_missing_labels: bool = False,
    labels_dict: dict = None,
    substitution_strategy: str = "uniform_number",
    uniform_prior: Optional[pd.DataFrame] = None,
    prior_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Aggregate predictions for each class across all reads at the DMR level.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with read-level predictions.
    group_cols : List[str]
        Columns to group by for aggregation. Default: ['dmr_label', 'file', 'original_label']
    prediction_cols : Optional[List[str]]
        List of prediction column names. If None, auto-detects columns starting with 'prediction_'.
    weight_col : str
        Column name to use for weighted aggregation. Default: 'total_marked_cpgs'
    create_weight_from_cpgs : bool
        If True and weight_col doesn't exist, creates it from methylated_CpGs + unmethylated_CpGs.
    fill_in_missing_labels: bool
        If True, fills in additional rows from labels_dict labels not presented in the data. All predictions in
        these rows will be zero
    labels_dict: dict
        Must be provided if fill_in_missing_labels is set to True
    substitution_strategy : str
        Strategy for filling missing labels.  One of ``"uniform_number"``,
        ``"prior_blending"``, ``"prior_imputation"``.  Default ``"uniform_number"``.
    uniform_prior : pd.DataFrame or None
        Pre-computed uniform prior matrix.  Required when
        *substitution_strategy* is not ``"uniform_number"``.
    prior_weight : float
        Weight of the prior in the blending formula.  Default ``1.0``.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with both simple average and weighted average predictions.
    """

    df = df.copy()
    if group_cols is None:
        group_cols = ["dmr_label", "file", "original_label"]
    if fill_in_missing_labels:
        if labels_dict is None:
            raise ValueError(
                "labels_dict must be provided when fill_in_missing_labels is set to True"
            )

    if prediction_cols is None:
        prediction_cols = [
            col
            for col in df.columns
            if col.startswith("prediction_") and col[11:].isdigit()
        ]
        prediction_cols = sorted(prediction_cols, key=lambda x: int(x.split("_")[1]))
        prediction_cols.append("methylation_level")

    # Create weight column if needed
    if weight_col not in df.columns and create_weight_from_cpgs:
        if "methylated_CpGs" in df.columns and "unmethylated_CpGs" in df.columns:
            df[weight_col] = df["methylated_CpGs"] + df["unmethylated_CpGs"]
        else:
            raise ValueError(
                f"Weight column '{weight_col}' not found and cannot create from CpG columns."
            )

    # Handle case where weights might be 0 (avoid division by zero)
    # (the explicit conversion to float is necessary to avoid silent downcasting to int
    # after clipping which would make the weights 0 again)
    df["_weight"] = df[weight_col].astype(float).clip(lower=1e-10)

    # Simple average aggregation
    simple_avg = df.groupby(group_cols)[prediction_cols].mean()
    simple_avg.columns = [f"{col}_avg" for col in simple_avg.columns]

    # Weighted average aggregation
    def weighted_average(group):
        weights = group["_weight"].values
        total_weight = weights.sum()

        result = {}
        for col in prediction_cols:
            values = group[col].values
            result[f"{col}_wavg"] = np.average(values, weights=weights)

        result["total_weight"] = total_weight
        result["n_reads"] = len(group)
        return pd.Series(result)

    weighted_avg = df.groupby(group_cols).apply(weighted_average, include_groups=False)
    # Combine results
    result = simple_avg.join(weighted_avg)
    result.reset_index(inplace=True)

    if fill_in_missing_labels:
        result = _fill_in_missing_labels(
            result,
            group_cols,
            labels_dict,
            substitution_strategy=substitution_strategy,
            uniform_prior=uniform_prior,
            prior_weight=prior_weight,
        )

    return result


def aggregate_by_grg_from_np_arrays(
    read_ids: np.ndarray,
    gr_group_idx_array: np.ndarray,
    weight_array: np.ndarray,
    pred_matrix: np.ndarray,
    n_gr_groups: int,
    n_pred_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultra-fast aggregation working directly on numpy arrays.

    This function avoids DataFrame creation entirely by operating on pre-extracted
    numpy arrays. It's designed to be called from pseudobulk generation where the
    source arrays are pre-computed once.

    Parameters
    ----------
    read_ids : np.ndarray
        Array of row indices to sample (shape: n_samples,).
    gr_group_idx_array : np.ndarray
        Full GR group indices array from source DataFrame (shape: n_rows,).
    weight_array : np.ndarray
        Full weights array from source DataFrame (shape: n_rows,).
    pred_matrix : np.ndarray
        Full prediction matrix from source DataFrame (shape: n_rows, n_pred_cols).
        Should be Fortran-order (column-major) for faster column access.
    n_gr_groups : int
        Number of unique GR groups (max GR group index + 1).
    n_pred_cols : int
        Number of prediction columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - weighted_avgs: shape (n_gr_groups, n_pred_cols)
        - counts: shape (n_gr_groups,)
        - total_weights: shape (n_gr_groups,)
    """
    # Slice 1D arrays (fast)
    sampled_groups = gr_group_idx_array[read_ids]
    raw_weights = weight_array[read_ids]
    sampled_weights = np.maximum(raw_weights.astype(np.float64), 1e-10)

    # Pre-compute aggregations for counts/weights
    weight_sums = np.bincount(
        sampled_groups, weights=sampled_weights, minlength=n_gr_groups
    )
    counts = np.bincount(sampled_groups, minlength=n_gr_groups)
    total_weights = np.bincount(
        sampled_groups, weights=raw_weights.astype(np.float64), minlength=n_gr_groups
    )

    # Pre-compute inverse for faster division
    inv_weight_sums = 1.0 / np.where(weight_sums > 0, weight_sums, 1.0)

    # For predictions, use indirect indexing per column to avoid full 2D slice
    weighted_avgs = np.empty((n_gr_groups, n_pred_cols), dtype=np.float64)
    for i in range(n_pred_cols):
        pred_col = pred_matrix[read_ids, i]  # Slice one column at a time
        weighted_sum = np.bincount(
            sampled_groups, weights=pred_col * sampled_weights, minlength=n_gr_groups
        )
        weighted_avgs[:, i] = weighted_sum * inv_weight_sums

    return weighted_avgs, counts, total_weights


def aggregate_chuncked_predictions_weighted(
    pred_df, weight_col="ncpgs_marked", group_col="read_name"
):
    """
    Calculates the weighted average of prediction columns grouped by read_name.
    """
    # 1. Identify prediction columns (prediction_0 ... prediction_39)
    pred_cols = [
        c for c in pred_df.columns if c.startswith("prediction_") and c[11:].isdigit()
    ]

    # 2. Create a working copy to avoid SettingWithCopy warnings
    df = pred_df.copy()

    # 3. Vectorized Weighting: Multiply all prediction columns by the weight column
    # This is much faster than doing it inside the groupby
    df[pred_cols] = df[pred_cols].multiply(df[weight_col], axis=0)

    # 4. Group by read_name and sum both the weighted predictions and the weights
    grouped = df.groupby(group_col)

    # Sum the weighted scores
    summed_preds = grouped[pred_cols].sum()

    # Sum the weights (the total number of CpGs seen across all chunks for this read)
    summed_weights = grouped[weight_col].sum()

    # 5. Divide to get the Weighted Average
    # We use .div with axis=0 to align by the index (read_name)
    final_averaged_df = summed_preds.div(summed_weights, axis=0)

    # 6. Cleanup
    # Handle cases where total weight might be 0 (avoid division by zero errors)
    # Though your filtering likely prevents this, it's good practice.
    final_averaged_df = final_averaged_df.fillna(0)

    # Reset index so 'read_name' becomes a column again
    return final_averaged_df.reset_index()


def get_final_prediction(
    aggregated_df: pd.DataFrame,
    methods: Optional[List[str]] = None,
    prediction_prefix: str = "prediction_",
) -> pd.DataFrame:
    """
    Get final class prediction from aggregated probabilities.

    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Output from aggregate_predictions_by_grg()
    method : str
        'avg' for simple average or 'wavg' for weighted average
    prediction_prefix : str
        Prefix for prediction columns

    Returns
    -------
    pd.DataFrame
        DataFrame with added final_prediction and final_confidence columns.
    """

    df = aggregated_df.copy()
    if methods is None:
        methods = ["avg", "wavg"]
    for method in methods:
        suffix = f"_{method}"

        # Find relevant prediction columns
        pred_cols = [
            col
            for col in df.columns
            if col.startswith(prediction_prefix)
            and col.endswith(suffix)
            and col.replace(prediction_prefix, "").replace(suffix, "").isdigit()
        ]
        pred_cols = sorted(
            pred_cols,
            key=lambda x: int(x.replace(prediction_prefix, "").replace(suffix, "")),
        )

        if not pred_cols:
            raise ValueError(f"No prediction columns found with suffix '{suffix}'")

        # Get predictions as array
        pred_array = df[pred_cols].values

        # Final prediction is argmax, confidence is max probability
        df[f"final_prediction_{method}"] = pred_array.argmax(axis=1)
        df[f"final_confidence_{method}"] = pred_array.max(axis=1)

    return df
