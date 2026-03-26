from typing import List, Optional

import numpy as np
import pandas as pd


def aggregate_predictions_by_dmr(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    prediction_cols: Optional[List[str]] = None,
    weight_col: str = "total_marked_cpgs",
    create_weight_from_cpgs: bool = True,
    fill_in_missing_labels: bool = False,
    labels_dict: dict = None,
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

    # Auto-detect prediction columns if not provided
    if prediction_cols is None:
        prediction_cols = [
            col
            for col in df.columns
            if col.startswith("prediction_") and col != "prediction"
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

    # Add additional metadata columns (take first value per group)
    metadata_cols = [
        col
        for col in df.columns
        if col not in prediction_cols + group_cols + ["_weight", weight_col]
        and col in ["ctype", "dmr_ctype", "label", "chromosome"]
    ]

    if metadata_cols:
        metadata = df.groupby(group_cols)[metadata_cols].first()
        result = result.join(metadata)
    result.reset_index(inplace=True)

    if fill_in_missing_labels:
        labels_dict_pd = pd.DataFrame(labels_dict, index=["dmr_ctype"]).T.reset_index()
        labels_dict_pd.columns = ["dmr_ctype_label", "dmr_ctype"]
        if set(labels_dict_pd["dmr_ctype_label"]).difference(
            set(result["dmr_ctype_label"])
        ):
            result = pd.merge(
                result,
                labels_dict_pd,
                on=["dmr_ctype_label", "dmr_ctype"],
                how="outer",
                indicator=True,
            )
            synthetic_rows = result["_merge"] == "right_only"
            result.drop(columns=["_merge"], inplace=True)
            fill_columns = [col for col in result.columns if col not in group_cols]
            string_fill_columns = [
                col
                for col in fill_columns
                if pd.api.types.is_string_dtype(result[col].dtype)
            ]
            if string_fill_columns:
                result = result.astype(
                    {col: object for col in string_fill_columns}, copy=False
                )
            result.loc[:, fill_columns] = result.loc[:, fill_columns].fillna(0)
            result.loc[synthetic_rows, "label"] = -1
            result["total_weight"] = result["total_weight"].apply(lambda x: max(x, 1))

    return result


def aggregate_predictions_by_dmr_optimized(df, group_cols):
    """Optimized aggregation using vectorized operations."""

    prediction_cols = [
        col
        for col in df.columns
        if col.startswith("prediction_") and col != "prediction"
    ]
    prediction_cols = sorted(prediction_cols, key=lambda x: int(x.split("_")[1]))
    prediction_cols.append("methylation_level")

    weight_col = "total_marked_cpgs"
    df = df.copy()
    if weight_col not in df.columns:
        if "methylated_CpGs" in df.columns and "unmethylated_CpGs" in df.columns:
            df[weight_col] = df["methylated_CpGs"] + df["unmethylated_CpGs"]
        else:
            raise ValueError(f"Weight column '{weight_col}' not found")

    # Clip weights to avoid division by zero
    # (the explicit conversion to float is necessary to avoid silent downcasting to int
    # after clipping which would make the weights 0 again)
    df["_weight"] = df[weight_col].astype(float).clip(lower=1e-10)
    # Pre-compute weighted values for each prediction column
    for col in prediction_cols:
        df[f"_weighted_{col}"] = df[col] * df["_weight"]
    # Build aggregation dictionary

    agg_dict = {}
    # Sum of weighted values and weights
    for col in prediction_cols:
        agg_dict[f"_weighted_{col}_sum"] = (f"_weighted_{col}", "sum")
    agg_dict["_weight_sum"] = ("_weight", "sum")
    agg_dict["n_reads"] = ("_weight", "count")
    agg_dict["total_weight"] = (weight_col, "sum")

    # Metadata columns - take first value
    metadata_cols = [col for col in ["label", "chromosome"] if col in df.columns]
    for col in metadata_cols:
        agg_dict[col] = (col, "first")

    # Perform aggregation
    grouped = df.groupby(group_cols, sort=True)
    result = grouped.agg(**agg_dict).reset_index()

    # Compute weighted averages from sums
    for col in prediction_cols:
        result[f"{col}_wavg"] = result[f"_weighted_{col}_sum"] / result["_weight_sum"]
        result.drop(columns=[f"_weighted_{col}_sum"], inplace=True)

    result.drop(columns=["_weight_sum"], inplace=True)
    return result


def aggregate_chuncked_predictions_weighted(
    pred_df, weight_col="ncpgs_marked", group_col="read_name"
):
    """
    Calculates the weighted average of prediction columns grouped by read_name.
    """
    # 1. Identify prediction columns (prediction_0 ... prediction_39)
    pred_cols = [c for c in pred_df.columns if c.startswith("prediction_")]

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
        Output from aggregate_predictions_by_dmr()
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
            if col.startswith(prediction_prefix) and col.endswith(suffix)
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
