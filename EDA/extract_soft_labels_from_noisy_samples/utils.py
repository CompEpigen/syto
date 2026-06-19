import os
from collections import defaultdict

import json
import numpy as np

global df_ctype_match
f_cell_type_match = "./data/files_to_cell_groups.json"
with open(f_cell_type_match, "r") as fp:
    df_ctype_match = json.load(fp)


def filename2ctype(f_name: str) -> str:
    """

    Extract cell types from the file name
    It returns the cell type matching to the cell types written in the region information
    e.g.,) Blood-Monocytes -> Blood-Mono+Macro

    """

    f_name = os.path.basename(f_name)
    f_name_short = "_".join(f_name.split("_")[1:]).split(".hg38")[0]
    return df_ctype_match.get(f_name_short)

    # try:
    #     return df_ctype_match[f_name_short]
    # except KeyError:
    #     print(f_name_short)
    #     print(f"{f_name} cannot be matched to the 39 cell types. NA is assigned for the cell type")
    #     # print(f_name, "---->", file_ctype)
    #     return "NA"


def nested_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: nested_defaultdict_to_dict(v) for k, v in d.items()}
    return d


# utilities to manipulate signatures as binary vectors and convert them to indices for easier handling
def get_idx_from_sig(sig):
    if isinstance(sig, str):
        sig = [int(bit) for bit in sig]
    idx = 0
    for i, bit in enumerate(sig):
        idx += bit * (2**i)
    return idx


def get_sig_from_idx(idx, sig_length, return_str=False):
    assert idx < 2**sig_length, "Index is too large for the given signature length"
    sig = []
    for i in range(sig_length):
        bit = (idx // (2**i)) % 2
        sig.append(bit)
    if return_str:
        return "".join(map(str, sig))
    return np.array(sig)


def nrmse(y_true, y_pred) -> float:
    """
    Compute the NRMSE (Normalized Root Mean Squared Error) between the true and predicted probability distributions,
    for all sample in y_pred.
    The normalization is done by dividing the RMSE by the worst possible RMSE for each sample.
    We return the average NRMSE over all samples.

    It is expected that all rows of y_true and y_pred sum to 1, and that they have the same shape.

    Args:
        y_true (np.ndarray): True probability distributions, shape (n_samples, n_classes)
        y_pred (np.ndarray): Predicted probability distributions, shape (n_samples, n_classes)
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    assert y_true.ndim == 2, "y_true and y_pred must be 2D arrays"
    rmse_per_sample = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    # the worst possible prediction is the one that predicts 1 for the class
    # with the lowest probability in y_true for each sample, and 0 for all other classes. This is equivalent to predicting the argmin of y_true for each sample.
    worst_possible_prediction_per_sample = np.zeros_like(y_true)
    worst_possible_prediction_per_sample[np.arange(y_true.shape[0]), y_true.argmin(axis=1)] = 1
    worst_possible_rmse = np.sqrt(np.mean((y_true - worst_possible_prediction_per_sample) ** 2, axis=1))
    avg_nrmse = np.mean(rmse_per_sample / worst_possible_rmse)
    return avg_nrmse