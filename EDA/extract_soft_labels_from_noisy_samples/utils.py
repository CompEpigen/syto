import os
from typing import Union, Dict
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


def nrmse(y_true, y_pred, return_per_sample=False) -> Union[float, np.ndarray]:
    """
    Compute the NRMSE (Normalized Root Mean Squared Error) between the true and predicted probability distributions,
    for all sample in y_pred.
    The normalization is done by dividing the RMSE by the worst possible RMSE for each sample.
    We return the average NRMSE over all samples.

    It is expected that all rows of y_true and y_pred sum to 1, and that they have the same shape.

    Args:
        y_true (np.ndarray): True probability distributions, shape (n_samples, n_classes)
        y_pred (np.ndarray): Predicted probability distributions, shape (n_samples, n_classes)
        return_per_sample (bool): Whether to return the NRMSE per sample, or the average NRMSE over all samples. Default is False.
    Returns:
        float: The average NRMSE over all samples, or the NRMSE per sample
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    assert y_true.ndim == 2, "y_true and y_pred must be 2D arrays"
    rmse_per_sample = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    # the worst possible prediction is the one that predicts 1 for the class
    # with the lowest probability in y_true for each sample, and 0 for all other classes. This is equivalent to predicting the argmin of y_true for each sample.
    worst_possible_prediction_per_sample = np.zeros_like(y_true)
    worst_possible_prediction_per_sample[
        np.arange(y_true.shape[0]), y_true.argmin(axis=1)
    ] = 1
    worst_possible_rmse = np.sqrt(
        np.mean((y_true - worst_possible_prediction_per_sample) ** 2, axis=1)
    )
    nrmse_per_sample = rmse_per_sample / worst_possible_rmse
    if return_per_sample:
        return nrmse_per_sample
    avg_nrmse = np.mean(nrmse_per_sample)
    return avg_nrmse


def compute_counts_sig_per_ctype_from_dataset(
    dataset, sig_positions, selected_ctypes_, n_ctypes
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Compute the counts of signatures per ctype from a given dataset.

    Returns:
        counts_sig_per_ctype: Dict[int, Dict[int, np.ndarray]]
            A dictionary where the keys are the start positions of the signatures,
            and the values are dictionaries where the keys are the read lengths and
            the values are 2D numpy arrays of shape (n_ctypes, 2**read_length) containing
            the counts of each signature for each ctype.
    """
    counts_sig_per_ctype = {}
    for start, read_lengths in sig_positions.items():
        counts_sig_per_ctype[start] = {}
        for read_length in read_lengths:
            counts_sig_per_ctype[start][read_length] = np.zeros(
                (n_ctypes, 2**read_length), dtype=int
            )
            for ctype_idx, ctype in enumerate(selected_ctypes_):
                pattern_counts = dataset[ctype][read_length][start]
                for pattern, n_reads in pattern_counts.items():
                    pattern_idx = get_idx_from_sig(pattern)
                    counts_sig_per_ctype[start][read_length][
                        ctype_idx, pattern_idx
                    ] += n_reads
    return counts_sig_per_ctype


def extract_complete_dataset(
    reads_ds: Dict[str, Dict[int, Dict[int, Dict[str, int]]]],
) -> Dict[str, Dict[int, Dict[int, Dict[str, int]]]]:
    """
    Create a complete dataset from a given reads_ds dataset
    by splitting long patterns into all possible subpatterns and storing the counts.
    """
    complete_dataset = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )
    for ctype, length_dict in reads_ds.items():
        for read_length in sorted(length_dict.keys()):
            for start, pattern_counts in length_dict[read_length].items():
                for pattern, n_reads in pattern_counts.items():
                    # add the original pattern to the complete dataset
                    complete_dataset[ctype][len(pattern)][start][pattern] += n_reads
                    # split the pattern into all possible subpatterns
                    for sub_length in range(1, len(pattern)):
                        for sub_start in range(len(pattern) - sub_length + 1):
                            subpattern = pattern[sub_start : sub_start + sub_length]
                            if len(subpattern) == 0:
                                continue  # skip empty subpatterns
                            complete_dataset[ctype][len(subpattern)][start + sub_start][
                                subpattern
                            ] += n_reads
    return complete_dataset


def compute_num_reads_in_dataset(
    dataset: Dict[str, Dict[int, Dict[int, Dict[str, int]]]],
) -> int:
    """
    Compute the total number of reads in a given dataset
    """
    total_reads = 0
    for ctype, length_dict in dataset.items():
        for read_length, start_dict in length_dict.items():
            for start, pattern_counts in start_dict.items():
                total_reads += sum(pattern_counts.values())
    return total_reads


def generate_bootstrap_sample_from_dataset(
    source_dataset: Dict[str, Dict[int, Dict[int, Dict[str, int]]]],
    ratio: float = None,
    dataset_size_to_sample: int = None,
) -> Dict[str, Dict[int, Dict[int, Dict[str, int]]]]:
    """
    Generate a bootstrap sample from a given dataset by sampling with replacement the reads.
    """
    assert (
        dataset_size_to_sample is not None or ratio is not None
    ), "Either dataset_size_to_sample or ratio must be provided"
    size_source_dataset = compute_num_reads_in_dataset(source_dataset)
    if ratio is not None:
        assert 0 < ratio, "Ratio must be > 0 (can be > 1 for oversampling)"
        dataset_size_to_sample = int(size_source_dataset * ratio)

    # create a list of all reads in the source dataset
    unique_reads = []
    counts_per_read = []
    for ctype, length_dict in source_dataset.items():
        for read_length, start_dict in length_dict.items():
            for start, pattern_counts in start_dict.items():
                for pattern, n_reads in pattern_counts.items():
                    unique_reads.append((ctype, read_length, start, pattern))
                    counts_per_read.append(n_reads)
    counts_per_read = np.array(counts_per_read)

    # sample with replacement the reads
    sampled_indices = np.random.choice(
        len(unique_reads),
        size=dataset_size_to_sample,
        replace=True,
        p=counts_per_read / counts_per_read.sum(),
    )
    sampled_reads = [unique_reads[i] for i in sampled_indices]

    # create the bootstrap sample dataset
    bootstrap_sample_dataset = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )
    for ctype, read_length, start, pattern in sampled_reads:
        bootstrap_sample_dataset[ctype][read_length][start][pattern] += 1
    return bootstrap_sample_dataset


def compute_pgt_sig_prior_from_counts(counts_sig_per_ctype, sig_pos_):
    """
    Compute the prior probabilities of signatures per ctype from the counts of signatures.
    """
    pgt_sig_prior = {}
    for start, read_lengths in sig_pos_.items():
        pgt_sig_prior[start] = {}
        for read_length in read_lengths:
            counts_sig_per_ctype_ = counts_sig_per_ctype[start][read_length]
            pgt_sig_prior[start][read_length] = (
                counts_sig_per_ctype_.sum(axis=0) / counts_sig_per_ctype_.sum()
            )
    return pgt_sig_prior


def compute_nrmse_per_sig_length(
    predicted_proba_sig_per_ctype: Dict[int, Dict[int, np.ndarray]],
    true_proba_sig_per_ctype: Dict[int, Dict[int, np.ndarray]],
    counts_sig_per_ctype: Dict[int, Dict[int, np.ndarray]] = None,
    min_count_threshold_per_ctype: int = None,
):
    """
    Compute the normalized root mean square error (NRMSE) between the predicted and true
    probabilities of signatures per ctype,
    for a whole set of signature positions (ie start and read_length combinations).
    The average is computed per signature length, to avoid confusion.

    It is possible to provide a counts_sig_per_ctype and a min_count_threshold_per_ctype
    to filter out the signature positions with too low counts
    for a given ctype.
    """
    nrmse_list_per_length = defaultdict(list)
    num_sig_ctype_pairs_considered_per_length = defaultdict(int)
    for start, read_lengths in predicted_proba_sig_per_ctype.items():
        for read_length, predicted_proba in read_lengths.items():
            true_proba = true_proba_sig_per_ctype[start][read_length]
            if min_count_threshold_per_ctype is not None:
                if counts_sig_per_ctype is None:
                    raise ValueError(
                        "If min_count_threshold_per_ctype is provided, counts_sig_per_ctype must also be provided"
                    )
                tmp_ctype_counts = counts_sig_per_ctype[start][read_length].sum(axis=1)
                ctype_mask = tmp_ctype_counts >= min_count_threshold_per_ctype
                predicted_proba = predicted_proba[ctype_mask]
                true_proba = true_proba[ctype_mask]
            # Skip computing NRMSE if there's no data after filtering
            if len(predicted_proba) == 0:
                continue
            nrmse_ = nrmse(true_proba, predicted_proba, return_per_sample=True)
            nrmse_list_per_length[read_length].extend(nrmse_)
            num_sig_ctype_pairs_considered_per_length[read_length] += len(
                predicted_proba
            )
    avg_nrmse_per_length = {
        read_length: np.mean(nrmse_list)
        for read_length, nrmse_list in nrmse_list_per_length.items()
        if len(nrmse_list) > 0
    }
    return avg_nrmse_per_length, num_sig_ctype_pairs_considered_per_length


def compute_nrmse_per_sig_length_ctype_given_sig(
    predicted_proba_ctype_given_sig: Dict[int, Dict[int, np.ndarray]],
    true_proba_ctype_given_sig: Dict[int, Dict[int, np.ndarray]],
    counts_sig_per_ctype: Dict[int, Dict[int, np.ndarray]] = None,
    min_count_threshold_per_signature: int = None,
    max_count_threshold_per_signature: int = None,
):
    """
    Compute NRMSE between predicted and true P(ctype | signature) distributions.

    (AVOID USE OUTSIDE OF THE CONTEXT OF THE NOTEBOOK WHERE IT IS USED)

    predicted_proba_ctype_given_sig[start][read_length] and
    true_proba_ctype_given_sig[start][read_length] must have shape
    (2**read_length, n_ctypes).

    Optionally, signatures with total counts below min_count_threshold_per_signature
    or above max_count_threshold_per_signature are filtered out before the NRMSE computation.
    """
    nrmse_list_per_length = defaultdict(list)
    num_sigs_considered_per_length = defaultdict(int)
    for start, read_lengths in predicted_proba_ctype_given_sig.items():
        for read_length, predicted_proba in read_lengths.items():
            true_proba = true_proba_ctype_given_sig[start][read_length]

            if (
                min_count_threshold_per_signature is not None
                or max_count_threshold_per_signature is not None
            ):
                if counts_sig_per_ctype is None:
                    raise ValueError(
                        "If min_count_threshold_per_signature or max_count_threshold_per_signature is provided, counts_sig_per_ctype must also be provided"
                    )
                tmp_sig_counts = counts_sig_per_ctype[start][read_length].sum(axis=0)
                if min_count_threshold_per_signature is not None:
                    sig_mask_min = tmp_sig_counts >= min_count_threshold_per_signature
                else:
                    sig_mask_min = np.ones_like(tmp_sig_counts, dtype=bool)
                if max_count_threshold_per_signature is not None:
                    sig_mask_max = tmp_sig_counts <= max_count_threshold_per_signature
                else:
                    sig_mask_max = np.ones_like(tmp_sig_counts, dtype=bool)
                sig_mask = sig_mask_min & sig_mask_max
                predicted_proba = predicted_proba[sig_mask]
                true_proba = true_proba[sig_mask]

            if len(predicted_proba) == 0:
                continue

            nrmse_ = nrmse(true_proba, predicted_proba, return_per_sample=True)
            nrmse_list_per_length[read_length].extend(nrmse_)
            num_sigs_considered_per_length[read_length] += len(predicted_proba)

    avg_nrmse_per_length = {
        read_length: np.mean(nrmse_list)
        for read_length, nrmse_list in nrmse_list_per_length.items()
        if len(nrmse_list) > 0
    }
    return avg_nrmse_per_length, num_sigs_considered_per_length
