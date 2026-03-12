
import multiprocessing as mp
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

from methyldl.deconvolution.uxm import (
    rearange_uxm_deconvolution_results,
    uxm_deconvolution,
)
from methyldl.inference.prediction_aggregation import (
    aggregate_predictions_by_dmr,
    aggregate_predictions_by_dmr_optimized,
)

# Global variables for worker processes (initialized once per worker)
_worker_data = {}

# TODO: make sure that reference cells are matching labels --> probably need to be reordered
def generate_pseudo_bulk(
    total_samples,
    labels,
    proportions,
    train_data,
    valid_data,
    test_data,
    atlas,
    ref_cells,
    labels_dict_reversed,
    return_reads=False,
):
    ref_pos = np.array(
        [
            labels_dict_reversed[cell] if cell in labels_dict_reversed.keys() else -1
            for cell in ref_cells
        ]
    )
    assert np.round(np.sum(proportions), 4) == 1, "Proportions must sum up to one"
    n_samples_list = [int(total_samples * x) for x in proportions]
    target_columns = [
        "dmr_ctype_label",
        "dmr_ctype",
        "prediction_0_wavg",
        "prediction_1_wavg",
        "prediction_2_wavg",
        "prediction_3_wavg",
        "prediction_4_wavg",
        "prediction_5_wavg",
        "prediction_6_wavg",
        "prediction_7_wavg",
        "prediction_8_wavg",
        "prediction_9_wavg",
        "prediction_10_wavg",
        "prediction_11_wavg",
        "prediction_12_wavg",
        "prediction_13_wavg",
        "prediction_14_wavg",
        "prediction_15_wavg",
        "prediction_16_wavg",
        "prediction_17_wavg",
        "prediction_18_wavg",
        "prediction_19_wavg",
        "prediction_20_wavg",
        "prediction_21_wavg",
        "prediction_22_wavg",
        "prediction_23_wavg",
        "prediction_24_wavg",
        "prediction_25_wavg",
        "prediction_26_wavg",
        "prediction_27_wavg",
        "prediction_28_wavg",
        "prediction_29_wavg",
        "prediction_30_wavg",
        "prediction_31_wavg",
        "prediction_32_wavg",
        "prediction_33_wavg",
        "prediction_34_wavg",
        "prediction_35_wavg",
        "prediction_36_wavg",
        "prediction_37_wavg",
        "prediction_38_wavg",
        "prediction_39_wavg",
        "methylation_level_wavg",
        "total_weight",
        "n_reads",
        "chromosome",
        "label",
    ]
    subs = []
    uxm_data = []
    reads = []
    sample_name = "pseudo_balk_sample"
    for df in [train_data, valid_data, test_data]:
        df["total_marked_cpgs"] = df["NCPGS"]
        df["methylation_level"] = df["M_rate"]
        df.rename(columns={"chr": "chromosome"}, inplace=True)
        df["direction"] = "U"
        grouped = df.groupby(["original_label", "dmr_ctype_label"], sort=False)
        sub = []
        for n, label in zip(n_samples_list, labels):
            for dmr_ctype_label in range(39):
                group = grouped.get_group((label, dmr_ctype_label))
                sub.append(group.sample(int(n / 39), replace=True))
        sub = pd.concat(sub)
        sub_aggregated = aggregate_predictions_by_dmr(
            sub, group_cols=["dmr_ctype_label", "dmr_ctype"]
        )
        sub_aggregated = sub_aggregated[target_columns]
        subs.append(sub_aggregated)
        results = sub
        results_agg = (
            results.groupby(["name", "direction"])
            .aggregate({"record_M": "sum", "record_U": "sum", "record_X": "sum"})
            .reset_index()
        )
        results_agg["count"] = (
            results_agg["record_M"] + results_agg["record_U"] + results_agg["record_X"]
        )
        results_agg["sf"] = results_agg["record_U"] / results_agg["count"]
        sf = deepcopy(results_agg[["name", "direction"]])
        sf[sample_name] = results_agg["sf"]
        counts = results_agg[["name", "direction", "count"]]
        counts.columns = ["name", "direction", sample_name]
        uxm_proportions = uxm_deconvolution(
            atlas, ref_cells, sf, counts, sample_names=[sample_name]
        )[0]
        uxm_deconv_results = {
            x: np.round(y, 4) for (x, y) in zip(ref_cells, uxm_proportions)
        }
        uxm_deconv_results_alligned = rearange_uxm_deconvolution_results(
            labels_dict_reversed, uxm_proportions, ref_cells
        )
        uxm_data.append((sf, counts, uxm_deconv_results, uxm_deconv_results_alligned))
        if return_reads:
            reads.append(sub)

    proportions_dict = {x: y for x, y in zip(labels, proportions)}
    proportions_full = [proportions_dict.get(x, 0) for x in range(39)]
    if return_reads:
        return labels, proportions_full, subs, uxm_data, reads
    return labels, proportions_full, subs, uxm_data





def init_worker(
    train_data, valid_data, test_data, allowed_labels, n_cells_max, n_read_per_split
):
    """Initialize worker process with shared data and pre-computed groups."""

    global _worker_data

    # Set unique random seed per process

    seed = mp.current_process().pid

    random.seed(seed)

    np.random.seed(seed)

    for df in [train_data, valid_data, test_data]:
        df["total_marked_cpgs"] = df["NCPGS"]
        df["methylation_level"] = df["M_rate"]
        df.rename(columns={"chr": "chromosome"}, inplace=True)
        df["direction"] = "U"
    # Pre-compute grouped dataframes (expensive operation done once per worker)

    _worker_data["grouped_train"] = train_data.groupby(
        ["original_label", "dmr_ctype_label"], sort=False
    )

    _worker_data["grouped_valid"] = valid_data.groupby(
        ["original_label", "dmr_ctype_label"], sort=False
    )

    _worker_data["grouped_test"] = test_data.groupby(
        ["original_label", "dmr_ctype_label"], sort=False
    )

    _worker_data["allowed_labels"] = allowed_labels

    _worker_data["n_cells_max"] = n_cells_max

    _worker_data["n_read_per_split"] = n_read_per_split

    _worker_data["target_columns"] = [
        "dmr_ctype_label",
        "dmr_ctype",
        "prediction_0_wavg",
        "prediction_1_wavg",
        "prediction_2_wavg",
        "prediction_3_wavg",
        "prediction_4_wavg",
        "prediction_5_wavg",
        "prediction_6_wavg",
        "prediction_7_wavg",
        "prediction_8_wavg",
        "prediction_9_wavg",
        "prediction_10_wavg",
        "prediction_11_wavg",
        "prediction_12_wavg",
        "prediction_13_wavg",
        "prediction_14_wavg",
        "prediction_15_wavg",
        "prediction_16_wavg",
        "prediction_17_wavg",
        "prediction_18_wavg",
        "prediction_19_wavg",
        "prediction_20_wavg",
        "prediction_21_wavg",
        "prediction_22_wavg",
        "prediction_23_wavg",
        "prediction_24_wavg",
        "prediction_25_wavg",
        "prediction_26_wavg",
        "prediction_27_wavg",
        "prediction_28_wavg",
        "prediction_29_wavg",
        "prediction_30_wavg",
        "prediction_31_wavg",
        "prediction_32_wavg",
        "prediction_33_wavg",
        "prediction_34_wavg",
        "prediction_35_wavg",
        "prediction_36_wavg",
        "prediction_37_wavg",
        "prediction_38_wavg",
        "prediction_39_wavg",
        "methylation_level_wavg",
        "total_weight",
        "n_reads",
        "chromosome",
        "label",
    ]


def worker_task(batch_indices):
    """Process a batch of examples."""

    global _worker_data
    results = []
    exceptions = []

    for idx in batch_indices:
        # try:
        labels, proportions = random_select_with_weights(
            _worker_data["allowed_labels"], _worker_data["n_cells_max"]
        )
        _, proportions, subs, uxm_data = generate_pseudo_bulk_optimized(
            _worker_data["n_read_per_split"],
            labels,
            proportions,
            _worker_data["grouped_train"],
            _worker_data["grouped_valid"],
            _worker_data["grouped_test"],
            _worker_data["target_columns"],
        )

        results.append((proportions, subs, uxm_data))

    # except Exception as e:

    #     exceptions.append((labels, proportions, str(e)))

    return results, exceptions


def generate_pseudo_bulk_optimized(
    total_samples,
    labels,
    proportions,
    grouped_train,
    grouped_valid,
    grouped_test,
    target_columns,
):
    """Optimized version using pre-computed groups with UXM deconvolution."""
    assert np.round(np.sum(proportions), 4) == 1, "Proportions must sum up to one"

    n_samples_list = [int(total_samples * x) for x in proportions]
    sample_name = "pseudo_balk_sample"

    subs = []
    uxm_data = []

    for grouped in [grouped_train, grouped_valid, grouped_test]:
        sub_parts = []
        samples_per_dmr = {
            label: int(n / 39) for label, n in zip(labels, n_samples_list)
        }

        for label in labels:
            n_per_dmr = samples_per_dmr[label]
            for dmr_ctype_label in range(39):
                try:
                    group = grouped.get_group((label, dmr_ctype_label))
                    sub_parts.append(group.sample(n_per_dmr, replace=True))
                except KeyError:
                    continue

        if sub_parts:
            sub = pd.concat(sub_parts, ignore_index=True)

            # Compute UXM deconvolution inputs before aggregation
            results = sub
            results_agg = (
                results.groupby(["name", "direction"])
                .aggregate({"record_M": "sum", "record_U": "sum", "record_X": "sum"})
                .reset_index()
            )
            results_agg["count"] = (
                results_agg["record_M"]
                + results_agg["record_U"]
                + results_agg["record_X"]
            )
            results_agg["sf"] = results_agg["record_U"] / results_agg["count"]

            sf = deepcopy(results_agg[["name", "direction"]])
            sf[sample_name] = results_agg["sf"]

            counts = results_agg[["name", "direction", "count"]].copy()
            counts.columns = ["name", "direction", sample_name]

            # uxm_proportions = uxm_deconvolution(atlas, ref_cells, sf, counts, sample_names=[sample_name])[0]
            # uxm_deconv_results = {x: np.round(y, 4) for (x, y) in zip(ref_cells, uxm_proportions)}
            # uxm_deconv_results_alligned = rearange_uxm_deconvolution_results(
            #     labels_dict_reversed, uxm_proportions, ref_cells
            # )
            # uxm_data.append((sf, counts, uxm_deconv_results, uxm_deconv_results_alligned))
            uxm_data.append((sf, counts))
            # Aggregate predictions
            sub = aggregate_predictions_by_dmr_optimized(
                sub, group_cols=["dmr_ctype_label", "dmr_ctype"]
            )
            sub = sub[target_columns]
            subs.append(sub)

    proportions_dict = dict(zip(labels, proportions))
    proportions_full = [proportions_dict.get(x, 0) for x in range(39)]

    return labels, proportions_full, subs, uxm_data


def run_ios_generation_parallel(
    train_data,
    valid_data,
    test_data,
    file_name,
    n_io_examples=30000,
    n_workers=None,
    batch_size=100,
    checkpoint_interval=1000,
    start_checkpoint_idx=0,
):
    """Main function to run parallel processing."""

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    allowed_labels = list(range(39))
    n_cells_max = 10
    n_read_per_split = int(4.75 * 1e5)

    # Create batches of indices
    all_indices = list(range(n_io_examples))
    batches = [
        all_indices[i : i + batch_size] for i in range(0, len(all_indices), batch_size)
    ]

    all_ios = []
    all_exceptions = []

    checkpoint_idx = start_checkpoint_idx
    # Use ProcessPoolExecutor with initializer

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(
            train_data,
            valid_data,
            test_data,
            allowed_labels,
            n_cells_max,
            n_read_per_split,
        ),
    ) as executor:

        # Submit all batches
        futures = {
            executor.submit(worker_task, batch): i for i, batch in enumerate(batches)
        }

        # Process results as they complete
        with tqdm(total=n_io_examples, desc="Generating examples") as pbar:
            completed = 0
            for future in as_completed(futures):
                results, exceptions = future.result()
                all_ios.extend(results)
                all_exceptions.extend(exceptions)
                completed += len(results) + len(exceptions)
                pbar.update(len(results) + len(exceptions))

                # Checkpoint
                if len(all_ios) % checkpoint_interval < batch_size:
                    checkpoint_idx += checkpoint_interval
                    with open(
                        file_name.replace(".pkl", f"_{checkpoint_idx}.pkl"), "wb"
                    ) as f:
                        pickle.dump(all_ios, f)
                    all_ios = (
                        []
                    )  # Initialize from the beggining so the object is not growing in memory

    return all_ios, all_exceptions



def random_select_with_weights(elements, n):
    """
    Randomly selects up to n elements from a list and generates
    corresponding weights that sum to 1.

    Args:
        elements: List of elements to select from
        n: Maximum number of elements to select

    Returns:
        tuple: (selected_elements, weights) where weights sum to 1
    """
    # Determine how many elements to select (up to n, but not more than available)
    k = random.randint(1, n)

    # Randomly select k elements without replacement
    selected = random.sample(elements, k)

    # Generate random weights and normalize to sum to 1
    weights = [random.random() for _ in range(k)]
    total = sum(weights)
    weights = [w / total for w in weights]

    return selected, weights

