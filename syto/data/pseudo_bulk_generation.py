import multiprocessing as mp
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from syto.deconvolution.uxm import (
    rearange_uxm_deconvolution_results,
    uxm_deconvolution,
)
from syto.classification.prediction_aggregation import (
    aggregate_predictions_by_grg,
    aggregate_predictions_by_grg_optimized,
    _fill_in_missing_labels,
)

# Global variables for worker processes (initialized once per worker)
_worker_data = {}


def _build_target_columns(num_prediction_classes: int = 39) -> list:
    """Build the default list of target columns for GR-aggregated output.

    Parameters
    ----------
    num_prediction_classes : int
        Number of classifier output classes (including background if present).
    """
    cols = ["dmr_ctype_label", "dmr_ctype"]
    cols += [f"prediction_{i}_wavg" for i in range(num_prediction_classes)]
    cols += ["methylation_level_wavg", "total_weight", "n_reads", "chromosome", "label"]
    return cols


def generate_pseudo_bulk(
    total_samples,
    labels,
    proportions,
    splits,
    atlas,
    ref_cells,
    labels_dict_reversed,
    return_reads=False,
    num_labels=39,
    num_prediction_classes=None,
):
    """Generate pseudo-bulk mixtures for the train, validation, and test splits.

    The function samples read-level predictions according to the requested cell-type
    proportions, distributing each selected cell type's reads uniformly across the
    DMR cell-type groups. For each input split, it returns both an aggregated
    DMR-level pseudo-bulk representation and the UXM deconvolution inputs/results
    computed from the sampled reads.

    Parameters
    ----------
    total_samples : int
        Total number of reads to sample before distributing them across the
        requested cell types.
    labels : list[int]
        Cell-type labels to include in the pseudo-bulk mixture.
    proportions : list[float]
        Mixture proportions associated with ``labels``. They must sum to 1.
    splits : dict
        Dict of read-level prediction dataframe for the splits.
    atlas : pd.DataFrame
        UXM atlas used to deconvolve the pseudo-bulk sample.
    ref_cells : list[str]
        Reference cell names expected by the UXM atlas and deconvolution routine.
    labels_dict_reversed : dict[str, int]
        Mapping from reference cell names to label ids used to align the UXM output.
    return_reads : bool, default=False
        If True, also return the sampled read-level pseudo-bulk dataframes.
    num_labels : int, default=39
        Number of DMR cell-type groups.
    num_prediction_classes : int, optional
        Number of classifier output classes (including background).
        Defaults to ``num_labels`` when not provided.

    Returns
    -------
    tuple
        If ``return_reads`` is False, returns ``(labels, proportions_full, subs,
        uxm_data)``. If ``return_reads`` is True, returns ``(labels,
        proportions_full, subs, uxm_data, reads)``.

        ``proportions_full`` is a dense length-``num_labels`` proportion vector
        aligned to all labels, ``subs`` contains one aggregated pseudo-bulk
        dataframe per split as a dictionary, and ``uxm_data`` contains one tuple per split with
        the UXM inputs and deconvolution outputs as a dictionary.

    Notes
    -----
    This function assumes that every requested ``(original_label,
    dmr_ctype_label)`` group exists in each split. It also uses
    ``int(n / num_labels)`` samples per DMR group, so small rounding losses
    are expected when distributing reads across DMR types.
    """
    if num_prediction_classes is None:
        num_prediction_classes = num_labels
    assert np.round(np.sum(proportions), 4) == 1, "Proportions must sum up to one"
    n_samples_list = [int(total_samples * x) for x in proportions]
    target_columns = _build_target_columns(num_prediction_classes)
    subs = {}
    uxm_data = {}
    reads = {}
    sample_name = "pseudo_bulk_sample"

    for split_name, df in splits.items():  # for each dataset (with predictions)
        df["total_marked_cpgs"] = df["NCPGS"]
        df["methylation_level"] = df["M_rate"]
        df.rename(columns={"chr": "chromosome"}, inplace=True)
        df["direction"] = "U"

        # sample from each ctype according to the proportions, sampling equally from each DMR type within each ctype
        grouped = df.groupby(["original_label", "dmr_ctype_label"], sort=False)
        sub = []
        for n, label in zip(n_samples_list, labels):
            for dmr_ctype_label in range(num_labels):
                group = grouped.get_group((label, dmr_ctype_label))
                sub.append(group.sample(int(n / num_labels), replace=True))
        sub = pd.concat(sub)

        # aggregate predictions by DMR type and chromosome
        sub_aggregated = aggregate_predictions_by_grg(
            sub, group_cols=["dmr_ctype_label", "dmr_ctype"]
        )
        sub_aggregated = sub_aggregated[target_columns]
        subs[split_name] = sub_aggregated

        # compute UXM deconvolution inputs before aggregation
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
        uxm_data[split_name] = (
            sf,
            counts,
            uxm_deconv_results,
            uxm_deconv_results_alligned,
        )

        if return_reads:
            reads[split_name] = sub

    # ensure that we have a proportion for each of the num_labels labels, filling in 0 for any missing ones
    proportions_dict = {x: y for x, y in zip(labels, proportions)}
    proportions_full = [
        proportions_dict.get(x, 0) for x in range(num_prediction_classes)
    ]
    if return_reads:
        return labels, proportions_full, subs, uxm_data, reads

    return labels, proportions_full, subs, uxm_data


def init_worker(
    splits,
    allowed_labels,
    n_cells_max,
    n_read_per_split,
    num_labels=39,
    num_prediction_classes=None,
    generate_uxm_inputs=True,
    target_proportions=None,
    dmr_sampling_variants=None,
):
    """Initialize worker process with shared data and pre-computed groups.

    Parameters
    ----------
    splits : dict
        Dict of per-split read-level DataFrames.
    allowed_labels : list[int]
        Labels allowed in random selection mode.
    n_cells_max : int
        Maximum number of cell types per mixture in random mode.
    n_read_per_split : int
        Total reads to sample per split per IO example.
    num_labels : int
        Number of DMR cell-type groups.  Default 39.
    num_prediction_classes : int, optional
        Number of classifier output classes (including background).
        Defaults to ``num_labels``.
    generate_uxm_inputs : bool
        Whether to compute UXM sf/counts tables.  Default True.
    target_proportions : list, optional
        Pre-defined proportions for target-proportion mode.
    dmr_sampling_variants : list[str], optional
        List of DMR sampling strategies to apply per IO example.
        Each entry must be ``"uniform_multinomial"``.
        Default ``["uniform_multinomial"]``.
    """

    global _worker_data

    if num_prediction_classes is None:
        num_prediction_classes = num_labels
    if dmr_sampling_variants is None:
        dmr_sampling_variants = ["uniform_multinomial"]

    # Set unique random seed per process
    seed = mp.current_process().pid
    random.seed(seed)
    np.random.seed(seed)

    grouped_splits = {}
    for name, df in splits.items():
        df["total_marked_cpgs"] = df["NCPGS"]
        df["methylation_level"] = df["M_rate"]
        df.rename(columns={"chr": "chromosome"}, inplace=True)
        df["direction"] = "U"
        # Pre-compute grouped dataframes (expensive operation done once per worker)
        grouped_splits[name] = df.groupby(
            ["original_label", "dmr_ctype_label"], sort=False
        )

    _worker_data["grouped_splits"] = grouped_splits

    _worker_data["allowed_labels"] = allowed_labels
    _worker_data["n_cells_max"] = n_cells_max
    _worker_data["n_read_per_split"] = n_read_per_split
    _worker_data["num_labels"] = num_labels
    _worker_data["generate_uxm_inputs"] = generate_uxm_inputs
    _worker_data["target_proportions"] = target_proportions
    _worker_data["dmr_sampling_variants"] = dmr_sampling_variants
    _worker_data["num_prediction_classes"] = num_prediction_classes
    _worker_data["target_columns"] = _build_target_columns(num_prediction_classes)


def worker_task(batch_args):
    """Process a batch of examples.

    Parameters
    ----------
    batch_args : tuple
        ``(batch_indices, batch_proportions)`` where ``batch_proportions``
        is either ``None`` (random mode) or a list of ``(labels, proportions)``
        tuples for target-proportion mode.

    Each proportion vector is processed once per DMR sampling variant
    stored in ``_worker_data["dmr_sampling_variants"]``.  Results are
    tagged with the variant name as a 4th tuple element:
    ``(proportions_out, subs, uxm_data, variant)``.
    """

    global _worker_data
    batch_indices, batch_proportions = batch_args
    results = []
    exceptions = []

    dmr_sampling_variants = _worker_data.get(
        "dmr_sampling_variants", ["uniform_multinomial"]
    )

    for i, _ in enumerate(batch_indices):
        try:
            if batch_proportions is not None:
                # Target-proportion mode: use pre-assigned proportions
                proportions_full = batch_proportions[i]
                labels = [j for j, p in enumerate(proportions_full) if p > 0]
                proportions = [proportions_full[j] for j in labels]
                # Re-normalize in case of floating-point drift
                total = sum(proportions)
                if total > 0:
                    proportions = [p / total for p in proportions]
            else:
                # Random mode
                labels, proportions = random_select_with_weights(
                    _worker_data["allowed_labels"], _worker_data["n_cells_max"]
                )

            for variant in dmr_sampling_variants:
                _, proportions_out, subs, uxm_data = generate_pseudo_bulk_optimized(
                    _worker_data["n_read_per_split"],
                    labels,
                    proportions,
                    _worker_data["grouped_splits"],
                    _worker_data["target_columns"],
                    num_labels=_worker_data["num_labels"],
                    num_prediction_classes=_worker_data["num_prediction_classes"],
                    generate_uxm_inputs=_worker_data["generate_uxm_inputs"],
                    dmr_sampling=variant,
                )

                results.append((proportions_out, subs, uxm_data, variant))

        except Exception as e:
            exceptions.append((labels, proportions, str(e)))

    return results, exceptions


def _compute_samples_per_dmr_multinomial(labels, n_samples_list, num_labels):
    """Realistic DMR allocation using multinomial sampling.

    Simulates natural sequencing noise by randomly distributing each
    cell type's total reads across the available DMRs. This causes
    total depth to fluctuate per locus, but guarantees the local
    proportions correctly center around the global mixture proportions.

    Parameters
    ----------
    labels : list[int]
        Cell-type labels in this mixture.
    n_samples_list : list[int]
        Total reads to sample for each label.
    num_labels : int
        Number of DMR groups.

    Returns
    -------
    dict
        ``{label: list[int]}`` with per-DMR read counts for each label.
    """
    result = {}

    # Assumption: A read has an equal baseline probability of landing in any DMR
    pvals = [1.0 / num_labels] * num_labels

    for label, n in zip(labels, n_samples_list):
        # np.random.multinomial perfectly distributes 'n' items into 'num_labels' bins
        counts = np.random.multinomial(n, pvals).tolist()
        result[label] = counts

    return result


def generate_pseudo_bulk_optimized(
    total_samples,
    labels,
    proportions,
    grouped_splits,
    target_columns,
    num_labels=39,
    num_prediction_classes=None,
    generate_uxm_inputs=True,
    dmr_sampling="uniform_multinomial",
):
    """Optimized version using pre-computed groups.

    Parameters
    ----------
    total_samples : int
        Total reads to sample per split.
    labels : list[int]
        Cell-type labels to include.
    proportions : list[float]
        Proportions for each label. Must sum to 1.
    grouped_splits : dict
        Dict of pre-computed groupby objects.
    target_columns : list[str]
        Columns to keep in aggregated output.
    num_labels : int
        Number of DMR cell-type groups. Default 39.
    num_prediction_classes : int, optional
        Number of classifier output classes (including background).
        Defaults to ``num_labels``.
    generate_uxm_inputs : bool
        If True, compute UXM sf/counts tables. Default True.
    dmr_sampling : str
        DMR sampling strategy. ``"uniform_multinomial"`` generates random weights per DMR,
        normalizes them, and multiplies by the target count. (only strategy available)
        Default ``"uniform_multinomial"``.
    """
    if num_prediction_classes is None:
        num_prediction_classes = num_labels
    assert np.round(np.sum(proportions), 4) == 1, "Proportions must sum up to one"
    assert dmr_sampling in (
        "uniform_multinomial",
    ), f"dmr_sampling must be 'uniform_multinomial', got '{dmr_sampling}'"

    n_samples_list = [int(total_samples * x) for x in proportions]
    sample_name = "pseudo_bulk_sample"

    # Compute per-DMR read counts based on the sampling strategy
    if dmr_sampling == "uniform_multinomial":
        samples_per_dmr = _compute_samples_per_dmr_multinomial(
            labels, n_samples_list, num_labels
        )
    else:
        raise ValueError(f"Unsupported dmr_sampling strategy: {dmr_sampling}")

    subs = {}
    uxm_data = {}

    for split_name, grouped in grouped_splits.items():
        sub_parts = []

        for label in labels:
            per_dmr = samples_per_dmr[label]
            for dmr_ctype_label in range(num_labels):
                # per_dmr is a list[int] for uniform_multinomial
                n_per_dmr = per_dmr[dmr_ctype_label]
                if n_per_dmr <= 0:
                    continue
                try:
                    group = grouped.get_group((label, dmr_ctype_label))
                    sub_parts.append(group.sample(n_per_dmr, replace=True))
                except KeyError:
                    continue

        if sub_parts:
            sub = pd.concat(sub_parts, ignore_index=True)

            if generate_uxm_inputs:
                # Compute UXM deconvolution inputs before aggregation
                results = sub
                results_agg = (
                    results.groupby(["name", "direction"])
                    .aggregate(
                        {"record_M": "sum", "record_U": "sum", "record_X": "sum"}
                    )
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

                uxm_data[split_name] = (sf, counts)
            else:
                uxm_data[split_name] = None

            # Aggregate predictions
            sub = aggregate_predictions_by_grg_optimized(
                sub, group_cols=["dmr_ctype_label", "dmr_ctype"]
            )
            sub = sub[target_columns]
            subs[split_name] = sub

    proportions_dict = dict(zip(labels, proportions))
    proportions_full = [
        proportions_dict.get(x, 0) for x in range(num_prediction_classes)
    ]

    return labels, proportions_full, subs, uxm_data


def run_ios_generation_parallel(
    splits,
    file_name,
    n_io_examples=30000,
    n_workers=None,
    batch_size=100,
    checkpoint_interval=1000,
    start_checkpoint_idx=0,
    allowed_labels=None,
    n_cells_max=10,
    n_read_per_split=None,
    num_labels=39,
    num_prediction_classes=None,
    generate_uxm_inputs=True,
    target_proportions=None,
    dmr_sampling_variants=None,
):
    """Parallel pseudo-bulk IO-example generation.

    Parameters
    ----------
    splits : dict
        Per-split read-level DataFrames (with predictions).
    file_name : str
        Base path for checkpoint pickle files.
    n_io_examples : int
        Number of IO examples to generate.  Ignored when
        ``target_proportions`` is provided (uses its length instead).
    n_workers : int, optional
        Number of worker processes.  Default: ``cpu_count - 1``.
    batch_size : int
        Examples per worker batch.
    checkpoint_interval : int
        Flush to disk every ``checkpoint_interval`` completed examples.
    start_checkpoint_idx : int
        Starting index for checkpoint file naming (for resumption).
    allowed_labels : list[int], optional
        Labels allowed in random mode.  Default: ``list(range(num_labels))``.
    n_cells_max : int
        Max cell types per mixture in random mode.
    n_read_per_split : int, optional
        Reads to sample per split.  Default: ``int(4.75e5)``.
    num_labels : int
        Number of DMR cell-type groups.  Default 39.
    num_prediction_classes : int, optional
        Number of classifier output classes (including background).
        Defaults to ``num_labels``.
    generate_uxm_inputs : bool
        Compute UXM sf/counts tables.  Default True.
    target_proportions : list of list[float], optional
        Pre-defined proportion vectors, each of length ``num_labels``.
        When provided, overrides ``n_io_examples`` and random selection.
    dmr_sampling_variants : list[str], optional
        List of DMR sampling strategies (``"uniform_multinomial"``,).
        Default ``["uniform_multinomial"]``.
    """
    if num_prediction_classes is None:
        num_prediction_classes = num_labels
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if allowed_labels is None:
        allowed_labels = list(range(num_labels))
    if n_read_per_split is None:
        n_read_per_split = int(4.75 * 1e5)
    if dmr_sampling_variants is None:
        dmr_sampling_variants = ["uniform_multinomial"]

    # Determine effective number of examples
    if target_proportions is not None:
        n_io_examples = len(target_proportions)

    # Create batches of indices (and optional proportions)
    all_indices = list(range(n_io_examples))
    batches_indices = [
        all_indices[i : i + batch_size]
        for i in range(start_checkpoint_idx, len(all_indices), batch_size)
    ]

    if target_proportions is not None:
        batches_proportions = [
            target_proportions[i : i + batch_size]
            for i in range(start_checkpoint_idx, len(target_proportions), batch_size)
        ]
    else:
        batches_proportions = [None] * len(batches_indices)

    # Zip indices with proportions for worker_task
    batches = list(zip(batches_indices, batches_proportions))

    all_ios = []
    all_exceptions = []

    checkpoint_idx = start_checkpoint_idx

    # Ensure output directory exists
    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)

    # Use ProcessPoolExecutor with initializer
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(
            splits,
            allowed_labels,
            n_cells_max,
            n_read_per_split,
            num_labels,
            num_prediction_classes,
            generate_uxm_inputs,
            None,  # target_proportions stored per-batch, not globally
            dmr_sampling_variants,
        ),
    ) as executor:

        # Submit all batches
        futures = {
            executor.submit(worker_task, batch): i for i, batch in enumerate(batches)
        }

        # Process results as they complete
        with tqdm(
            total=n_io_examples,
            desc="Generating examples",
            initial=start_checkpoint_idx,
        ) as pbar:
            for future in as_completed(futures):
                results, exceptions = future.result()
                all_ios.extend(results)
                all_exceptions.extend(exceptions)
                pbar.update(len(results) + len(exceptions))

                # Checkpoint
                if len(all_ios) >= checkpoint_interval:
                    checkpoint_idx += len(all_ios)
                    with open(
                        file_name.replace(".pkl", f"_{checkpoint_idx}.pkl"), "wb"
                    ) as f:
                        pickle.dump(all_ios, f)
                    # Initialize from the beginning so the object is not growing in memory
                    all_ios = []

    # Final flush of remaining examples
    if all_ios:
        checkpoint_idx += len(all_ios)
        with open(file_name.replace(".pkl", f"_{checkpoint_idx}.pkl"), "wb") as f:
            pickle.dump(all_ios, f)
        all_ios = []

    return all_ios, all_exceptions


def consolidate_ios_pickles(
    ios_dir: str,
    output_path: str,
    labels_dict: dict,
    num_labels: int = 39,
    num_prediction_classes: int = None,
):
    """Load partial pickle checkpoints and consolidate into a single ``.npz``.

    Each pickle is expected to contain a list of tuples.  The tuples may
    be in one of this one format (format may be extended later):

    **Variant-tagged (4-element):**
        ``(proportions, subs, uxm_data, variant)``

    where ``subs`` is a dictionary of DataFrames (e.g. 'train', 'valid',
    'test'), ``uxm_data`` is the corresponding UXM data (or ``None``),
    and ``variant`` is a string like ``"uniform"`` or ``"random"``.
    Features are stored under
    ``features_{split}_{variant}`` keys and proportions under
    ``proportions_{split}`` keys (deduplicated per split).  A top-level
    ``proportions`` key is additionally written **only** when all splits
    share the exact same proportions array.

    Parameters
    ----------
    ios_dir : str
        Directory containing the checkpoint ``.pkl`` files.
    output_path : str
        Path for the output ``.npz`` file.
    num_labels : int
        Number of DMR cell-type groups.  Default 39.
    num_prediction_classes : int, optional
        Number of classifier output classes (including background).
        Defaults to ``num_labels``.

    Returns
    -------
    dict
        Numpy arrays keyed as described above.
    """
    if num_prediction_classes is None:
        num_prediction_classes = num_labels
    pred_cols = [f"prediction_{i}_wavg" for i in range(num_prediction_classes)]

    pkl_files = sorted(f for f in os.listdir(ios_dir) if f.endswith(".pkl"))

    # ── Variant-aware consolidation ────────────────────────────────
    return _consolidate_variant_aware(
        ios_dir, output_path, num_labels, pred_cols, pkl_files, labels_dict
    )


def _consolidate_variant_aware(
    ios_dir: str,
    output_path: str,
    num_labels: int,
    pred_cols: list,
    pkl_files: list,
    labels_dict: dict,
):
    """Variant-aware consolidation for ``(proportions, subs, uxm_data, variant)`` tuples.

    Stores proportions as ``proportions_{split}`` (deduplicated per
    proportion vector).  Features are stored under
    ``features_{split}_{variant}``.
    """
    # {split_name: {variant: [arrays]}}
    features_by_split_variant = {}
    # {split_name: {variant: [proportion_arrays]}}
    proportions_by_split_variant = {}

    for pkl_name in tqdm(pkl_files, desc="Consolidating pickles (variant-aware)"):
        pkl_path = os.path.join(ios_dir, pkl_name)
        try:
            with open(pkl_path, "rb") as f:
                part_ios = pickle.load(f)
        except Exception:
            import warnings

            warnings.warn(f"{pkl_name} is corrupted and cannot be opened")
            continue

        for item in part_ios:
            props = item[0]
            subs = item[1]
            # item[2] is uxm_data (not stored in npz)
            variant = item[3]

            gt = np.expand_dims(np.array(props), 0)

            for split_name, df in subs.items():
                key = (split_name, variant)

                if split_name not in features_by_split_variant:
                    features_by_split_variant[split_name] = {}
                    proportions_by_split_variant[split_name] = {}

                if variant not in features_by_split_variant[split_name]:
                    features_by_split_variant[split_name][variant] = []
                    proportions_by_split_variant[split_name][variant] = []
                if df.shape[0] < num_labels:
                    df = _fill_in_missing_labels(
                        df,
                        group_cols=["dmr_ctype_label", "dmr_ctype"],
                        labels_dict=labels_dict,
                    )
                    df = df.sort_values("dmr_ctype_label").reset_index(drop=True)
                features_by_split_variant[split_name][variant].append(
                    np.expand_dims(df[pred_cols].to_numpy(), 0)
                )
                proportions_by_split_variant[split_name][variant].append(gt)

    result = {}

    # Build per-split proportions (deduplicated: same across variants for one split)
    for split_name, variant_dict in proportions_by_split_variant.items():
        # Collect all proportion arrays across variants for this split
        first_variant = sorted(variant_dict.keys())[0]
        result[f"proportions_{split_name}"] = np.concatenate(
            variant_dict[first_variant], axis=0
        )

    # Build per-split-per-variant features
    for split_name, variant_dict in features_by_split_variant.items():
        for variant, feat_list in sorted(variant_dict.items()):
            if feat_list:
                result[f"features_{split_name}_{variant}"] = np.concatenate(
                    feat_list, axis=0
                )

    # Check if all splits share the same proportions; if so, also store
    # a top-level "proportions" key for convenience.
    prop_keys = [k for k in result if k.startswith("proportions_")]
    if len(prop_keys) >= 2:
        first = result[prop_keys[0]]
        all_same = all(np.array_equal(first, result[k]) for k in prop_keys[1:])
        if all_same:
            result["proportions"] = first

    np.savez_compressed(output_path, **result)
    return result


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
    if not elements or n <= 0:
        return [], []

    # Determine how many elements to select (up to n, but not more than available).
    max_selectable = min(n, len(elements))
    k = random.randint(1, max_selectable)
    k = min(k, max_selectable)

    # Randomly select k elements without replacement
    selected = random.sample(elements, k)

    # Generate random weights and normalize to sum to 1
    weights = [random.random() for _ in range(k)]
    total = sum(weights)
    weights = [w / total for w in weights]

    return selected, weights
