"""
This module  reimplements pieces of UXM code from original work by Loyfer et. al: https://github.com/nloyfer/UXM_deconv.
The focus is on creating a minimum setup sufficient to run deconvolution in python in a manner compatible with overall pipeline without introducing dependencies to original code.
Some of the methods are directly copied while other are specific to this repo.

Use and distribution of the original UXM_deconv code reproduced here is subject to the
Software Research License included in LICENSE.md alongside this module.
"""

import sys
import os.path as op
import logging

import pandas as pd
import numpy as np
from scipy import optimize
from syto.data.dataset import resolve_column
from baselines.deconvolution.utils import rearange_deconvolution_results

### Selected original deconvolution code from https://github.com/nloyfer/UXM_deconv ###

_module_logger = logging.getLogger(__name__)


def validate_ref_tissues(df, tissue_list):
    """Validate that the provided tissue list is present in the atlas DataFrame columns"""
    for col in tissue_list:
        if col not in df.columns:
            _module_logger.error("Invalid cell type (not in atlas): %s", col)
            sys.exit(1)


def validate_file(fpath):
    """Validate that the provided file path exists and is a file"""
    if not op.isfile(fpath):
        _module_logger.error("Invalid file: %s", fpath)
        sys.exit(1)
    return fpath


def load_atlas(atlas_path, ignore=None, include=None):
    """Load the reference atlas and optionally filter tissues"""
    if not op.isfile(atlas_path):
        _module_logger.error("Invalid reference atlas (--atlas flag): %s", atlas_path)
    validate_file(atlas_path)

    # take a peek:
    df = pd.read_csv(atlas_path, sep="\t", nrows=2)
    if df.shape[1] < 8:
        _module_logger.error("Invalid atlas: %s", atlas_path)
        sys.exit(1)
    df = pd.read_csv(atlas_path, sep="\t")
    if not all(df["name"].str.startswith("chr")):
        _module_logger.error(
            'Invalid atlas: %s. "name" column must all start with "chr"', atlas_path
        )
        sys.exit(1)

    if ignore is not None:
        validate_ref_tissues(df, ignore)
        for col in ignore:
            del df[col]
            df = df[df.target != col]
    elif include is not None:
        validate_ref_tissues(df, include)
        df = df[df.target.isin(include)]
        keep = list(df.columns[:8]) + include
        df = df[keep]

    df.reset_index(inplace=True, drop=True)
    return df, list(df.columns[8:])


def decon_single_samp(samp, atlas, counts, verbose, debug=False):
    """
    Deconvolve a single sample, using NNLS, to get the mixture coefficients.
    :param samp: a vector of a single sample
    :param atlas: the atlas DataFrame
    :return: the mixture coefficients
    """

    name = samp.columns[2]
    counts.columns = ["name", "direction", "counts"]

    # remove missing sites from both sample and atlas:
    # TODO: imputation for the atlas?
    nd_cols = ["name", "direction"]
    data = (
        samp.merge(
            atlas.drop_duplicates(nd_cols, ignore_index=True), on=nd_cols, how="inner"
        )
        .copy()
        .dropna(axis=0)
    )
    data = data.merge(
        counts.drop_duplicates(nd_cols, ignore_index=True), on=nd_cols, how="left"
    )

    if data.empty:
        _module_logger.warning("Skipping an empty sample: %s", name)
        return np.nan, np.nan

    if data.shape[0] > atlas.shape[0]:
        _module_logger.error("Merge went wrong. Validate your atlas")
        return None, None
    if verbose:
        _module_logger.info("%s: %d \\ %d markers", name, data.shape[0], atlas.shape[0])
    del data["name"], data["direction"]

    samp = data.iloc[:, 0]
    counts = data.iloc[:, -1]
    red_atlas = data.iloc[:, 1:-1]

    # apply weights:
    red_atlas = red_atlas * counts.values[:, np.newaxis]
    samp = samp * counts

    # get the mixture coefficients by deconvolution
    # (non-negative least squares)
    mixture, residual = optimize.nnls(red_atlas, samp)
    mixture /= np.sum(mixture)
    return mixture


def uxm_deconvolution(
    atlas, ref_cells, sf, counts, sample_names=["pseudo_bulk_sample"]
):
    params = [
        (
            sf[["name", "direction", samp]],
            atlas[["name", "direction"] + ref_cells],
            counts[["name", "direction", samp]],
            False,
            False,
        )
        for samp in sample_names
    ]

    arr = [decon_single_samp(*p) for p in params]
    return arr


### End of original UXM code section ###

### Additional utilities, specific to this repo ###



def build_uxm_input(reads: pd.DataFrame, min_cpgs_count=4) -> dict:
    """
    Build UXM-compatible scaling factors and counts from prepared reads.

    Expects reads to already have NCPGS, record_M, record_U, record_X (from
    mark_records_methyl_state) and a 'name' column identifying the atlas region.
    """
    from copy import deepcopy

    results_agg = (
        reads[reads["NCPGS"] >= min_cpgs_count]
        .groupby("name")
        .aggregate({"record_M": "sum", "record_U": "sum", "record_X": "sum"})
        .reset_index()
    )
    results_agg["count"] = (
        results_agg["record_M"] + results_agg["record_U"] + results_agg["record_X"]
    )
    results_agg["sf"] = results_agg["record_U"] / results_agg["count"]
    results_agg["direction"] = "U"

    sf = deepcopy(results_agg[["name", "direction"]])
    sf["sample"] = results_agg["sf"]
    counts = results_agg[["name", "direction", "count"]].copy()
    counts.columns = ["name", "direction", "sample"]

    return {"scaling_factors": sf, "counts": counts}


def mark_records_methyl_state(
    reads_data,
    methyl_tr=0.75,
    unmethyl_tr=0.25,
):
    """Classify each CpG read as methylated (M), unmethylated (U), or ambiguous (X).

    Adds columns ``M``, ``U``, ``NCPGS``, ``M_rate``, ``record_M``,
    ``record_U``, ``record_X`` to ``reads_data`` in-place and returns it.
    """
    meth_col = resolve_column(reads_data.columns, "methylation_ids")
    pat = reads_data[meth_col]
    reads_data["M"] = pat.apply(lambda x: x.count("1"))
    reads_data["U"] = pat.apply(lambda x: x.count("0"))
    reads_data["NCPGS"] = reads_data["M"] + reads_data["U"]
    reads_data["M_rate"] = reads_data["M"] / reads_data["NCPGS"].replace(0, np.nan)

    is_M = reads_data["M_rate"] >= methyl_tr
    is_U = reads_data["M_rate"] <= unmethyl_tr
    reads_data["record_M"] = is_M.astype(int)
    reads_data["record_U"] = is_U.astype(int)
    reads_data["record_X"] = (~is_M & ~is_U).astype(int)
    return reads_data


def run_uxm_deconvolution(
    reads: pd.DataFrame,
    atlas_df: pd.DataFrame,
    ref_cells: list,
    labels_dict_reversed: dict,
    n_labels: int = None,
):
    """Mark read methylation state, build UXM input, deconvolve, and align proportions.

    Returns a list of floats (one per label in labels_dict order), or None if
    deconvolution fails (e.g. no overlapping reads).
    """
    reads = mark_records_methyl_state(reads.copy())
    uxm_in = build_uxm_input(reads)
    proportions = uxm_deconvolution(
        atlas_df,
        ref_cells,
        uxm_in["scaling_factors"],
        uxm_in["counts"],
        sample_names=["sample"],
    )[0]
    if not isinstance(proportions, np.ndarray):
        return None
    return rearange_deconvolution_results(
        labels_dict_reversed, proportions, ref_cells, n_labels=n_labels
    )


def prepare_reads_for_uxm(
    reads_data,
    atlas,
    labels_dict,
    cell_type_match_dict,
    debug=False,
    unmethyl_tr=0.25,
    methyl_tr=0.75,
    seq_column="seq",
    methylation_pattern_column="methylation_encoding",
    read_name_column="read_name",
    progress_prefix="",
):

    results = []

    n_records = len(reads_data)

    is_soft_label_avaliable = "soft_label" in reads_data.columns

    # Build chromosome start index mapping
    chrom_start_idx = {}
    if n_records > 0:
        current_chrom = reads_data.iloc[0]["chromosome"]
        chrom_start_idx[current_chrom] = 0
        for i in range(1, n_records):
            chrom = reads_data.iloc[i]["chromosome"]
            if chrom != current_chrom:
                chrom_start_idx[chrom] = i
                current_chrom = chrom

    chrom_base_pointer = {}

    n_atlas = len(atlas)
    for i_atlas, (_, row) in enumerate(atlas.iterrows()):
        if i_atlas % 100 == 0 or i_atlas == n_atlas - 1:
            current_chrom_ptr = chrom_base_pointer.get(row["chr"], 0)
            _module_logger.info(
                "\r%sProcessed %d/%d atlas regions (reads scanned: %d/%d)",
                progress_prefix,
                i_atlas,
                n_atlas,
                current_chrom_ptr,
                n_records,
            )

        name = row["name"]
        chromosome = row["chr"]
        start = row["start"] - 1  # 1-based to 0-based
        end = row["end"]  # EXCLUSIVE

        if chromosome not in chrom_base_pointer:
            chrom_base_pointer[chromosome] = chrom_start_idx.get(chromosome, n_records)

        base_ptr = chrom_base_pointer[chromosome]

        # Advance past records that can't overlap with this or future regions
        while base_ptr < n_records:
            record = reads_data.iloc[base_ptr]
            if record["chromosome"] != chromosome:
                break
            if record["read_end"] < start:
                base_ptr += 1
            else:
                break

        chrom_base_pointer[chromosome] = base_ptr

        scan_ptr = base_ptr
        while scan_ptr < n_records:
            record = reads_data.iloc[scan_ptr]

            if record["chromosome"] != chromosome:
                break

            if record["read_start"] >= end:
                break

            if record["read_end"] >= start:
                pattern = record[methylation_pattern_column]
                seq = record[seq_column]

                trimmed_start = max(start, record["read_start"])
                trimmed_end_inclusive = min(end - 1, record["read_end"])

                offset_start = trimmed_start - record["read_start"]
                offset_end = (
                    len(pattern) - (record["read_end"] - trimmed_end_inclusive) + 1
                )

                new_pattern = pattern[offset_start:offset_end]
                new_seq = seq[offset_start:offset_end]
                if is_soft_label_avaliable:
                    soft_label = record["soft_label"]
                else:
                    soft_label = np.nan

                if debug:
                    results.append(
                        (
                            chromosome,
                            record[read_name_column],
                            record["read_start"],
                            record["read_end"],
                            trimmed_start,
                            trimmed_end_inclusive,
                            offset_start,
                            offset_end,
                            pattern,
                            new_pattern,
                            seq,
                            new_seq,
                            name,
                            start,
                            end,
                            record["original_label"],
                            record["label"],
                            soft_label,
                        )
                    )
                else:
                    results.append(
                        (
                            chromosome,
                            record[read_name_column],
                            trimmed_start,
                            trimmed_end_inclusive,
                            new_pattern,
                            new_seq,
                            name,
                            start,
                            end,
                            record["original_label"],
                            record["label"],
                            soft_label,
                        )
                    )

            scan_ptr += 1

    _module_logger.info("")  # Add a newline after the progress bar finishes

    # Update column names to reflect actual content
    if debug:
        columns = [
            "chr",
            "read_name",
            "record_start",
            "record_end",
            "trimmed_start",
            "trimmed_end",
            "offset_start",
            "offset_end",
            "original_pattern",
            "pattern",
            "original_seq",
            "seq",
            "name",
            "region_start",
            "region_end",
            "original_label",
            "label",
            "soft_label",
        ]
    else:
        columns = [
            "chr",
            "read_name",
            "trimmed_start",
            "trimmed_end",
            "pattern",
            "seq",
            "name",
            "region_start",
            "region_end",
            "original_label",
            "label",
            "soft_label",
        ]

    results = pd.DataFrame(results, columns=columns)

    results = mark_records_methyl_state(
        results, methyl_tr=methyl_tr, unmethyl_tr=unmethyl_tr
    )

    results = pd.merge(results, atlas[["name", "target"]], on="name")
    results.rename(columns={"target": "dmr_ctype"}, inplace=True)
    labels_dict_reversed = {y: int(x) for (x, y) in labels_dict.items()}
    results["dmr_ctype_matched"] = results["dmr_ctype"].apply(
        lambda x: cell_type_match_dict.get(x, x)
    )
    results["dmr_ctype_label"] = results["dmr_ctype_matched"].apply(
        lambda x: (labels_dict_reversed.get(x, None))
    )
    results.dropna(subset=["dmr_ctype_label"], inplace=True)
    results["dmr_ctype_label"] = results["dmr_ctype_label"].astype(int)

    results["pattern_len"] = results["pattern"].apply(len)
    return results
