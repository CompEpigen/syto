"""
This module  reimplements pieces of UXM code from original work by Loyfer et. al: https://github.com/nloyfer/UXM_deconv.
The focus is on creating a minimum setup sufficient to run deconvolution in python in a manner compatible with overall pipeline without introducing dependencies to original code.
Some of the methods are directly copied while other are specific to this repo.
"""

import pandas as pd
import numpy as np
import os
from scipy import optimize
import sys
import os.path as op

### Selected original deconvolution code from https://github.com/nloyfer/UXM_deconv ###


def eprint(*args, **kargs):
    print(*args, file=sys.stderr, **kargs)


def validate_ref_tissues(df, tissue_list):
    for col in tissue_list:
        if col not in df.columns:
            eprint("Invalid cell type (not in atlas):", col)
            exit()


def validate_file(fpath):
    if not op.isfile(fpath):
        eprint("Invalid file", fpath)
        exit()
    return fpath


def load_atlas(atlas_path, ignore=None, include=None):
    if not op.isfile(atlas_path):
        eprint("Invalid reference atlas (--atlas flag)")
    validate_file(atlas_path)

    # take a peek:
    df = pd.read_csv(atlas_path, sep="\t", nrows=2)
    if df.shape[1] < 8:
        eprint(f"Invalid atlas: {atlas_path}")
        exit(1)
    df = pd.read_csv(atlas_path, sep="\t")
    if df["name"].str.startswith("chr").sum() != df.shape[0]:
        eprint(f'Invalid atlas: {atlas_path}. "name" column must all start with "chr"')
        exit(1)

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
        eprint(f"Warning: skipping an empty sample {name}")
        return np.nan, np.nan

    if data.shape[0] > atlas.shape[0]:
        eprint("ERROR: merge went wrong. Validate your atlas")
        return None, None
    if verbose:
        eprint("{}: {} \ {} markers".format(name, data.shape[0], atlas.shape[0]))
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
    atlas, ref_cells, sf, counts, sample_names=["pseudo_balk_sample"]
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


def rearange_uxm_deconvolution_results(
    labels_dict_reversed, uxm_proportions, ref_cells
):
    """
    The function rearanges uxm deconvolution results to match target labels order encoded in the input dictionary
    """
    ref_pos = np.array(
        [
            labels_dict_reversed[cell] if cell in labels_dict_reversed.keys() else -1
            for cell in ref_cells
        ]
    )
    uxm_proportions_alligned = [
        uxm_proportions[i]
        for i in [int(np.where(ref_pos == i)[0][0]) for i in range(0, 39)]
    ]
    return uxm_proportions_alligned


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

    for _, row in atlas.iterrows():
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

    results["M"] = results["pattern"].apply(lambda x: x.count("1"))
    results["U"] = results["pattern"].apply(lambda x: x.count("0"))

    # For classification, use only confident calls
    results["NCPGS"] = results["M"] + results["U"]
    results["M_rate"] = results["M"] / results["NCPGS"].replace(0, np.nan)

    is_M = results["M_rate"] >= methyl_tr
    is_U = results["M_rate"] <= unmethyl_tr
    results["record_M"] = is_M.astype(int)
    results["record_U"] = is_U.astype(int)
    results["record_X"] = (~is_M & ~is_U).astype(int)

    results = pd.merge(results, atlas[["name", "target"]], on="name")
    results.rename(columns={"target": "dmr_ctype"}, inplace=True)
    labels_dict_reversed = {y: x for (x, y) in labels_dict.items()}
    results["dmr_ctype_matched"] = results["dmr_ctype"].apply(
        lambda x: cell_type_match_dict[x] if x in cell_type_match_dict.keys() else x
    )
    results["dmr_ctype_label"] = results["dmr_ctype_matched"].apply(
        lambda x: (
            int(labels_dict_reversed[x]) if x in labels_dict_reversed.keys() else None
        )
    )
    results.dropna(subset=["dmr_ctype_label"], inplace=True)
    results["dmr_ctype_label"] = results["dmr_ctype_label"].astype(int)

    results["pattern_len"] = results["pattern"].apply(len)
    return results
