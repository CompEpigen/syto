""" """

import pandas as pd

RECOVERED_READS_REQUIRED = {
    "ref_name",
    "ref_pos",
    "original_seq",
    "methyl_seq",
    "ctype",
}


def bam_reads_to_recovered(
    df: pd.DataFrame, ctype: str, chrom_prefix: str = ""
) -> pd.DataFrame:
    """Reshape BAM-parsed reads into the recovered-reads CSV layout.

    ``df`` is the output of :func:`syto.data.sequencing.bam_processing.read_bam_regions`.
    Every read is labelled with the sample-level ``ctype``; ``chrom_prefix`` is
    prepended to chromosome names so BAMs aligned to e.g. hs37d5 (``1``) match
    ``chrN`` atlases.  The result feeds :func:`adapt_recovered_reads`, so both
    input types derive ``read_end`` from the sequence length the same way.
    """
    return pd.DataFrame(
        {
            "ref_name": chrom_prefix + df["chromosome"].astype(str),
            "ref_pos": df["read_start"].astype(int),
            "original_seq": df["seq"],
            "methyl_seq": df["methylation_encoding"],
            "ctype": ctype,
        }
    )


def adapt_recovered_reads(
    df: pd.DataFrame, labels_dict: dict, cell_type_match_dict: dict = None
) -> pd.DataFrame:
    """Map a recovered-reads CSV frame to the syto read schema.

    Output columns: chromosome, read_start, read_end (0-based inclusive),
    input_ids, methylation_ids, original_label. Rows whose ``ctype`` is not in
    ``labels_dict`` are dropped.
    """
    missing = RECOVERED_READS_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Recovered-reads frame missing required columns: {missing}")

    ctype_to_label = {v: int(k) for k, v in labels_dict.items()}
    out = pd.DataFrame(
        {
            "chromosome": df["ref_name"].to_numpy(),
            "read_start": df["ref_pos"].astype(int).to_numpy(),
            "input_ids": df["original_seq"].to_numpy(),
            "methylation_ids": df["methyl_seq"].to_numpy(),
        }
    )

    if cell_type_match_dict is not None:
        df["ctype"] = df["ctype"].apply(
            lambda x: cell_type_match_dict[x] if x in cell_type_match_dict.keys() else x
        )

    out["read_end"] = out["read_start"] + df["original_seq"].str.len().to_numpy() - 1
    out["original_label"] = df["ctype"].map(ctype_to_label)
    out = out[out["original_label"].notna()].copy()
    out["original_label"] = out["original_label"].astype(int)
    return out.reset_index(drop=True)
