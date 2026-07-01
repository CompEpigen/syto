""" """

import pandas as pd

RECOVERED_READS_REQUIRED = {
    "ref_name",
    "ref_pos",
    "original_seq",
    "methyl_seq",
    "ctype",
}


def adapt_recovered_reads(df: pd.DataFrame, labels_dict: dict, cell_type_match_dict:dict=None) -> pd.DataFrame:
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
        df["ctype"] = df["ctype"].apply(lambda x : cell_type_match_dict[x] if x in cell_type_match_dict.keys() else x)

    out["read_end"] = out["read_start"] + df["original_seq"].str.len().to_numpy() - 1
    out["original_label"] = df["ctype"].map(ctype_to_label)
    out = out[out["original_label"].notna()].copy()
    out["original_label"] = out["original_label"].astype(int)
    return out.reset_index(drop=True)
