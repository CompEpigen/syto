""" """

import pandas as pd


def merge_paired_reads(df, verbose=True):
    """
    Merge paired-end reads into single fragments, similar to wgbs_tools' bam2pat.

    For paired-end sequencing, both mates represent the same DNA fragment and should
    be merged. This function:
    1. Groups reads by read_name AND dmr_label (both mates share the same name, merge per DMR)
    2. Merges methylation information, handling overlapping CpGs
    3. Creates a single fragment entry spanning both mates

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from process_bam_with_chunking with columns:
        - read_name: Read identifier (shared by both mates)
        - dmr_label: Single DMR associated with this entry
        - read_start, read_end: Genomic coordinates
        - chromosome: Chromosome
        - methylation_encoding: String encoding methylation states
        - seq: Sequence
        - clipped_methylated, clipped_unmethylated, clipped_total: CpG counts in clipped region
        - and other metadata columns
    verbose : bool
        Print progress information

    Returns:
    --------
    pd.DataFrame
        DataFrame with paired mates merged into single fragment entries.
        Each entry is for a single DMR with clipped CpG counts for UXM classification.
    """
    if len(df) == 0:
        return df

    # Determine grouping columns - use dmr_label if available
    if "dmr_label" in df.columns:
        group_cols = ["read_name", "dmr_label"]
    else:
        group_cols = ["read_name"]

    # Count reads per (name, dmr_label) to identify pairs vs singletons
    read_counts = df.groupby(group_cols).size()
    singletons = read_counts[read_counts == 1].index
    pairs = read_counts[read_counts == 2].index
    multiplets = read_counts[read_counts > 2].index

    if verbose:
        print(f"Read distribution (grouped by {group_cols}):")
        print(f"  Singletons: {len(singletons)}")
        print(f"  Paired (2 mates): {len(pairs)}")
        print(
            f"  Multiplets (>2): {len(multiplets)} (will be treated as separate entries)"
        )

    merged_data = []

    # Process singletons - keep as is
    if len(group_cols) == 2:
        singleton_mask = df.set_index(group_cols).index.isin(singletons)
        singleton_df = df[singleton_mask]
    else:
        singleton_df = df[df["read_name"].isin(singletons)]
    for _, row in singleton_df.iterrows():
        merged_data.append(row.to_dict())

    # Process pairs - merge mates
    if len(group_cols) == 2:
        pairs_mask = df.set_index(group_cols).index.isin(pairs)
        pairs_df = df[pairs_mask]
    else:
        pairs_df = df[df["read_name"].isin(pairs)]

    for key, group in pairs_df.groupby(group_cols):
        if len(group) != 2:
            continue

        mate1, mate2 = group.iloc[0], group.iloc[1]

        # Ensure mate1 is the one with smaller start position
        if mate2["read_start"] < mate1["read_start"]:
            mate1, mate2 = mate2, mate1

        # Merge the pair
        merged_fragment = _merge_mate_pair(mate1, mate2)
        merged_data.append(merged_fragment)

    # Process multiplets - keep each entry separately (unusual case)
    if len(group_cols) == 2:
        multiplet_mask = df.set_index(group_cols).index.isin(multiplets)
        multiplet_df = df[multiplet_mask]
    else:
        multiplet_df = df[df["read_name"].isin(multiplets)]
    for _, row in multiplet_df.iterrows():
        merged_data.append(row.to_dict())

    result_df = pd.DataFrame(merged_data)

    if verbose:
        print(
            f"Merged result: {len(result_df)} fragments (from {len(df)} read entries)"
        )

    return result_df


def _merge_mate_pair(mate1, mate2):
    """
    Merge two mates of a paired-end read into a single fragment.

    Parameters:
    -----------
    mate1 : pd.Series
        First mate (should have smaller read_start)
    mate2 : pd.Series
        Second mate

    Returns:
    --------
    dict
        Merged fragment with combined methylation information
    """
    # Calculate fragment coordinates
    frag_start = min(mate1["read_start"], mate2["read_start"])
    frag_end = max(mate1["read_end"], mate2["read_end"])
    frag_length = frag_end - frag_start

    # Merge methylation encodings
    # Create a position-aware mapping for CpGs
    merged_encoding, merged_cpg_stats = _merge_methylation_encodings(
        mate1["seq"],
        mate1["methylation_encoding"],
        mate1["read_start"],
        mate2["seq"],
        mate2["methylation_encoding"],
        mate2["read_start"],
        frag_start,
        frag_end,
    )

    # Handle DMR info - use singular dmr_label if available, fall back to dmr_labels
    dmr_label = mate1.get("dmr_label", "") or mate1.get("dmr_labels", "")
    dmr_type = mate1.get("dmr_type", "") or mate1.get("dmr_types", "")
    dmr_start = mate1.get("dmr_start", 0)
    dmr_end = mate1.get("dmr_end", 0)

    # Merge clipped sequences if available
    # For merged fragments, we need to recalculate clipped data within DMR boundaries
    if "seq_clipped" in mate1.index and dmr_start and dmr_end:
        # Merge clipped sequences within DMR boundaries
        clipped_encoding, clipped_stats = _merge_methylation_encodings(
            mate1.get("seq_clipped", ""),
            mate1.get("methylation_clipped", ""),
            mate1.get("clip_start", mate1["read_start"]),
            mate2.get("seq_clipped", ""),
            mate2.get("methylation_clipped", ""),
            mate2.get("clip_start", mate2["read_start"]),
            dmr_start,
            dmr_end,
        )
        clipped_methylated = clipped_stats["methylated"]
        clipped_unmethylated = clipped_stats["unmethylated"]
        clipped_total = clipped_stats["total"]
    else:
        # Fallback if no clipped data
        clipped_methylated = mate1.get("clipped_methylated", 0) + mate2.get(
            "clipped_methylated", 0
        )
        clipped_unmethylated = mate1.get("clipped_unmethylated", 0) + mate2.get(
            "clipped_unmethylated", 0
        )
        clipped_total = clipped_methylated + clipped_unmethylated
        clipped_encoding = {"seq": "", "methylation_encoding": ""}

    clipped_meth_rate = clipped_methylated / clipped_total if clipped_total > 0 else 0.0

    # Build merged fragment entry
    merged = {
        "read_name": mate1["read_name"],
        "chromosome": mate1["chromosome"],
        "read_start": frag_start,  # Fragment start (earliest)
        "read_end": frag_end,  # Fragment end (latest)
        "read_length": frag_length,
        "seq": merged_encoding["seq"],
        "methylation_encoding": merged_encoding["methylation_encoding"],
        "total_cpgs": merged_cpg_stats["total"],
        "methylated_cpgs": merged_cpg_stats["methylated"],
        "unmethylated_cpgs": merged_cpg_stats["unmethylated"],
        "methylation_rate": merged_cpg_stats["methylation_rate"],
        "read_total_cpgs": merged_cpg_stats["total"],  # For compatibility
        "mapping_quality": min(mate1["mapping_quality"], mate2["mapping_quality"]),
        "is_reverse": mate1["is_reverse"],  # Keep first mate's orientation
        "data_type": mate1["data_type"],
        "is_merged_pair": True,
        "mate1_start": mate1["read_start"],
        "mate1_end": mate1["read_end"],
        "mate2_start": mate2["read_start"],
        "mate2_end": mate2["read_end"],
        # DMR info (singular)
        "overlaps_dmr": mate1.get("overlaps_dmr", False)
        or mate2.get("overlaps_dmr", False),
        "dmr_label": dmr_label,
        "dmr_type": dmr_type,
        "dmr_start": dmr_start,
        "dmr_end": dmr_end,
        # Clipped data for UXM classification
        "seq_clipped": clipped_encoding.get("seq", ""),
        "methylation_clipped": clipped_encoding.get("methylation_encoding", ""),
        "clipped_methylated": clipped_methylated,
        "clipped_unmethylated": clipped_unmethylated,
        "clipped_total": clipped_total,
        "clipped_meth_rate": clipped_meth_rate,
        # Chunk info (use fragment coordinates)
        "chunk_start": frag_start,
        "chunk_end": frag_end,
        "chunk_length": frag_length,
    }

    return merged


def _merge_methylation_encodings(
    seq1, enc1, start1, seq2, enc2, start2, frag_start, frag_end
):
    """
    Merge methylation encodings from two mates, matching wgbstools' merge_PE behavior.

    Consensus logic:
    - If both mates report same state, use that state
    - If one is unknown ('2'), use the other's value
    - If they DISAGREE (both known but different), use '2' (unknown)
    """
    frag_length = frag_end - frag_start

    # Initialize merged arrays with 'unknown' state
    merged_enc = ["2"] * frag_length
    merged_seq = ["N"] * frag_length

    # Fill in mate1 data
    for i in range(len(seq1)):
        if i >= len(enc1):
            break
        pos = (start1 - frag_start) + i
        if 0 <= pos < frag_length:
            merged_seq[pos] = seq1[i]
            merged_enc[pos] = enc1[i]

    # Fill in mate2 data with consensus logic matching wgbstools
    for i in range(len(seq2)):
        if i >= len(enc2):
            break
        pos = (start2 - frag_start) + i
        if 0 <= pos < frag_length:
            existing = merged_enc[pos]
            merged_seq[pos] = seq2[i]

            meth = enc2[i]

            if existing == "2":
                # No data from mate1 (or unknown), use mate2
                merged_enc[pos] = meth
            elif meth == "2":
                # No data from mate2, keep mate1
                pass
            elif existing == meth:
                # Both agree, keep the value
                pass
            else:
                # CONFLICT: both are known ('0' or '1') but different
                # Mark as unknown, matching wgbstools behavior
                merged_enc[pos] = "2"

    # Calculate CpG statistics from merged encoding
    # Only count confident calls ('1' and '0'), not unknowns ('2')
    methylated = sum(1 for x in merged_enc if x == "1")
    unmethylated = sum(1 for x in merged_enc if x == "0")
    total = methylated + unmethylated
    meth_rate = methylated / total if total > 0 else 0.0

    return {"seq": "".join(merged_seq), "methylation_encoding": "".join(merged_enc)}, {
        "total": total,
        "methylated": methylated,
        "unmethylated": unmethylated,
        "methylation_rate": meth_rate,
    }
