""""""

from methyldl.data.sequencing.dmr_overlap_analysis import get_overlapping_dmrs


def chunk_tokens(tokens, window_size, stride):
    """
    Splits a list of tokens into overlapping chunks of fixed size.
    Ensures no chunk is shorter than window_size (unless the total read is shorter).
    """
    total_len = len(tokens)

    # Case 1: Read is shorter than the window. Return as is.
    if total_len <= window_size:
        yield tokens
        return

    # Case 2: Sliding window
    # We iterate until the window would go out of bounds
    for i in range(0, total_len - window_size + 1, stride):
        yield tokens[i : i + window_size]

    # Case 3: Handle the "Tail"
    # If the last sliding window didn't exactly align with the end,
    # we yield one final chunk containing the *last* window_size elements.
    # This creates a variable overlap for the last segment, but ensures full context.
    last_window_start = ((total_len - window_size) // stride) * stride
    tail_window_start = total_len - window_size
    if last_window_start != tail_window_start:
        yield tokens[-window_size:]


def generate_valid_tokens(read_data, k=3):
    """
    Yields valid k-mers and their ORIGINAL indices.
    Skips any k-mer containing 'N'.
    """
    seq = read_data["input_ids"]
    pattern = read_data["methylation_ids"]

    # We iterate up to len(seq) - k + 1
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]

        # 1. Check for 'N' in the window
        if "N" in kmer:
            continue

        center_idx = i + k // 2
        methylation_code = pattern[center_idx]

        # Yield the clean k-mer and its specific methylation label
        yield [kmer, methylation_code]


def prepare_methylbert_list_inference(
    results_df, dmr_label_column, seq_length=150, stride=75, soft_labels=False, is_binary=False
):
    """
    Prepares inference data with sliding window chunking.
    params:
        stride: How far to move the window (75 = 50% overlap for 150bp window)
    """
    data_list = [
        [
            "dna_seq",
            "methyl_seq",
            "dmr_ctype",
            "dmr_label",
            "ctype",
            "original_label",
            "read_name",
            "ncpgs_marked",
        ]
    ]

    for i, row in results_df.iterrows():
        # 1. Get the CLEAN stream of tokens (Ns removed)
        processed_read_full = list(generate_valid_tokens(row))
        read_name = row["read_name"]
        # If read was entirely Ns or empty, skip
        if not processed_read_full:
            continue

        # 2. Chunk the valid tokens
        # We process the read in chunks of 'seq_length'
        for chunk in chunk_tokens(
            processed_read_full, window_size=seq_length, stride=stride
        ):
            dna = " ".join([x[0] for x in chunk])
            methyl = "".join([x[1] for x in chunk])
            ncpgs_marked = methyl.count("0") + methyl.count("1")
            if is_binary:
                label = int(row["dmr_ctype_label"] == row["label"])
            else:
                label = row["soft_label"] if soft_labels else row["label"]

            o_label = row["original_label"]
            dmr_label = row[dmr_label_column]
            dmr_ctype = row["dmr_ctype_label"]

            data_list.append(
                [
                    dna,
                    methyl,
                    dmr_ctype,
                    dmr_label,
                    label,
                    o_label,
                    read_name,
                    ncpgs_marked,
                ]
            )

    return data_list


def chunk_read_data(
    seq,
    methylation_encoding,
    cpg_positions,
    meth_states,
    read_start,
    read_end,
    chrom,
    chunk_size,
    dmr_trees,
    strict,
):
    """
    Split a read into chunks of specified size and filter for DMR overlaps.
    Creates ONE ENTRY PER OVERLAPPING DMR with sequences clipped to DMR boundaries.

    This matches wgbstools behavior where fragments are clipped to marker boundaries
    before U/X/M classification.

    Returns:
    --------
    list of dict
        List of chunk dictionaries, one per overlapping DMR with:
        - seq, methylation_encoding: Original chunk data
        - seq_clipped, methylation_clipped: Clipped to DMR boundaries
        - clipped_methylated, clipped_unmethylated: CpG counts in clipped region
        - dmr_label, dmr_type: Single DMR info (not comma-separated)
    """
    chunks = []
    read_length = len(seq)

    for chunk_start_offset in range(0, read_length, chunk_size):
        chunk_end_offset = min(chunk_start_offset + chunk_size, read_length)

        chunk_gen_start = read_start + chunk_start_offset
        chunk_gen_end = read_start + chunk_end_offset

        chunk_seq = seq[chunk_start_offset:chunk_end_offset]
        chunk_meth_enc = methylation_encoding[chunk_start_offset:chunk_end_offset]

        # Collect CpGs for this chunk
        chunk_cpgs_pos = []
        chunk_cpg_states = []
        for cpg_pos, cpg_state in zip(cpg_positions, meth_states):
            if chunk_gen_start <= cpg_pos < chunk_gen_end:
                chunk_cpgs_pos.append(cpg_pos)
                chunk_cpg_states.append(cpg_state)

        # Original chunk CpG stats
        total_cpgs = len(chunk_cpgs_pos)
        methylated_cpgs = sum(1 for s in chunk_cpg_states if s == 1)
        unmethylated_cpgs = sum(1 for s in chunk_cpg_states if s == 0)
        methylation_rate = (methylated_cpgs / total_cpgs) if total_cpgs > 0 else 0.0

        # Get list of overlapping DMRs (not aggregated)
        overlapping_dmrs = get_overlapping_dmrs(
            chunk_gen_start, chunk_gen_end, chrom, dmr_trees, strict
        )

        # Create ONE ENTRY PER DMR with clipped sequences
        for dmr_info in overlapping_dmrs:
            dmr_start = dmr_info["dmr_start"]
            dmr_end = dmr_info["dmr_end"]

            # Calculate clipped region (intersection of chunk and DMR)
            clip_gen_start = max(chunk_gen_start, dmr_start)
            clip_gen_end = min(chunk_gen_end, dmr_end)

            # Calculate offsets within chunk for clipping
            clip_offset_start = clip_gen_start - chunk_gen_start
            clip_offset_end = clip_gen_end - chunk_gen_start

            # Clip sequence and methylation encoding to DMR boundaries
            seq_clipped = chunk_seq[clip_offset_start:clip_offset_end]
            meth_clipped = chunk_meth_enc[clip_offset_start:clip_offset_end]

            # Count CpGs in clipped region only (for UXM classification)
            clipped_methylated = 0
            clipped_unmethylated = 0
            clipped_total = 0
            for cpg_pos, cpg_state in zip(chunk_cpgs_pos, chunk_cpg_states):
                if clip_gen_start <= cpg_pos < clip_gen_end:
                    clipped_total += 1
                    if cpg_state == 1:
                        clipped_methylated += 1
                    elif cpg_state == 0:
                        clipped_unmethylated += 1

            clipped_meth_rate = (
                (clipped_methylated / clipped_total) if clipped_total > 0 else 0.0
            )

            chunk_data = {
                # Original chunk data
                "seq": chunk_seq,
                "methylation_encoding": chunk_meth_enc,
                "chromosome": chrom,
                "chunk_start": chunk_gen_start,
                "chunk_end": chunk_gen_end,
                "chunk_length": chunk_end_offset - chunk_start_offset,
                "chunk_offset_in_read": chunk_start_offset,
                "total_cpgs": total_cpgs,
                "methylated_cpgs": methylated_cpgs,
                "unmethylated_cpgs": unmethylated_cpgs,
                "methylation_rate": methylation_rate,
                # Clipped data (for UXM classification)
                "seq_clipped": seq_clipped,
                "methylation_clipped": meth_clipped,
                "clip_start": clip_gen_start,
                "clip_end": clip_gen_end,
                "clip_length": len(seq_clipped),
                "clipped_methylated": clipped_methylated,
                "clipped_unmethylated": clipped_unmethylated,
                "clipped_total": clipped_total,
                "clipped_meth_rate": clipped_meth_rate,
                # Single DMR info (not comma-separated)
                "overlaps_dmr": True,
                "dmr_label": dmr_info["name"],
                "dmr_type": dmr_info["type"],
                "dmr_start": dmr_start,
                "dmr_end": dmr_end,
                "overlap_bp": dmr_info["overlap_bp"],
                "overlap_pct": dmr_info["overlap_pct"],
            }
            chunks.append(chunk_data)

    return chunks
