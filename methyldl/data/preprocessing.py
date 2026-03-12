""""""

import multiprocessing
from functools import partial

from .dmr_overlap_analysis import _get_overlapping_dmrs
from .genome import process_chunk


def pretrain_data_preprocess(
    f_ref: str,
    k: int = 3,
    seq_len: int = 510,
    f_output: str = None,
    num_cores: int = None,
) -> None:
    """
    Parallel version of generating N bp length k-mers sequence from reference genome as pretrain data.

    f_ref : str
        Path to the reference fasta file
    k : int
        Number for k-mers (default=3)
    seq_len: int
        Base-pair length of generated sequences (default=510)
    f_output : str
        Path to the output file, an appropriate name
        will be automatically assigned if not given
    """
    # TODO adapt to all models by allowing argument k = -1
    # Get number of CPU cores and use 90% of them, but not less than one
    avaliable_cpus = multiprocessing.cpu_count()
    if num_cores is None:
        num_cores = max(1, int(avaliable_cpus * 0.9))
    else:
        assert (
            num_cores <= avaliable_cpus
        ), f"Number of cores should not be bigger than total number of cores avaliable: {avaliable_cpus}"

    # Assign default output file name if not provided
    if f_output is None:
        f_output = f_ref.replace("Raw", "Refined") + f"_{k}mers_seqlen{seq_len}.txt"

    valid_chromosomes = ["CHR" + str(i) for i in range(1, 23)] + ["CHRX", "CHRY"]

    # Read the file in chunks and process in parallel
    with open(f_ref, "r") as fp_ref:
        lines = fp_ref.readlines()

    # Split the file into chunks for parallel processing
    chunk_size = len(lines) // num_cores
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

    # Create a partial function with fixed arguments for parallel processing
    process_partial = partial(
        process_chunk, k=k, seq_len=seq_len, valid_chromosomes=valid_chromosomes
    )

    # Set up multiprocessing pool and process chunks in parallel
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(process_partial, chunks)

    # Flatten the results and write to the output file
    with open(f_output, "w") as fp_out:
        for result in results:
            for sentence in result:
                fp_out.write(sentence + "\n")

    print(f"Processing completed. Output saved to {f_output}")


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
    if total_len % stride != 0:
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

        center_idx = i + 1
        methylation_code = pattern[center_idx]

        # Yield the clean k-mer and its specific methylation label
        yield [kmer, methylation_code]


def prepare_methylbert_list_inference(
    results_df, dmr_label_column, seq_length=150, stride=75
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
            # Metadata propagation
            # Note: You might want to track which chunk this is (e.g., read_id_0, read_id_1)
            # but for bulk inference, this format works.
            label = 39
            o_label = 39
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
        chunk_cpgs = []
        chunk_cpg_states = []
        for cpg_pos, cpg_state in zip(cpg_positions, meth_states):
            if chunk_gen_start <= cpg_pos < chunk_gen_end:
                chunk_cpgs.append(cpg_pos)
                chunk_cpg_states.append(cpg_state)

        # Original chunk CpG stats
        total_cpgs = len(chunk_cpgs)
        methylated_cpgs = sum(1 for s in chunk_cpg_states if s == 1)
        unmethylated_cpgs = sum(1 for s in chunk_cpg_states if s == 0)
        methylation_rate = (methylated_cpgs / total_cpgs) if total_cpgs > 0 else 0.0

        # Get list of overlapping DMRs (not aggregated)
        overlapping_dmrs = _get_overlapping_dmrs(
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
            for cpg_pos, cpg_state in zip(chunk_cpgs, chunk_cpg_states):
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
