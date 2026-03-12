import argparse
import multiprocessing
from functools import partial
from typing import List

import pysam


def generate_kmer_str_with_overlap(sequence: str, kmer: int = 3) -> str:
    """
    Construct k-mers with overlap from the original DNA sequence

    sequence: str
        sequence of nucleotides to convert to k-mers
    kmer : int
        Number for k-mers (default=3)
    """
    return " ".join([sequence[i : i + kmer] for i in range(len(sequence) - kmer + 1)])


def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])


def process_chunk(chunk, k, seq_len, valid_chromosomes):
    """Processes a chunk of the reference file."""
    cur_line = ""
    collect_data = False
    results = []

    for line in chunk:
        line = line.strip().upper()

        if line.count("N") > 0:
            cur_line = ""
            continue
        elif line.count(">") > 0:
            chromosome = line.split(">")[1]
            collect_data = chromosome in valid_chromosomes
            cur_line = ""
            continue

        if collect_data:
            cur_line += line
            while len(cur_line) >= seq_len:
                new_line = cur_line[:seq_len]
                cur_line = cur_line[seq_len:]
                sentence = generate_kmer_str_with_overlap(new_line, kmer=k)
                results.append(sentence)

    return results


def collapse_methylation(seq: List) -> int:
    """
    Computes aggregated methylation tag based on the tags applied to individual nucleotides in a given sequence seq
    """
    if not seq:
        return 2  # Other for empty sequences
    if any(x == 1 for x in seq):
        return 1  #  Methylated cytosine at CpG context
    elif any(x == 0 for x in seq):
        return 0  #  Unmethylated cytosine at CpG context
    else:
        return 2  #  Other


def count_reference_cpgs(ref_fasta, chrom, start, end):
    """
    Count CpG dinucleotides in the reference genome for a given region.

    This is used to count the total number of CpGs a fragment spans,
    including those in the insert region between paired-end mates.

    Parameters:
    -----------
    ref_fasta : pysam.FastaFile
        Open reference genome file
    chrom : str
        Chromosome name
    start : int
        Start position (0-based)
    end : int
        End position (exclusive)

    Returns:
    --------
    int
        Number of CpG dinucleotides in the region
    """
    try:
        seq = ref_fasta.fetch(chrom, start, end + 1).upper()
        return seq.count("CG")
    except:
        return 0


def add_reference_cpg_counts(df, reference_path, verbose=True):
    """
    Add reference-based CpG counts to the DataFrame.

    This counts ALL CpGs in each fragment's genomic span using the reference genome,
    matching wgbs_tools' behavior. The difference between reference CpGs and
    called CpGs (M + U) gives the number of "unknown" CpGs.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with read/fragment data containing:
        - chromosome, read_start, read_end: Fragment coordinates
        - methylated_cpgs, unmethylated_cpgs: Called CpG counts
    reference_path : str
        Path to reference genome FASTA
    verbose : bool
        Print progress

    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns:
        - ref_cpg_count: Total CpGs in fragment span (from reference)
        - unknown_cpgs: CpGs in span but not called (ref_cpg_count - M - U)
    """
    if len(df) == 0:
        return df

    if verbose:
        print(f"Counting reference CpGs for {len(df)} fragments...")

    ref_fasta = pysam.FastaFile(reference_path)

    ref_counts = []
    for _, row in df.iterrows():
        count = count_reference_cpgs(
            ref_fasta, row["chromosome"], row["read_start"], row["read_end"]
        )
        ref_counts.append(count)

    ref_fasta.close()

    df = df.copy()
    df["ref_cpg_count"] = ref_counts
    df["unknown_cpgs"] = (
        df["ref_cpg_count"] - df["methylated_cpgs"] - df["unmethylated_cpgs"]
    )
    # Ensure non-negative (edge cases)
    df["unknown_cpgs"] = df["unknown_cpgs"].clip(lower=0)

    if verbose:
        total_ref = df["ref_cpg_count"].sum()
        total_m = df["methylated_cpgs"].sum()
        total_u = df["unmethylated_cpgs"].sum()
        total_x = df["unknown_cpgs"].sum()
        print(
            f"Reference CpG counts - Total: {total_ref}, M: {total_m}, U: {total_u}, X: {total_x}"
        )

    return df
