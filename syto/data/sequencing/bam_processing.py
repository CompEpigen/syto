"""
BAM file processing for read-level CpG methylation extraction.

Supports two sequencing technologies:
- ONT (Oxford Nanopore): Uses MM/ML tags for methylation calls
- WGBS (Whole Genome Bisulfite Sequencing): Uses reference genome comparison

Pipeline flow:
    BAM file → genomic chunking → parallel read processing →
    methylation extraction → optional paired-end merging → DataFrame

Coordinate system: 0-based, half-open [start, end) matching BAM/pysam conventions.
Default quality filters match SAMtools: -f 3 -F 1796 -q 10
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool

import numba as nb
import numpy as np
import pandas as pd
import pysam

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================


class DataType(Enum):
    """Supported sequencing data types."""

    ONT = "ont"
    WGBS = "wgbs"


# Methylation states used in CpG scan results and encodings.
# These are used as literal ints in numba functions and as
# string characters ("0", "1", "2") in methylation encoding strings.
METHYLATED = 1
UNMETHYLATED = 0
UNKNOWN = 2

# ASCII values for CpG scanning in numba
CG_ASCII = np.array([ord("C"), ord("G")], dtype=np.uint8)


# ============================================================
# Data classes
# ============================================================


@dataclass
class ChunkTask:
    """Parameters for processing a single genomic chunk in parallel."""

    chromosome: str
    chunk_start: int
    chunk_end: int
    bam_path: str
    interesting_chromosomes: list
    methyl_tr: int
    data_type: str
    reference_path: str | None
    min_mapq: int
    require_flags: int
    exclude_flags: int
    min_cpgs: int


# ============================================================
# Data type detection
# ============================================================


def detect_bam_data_type(bam_path, sample_size=1000):
    """
    Detect if BAM file is ONT (has ML tags) or WGBS (no ML tags).
    """
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        reads_checked = 0
        has_ml_tag = 0

        for read in bam:
            if read.is_unmapped:
                continue

            try:
                read.get_tag("ML")
                has_ml_tag += 1
            except KeyError:
                pass

            reads_checked += 1
            if reads_checked >= sample_size:
                break

        if reads_checked == 0:
            raise ValueError("No mapped reads found in BAM file")

        if has_ml_tag / reads_checked > 0.5:
            return DataType.ONT.value
        else:
            return DataType.WGBS.value


# ============================================================
# MM/ML tag parsing
# ============================================================


def parse_mm_tag(mm_tag):
    """
    Parse MM tag to extract modification types and their positions.

    MM tag format: "A+a,10,5,3;C+m,2,1,4;G+g,5,2;"
    Returns dict with modification info and cumulative counts for ML indexing.

    Parameters
    ----------
    mm_tag : str
        MM tag string from BAM file

    Returns
    -------
    dict with keys:
        'modifications': list of tuples (base, modification_code, skip_positions)
        'cpg_mod_index': index of C+m modification (-1 if not found)
        'cpg_ml_offset': offset in ML array where C+m probabilities start
        'cpg_count': number of C+m positions
    """
    if not mm_tag:
        return None

    # Split by semicolon to get each modification type
    mod_groups = mm_tag.split(";")

    modifications = []
    cumulative_count = 0
    cpg_mod_index = -1
    cpg_ml_offset = 0
    cpg_count = 0

    for idx, group in enumerate(mod_groups):
        if not group.strip():
            continue

        # Parse format: "C+m,2,1,4" or "C+m?,2,1,4"
        parts = group.split(",")
        if len(parts) < 1:
            continue

        # Extract base and modification code
        mod_info = parts[0]
        # Match patterns like "C+m" or "C+m?" or "A+a"
        match = re.match(r"([ACGT])\+([a-z\?]+)", mod_info)
        if not match:
            continue

        base = match.group(1)
        mod_code = match.group(2)

        # Extract skip positions
        skip_positions = [int(x) for x in parts[1:] if x.strip()]
        mod_count = len(skip_positions)

        modifications.append((base, mod_code, skip_positions))

        # Check if this is C+m (CpG methylation)
        # Can be "C+m" or "C+m?" where ? indicates uncertainty
        if base == "C" and mod_code.startswith("m"):
            cpg_mod_index = idx
            cpg_ml_offset = cumulative_count
            cpg_count = mod_count

        cumulative_count += mod_count

    return {
        "modifications": modifications,
        "cpg_mod_index": cpg_mod_index,
        "cpg_ml_offset": cpg_ml_offset,
        "cpg_count": cpg_count,
        "total_mods": cumulative_count,
    }


def extract_cpg_ml_values(ml_array, mm_info):
    """
    Extract only the ML probability values corresponding to C+m modifications.

    Parameters
    ----------
    ml_array : array-like
        Full ML probability array
    mm_info : dict
        Output from parse_mm_tag()

    Returns
    -------
    np.array
        ML values for C+m positions only
    """
    if mm_info is None or mm_info["cpg_mod_index"] == -1:
        return np.array([])

    offset = mm_info["cpg_ml_offset"]
    count = mm_info["cpg_count"]

    # Extract the slice of ML values for C+m
    if offset + count <= len(ml_array):
        return np.array(ml_array[offset : offset + count])
    else:
        # Handle edge case where ML array is shorter than expected
        return np.array(ml_array[offset:])


# ============================================================
# Numba-accelerated CpG scanning
# ============================================================


@nb.njit(fastmath=True, cache=True)
def cpg_scan(seq_bytes, ml_values, tr=122):
    """Return CpG offsets (0-based) and methylation states for one read (ONT)."""
    n = len(seq_bytes) - 1
    pos_buf = []
    state_buf = []
    cg_idx = 0
    for i in range(n):
        if seq_bytes[i] == CG_ASCII[0] and seq_bytes[i + 1] == CG_ASCII[1]:
            pos_buf.append(i)
            if cg_idx < len(ml_values):
                state_buf.append(METHYLATED if ml_values[cg_idx] > tr else UNMETHYLATED)
            else:
                state_buf.append(UNKNOWN)
            cg_idx += 1
    return pos_buf, state_buf


@nb.njit(fastmath=True, cache=True)
def cpg_scan_wgbs_forward(ref_bytes, read_bytes):
    """
    Scan CpGs in WGBS data (forward strand).
    Compare reference vs read to infer methylation.
    """
    n = len(ref_bytes) - 1
    pos_buf = []
    state_buf = []

    for i in range(n):
        if ref_bytes[i] == ord("C") and ref_bytes[i + 1] == ord("G"):
            pos_buf.append(i)

            if i < len(read_bytes):
                if read_bytes[i] == ord("C"):
                    state_buf.append(METHYLATED)
                elif read_bytes[i] == ord("T"):
                    state_buf.append(UNMETHYLATED)
                else:
                    state_buf.append(UNKNOWN)
            else:
                state_buf.append(UNKNOWN)

    return pos_buf, state_buf


@nb.njit(fastmath=True, cache=True)
def cpg_scan_wgbs_reverse(ref_bytes, read_bytes):
    """
    Scan CpGs in WGBS data (reverse strand).
    """
    n = len(ref_bytes) - 1
    pos_buf = []
    state_buf = []

    for i in range(n):
        if ref_bytes[i] == ord("C") and ref_bytes[i + 1] == ord("G"):
            pos_buf.append(i)

            if i + 1 < len(read_bytes):
                if read_bytes[i + 1] == ord("G"):
                    state_buf.append(METHYLATED)
                elif read_bytes[i + 1] == ord("A"):
                    state_buf.append(UNMETHYLATED)
                else:
                    state_buf.append(UNKNOWN)
            else:
                state_buf.append(UNKNOWN)

    return pos_buf, state_buf


# ============================================================
# CIGAR sequence processing
# ============================================================


def clean_cigar_sequence(read):
    """
    Process CIGAR string to align read sequence to reference coordinates.

    Similar to wgbstools' clean_CIGAR function:
    - Removes inserted bases (I) from the read sequence
    - Adds 'N' placeholders for deleted bases (D) in reference
    - Removes soft-clipped bases (S)

    Returns
    -------
    str: Processed read sequence that aligns 1:1 with reference positions
    """
    if read.cigartuples is None:
        return read.query_alignment_sequence

    seq = read.query_sequence  # Full query sequence including soft clips
    result = []
    seq_pos = 0  # Position in query sequence

    for op, length in read.cigartuples:
        if op == 0:  # M - Match/mismatch: consume both query and reference
            result.append(seq[seq_pos : seq_pos + length])
            seq_pos += length
        elif op == 1:  # I - Insertion: consume query only, skip these bases
            seq_pos += length  # Skip inserted bases
        elif op == 2:  # D - Deletion: consume reference only, add placeholder
            result.append("N" * length)  # Add N's for deleted reference positions
        elif op == 3:  # N - Reference skip (intron): add placeholder
            result.append("N" * length)
        elif op == 4:  # S - Soft clip: consume query only, skip these bases
            seq_pos += length
        elif op == 5:  # H - Hard clip: doesn't consume query
            pass
        elif op == 7:  # = - Sequence match: same as M
            result.append(seq[seq_pos : seq_pos + length])
            seq_pos += length
        elif op == 8:  # X - Sequence mismatch: same as M
            result.append(seq[seq_pos : seq_pos + length])
            seq_pos += length

    return "".join(result)


# ============================================================
# CpG statistics
# ============================================================


def compute_cpg_stats(meth_states):
    """
    Compute CpG methylation statistics from a list of states.

    Only METHYLATED and UNMETHYLATED calls are counted in the total;
    UNKNOWN states are excluded.

    Parameters
    ----------
    meth_states : list of int
        Methylation states (METHYLATED=1, UNMETHYLATED=0, UNKNOWN=2)

    Returns
    -------
    dict with 'methylated', 'unmethylated', 'total', 'methylation_rate'
    """
    methylated = sum(1 for s in meth_states if s == METHYLATED)
    unmethylated = sum(1 for s in meth_states if s == UNMETHYLATED)
    total = methylated + unmethylated
    rate = methylated / total if total > 0 else 0.0
    return {
        "methylated": methylated,
        "unmethylated": unmethylated,
        "total": total,
        "methylation_rate": rate,
    }


# ============================================================
# Read filtering and CpG extraction helpers
# ============================================================


def _passes_quality_filters(
    read,
    interesting_chromosomes=None,
    min_mapq=10,
    require_flags=3,
    exclude_flags=1796,
):
    """
    Check if a read passes quality filters matching SAMtools -f 3 -F 1796 -q 10.

    Parameters
    ----------
    read : pysam.AlignedSegment
        BAM read object
    interesting_chromosomes : list or None
        List of chromosomes to process (None = process all)
    min_mapq : int
        Minimum mapping quality threshold
    require_flags : int
        SAM flags that must ALL be set (SAMtools -f)
    exclude_flags : int
        SAM flags to exclude if ANY are set (SAMtools -F)

    Returns
    -------
    bool
        True if the read passes all filters
    """
    # Filter by chromosome if specified
    if interesting_chromosomes and read.reference_name not in interesting_chromosomes:
        return False

    # Check if ALL require_flags are set (SAMtools -f)
    if require_flags and (read.flag & require_flags) != require_flags:
        return False

    # Check if ANY of the exclude_flags are set (SAMtools -F)
    if exclude_flags and (read.flag & exclude_flags):
        return False

    # Skip reads below minimum mapping quality (SAMtools -q)
    if read.mapping_quality < min_mapq:
        return False

    return True


def _extract_cpgs_ont(read, methyl_tr=122):
    """
    Extract CpG positions and methylation states from an ONT read.

    Uses MM/ML tags to identify C+m modifications and their probabilities.

    Parameters
    ----------
    read : pysam.AlignedSegment
        BAM read with MM and ML tags
    methyl_tr : int
        Methylation probability threshold (default: 122, ~48% on 0-255 scale)

    Returns
    -------
    tuple of (pos_in_read, meth_states, seq, read_length) or None
        Returns None if tags are missing or no CpG modifications found.
    """
    try:
        ml_values_full = read.get_tag("ML")
        mm_tag = read.get_tag("MM")
    except KeyError:
        return None

    # Parse MM tag to find C+m modifications
    mm_info = parse_mm_tag(mm_tag)
    if mm_info is None or mm_info["cpg_mod_index"] == -1:
        return None

    # Extract only the ML values for C+m
    ml_values = extract_cpg_ml_values(ml_values_full, mm_info)
    if len(ml_values) == 0:
        return None

    seq = read.get_forward_sequence()
    read_length = len(seq)

    # Scan for CpGs using extracted C+m probabilities
    pos_in_read, meth_states = cpg_scan(seq.encode(), ml_values, methyl_tr)
    return pos_in_read, meth_states, seq, read_length


def _extract_cpgs_wgbs(read, ref_fasta):
    """
    Extract CpG positions and methylation states from a WGBS read.

    Compares bisulfite-converted read sequence against the reference genome
    to infer methylation at CpG sites.

    Parameters
    ----------
    read : pysam.AlignedSegment
        BAM read object
    ref_fasta : pysam.FastaFile
        Open reference genome file

    Returns
    -------
    tuple of (pos_in_read, meth_states, ref_seq, read_length) or None
        Returns None if reference fetch or CIGAR cleaning fails.
    """
    chromosome = read.reference_name
    r_beg, r_end = read.reference_start, read.reference_end

    # Get reference sequence for this region
    try:
        # Fetch +1 character in case region ends with CG
        ref_seq = ref_fasta.fetch(chromosome, r_beg, r_end + 1).upper()
        # Only keep the extra character if it completes a terminal CG
        if ref_seq[-2:] != "CG":
            ref_seq = ref_seq[:-1]
    except (ValueError, KeyError, IndexError, RuntimeError, OSError):
        return None

    read_seq = clean_cigar_sequence(read)
    if read_seq is None:
        return None

    read_length = len(ref_seq)

    # Scan for CpGs and infer methylation from bisulfite conversion
    if read.is_reverse:
        pos_in_read, meth_states = cpg_scan_wgbs_reverse(
            ref_seq.encode(), read_seq.encode()
        )
    else:
        pos_in_read, meth_states = cpg_scan_wgbs_forward(
            ref_seq.encode(), read_seq.encode()
        )

    return pos_in_read, meth_states, ref_seq, read_length


def _build_methylation_encoding(pos_in_read, meth_states, encoding_length):
    """
    Build a per-base methylation encoding string.

    Parameters
    ----------
    pos_in_read : list of int
        0-based positions of CpGs within the sequence
    meth_states : list of int
        Methylation states at each CpG position
    encoding_length : int
        Length of the output encoding string

    Returns
    -------
    str
        String of length encoding_length where each character is
        '0' (unmethylated), '1' (methylated), or '2' (unknown/no CpG)
    """
    meth_enc = [str(UNKNOWN)] * encoding_length
    for pos, state in zip(pos_in_read, meth_states):
        if pos < encoding_length:
            meth_enc[pos] = str(state)
    return "".join(meth_enc)


def _build_read_record(read, pos_in_read, meth_states, seq, data_type):
    """
    Assemble the output dictionary for a single processed read.

    Parameters
    ----------
    read : pysam.AlignedSegment
        BAM read object
    pos_in_read : list of int
        0-based CpG positions within the sequence
    meth_states : list of int
        Methylation states at each CpG
    seq : str
        Sequence (forward sequence for ONT, reference for WGBS)
    data_type : str
        'ont' or 'wgbs'

    Returns
    -------
    dict
        Read-level methylation data
    """
    r_beg, r_end = read.reference_start, read.reference_end
    cpg_positions = [r_beg + off for off in pos_in_read]
    methylation_encoding = _build_methylation_encoding(
        pos_in_read, meth_states, len(seq)
    )

    total_cpgs = len(pos_in_read)
    methylated_cpgs = sum(1 for s in meth_states if s == METHYLATED)
    unmethylated_cpgs = sum(1 for s in meth_states if s == UNMETHYLATED)
    methylation_rate = methylated_cpgs / total_cpgs if total_cpgs > 0 else 0.0

    return {
        "read_name": read.query_name,
        "chromosome": read.reference_name,
        "read_start": r_beg,
        "read_end": r_end,
        "read_length": r_end - r_beg,
        "seq": seq,
        "methylation_encoding": methylation_encoding,
        "cpg_positions": cpg_positions,
        "meth_states": list(meth_states),
        "total_cpgs": total_cpgs,
        "methylated_cpgs": methylated_cpgs,
        "unmethylated_cpgs": unmethylated_cpgs,
        "methylation_rate": methylation_rate,
        "mapping_quality": read.mapping_quality,
        "is_reverse": read.is_reverse,
        "data_type": data_type,
    }


# ============================================================
# Single read processing
# ============================================================


def process_single_read(
    read,
    data_type,
    methyl_tr=122,
    ref_fasta=None,
    interesting_chromosomes=None,
    min_mapq=10,
    require_flags=3,
    exclude_flags=1796,
    min_cpgs=1,
):
    """
    Process a single read and return its methylation data.

    Parameters
    ----------
    read : pysam.AlignedSegment
        BAM read object
    data_type : str
        'ont' or 'wgbs'
    methyl_tr : int
        Methylation threshold for ONT (default: 122)
    ref_fasta : pysam.FastaFile or None
        Reference genome (required for WGBS)
    interesting_chromosomes : list or None
        List of chromosomes to process (None = process all)
    min_mapq : int
        Minimum mapping quality threshold (default: 10, matching SAMtools -q 10)
    require_flags : int
        SAM flags that must ALL be set (default: 3 = paired + properly paired,
        matching SAMtools -f 3). Set to 0 to disable required flag filtering.
        Flag breakdown: 1 (paired) + 2 (properly paired)
    exclude_flags : int
        SAM flags to exclude if ANY are set (default: 1796 = unmapped + secondary
        + failed QC + duplicate, matching SAMtools -F 1796). Set to 0 to disable.
        Flag breakdown: 4 (unmapped) + 256 (secondary) + 512 (failed QC)
        + 1024 (duplicate)
    min_cpgs : int
        Minimum number of CpG sites required in the read (default: 1).
        Reads with fewer CpGs are skipped.

    Returns
    -------
    list of dict
        List with a single dictionary containing read-level methylation data,
        or empty list if read is filtered out.
    """
    # Quality filtering
    if not _passes_quality_filters(
        read, interesting_chromosomes, min_mapq, require_flags, exclude_flags
    ):
        return []

    # Data-type-specific CpG extraction
    if data_type == DataType.ONT.value:
        result = _extract_cpgs_ont(read, methyl_tr)
    elif data_type == DataType.WGBS.value:
        if ref_fasta is None:
            raise ValueError("Reference genome (ref_fasta) required for WGBS data")
        result = _extract_cpgs_wgbs(read, ref_fasta)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if result is None:
        return []

    pos_in_read, meth_states, seq, _read_length = result

    # Filter by minimum CpG count
    if len(pos_in_read) < min_cpgs:
        return []

    return [_build_read_record(read, pos_in_read, meth_states, seq, data_type)]


# ============================================================
# Chunk-level processing
# ============================================================


def process_tabular_chunk(task):
    """
    Process a genomic chunk and extract read-level methylation data.

    Supports both ONT and WGBS data. Quality filtering defaults match
    SAMtools -f 3 -F 1796 -q 10 with a minimum CpG count filter.

    Parameters
    ----------
    task : ChunkTask
        Processing parameters for this genomic chunk.
    """
    tabular_data = []

    # Open reference genome if WGBS
    ref_fasta = None
    if task.data_type == DataType.WGBS.value:
        if task.reference_path is None:
            raise ValueError("Reference genome path required for WGBS data")
        ref_fasta = pysam.FastaFile(task.reference_path)

    with pysam.AlignmentFile(task.bam_path, "rb") as bam:
        for read in bam.fetch(task.chromosome, task.chunk_start, task.chunk_end):
            # Skip reads that don't START within this chunk to avoid duplicate
            # counting.  bam.fetch() returns all reads that OVERLAP the region,
            # but we only want to process each read once (in the chunk where
            # it starts).
            if (
                read.reference_start < task.chunk_start
                or read.reference_start >= task.chunk_end
            ):
                continue

            # Process single read with quality filters
            read_data = process_single_read(
                read=read,
                data_type=task.data_type,
                methyl_tr=task.methyl_tr,
                ref_fasta=ref_fasta,
                interesting_chromosomes=task.interesting_chromosomes,
                min_mapq=task.min_mapq,
                require_flags=task.require_flags,
                exclude_flags=task.exclude_flags,
                min_cpgs=task.min_cpgs,
            )

            # Add all read entries (each read returns a list with one dict)
            tabular_data.extend(read_data)

    if ref_fasta is not None:
        ref_fasta.close()

    return tabular_data


# ============================================================
# BAM-level orchestration
# ============================================================


def process_bam_with_chunking(
    bam_path,
    chromosomes,
    n_jobs=4,
    chunk_size_genomic=1_000_000,
    methyl_tr=122,
    reference_path=None,
    data_type=None,
    min_mapq=10,
    require_flags=3,
    exclude_flags=1796,
    min_cpgs=1,
    merge_pairs=True,
):
    """
    Process BAM file and extract read-level methylation data.
    Supports both ONT and WGBS data.

    For ONT: Properly handles MM tags with multiple modifications (A+a, C+m, etc.)
    For WGBS: Uses reference genome comparison

    Parameters
    ----------
    bam_path : str
        Path to BAM file
    chromosomes : list
        List of chromosomes to process
    n_jobs : int
        Number of parallel jobs (default: 4)
    chunk_size_genomic : int
        Genomic chunk size for parallel processing (default: 1M)
    methyl_tr : int
        Methylation threshold for ONT (default: 122)
    reference_path : str
        Path to reference genome FASTA (required for WGBS)
    data_type : str
        'ont' or 'wgbs'. If None, will auto-detect.
    min_mapq : int
        Minimum mapping quality threshold (default: 10, matching SAMtools -q 10).
        Set to 0 to disable MAPQ filtering.
    require_flags : int
        SAM flags that must ALL be set (default: 3 = paired + properly paired,
        matching SAMtools -f 3). Set to 0 to disable required flag filtering.
        Flag breakdown: 1 (paired) + 2 (properly paired)
    exclude_flags : int
        SAM flags to exclude if ANY are set (default: 1796, matching SAMtools -F 1796).
        Flag breakdown: 4 (unmapped) + 256 (secondary) + 512 (failed QC) + 1024 (duplicate)
        Set to 0 to disable flag filtering.
    min_cpgs : int
        Minimum number of CpG sites required in the read (default: 1).
        Reads with fewer CpGs are skipped.
    merge_pairs : bool
        If True, merge paired-end reads (mates) into single fragments, similar to
        wgbs_tools' bam2pat. Both mates are combined with their methylation patterns
        merged, spanning the full fragment. Also adds reference-based CpG counts
        including 'unknown_cpgs' for CpGs in the insert region. (default: True)

    Returns
    -------
    pd.DataFrame
        DataFrame with read-level methylation data.
        If merge_pairs=True, returns merged fragment entries with additional columns:
        - ref_cpg_count: Total CpGs in fragment span (from reference genome)
        - unknown_cpgs: CpGs in span but not called (in insert region between mates)
        These match wgbs_tools' behavior for counting CpGs.
    """

    # Auto-detect data type if not specified
    if data_type is None:
        logger.info("Auto-detecting data type...")
        data_type = detect_bam_data_type(bam_path)
        logger.info("Detected data type: %s", data_type.upper())

    # Validate inputs
    if data_type == DataType.WGBS.value and reference_path is None:
        raise ValueError("Reference genome path is required for WGBS data")

    # Get chromosome lengths from BAM
    logger.info("Reading BAM file...")
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        chr_lengths = {ref: length for ref, length in zip(bam.references, bam.lengths)}

    # Generate genomic chunks for parallel processing
    tasks = []
    for chromosome in chromosomes:
        if chromosome not in chr_lengths:
            logger.warning("%s not found in BAM file", chromosome)
            continue

        chr_len = chr_lengths[chromosome]
        for start in range(0, chr_len, chunk_size_genomic):
            end = min(start + chunk_size_genomic, chr_len)
            tasks.append(
                ChunkTask(
                    chromosome=chromosome,
                    chunk_start=start,
                    chunk_end=end,
                    bam_path=bam_path,
                    interesting_chromosomes=chromosomes,
                    methyl_tr=methyl_tr,
                    data_type=data_type,
                    reference_path=reference_path,
                    min_mapq=min_mapq,
                    require_flags=require_flags,
                    exclude_flags=exclude_flags,
                    min_cpgs=min_cpgs,
                )
            )

    logger.info("Processing %d genomic chunks with %d workers...", len(tasks), n_jobs)

    # Process in parallel
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            results = pool.map(process_tabular_chunk, tasks)
    else:
        results = [process_tabular_chunk(task) for task in tasks]

    # Flatten results
    all_data = []
    for result in results:
        all_data.extend(result)

    logger.info("Collected %d reads", len(all_data))

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Optionally merge paired-end reads into fragments
    if merge_pairs and len(df) > 0:
        logger.info("Merging paired-end reads into fragments...")
        df = merge_paired_reads(df, verbose=True)

    return df


# ============================================================
# Paired-end read merging
# ============================================================


def merge_paired_reads(df, verbose=True):
    """
    Merge paired-end reads into single fragments, similar to wgbs_tools' bam2pat.

    For paired-end sequencing, both mates represent the same DNA fragment and should
    be merged. This function:
    1. Groups reads by read_name AND grg_label (both mates share the same name, merge per grg)
    2. Merges methylation information, handling overlapping CpGs
    3. Creates a single fragment entry spanning both mates

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from process_bam_with_chunking with columns:
        - read_name: Read identifier (shared by both mates)
        - grg_label: Single GRG associated with this entry
        - read_start, read_end: Genomic coordinates
        - chromosome: Chromosome
        - methylation_encoding: String encoding methylation states
        - seq: Sequence
        - clipped_methylated, clipped_unmethylated, clipped_total: CpG counts in clipped region
        - and other metadata columns
    verbose : bool
        Print progress information

    Returns
    -------
    pd.DataFrame
        DataFrame with paired mates merged into single fragment entries.
        Each entry is for a single GRG with clipped CpG counts for UXM classification.
    """
    if len(df) == 0:
        return df

    # Determine grouping columns - use grg_label if available
    if "grg_label" in df.columns:
        group_cols = ["read_name", "grg_label"]
    else:
        group_cols = ["read_name"]

    # Count reads per (name, grg_label) to identify pairs vs singletons
    read_counts = df.groupby(group_cols).size()
    singletons = read_counts[read_counts == 1].index
    pairs = read_counts[read_counts == 2].index
    multiplets = read_counts[read_counts > 2].index

    if verbose:
        logger.info("Read distribution (grouped by %s):", group_cols)
        logger.info("  Singletons: %d", len(singletons))
        logger.info("  Paired (2 mates): %d", len(pairs))
        logger.info(
            "  Multiplets (>2): %d (will be treated as separate entries)",
            len(multiplets),
        )

    # Separate singletons, pairs, and multiplets using DataFrame slicing
    # (avoids row-wise iteration for singletons and multiplets)
    if len(group_cols) == 2:
        idx = df.set_index(group_cols).index
        singleton_df = df[idx.isin(singletons)]
        pairs_df = df[idx.isin(pairs)]
        multiplet_df = df[idx.isin(multiplets)]
    else:
        singleton_df = df[df["read_name"].isin(singletons)]
        pairs_df = df[df["read_name"].isin(pairs)]
        multiplet_df = df[df["read_name"].isin(multiplets)]

    # Process pairs — merge mates (requires per-pair logic)
    merged_pairs = []
    for _key, group in pairs_df.groupby(group_cols):
        if len(group) != 2:
            continue

        mate1, mate2 = group.iloc[0], group.iloc[1]

        # Ensure mate1 is the one with smaller start position
        if mate2["read_start"] < mate1["read_start"]:
            mate1, mate2 = mate2, mate1

        merged_pairs.append(_merge_mate_pair(mate1, mate2))

    merged_pairs_df = pd.DataFrame(merged_pairs) if merged_pairs else pd.DataFrame()

    # Concatenate all groups efficiently
    result_df = pd.concat(
        [singleton_df, merged_pairs_df, multiplet_df], ignore_index=True
    )

    if verbose:
        logger.info(
            "Merged result: %d fragments (from %d read entries)",
            len(result_df),
            len(df),
        )

    return result_df


def _merge_mate_pair(mate1, mate2):
    """
    Merge two mates of a paired-end read into a single fragment.

    Parameters
    ----------
    mate1 : pd.Series
        First mate (should have smaller read_start)
    mate2 : pd.Series
        Second mate

    Returns
    -------
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

    # Handle grg info - use singular grg_label if available, fall back to grg_labels
    grg_label = mate1.get("grg_label", "") or mate1.get("grg_labels", "")
    grg_type = mate1.get("grg_type", "") or mate1.get("grg_types", "")
    grg_start = mate1.get("grg_start", 0)
    grg_end = mate1.get("grg_end", 0)

    # Merge clipped sequences if available
    # For merged fragments, we need to recalculate clipped data within grg boundaries
    if "seq_clipped" in mate1.index and grg_start and grg_end:
        # Merge clipped sequences within grg boundaries
        clipped_encoding, clipped_stats = _merge_methylation_encodings(
            mate1.get("seq_clipped", ""),
            mate1.get("methylation_clipped", ""),
            mate1.get("clip_start", mate1["read_start"]),
            mate2.get("seq_clipped", ""),
            mate2.get("methylation_clipped", ""),
            mate2.get("clip_start", mate2["read_start"]),
            grg_start,
            grg_end,
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
        # grg info (singular)
        "overlaps_grg": mate1.get("overlaps_grg", False)
        or mate2.get("overlaps_grg", False),
        "grg_label": grg_label,
        "grg_type": grg_type,
        "grg_start": grg_start,
        "grg_end": grg_end,
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


# ============================================================
# Methylation encoding merging
# ============================================================


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
    unknown_str = str(UNKNOWN)

    # Initialize merged arrays with 'unknown' state
    merged_enc = [unknown_str] * frag_length
    merged_seq = ["N"] * frag_length

    # Fill in mate1 data
    for i, base1 in enumerate(seq1):
        if i >= len(enc1):
            break
        pos = (start1 - frag_start) + i
        if 0 <= pos < frag_length:
            merged_seq[pos] = base1
            merged_enc[pos] = enc1[i]

    # Fill in mate2 data with consensus logic matching wgbstools
    for i, base2 in enumerate(seq2):
        if i >= len(enc2):
            break
        pos = (start2 - frag_start) + i
        if 0 <= pos < frag_length:
            existing = merged_enc[pos]
            merged_seq[pos] = base2
            meth = enc2[i]

            if existing == unknown_str:
                # No data from mate1 (or unknown), use mate2
                merged_enc[pos] = meth
            elif meth == unknown_str:
                # No data from mate2, keep mate1
                pass
            elif existing == meth:
                # Both agree, keep the value
                pass
            else:
                # CONFLICT: both are known ('0' or '1') but different
                # Mark as unknown, matching wgbstools behavior
                merged_enc[pos] = unknown_str

    # Calculate CpG statistics from merged encoding
    stats = compute_cpg_stats([int(x) for x in merged_enc])

    return {
        "seq": "".join(merged_seq),
        "methylation_encoding": "".join(merged_enc),
    }, stats
