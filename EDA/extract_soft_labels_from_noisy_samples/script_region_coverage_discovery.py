"""
In this script,

we compute, for every dmr region in the atlas Data/Atlas.U25.l4.hg38.full.tsv,
the read coverage of every cell type.

The cell type files are located in /staging/leuven/stg_00118/methylDL/data/loyfer2023/hg38/data.

There are both .hg38.pat.gz files and their corresponding .hg38.pat.gz.csi files.
The csi files are index files for the .gz files, and are required for efficient reading of the .gz files.

Examples of how to extract specific regions from the .pat.gz files using the csi index files can be found in EDA/extract_soft_labels_from_noisy_samples/extract_soft_labels_from_noisy_samples.ipynb .

The output of this script should be a csv file with the following columns:
- chr, start, end, startCpG, endCpG: the coordinates of the dmr region (as in the atlas file)
- for every cell type, the read coverage of that cell type in that dmr region
"""

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pysam

from utils import filename2ctype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PATH_TO_DATA = Path(
    "/staging/leuven/stg_00118/methylDL/data/loyfer2023/hg38/data/GSE186458"
)
PATH_TO_ATLAS = Path(
    "/data/leuven/389/vsc38912/Projects/syto/Data/Atlas.U25.l4.hg38.full.tsv"
)
OUTPUT_CSV = Path(
    "/data/leuven/389/vsc38912/Projects/syto/EDA/extract_soft_labels_from_noisy_samples/data/region_coverage.csv"
)

# How many CpG positions to look back before the region start to capture reads
# that begin upstream but still overlap the DMR region.
UPSTREAM_BUFFER_CPGS = 100


def load_atlas(path: Path) -> pd.DataFrame:
    """Load the atlas and return only coordinate columns."""
    atlas = pd.read_csv(path, sep="\t")
    log.info("Loaded atlas with %d regions from %s", len(atlas), path)
    return atlas[["chr", "start", "end", "startCpG", "endCpG"]]


def group_files_by_ctype(data_dir: Path) -> dict[str, list[Path]]:
    """Map each atlas cell-type label to the list of matching .pat.gz files."""
    ctype_to_files: dict[str, list[Path]] = defaultdict(list)
    all_pat_files = sorted(data_dir.glob("*.pat.gz"))
    log.info("Found %d .pat.gz files in %s", len(all_pat_files), data_dir)
    skipped = 0
    for f in all_pat_files:
        ctype = filename2ctype(f.name)
        if ctype is None:
            skipped += 1
            continue
        ctype_to_files[ctype].append(f)
    log.info(
        "Mapped files to %d distinct cell types (%d files skipped / unmapped)",
        len(ctype_to_files),
        skipped,
    )
    return dict(ctype_to_files)


def coverage_for_region(
    tbx: pysam.TabixFile,
    chrom: str,
    start_cpg: int,
    end_cpg: int,
    upstream_buffer: int = UPSTREAM_BUFFER_CPGS,
) -> int:
    """
    Return the total number of reads in *tbx* that overlap the region's CpG
    range.  Following the wgbstools convention, *end_cpg* is EXCLUSIVE, so the
    region covers CpG indices [start_cpg, end_cpg).

    A read starting at CpG index *i* with pattern of length *k* covers
    CpGs [i, i+k-1].  It overlaps [start_cpg, end_cpg) when:
        i < end_cpg  AND  i + k - 1 >= start_cpg

    The .pat tabix index is keyed on the read's start CpG (column 2), so a read
    that begins upstream of the region is not returned by a fetch starting at
    start_cpg.  We therefore query a generous window
    [start_cpg - upstream_buffer, end_cpg) and filter precisely in Python.
    """
    fetch_start = max(0, start_cpg - upstream_buffer)
    total = 0
    try:
        # Fetch a generous superset; exactness is enforced by the filter below.
        for row in tbx.fetch(chrom, fetch_start, end_cpg + 1):
            parts = row.split("\t")
            i = int(parts[1])
            pattern = parts[2]
            n_reads = int(parts[3])
            read_end = i + len(pattern) - 1
            if i < end_cpg and read_end >= start_cpg:
                total += n_reads
    except ValueError:
        # Chromosome not present in this file
        pass
    return total


def compute_coverage_for_file(
    pat_file: Path,
    regions_by_chrom: dict[str, pd.DataFrame],
) -> dict[int, int]:
    """
    Open *pat_file* once and compute coverage for every region.

    Returns a dict {region_row_index -> coverage}.
    """
    csi_file = Path(str(pat_file) + ".csi")
    coverage: dict[int, int] = {}
    try:
        tbx = pysam.TabixFile(str(pat_file), index=str(csi_file))
    except Exception as exc:
        log.warning("Cannot open %s: %s", pat_file.name, exc)
        return coverage

    for chrom, chrom_regions in regions_by_chrom.items():
        for row in chrom_regions.itertuples():
            cov = coverage_for_region(tbx, chrom, row.startCpG, row.endCpG)
            coverage[row.Index] = coverage.get(row.Index, 0) + cov

    tbx.close()
    return coverage


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ────────────────────────────────────────────────────────────
    atlas = load_atlas(PATH_TO_ATLAS)
    ctype_to_files = group_files_by_ctype(PATH_TO_DATA)
    all_ctypes = sorted(ctype_to_files.keys())

    # Pre-group atlas regions by chromosome for efficient per-file iteration
    regions_by_chrom: dict[str, pd.DataFrame] = {
        chrom: grp for chrom, grp in atlas.groupby("chr")
    }
    log.info("Atlas spans %d chromosomes", len(regions_by_chrom))

    # ── Accumulate coverage ────────────────────────────────────────────────────
    # coverage_matrix[region_index][ctype] = total reads
    coverage_matrix: dict[int, dict[str, int]] = {idx: {} for idx in atlas.index}

    n_ctypes = len(all_ctypes)
    for ctype_idx, ctype in enumerate(all_ctypes, start=1):
        files = ctype_to_files[ctype]
        log.info(
            "[%d/%d] Processing cell type '%s' (%d file(s))",
            ctype_idx,
            n_ctypes,
            ctype,
            len(files),
        )
        ctype_coverage: dict[int, int] = {}
        for file_idx, pat_file in enumerate(files, start=1):
            log.debug("  [%d/%d] Reading %s", file_idx, len(files), pat_file.name)
            file_cov = compute_coverage_for_file(pat_file, regions_by_chrom)
            for region_idx, cov in file_cov.items():
                ctype_coverage[region_idx] = ctype_coverage.get(region_idx, 0) + cov

        for region_idx, cov in ctype_coverage.items():
            coverage_matrix[region_idx][ctype] = cov

    log.info("Coverage accumulation complete. Building output dataframe…")

    # ── Build output dataframe ─────────────────────────────────────────────────
    coverage_df = pd.DataFrame(coverage_matrix).T  # regions × ctypes
    coverage_df = coverage_df.reindex(columns=all_ctypes).fillna(0).astype(int)
    output_df = pd.concat([atlas, coverage_df], axis=1)

    output_df.to_csv(OUTPUT_CSV, index=False)
    log.info(
        "Saved output (%d regions × %d columns) to %s",
        len(output_df),
        len(output_df.columns),
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
