import pandas as pd
from typing import List, Dict
import pandas as pd


def count_unique_positions(df: pd.DataFrame) -> int:
    """
    Count the number of unique genomic positions covered by all reads.

    Parameters:
        df (pd.DataFrame): Must contain columns 'chromosome', 'read_start', and 'read_end'.

    Returns:
        int: Number of unique genomic positions covered by all reads.
    """
    total_unique_positions = 0

    # Process by chromosome
    for chrom, group in df.groupby("chromosome"):
        # Sort by start coordinate
        intervals = group[["read_start", "read_end"]].sort_values("read_start").values

        merged_intervals = []
        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:
            if start <= current_end:  # Overlapping
                current_end = max(current_end, end)
            else:  # No overlap, push current interval and start new one
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end

        # Add the last interval
        merged_intervals.append((current_start, current_end))

        # Sum lengths of merged intervals
        for start, end in merged_intervals:
            total_unique_positions += end - start + 1

    return total_unique_positions


def split_long_reads(df: pd.DataFrame, max_read_length: int) -> pd.DataFrame:
    """
    Split DNA reads longer than max_read_length into smaller chunks.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing DNA read data
    max_read_length : int
        Maximum allowed read length for splitting

    Returns:
    --------
    pd.DataFrame
        Transformed dataset with split reads and calculated CpG counts
    """

    def count_cpgs(methylation_string: str) -> int:
        """Count CpG sites (represented by '1' or '2' in methylation_ids)
        0 - unmethylated C, 1 - methylated C, 2 - methylation status unknown"""
        return sum(1 for char in methylation_string if char in ["0", "1"])

    def split_single_read(row: pd.Series) -> List[Dict]:
        """Split a single read into chunks if it exceeds max_read_length"""
        # Extract relevant fields directly from the pandas Series
        methylation_ids = row["methylation_ids"]
        # Get the actual sequence length
        sequence_length = len(methylation_ids)

        # If the read is within the max length, return as is
        if sequence_length <= max_read_length:
            return [{**row.to_dict(), "num_cpgs": count_cpgs(methylation_ids)}]

        # Split the read into chunks
        chunks = []
        for start in range(0, sequence_length, max_read_length):
            end = start + max_read_length

            chunk = row.to_dict()
            chunk["input_ids"] = row["input_ids"][start:end]
            chunk["methylation_ids"] = methylation_ids[start:end]
            chunk["num_cpgs"] = count_cpgs(chunk["methylation_ids"])

            chunks.append(chunk)

        return chunks

    # Process all rows
    all_chunks = []
    for _, row in df.iterrows():
        chunks = split_single_read(row)
        all_chunks.extend(chunks)

    # Convert to DataFrame
    result_df = pd.DataFrame(all_chunks)

    return result_df
