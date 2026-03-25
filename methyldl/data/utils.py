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


def _merge_comma_separated(str1, str2):
    """Merge two comma-separated strings, keeping unique values."""
    if not str1 and not str2:
        return ""
    items1 = set(str1.split(",")) if str1 else set()
    items2 = set(str2.split(",")) if str2 else set()
    merged = items1 | items2
    merged.discard("")  # Remove empty strings
    return ",".join(sorted(merged))

import numpy as np
from sklearn.model_selection import train_test_split

def split_by_file_and_class(df, target_col='original_label', file_col='file', 
                            train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, 
                            random_state=42):
    """
    Splits the dataset into train, valid, and test sets.
    Prioritizes file-level isolation to prevent data leakage, 
    while EXPLICITLY guaranteeing every class is represented in all splits.
    """
    assert np.isclose(train_ratio + valid_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
    
    train_chunks, valid_chunks, test_chunks = [], [], []
    
    # Group the entire dataset by the ground truth cell type
    grouped = df.groupby(target_col)
    
    for label, group in grouped:
        # Get read counts per file for this specific cell type
        file_counts = group[file_col].value_counts().to_dict()
        files_sorted = sorted(file_counts.keys(), key=lambda k: file_counts[k], reverse=True)
        n_files = len(files_sorted)
        total_reads = sum(file_counts.values())
        
        if n_files >= 3:
            # REVISED GREEDY FILE-LEVEL ALLOCATION WITH COVERAGE OVERRIDE
            targets = {
                'train': total_reads * train_ratio,
                'valid': total_reads * valid_ratio,
                'test': total_reads * test_ratio
            }
            current_counts = {'train': 0, 'valid': 0, 'test': 0}
            allocations = {'train': [], 'valid': [], 'test': []}
            
            for i, file in enumerate(files_sorted):
                f_count = file_counts[file]
                remaining_files = n_files - i
                
                # Check which buckets still have 0 files
                empty_buckets = [b for b in targets.keys() if not allocations[b]]
                
                # FORCE COVERAGE: If we only have just enough files left to fill the empty 
                # buckets, we must abandon the ratio math and fill the empty buckets.
                if len(empty_buckets) >= remaining_files:
                    best_bucket = max(empty_buckets, key=lambda b: targets[b])
                else:
                    # Otherwise, use standard greedy: give to bucket with highest deficit
                    deficits = {k: targets[k] - current_counts[k] for k in targets}
                    best_bucket = max(deficits, key=deficits.get)
                    
                allocations[best_bucket].append(file)
                current_counts[best_bucket] += f_count
                
            train_chunks.append(group[group[file_col].isin(allocations['train'])])
            valid_chunks.append(group[group[file_col].isin(allocations['valid'])])
            test_chunks.append(group[group[file_col].isin(allocations['test'])])
            
        elif n_files == 2:
            # 2 FILES: Keep Train isolated. Split File 2 into Valid/Test.
            file1, file2 = files_sorted[0], files_sorted[1]
            
            # Put the largest file in Train
            train_chunks.append(group[group[file_col] == file1])
            
            # Split the second file between valid and test
            file2_df = group[group[file_col] == file2]
            val_prop = valid_ratio / (valid_ratio + test_ratio)
            
            val_df, test_df = train_test_split(file2_df, train_size=val_prop, random_state=random_state)
            valid_chunks.append(val_df)
            test_chunks.append(test_df)
            
        else:
            # 1 FILE: Unavoidable read-level split to ensure class coverage.
            file_df = group
            train_df, temp_df = train_test_split(file_df, train_size=train_ratio, random_state=random_state)
            
            val_prop = valid_ratio / (valid_ratio + test_ratio)
            val_df, test_df = train_test_split(temp_df, train_size=val_prop, random_state=random_state)
            
            train_chunks.append(train_df)
            valid_chunks.append(val_df)
            test_chunks.append(test_df)

    # Combine all the chunks
    df_train = pd.concat(train_chunks, ignore_index=True)
    df_valid = pd.concat(valid_chunks, ignore_index=True)
    df_test = pd.concat(test_chunks, ignore_index=True)
    
    return df_train, df_valid, df_test
