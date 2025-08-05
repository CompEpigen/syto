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
        intervals = group[['read_start', 'read_end']].sort_values('read_start').values
        
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
            total_unique_positions += (end - start + 1)
    
    return total_unique_positions