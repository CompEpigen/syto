""""""


def _get_overlapping_dmrs(read_start, read_end, chrom, dmr_trees, strict=False):
    """
    Get list of overlapping DMRs with individual info (not aggregated).
    Returns one dict per overlapping DMR for iteration.
    """
    if chrom not in dmr_trees:
        return []

    overlapping = dmr_trees[chrom][read_start:read_end]

    if strict and overlapping:
        overlapping = {
            iv for iv in overlapping if iv.begin <= read_start and iv.end >= read_end
        }

    if not overlapping:
        return []

    read_len = read_end - read_start
    results = []

    for interval in overlapping:
        dmr_data = interval.data
        dmr_start, dmr_end = interval.begin, interval.end
        dmr_len = dmr_end - dmr_start

        # Calculate overlap
        intersect_start = max(read_start, dmr_start)
        intersect_end = min(read_end, dmr_end)
        overlap_len = max(0, intersect_end - intersect_start)

        denominator = min(read_len, dmr_len)
        pct = (overlap_len / denominator) * 100.0 if denominator > 0 else 0.0

        results.append(
            {
                "name": str(dmr_data["name"]),
                "type": str(dmr_data["type"]),
                "dmr_start": dmr_start,
                "dmr_end": dmr_end,
                "overlap_bp": overlap_len,
                "overlap_pct": pct,
            }
        )

    return results


def analyze_read_dmr_overlap(read_start, read_end, chrom, dmr_trees, strict=False):
    """
    Analyze overlap between a read segment and DMRs with calculated overlap metrics.

    Parameters:
    - strict: If True, only returns overlaps where read is FULLY contained in DMR.
    """
    if chrom not in dmr_trees:
        return _empty_overlap_result()

    # Query the tree
    overlapping = dmr_trees[chrom][read_start:read_end]

    # Optional Strict Filter (Read must be fully inside DMR)
    if strict and overlapping:
        overlapping = {
            iv for iv in overlapping if iv.begin <= read_start and iv.end >= read_end
        }

    if not overlapping:
        return _empty_overlap_result()

    # Initialize lists to store data for all overlapping DMRs
    dmr_labels = []
    dmr_types = []
    overlap_bps = []
    overlap_pcts = []

    read_len = read_end - read_start

    for interval in overlapping:
        dmr_data = interval.data
        dmr_start, dmr_end = interval.begin, interval.end
        dmr_len = dmr_end - dmr_start

        # 1. Calculate Overlap in BP
        # Intersection = max of starts to min of ends
        intersect_start = max(read_start, dmr_start)
        intersect_end = min(read_end, dmr_end)
        overlap_len = max(0, intersect_end - intersect_start)

        # 2. Calculate Percentage (Maximal Percentage Intersection)
        # Denominator is the length of the shorter entity (Read vs DMR)
        denominator = min(read_len, dmr_len)

        if denominator > 0:
            pct = (overlap_len / denominator) * 100.0
        else:
            pct = 0.0

        # Append to lists
        dmr_labels.append(str(dmr_data["name"]))
        dmr_types.append(str(dmr_data["type"]))
        overlap_bps.append(str(overlap_len))
        overlap_pcts.append(f"{pct:.1f}")  # Format as "75.0"

    return {
        "overlaps_dmr": True,
        "num_dmrs_overlapped": len(overlapping),
        "dmr_labels": ",".join(dmr_labels),
        "dmr_types": ",".join(dmr_types),
        "overlap_bps": ",".join(overlap_bps),  # e.g., "30,15"
        "overlap_pcts": ",".join(overlap_pcts),  # e.g., "75.0,20.0"
        # 'dmr_mean_meth_target': ...
    }


def _empty_overlap_result():
    return {
        "overlaps_dmr": False,
        "num_dmrs_overlapped": 0,
        "dmr_labels": "",
        "dmr_types": "",
        "overlap_bps": "",
        "overlap_pcts": "",
        "dmr_mean_meth_target": None,
        "dmr_mean_meth_background": None,
        "dmr_mean_diff": None,
    }
