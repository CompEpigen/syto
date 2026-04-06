import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_cpg_signature(row):
    """
    Standardize the pattern into absolute genomic coordinates
    """
    start = row["trimmed_start"]
    pattern = str(row["pattern"])
    return tuple(
        (start + i, int(val)) for i, val in enumerate(pattern) if val in ("0", "1")
    )


def signature_distance(sig1, sig2):
    """
    Computes Jaccard distance between two CpG signatures.
    Each signature is treated as a set of (position, state) tuples.
    Distance = 1.0 - (|Intersection| / |Union|)
    """
    set1 = set(sig1)
    set2 = set(sig2)
    
    if not set1 and not set2:
        return 0.0
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return 1.0 - (intersection / union)


def apply_normalized_knn_smoothing(
    base_counts, min_reads=30, max_distance=0.5, num_classes=40
):
    """
    Applies KNN smoothing with strict distance constraints and global
    sequencing depth normalization.
    """
    class_cols = list(range(num_classes))

    # ==========================================
    # 1. Calculate Global Normalization Weights
    # ==========================================
    # Sum all reads across all regions and signatures for each class
    global_class_counts = base_counts[class_cols].sum(axis=0)

    # Use the median count of the present classes as the target normalization baseline
    nonzero_counts = global_class_counts[global_class_counts > 0]
    if len(nonzero_counts) == 0:
        raise ValueError("No reads found for any class in base_counts.")

    median_count = nonzero_counts.median()

    # Calculate weights (add epsilon to avoid division by zero for absent classes)
    epsilon = 1e-9
    class_weights = (median_count / (global_class_counts + epsilon)).values

    # print(f"Calculated normalization weights. Max weight: {class_weights.max():.2f}, Min weight: {class_weights.min():.2f}")

    # ==========================================
    # 2. Region-by-Region Smoothing
    # ==========================================
    smoothed_records = []
    grouped = base_counts.groupby("name")

    for region, group in tqdm(grouped, desc="Smoothing Regions"):
        group = group.reset_index(drop=True)
        n_sigs = len(group)

        sigs = group["cpg_sig"].values
        counts_mat = group[class_cols].values.astype(float)
        total_reads = group["total_reads"].values

        # Precompute distance matrix for this region
        dist_mat = np.zeros((n_sigs, n_sigs))
        for i in range(n_sigs):
            for j in range(i + 1, n_sigs):
                d = signature_distance(sigs[i], sigs[j])
                dist_mat[i, j] = d
                dist_mat[j, i] = d

        # Apply smoothing and normalization
        for i in range(n_sigs):
            accumulated_counts = np.zeros(num_classes)
            accumulated_total = 0
            sigs_pooled = 0

            if total_reads[i] >= min_reads:
                # Target already met: use exact raw counts
                accumulated_counts = counts_mat[i]
                accumulated_total = total_reads[i]
                sigs_pooled = 1
            else:
                # Target not met: borrow from valid neighbors
                neighbor_indices = np.argsort(dist_mat[i])

                for idx in neighbor_indices:
                    if dist_mat[i, idx] > max_distance:
                        break

                    accumulated_counts += counts_mat[idx]
                    accumulated_total += total_reads[idx]
                    sigs_pooled += 1

                    if accumulated_total >= min_reads:
                        break

            # --- CRITICAL STEP: Apply Global Normalization Weights ---
            normalized_counts = accumulated_counts * class_weights
            total_normalized = normalized_counts.sum()

            # Compute probability on the NORMALIZED pool
            if total_normalized > 0:
                prob = normalized_counts / total_normalized
            else:
                prob = np.zeros(num_classes)

            smoothed_records.append(
                {
                    "name": region,
                    "cpg_sig": sigs[i],
                    "normalized_counts": normalized_counts.tolist(),
                    "soft_label": prob.tolist(),
                    "raw_reads_pooled": int(
                        accumulated_total
                    ),  # Track actual physical reads
                    "total_normalized_reads": int(total_normalized),
                    "num_signatures_pooled": sigs_pooled,
                }
            )

    return pd.DataFrame(smoothed_records)
