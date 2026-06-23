from typing import Dict

import numpy as np
from collections import defaultdict

from utils import get_sig_from_idx, get_idx_from_sig

## METHODS FOR PURE SAMPLES ONLY


# Naive soft labels without pooling
def naive_sl_without_pooling(counts_sig_per_ctype: Dict[int, Dict[int, np.ndarray]]):
    """
    Compute P(sig | ctype) by normalizing the counts of signatures per ctype.
    Can only be applied to PURE samples, where we know exactly the ctype from which
    each read comes.
    """
    return {
        start: {
            read_length: counts / counts.sum(axis=1, keepdims=True)
            for read_length, counts in read_lengths.items()
        }
        for start, read_lengths in counts_sig_per_ctype.items()
    }


def naive_sl_without_pooling_ctype_given_sig(
    counts_sig_per_ctype: Dict[int, Dict[int, np.ndarray]],
):
    """
    Compute P(ctype | sig) by normalizing the counts of signatures per ctype.
    Can only be applied to PURE samples, where we know exactly the ctype from which
    each read comes.
    """
    return {
        start: {
            read_length: (counts / counts.sum(axis=0, keepdims=True)).T
            for read_length, counts in read_lengths.items()
        }
        for start, read_lengths in counts_sig_per_ctype.items()
    }


# Naive soft labels with pooling
def jaccard_distance(sig1_data, sig2_data) -> float:
    """
    Compute the Jaccard distance between two signatures represented as (start, length, sig_idx).
    First we reconstruct the real signatures from the signature indices, then we compute the
    Jaccard distance between the two sets of CpG positions with status 1.
    """
    start1, len1, sig_idx1 = sig1_data
    start2, len2, sig_idx2 = sig2_data
    sig1 = get_sig_from_idx(sig_idx1, len1)
    sig2 = get_sig_from_idx(sig_idx2, len2)
    complete_sig1 = set((start1 + i, int(state)) for i, state in enumerate(sig1))
    complete_sig2 = set((start2 + i, int(state)) for i, state in enumerate(sig2))
    intersection = complete_sig1 & complete_sig2
    union = complete_sig1 | complete_sig2
    return 1 - len(intersection) / len(union) if union else 0.0


def naive_sl_with_pooling(
    counts_sig_per_ctype: Dict[int, Dict[int, np.ndarray]],
    pgt_sig_prior: Dict[int, Dict[int, np.ndarray]],
    signature_positions: Dict[int, set[int]],
    dist_threshold: float = 0.41,
    min_counts: int = 30,
    use_pooled_counts_for_prior: bool = False,
):
    """
    Compute P(sig | ctype) for every signature by first computing
    P(ctype | sig) (using soft labels with pooling) and then converting it
    to P(sig | ctype) using a prior P(sig).

    If use_pooled_counts_for_prior=True, the prior P(sig) is estimated from
    pooled counts (signature-wise pooled totals) (this gives bad results).
    Otherwise, pgt_sig_prior is used.

    Can only be applied to PURE samples, where we know exactly the ctype from which
    each read comes.
    """
    tmp_first_counts = next(iter(next(iter(counts_sig_per_ctype.values())).values()))
    n_ctypes = tmp_first_counts.shape[0]
    del tmp_first_counts

    # first we construct the list of all signatures with their start, length and index
    all_signatures = []
    for start, read_lengths in signature_positions.items():
        for read_length in read_lengths:
            n_signatures = 2**read_length
            for sig_idx in range(n_signatures):
                all_signatures.append((start, read_length, sig_idx))

    signature_to_global_idx = {
        signature: global_idx for global_idx, signature in enumerate(all_signatures)
    }

    # then we compute the pairwise Jaccard distances between all signatures
    n_all_signatures = len(all_signatures)
    jaccard_distances = np.zeros((n_all_signatures, n_all_signatures))
    for i in range(n_all_signatures):
        for j in range(i + 1, n_all_signatures):
            jaccard_distances[i, j] = jaccard_distance(
                all_signatures[i], all_signatures[j]
            )
            jaccard_distances[j, i] = jaccard_distances[i, j]

    # for all signatures, we iterate over all signatures by ascending order of Jaccard distance
    # and we pool the signatures that are within the distance threshold until
    # we reach the minimum counts threshold
    pooled_counts = np.zeros((n_all_signatures, n_ctypes), dtype=int)
    for sig_global_idx, (start, read_length, sig_idx) in enumerate(all_signatures):
        ordered_closed_signatures = np.argsort(jaccard_distances[sig_global_idx])
        for close_sig_global_idx in ordered_closed_signatures:
            close_start, close_read_length, close_sig_idx = all_signatures[
                close_sig_global_idx
            ]
            if jaccard_distances[sig_global_idx, close_sig_global_idx] > dist_threshold:
                break
            pooled_counts[sig_global_idx] += counts_sig_per_ctype[close_start][
                close_read_length
            ][:, close_sig_idx]
            if pooled_counts[sig_global_idx].sum() >= min_counts:
                break

    # now compute the naive soft labels P(ctype | sig) by normalizing the pooled counts
    naive_sl_with_pooling_ctype_given_sig = {}
    naive_sl_with_pooling_sig_given_ctype = {}
    for start, read_lengths in signature_positions.items():
        naive_sl_with_pooling_ctype_given_sig[start] = {}
        naive_sl_with_pooling_sig_given_ctype[start] = {}
        for read_length in read_lengths:
            n_signatures = 2**read_length
            naive_sl_with_pooling_ctype_given_sig[start][read_length] = np.zeros(
                (n_signatures, n_ctypes)
            )

            # compute the naive soft labels P(ctype | sig) by normalizing the pooled counts
            for sig_idx in range(n_signatures):
                sig_global_idx = signature_to_global_idx[(start, read_length, sig_idx)]
                naive_sl_with_pooling_ctype_given_sig[start][read_length][sig_idx] = (
                    pooled_counts[sig_global_idx] / pooled_counts[sig_global_idx].sum()
                    if pooled_counts[sig_global_idx].sum() > 0
                    else 0
                )

            # compute P(sig | ctype) by converting P(ctype | sig) using a prior P(sig)
            if use_pooled_counts_for_prior:
                pooled_sig_counts = np.array(
                    [
                        pooled_counts[
                            signature_to_global_idx[(start, read_length, sig_idx)]
                        ].sum()
                        for sig_idx in range(n_signatures)
                    ],
                    dtype=float,
                )
                if pooled_sig_counts.sum() > 0:
                    tmp_sig_prior = pooled_sig_counts / pooled_sig_counts.sum()
                else:
                    tmp_sig_prior = pgt_sig_prior[start][read_length]
            else:
                tmp_sig_prior = pgt_sig_prior[start][read_length]

            proba_sig_given_ctype = naive_sl_with_pooling_ctype_given_sig[start][
                read_length
            ] * tmp_sig_prior.reshape(-1, 1)
            proba_sig_given_ctype = proba_sig_given_ctype / proba_sig_given_ctype.sum(
                axis=0, keepdims=True
            )
            naive_sl_with_pooling_sig_given_ctype[start][
                read_length
            ] = proba_sig_given_ctype.T

    return naive_sl_with_pooling_sig_given_ctype, naive_sl_with_pooling_ctype_given_sig


def naive_sl_with_pooling_using_extended_data(
    counts_sig_per_ctype: Dict[int, Dict[int, np.ndarray]],
    pgt_sig_prior: Dict[int, Dict[int, np.ndarray]],
    signature_positions: Dict[int, set[int]],
    dist_threshold: float = 0.41,
    min_counts: int = 5,
):
    """
    Compute P(sig | ctype) for every signature by first computing
    P(ctype | sig) (using soft labels with pooling) and then converting it
    to P(sig | ctype) using a prior P(sig).

    Contrarily to naive_sl_with_pooling, this method is design to work with
    a form of extended data. More specifically, s1 can pool from s2 or from
    any subpattern of s2. However they cannot be counted twice
    """
    tmp_first_counts = next(iter(next(iter(counts_sig_per_ctype.values())).values()))
    n_ctypes = tmp_first_counts.shape[0]
    del tmp_first_counts

    # first we construct the dict of all signatures with their start, length and sig_idx
    all_signatures = []
    for start, read_lengths in signature_positions.items():
        for read_length in read_lengths:
            n_signatures = 2**read_length
            for sig_idx in range(n_signatures):
                all_signatures.append((start, read_length, sig_idx))
    n_all_signatures = len(all_signatures)
    signature_to_global_idx = {
        signature: global_idx for global_idx, signature in enumerate(all_signatures)
    }

    # then we create the dict of all extended signatures with:
    # start, length, sig_idx, and parent_global_idx (the signature from which it is extracted)
    extended_signatures = []
    for start, read_lengths in signature_positions.items():
        for read_length in read_lengths:
            n_signatures = 2**read_length
            for sig_idx in range(n_signatures):
                sig_global_idx = signature_to_global_idx[(start, read_length, sig_idx)]
                sig_pattern = get_sig_from_idx(sig_idx, read_length)
                for i in range(read_length):# start index of the subpattern
                    for j in range(i + 1, read_length+1): # end index of the subpattern (exclusive)
                        subpattern = sig_pattern[i:j]
                        subpattern_length = j - i
                        subpattern_sig_idx = get_idx_from_sig(subpattern)
                        extended_signatures.append(
                            (start + i, subpattern_length, subpattern_sig_idx, sig_global_idx)
                        )
    n_extended_signatures = len(extended_signatures)
    
    # then we compute the pairwise Jaccard distances between all signatures and extended signatures
    jaccard_distances = np.zeros((n_all_signatures, n_extended_signatures))
    for i in range(n_all_signatures):
        for j in range(n_extended_signatures):
            jaccard_distances[i, j] = jaccard_distance(
                all_signatures[i], extended_signatures[j][:3]
            )
    
    # for all signatures, we iterate over all extended signatures by ascending order
    # of Jaccard distance and we pool the signatures that are within the distance threshold until
    # we reach the minimum counts threshold.
    # We must be careful not to pool twice from the same parent signature
    pooled_counts = np.zeros((n_all_signatures, n_ctypes), dtype=int)
    for sig_global_idx, (start, read_length, sig_idx) in enumerate(all_signatures):
        ordered_closed_extended_signatures = np.argsort(jaccard_distances[sig_global_idx])
        # we create a set to keep track of which parent signatures we have already pooled from
        parent_counts_pooled = set()
        # we iterate over the ordered extended signatures
        for close_ext_sig_idx in ordered_closed_extended_signatures:
            close_start, close_read_length, close_sig_idx, parent_global_idx = extended_signatures[close_ext_sig_idx]
            if jaccard_distances[sig_global_idx, close_ext_sig_idx] > dist_threshold:
                break
            # we check if we have already pooled from this parent signature
            if parent_global_idx in parent_counts_pooled:
                continue
            parent_start, parent_read_length, parent_sig_idx = all_signatures[parent_global_idx]
            pooled_counts[sig_global_idx] += counts_sig_per_ctype[parent_start][parent_read_length][:, parent_sig_idx]
            parent_counts_pooled.add(parent_global_idx)
            if pooled_counts[sig_global_idx].sum() >= min_counts:
                break
            
    # now compute the naive soft labels P(ctype | sig) by normalizing the pooled counts
    naive_sl_with_pooling_ctype_given_sig = {}
    naive_sl_with_pooling_sig_given_ctype = {}
    for start, read_lengths in signature_positions.items():
        naive_sl_with_pooling_ctype_given_sig[start] = {}
        naive_sl_with_pooling_sig_given_ctype[start] = {}
        for read_length in read_lengths:
            n_signatures = 2**read_length
            naive_sl_with_pooling_ctype_given_sig[start][read_length] = np.zeros(
                (n_signatures, n_ctypes)
            )

            # compute the naive soft labels P(ctype | sig) by normalizing the pooled counts
            for sig_idx in range(n_signatures):
                sig_global_idx = signature_to_global_idx[(start, read_length, sig_idx)]
                naive_sl_with_pooling_ctype_given_sig[start][read_length][sig_idx] = (
                    pooled_counts[sig_global_idx] / pooled_counts[sig_global_idx].sum()
                    if pooled_counts[sig_global_idx].sum() > 0
                    else 0
                )

            # compute P(sig | ctype) by converting P(ctype | sig) using a prior P(sig)
            tmp_sig_prior = pgt_sig_prior[start][read_length]
            proba_sig_given_ctype = naive_sl_with_pooling_ctype_given_sig[start][
                read_length
            ] * tmp_sig_prior.reshape(-1, 1)
            proba_sig_given_ctype = proba_sig_given_ctype / proba_sig_given_ctype.sum(
                axis=0, keepdims=True
            )
            naive_sl_with_pooling_sig_given_ctype[start][
                read_length
            ] = proba_sig_given_ctype.T

    return naive_sl_with_pooling_sig_given_ctype, naive_sl_with_pooling_ctype_given_sig



## METHODS THAT CAN BE APPLIED TO PURE AND NOISY SAMPLES


def generalized_naive_sl_without_pooling(
    counts_sig_per_sample: Dict[int, Dict[int, np.ndarray]],
    ctype_prop_per_sample: np.ndarray,
    reestimate_ctype_prop_per_sample: bool = False,
):
    """
    Compute P(sig | ctype) for all sigs in counts_sig_per_sample,
    using the generalization of the naive soft labels without pooling to
    noisy samples that leverages EM.

    counts_sig_per_sample has the structure {start: {read_length: np.ndarray of shape (n_samples, 2**read_length)}}
    ctype_prop_per_sample has the structure np.ndarray of shape (n_samples, n_ctypes)
    """
    # TODO


def sl_with_simple_archetypes(
    counts_sig_per_sample: Dict[int, Dict[int, np.ndarray]],
    ctype_proba_per_sample: np.ndarray,
    reestimate_ctype_proba_per_sample: bool = False,
    max_num_iterations: int = 100,
    delta_tol: float = 1e-4,
):
    """
    Compute P(sig | ctype) for all sigs in counts_sig_per_sample,
    using the underlying assumption that each ctype
    has a single methylation archetype (CelFiE-style model).

    Since all the reads with data (sample, signature) undergo the same update rules,
    we update all of them at once by keeping track of the counts per signature.

    Args:
        counts_sig_per_sample: has the structure
            {start: {read_length: np.ndarray of shape (n_samples, 2**read_length)}}
        ctype_proba_per_sample: has the structure np.ndarray of shape (n_samples, n_ctypes)
        reestimate_ctype_proba_per_sample: whether to reestimate the ctype proportions
            per sample at each iteration
        max_num_iterations: maximum number of iterations for the EM-like procedure
        delta_tol: convergence threshold for the change in archetypes

    Returns:
        Dict[int, Dict[int, np.ndarray]] with shape (n_ctypes, 2**read_length)
        per (start, read_length), corresponding to P(signature | ctype).
    """
    eps = 1e-10
    n_samples, n_ctypes = ctype_proba_per_sample.shape

    # Determine the full CpG span covered by all signatures.
    start_CpG_index = min(counts_sig_per_sample.keys())
    end_CpG_index = max(
        start + read_length
        for start, read_lengths in counts_sig_per_sample.items()
        for read_length in read_lengths.keys()
    )  # exclusive
    n_CpG_positions = end_CpG_index - start_CpG_index

    # Initialize archetypes (mu) and sample-specific ctype proportions (p).
    archetypes = np.random.rand(n_ctypes, n_CpG_positions)
    archetypes = np.clip(archetypes, eps, 1 - eps)
    ctype_proba_per_sample = ctype_proba_per_sample.copy()

    ctype_proba_per_read = {
        start: {
            read_length: np.zeros((n_samples, 2**read_length, n_ctypes), dtype=float)
            for read_length in read_lengths
        }
        for start, read_lengths in counts_sig_per_sample.items()
    }

    for _ in range(max_num_iterations):
        # E-step expected counts.
        expected_n_bc = np.zeros((n_samples, n_ctypes), dtype=float)
        expected_o1 = np.zeros((n_ctypes, n_CpG_positions), dtype=float)
        expected_o0 = np.zeros((n_ctypes, n_CpG_positions), dtype=float)

        for start, read_lengths in counts_sig_per_sample.items():
            for read_length, counts_per_sample in read_lengths.items():
                n_signatures = 2**read_length
                sig_patterns = np.array(
                    [
                        get_sig_from_idx(sig_idx, read_length)
                        for sig_idx in range(n_signatures)
                    ],
                    dtype=float,
                )  # (n_signatures, read_length)

                start_offset = start - start_CpG_index
                archetype_subset = archetypes[
                    :, start_offset : start_offset + read_length
                ]
                archetype_subset = np.clip(archetype_subset, eps, 1 - eps)

                # log phi_c(s) for numerical stability.
                log_phi = (
                    sig_patterns @ np.log(archetype_subset).T
                    + (1.0 - sig_patterns) @ np.log(1.0 - archetype_subset).T
                )  # (n_signatures, n_ctypes)

                # gamma_{b,s,c} proportional to p_{b,c} * phi_c(s)
                log_num = (  # log of numerator of gamma_{b,s,c}
                    np.log(np.clip(ctype_proba_per_sample, eps, 1.0))[:, None, :]
                    + log_phi[None, :, :]
                )  # (n_samples, n_signatures, n_ctypes)
                log_num = log_num - np.max(log_num, axis=2, keepdims=True)
                num = np.exp(log_num)
                gamma = num / np.clip(num.sum(axis=2, keepdims=True), eps, None)
                ctype_proba_per_read[start][read_length] = gamma

                weighted_gamma = (
                    counts_per_sample[:, :, None] * gamma
                )  # (n_samples, n_signatures, n_ctypes)

                # Expected sample/ctype counts: \tilde{n}_{b,c}
                expected_n_bc += weighted_gamma.sum(axis=1)  # (n_samples, n_ctypes)

                # Expected methylated/unmethylated counts per ctype/CpG.
                sum_weighted_over_samples = weighted_gamma.sum(
                    axis=0
                )  # (n_signatures, n_ctypes)
                for local_k in range(read_length):
                    global_k = start_offset + local_k
                    s_k = sig_patterns[:, local_k]  # (n_signatures,)
                    expected_o1[:, global_k] += (
                        s_k @ sum_weighted_over_samples
                    )  # (n_ctypes,)
                    expected_o0[:, global_k] += (
                        1.0 - s_k
                    ) @ sum_weighted_over_samples  # (n_ctypes,)

        # M-step for p (optional)
        if reestimate_ctype_proba_per_sample:
            row_sums = expected_n_bc.sum(axis=1, keepdims=True)
            new_ctype_proba_per_sample = np.divide(
                expected_n_bc,
                np.clip(row_sums, eps, None),
            )
            zero_rows = row_sums[:, 0] <= eps
            if np.any(zero_rows):
                new_ctype_proba_per_sample[zero_rows] = ctype_proba_per_sample[
                    zero_rows
                ]
            ctype_proba_per_sample = new_ctype_proba_per_sample

        # M-step for mu
        new_archetypes = np.divide(
            expected_o1,
            np.clip(expected_o1 + expected_o0, eps, None),
        )
        no_coverage_mask = (expected_o1 + expected_o0) <= eps
        new_archetypes[no_coverage_mask] = archetypes[no_coverage_mask]
        new_archetypes = np.clip(new_archetypes, eps, 1 - eps)

        delta = np.max(np.abs(new_archetypes - archetypes))
        archetypes = new_archetypes
        if delta < delta_tol:
            break

    # Build final P(signature | ctype) from converged archetypes.
    proba_sig_given_ctype = {}
    for start, read_lengths in counts_sig_per_sample.items():
        proba_sig_given_ctype[start] = {}
        for read_length in read_lengths:
            n_signatures = 2**read_length
            sig_patterns = np.array(
                [
                    get_sig_from_idx(sig_idx, read_length)
                    for sig_idx in range(n_signatures)
                ],
                dtype=float,
            )
            start_offset = start - start_CpG_index
            archetype_subset = np.clip(
                archetypes[:, start_offset : start_offset + read_length],
                eps,
                1 - eps,
            )
            phi = np.exp(
                sig_patterns @ np.log(archetype_subset).T
                + (1.0 - sig_patterns) @ np.log(1.0 - archetype_subset).T
            )  # (n_signatures, n_ctypes)
            phi = phi / np.clip(phi.sum(axis=0, keepdims=True), eps, None)
            proba_sig_given_ctype[start][read_length] = phi.T

    return proba_sig_given_ctype


def sl_with_generalized_archetypes(
    counts_sig_per_sample: Dict[int, Dict[int, np.ndarray]],
    ctype_proba_per_sample: np.ndarray,
    n_archetypes: int = None,
    reestimate_ctype_proba_per_sample: bool = False,
    max_num_iterations: int = 100,
    delta_tol: float = 1e-4,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Compute P(sig | ctype) for all sigs in counts_sig_per_sample,
    using the underlying assumption that each ctype
    is generated from a linear combination of methylation archetypes,
    the archetypes being shared across ctypes.

    Generative model for a read r of length L CpGs:
        b ~ Categorical(pi)
        c | b ~ Categorical(p_{b, .})
        a | c ~ Categorical(theta_{c, .})
        s_k | a ~ Ber(mu_{a, k})  for each CpG k in dom(s)

    so that
        P(s | c) = sum_a theta_{c, a} * prod_k mu_{a, k}^{s_k} (1 - mu_{a, k})^{1 - s_k}.

    Args:
        counts_sig_per_sample: has the structure
            {start: {read_length: np.ndarray of shape (n_samples, 2**read_length)}}
        ctype_proba_per_sample: np.ndarray of shape (n_samples, n_ctypes), i.e. p_{b, c}.
        n_archetypes: number of shared methylation archetypes A. Defaults to n_ctypes.
        reestimate_ctype_proba_per_sample: whether to reestimate p_{b, c} at each
            iteration (M-step). If False, the oracle ctype_proba_per_sample is kept fixed.
        max_num_iterations: maximum number of EM iterations.
        delta_tol: convergence threshold on the maximum change in (mu, theta).

    Returns:
        Dict[int, Dict[int, np.ndarray]] with shape (n_ctypes, 2**read_length)
        per (start, read_length), corresponding to P(signature | ctype).
    """
    eps = 1e-10
    n_samples, n_ctypes = ctype_proba_per_sample.shape
    if n_archetypes is None:
        n_archetypes = n_ctypes

    # Determine the full CpG span covered by all signatures.
    start_CpG_index = min(counts_sig_per_sample.keys())
    end_CpG_index = max(
        start + read_length
        for start, read_lengths in counts_sig_per_sample.items()
        for read_length in read_lengths.keys()
    )  # exclusive
    n_CpG_positions = end_CpG_index - start_CpG_index

    # Initialize archetypes (mu), archetype proportions per ctype (theta)
    # and sample-specific ctype proportions (p).
    archetypes = np.clip(np.random.rand(n_archetypes, n_CpG_positions), eps, 1 - eps)
    theta = np.random.rand(n_ctypes, n_archetypes)
    theta = theta / theta.sum(axis=1, keepdims=True)
    ctype_proba_per_sample = ctype_proba_per_sample.copy()

    for _ in range(max_num_iterations):
        # E-step expected complete-data counts.
        expected_n_bc = np.zeros((n_samples, n_ctypes), dtype=float)  # \tilde{n}_{b,c}
        expected_m_ca = np.zeros(
            (n_ctypes, n_archetypes), dtype=float
        )  # \tilde{m}_{c,a}
        expected_o1 = np.zeros((n_archetypes, n_CpG_positions), dtype=float)
        expected_o0 = np.zeros((n_archetypes, n_CpG_positions), dtype=float)

        log_p = np.log(
            np.clip(ctype_proba_per_sample, eps, 1.0)
        )  # (n_samples, n_ctypes)
        log_theta = np.log(np.clip(theta, eps, 1.0))  # (n_ctypes, n_archetypes)

        for start, read_lengths in counts_sig_per_sample.items():
            for read_length, counts_per_sample in read_lengths.items():
                n_signatures = 2**read_length
                sig_patterns = np.array(
                    [
                        get_sig_from_idx(sig_idx, read_length)
                        for sig_idx in range(n_signatures)
                    ],
                    dtype=float,
                )  # (n_signatures, read_length)

                start_offset = start - start_CpG_index
                archetype_subset = np.clip(
                    archetypes[:, start_offset : start_offset + read_length],
                    eps,
                    1 - eps,
                )  # (n_archetypes, read_length)

                # log phi_a(s) for numerical stability.
                log_phi = (
                    sig_patterns @ np.log(archetype_subset).T
                    + (1.0 - sig_patterns) @ np.log(1.0 - archetype_subset).T
                )  # (n_signatures, n_archetypes)

                # gamma_{b,s,c,a} proportional to p_{b,c} * theta_{c,a} * phi_a(s).
                log_num = (  # log of the numerator of gamma_{b,s,c,a}
                    log_p[:, None, :, None]
                    + log_theta[None, None, :, :]
                    + log_phi[None, :, None, :]
                )  # (n_samples, n_signatures, n_ctypes, n_archetypes)
                log_num = log_num - np.max(log_num, axis=(2, 3), keepdims=True)
                num = np.exp(log_num)
                gamma = num / np.clip(
                    num.sum(axis=(2, 3), keepdims=True), eps, None
                )  # (n_samples, n_signatures, n_ctypes, n_archetypes)

                weighted_gamma = (
                    counts_per_sample[:, :, None, None] * gamma
                )  # (n_samples, n_signatures, n_ctypes, n_archetypes)

                # Expected sample/ctype counts: \tilde{n}_{b,c} = sum_i gamma_i(c).
                expected_n_bc += weighted_gamma.sum(
                    axis=(1, 3)
                )  # (n_samples, n_ctypes)

                # Expected ctype/archetype counts: \tilde{m}_{c,a} = sum_i gamma_i(c,a).
                expected_m_ca += weighted_gamma.sum(
                    axis=(0, 1)
                )  # (n_ctypes, n_archetypes)

                # Expected methylated/unmethylated counts per archetype/CpG,
                # weighted by the marginal posterior gamma_i(a) = sum_c gamma_i(c,a).
                weighted_gamma_a = weighted_gamma.sum(
                    axis=(0, 2)
                )  # (n_signatures, n_archetypes)
                for local_k in range(read_length):
                    global_k = start_offset + local_k
                    s_k = sig_patterns[:, local_k]  # (n_signatures,)
                    expected_o1[:, global_k] += (
                        s_k @ weighted_gamma_a
                    )  # (n_archetypes,)
                    expected_o0[:, global_k] += (
                        1.0 - s_k
                    ) @ weighted_gamma_a  # (n_archetypes,)

        # M-step for p (optional).
        if reestimate_ctype_proba_per_sample:
            row_sums = expected_n_bc.sum(axis=1, keepdims=True)
            new_ctype_proba_per_sample = np.divide(
                expected_n_bc,
                np.clip(row_sums, eps, None),
            )
            zero_rows = row_sums[:, 0] <= eps
            if np.any(zero_rows):
                new_ctype_proba_per_sample[zero_rows] = ctype_proba_per_sample[
                    zero_rows
                ]
            ctype_proba_per_sample = new_ctype_proba_per_sample

        # M-step for theta.
        theta_row_sums = expected_m_ca.sum(axis=1, keepdims=True)
        new_theta = np.divide(
            expected_m_ca,
            np.clip(theta_row_sums, eps, None),
        )
        zero_theta_rows = theta_row_sums[:, 0] <= eps
        if np.any(zero_theta_rows):
            new_theta[zero_theta_rows] = theta[zero_theta_rows]

        # M-step for mu.
        new_archetypes = np.divide(
            expected_o1,
            np.clip(expected_o1 + expected_o0, eps, None),
        )
        no_coverage_mask = (expected_o1 + expected_o0) <= eps
        new_archetypes[no_coverage_mask] = archetypes[no_coverage_mask]
        new_archetypes = np.clip(new_archetypes, eps, 1 - eps)

        delta = max(
            np.max(np.abs(new_archetypes - archetypes)),
            np.max(np.abs(new_theta - theta)),
        )
        archetypes = new_archetypes
        theta = new_theta
        if delta < delta_tol:
            break

    # Build final P(signature | ctype) from converged archetypes and theta.
    proba_sig_given_ctype = {}
    for start, read_lengths in counts_sig_per_sample.items():
        proba_sig_given_ctype[start] = {}
        for read_length in read_lengths:
            n_signatures = 2**read_length
            sig_patterns = np.array(
                [
                    get_sig_from_idx(sig_idx, read_length)
                    for sig_idx in range(n_signatures)
                ],
                dtype=float,
            )
            start_offset = start - start_CpG_index
            archetype_subset = np.clip(
                archetypes[:, start_offset : start_offset + read_length],
                eps,
                1 - eps,
            )
            phi = np.exp(
                sig_patterns @ np.log(archetype_subset).T
                + (1.0 - sig_patterns) @ np.log(1.0 - archetype_subset).T
            )  # (n_signatures, n_archetypes)
            # P(s | c) = sum_a theta_{c,a} phi_a(s).
            proba = phi @ theta.T  # (n_signatures, n_ctypes)
            proba = proba / np.clip(proba.sum(axis=0, keepdims=True), eps, None)
            proba_sig_given_ctype[start][read_length] = proba.T

    return proba_sig_given_ctype
