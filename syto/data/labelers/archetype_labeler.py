from typing import Literal, Optional, Sequence

from tqdm import tqdm
import pandas as pd
import numpy as np

from syto.data.labelers.abstract_labeler import AbstractLabeler
from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)

# pylint: disable-next=invalid-name
CtypePriorType = Literal["uniform", "inv_global_freq", "inv_region_freq", "predefined"]


class ArchetypeLabeler(AbstractLabeler):
    """
    Perform end to end computation of soft labels for a given dataframe of reads,
    using an underlying model of a single methylation archetype per ctype.

    Designed to be used only with binary CpG signatures.

    Generative model for a read r covering CpGs k in dom(s):
        - a biological sample b is drawn,
        - a cell type c is drawn given the sample composition,
        - the signature s is emitted by drawing the methylation status of each
          CpG k from a Bernoulli with parameter mu[c, k].

    Because the cell type of origin of each read is observed (pure samples), the
    responsibilities are simply the one-hot encoded labels of the reads, so no EM
    is required: the archetype methylation probabilities have a closed form

        mu[c, k] = o1[c, k] / (o1[c, k] + o0[c, k])

    where o1[c, k] (resp. o0[c, k]) is the number of reads of ctype c covering CpG
    k observed methylated (resp. unmethylated). Archetypes are estimated
    independently within each genomic region (CpG positions are region specific).
    If one CpG k is uncovered for a ctype c, mu[c, k] is set to the average of the
    covered mu[c, k'] of that same ctype within the region.
    The resulting mu[c, k] values are always clipped to the open interval (eps, 1 - eps)
    to avoid log-likelihoods of -inf.

    The native output is the per-ctype likelihood of a signature

        phi[c] = P(s | c) = prod_{k in dom(s)} mu[c, k]^{s_k} (1 - mu[c, k])^{1 - s_k}

    from which the soft label P(c | s) is obtained by Bayes' theorem

        P(c | s) = prior[c] * phi[c] / sum_c' prior[c'] * phi[c'].
    """

    def __init__(
        self, signature_handler: BinaryCpGSignatureHandler, eps: float = 1e-10
    ):
        """
        Args:
            signature_handler: an instance of BinaryCpGSignatureHandler to extract signatures
                (instantiated with the appropriate column names)
            eps: small value used to clip the archetype methylation probabilities
                mu[c, k] to the open interval (eps, 1 - eps), keeping the
                log-likelihoods finite.
        """
        self.signature_handler = signature_handler
        self.eps = eps

    def compute_labels(
        self,
        reads_df: pd.DataFrame,
        num_classes: int = 39,
        ctype_prior_type: CtypePriorType = "uniform",
        predefined_prior: Optional[Sequence[float]] = None,
        keep_likelihoods: bool = True,
        precomputed_signature_column: Optional[str] = None,
        fit_mask=None,
        fallback_label=None,
    ) -> pd.DataFrame:
        """
        Compute the archetype-based soft labels for the given dataframe of reads.
        The DataFrame should have the following columns:
        - "name": the name of the genomic region (e.g. "chr1:1000-2000" or "colon-fibro-specific")
        - "original_label": the (observed) class label of origin for this read
            (expects integer from 0 to num_classes-1)

        Args:
            reads_df: DataFrame containing the reads and their original labels
            num_classes: total number of classes in the original labels
            ctype_prior_type: how to build the prior P(ctype) used in Bayes' theorem:
                - "uniform": every ctype is equally likely,
                - "inv_global_freq": prior proportional to the inverse of the global
                    ctype frequencies (estimated over all regions),
                - "inv_region_freq": prior proportional to the inverse of the ctype
                    frequencies estimated within each region,
                - "predefined": use the prior passed through ``predefined_prior``.
            predefined_prior: a sequence of length num_classes used as the prior over
                ctypes when ``ctype_prior_type`` is "predefined" (it is normalized to
                sum to one). Ignored for the other prior types.
            keep_likelihoods: whether to store the per-ctype likelihood vector
                P(sig | ctype) in a "likelihoods" column of the output.
            precomputed_signature_column: if None, the signatures are computed from the
                reads using the signature handler. If not None, the column with this name
                is used as the precomputed signature for each read (and signatures are not
                recomputed).

        Returns:
            DataFrame with original columns plus:
            - "signature": the extracted omics signature for each read
            - "soft_label": the computed soft label P(ctype | sig)
                (probability distribution over classes) for each read
            If keep_likelihoods is True, also includes:
            - "likelihoods": the per-ctype likelihood vector P(sig | ctype)
        """
        assert all(
            col in reads_df.columns for col in ["name", "original_label"]
        ), "Input DataFrame must contain 'name' and 'original_label' columns."
        assert (
            reads_df["original_label"].between(0, num_classes - 1).all()
        ), f"All 'original_label' values must be between 0 and {num_classes - 1}."
        assert ctype_prior_type in (
            "uniform",
            "inv_global_freq",
            "inv_region_freq",
            "predefined",
        ), (
            "ctype_prior_type must be one of "
            "'uniform', 'inv_global_freq', 'inv_region_freq', 'predefined'."
        )
        reads_df = reads_df.copy()

        ## Compute the signatures for each read (or reuse precomputed ones)
        if precomputed_signature_column is None:
            tqdm.pandas(desc="Extracting signatures")
            reads_df["signature"] = reads_df.progress_apply(
                self.signature_handler.extract_signature, axis=1
            )
        else:
            assert (
                precomputed_signature_column in reads_df.columns
            ), f"Column '{precomputed_signature_column}' not found in the input DataFrame."
            reads_df["signature"] = reads_df[precomputed_signature_column]

        ## Estimate archetypes/prior from the fit subset; score all reads.
        fit_df = reads_df if fit_mask is None else reads_df[np.asarray(fit_mask)]

        ## Aggregate the counts of classes by (region, signature) over the fit subset
        count_columns = ["raw_counts_" + str(c) for c in range(num_classes)]
        fit_counts = (
            fit_df.groupby(["name", "signature", "original_label"])
            .size()
            .unstack(fill_value=0)
        )  # df indexed by (name, signature) with columns original_label
        fit_counts.reset_index(inplace=True)
        # add columns with 0 counts for classes that never appear
        for c in range(num_classes):
            if c not in fit_counts.columns:
                fit_counts[c] = 0
        fit_counts.rename(
            columns={c: "raw_counts_" + str(c) for c in range(num_classes)},
            inplace=True,
        )
        fit_counts = fit_counts[["name", "signature"] + count_columns]

        ## Build the (region-independent) prior over ctypes from the fit subset
        base_prior = self._compute_base_prior(
            fit_df,
            num_classes=num_classes,
            ctype_prior_type=ctype_prior_type,
            predefined_prior=predefined_prior,
        )

        fit_groups = {name: g for name, g in fit_counts.groupby("name")}
        fallback = (
            np.asarray(fallback_label, dtype=float)
            if fallback_label is not None
            else np.full(num_classes, 1.0 / num_classes)
        )

        ## Compute archetypes and score every read's signature, region by region
        full_pairs = reads_df[["name", "signature"]].drop_duplicates()
        records = []
        for region, region_pairs in tqdm(
            full_pairs.groupby("name"),
            desc="Computing archetype soft labels per region",
        ):
            sigs_to_score = region_pairs["signature"].tolist()

            # region with no fit reads: everything falls back
            if region not in fit_groups:
                for sig in sigs_to_score:
                    records.append(
                        {
                            "name": region,
                            "signature": sig,
                            "soft_label": fallback.tolist(),
                            "likelihoods": [float("nan")] * num_classes,
                        }
                    )
                continue

            group = fit_groups[region].reset_index(drop=True)
            count_matrix = group[count_columns].values.astype(float)  # (n_sigs, C)

            # estimate the per-ctype archetype methylation probabilities mu[c, k]
            mu, pos_to_idx = self._estimate_region_archetypes(
                group["signature"].values,
                count_matrix,
                num_classes=num_classes,
                region=region,
            )

            # region specific prior, if requested
            if ctype_prior_type == "inv_region_freq":
                region_counts = count_matrix.sum(axis=0)  # (C,)
                prior = self._normalize_inverse_prior(region_counts, num_classes)
            else:
                prior = base_prior

            for sig in sigs_to_score:
                log_phi = self._signature_log_likelihood(sig, mu, pos_to_idx)
                likelihoods = np.exp(log_phi)
                soft_label = self._bayes_posterior(log_phi, prior)

                record = {
                    "name": region,
                    "signature": sig,
                    "soft_label": soft_label.tolist(),
                    "likelihoods": likelihoods.tolist(),
                }
                records.append(record)

        result_df = pd.DataFrame(records)

        ## Select the columns to keep and merge back to the reads
        keep_cols = ["name", "signature", "soft_label"]
        if keep_likelihoods:
            keep_cols.append("likelihoods")
        result_df = result_df[keep_cols]

        reads_df = reads_df.merge(result_df, on=["name", "signature"], how="left")
        return reads_df

    def _compute_base_prior(
        self,
        reads_df: pd.DataFrame,
        num_classes: int,
        ctype_prior_type: CtypePriorType,
        predefined_prior: Optional[Sequence[float]],
    ) -> Optional[np.ndarray]:
        """
        Build the region-independent prior over ctypes.

        Returns None for "inv_region_freq" (the prior is then computed per region).
        """
        if ctype_prior_type == "uniform":
            return np.full(num_classes, 1.0 / num_classes)

        if ctype_prior_type == "inv_global_freq":
            global_counts = np.zeros(num_classes)
            value_counts = reads_df["original_label"].value_counts()
            for ctype, count in value_counts.items():
                global_counts[int(ctype)] = count
            return self._normalize_inverse_prior(global_counts, num_classes)

        if ctype_prior_type == "predefined":
            assert (
                predefined_prior is not None
            ), "predefined_prior must be provided when ctype_prior_type is 'predefined'."
            prior = np.asarray(predefined_prior, dtype=float)
            assert prior.shape == (num_classes,), (
                f"predefined_prior must have length {num_classes}, "
                f"got {prior.shape}."
            )
            assert (prior >= 0).all(), "predefined_prior must be non-negative."
            return self._normalize_prior(prior, num_classes)

        # "inv_region_freq": handled per region in compute_labels
        return None

    @staticmethod
    def _normalize_prior(counts: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Normalize a vector of non-negative weights into a probability distribution.

        Falls back to a uniform distribution when all weights are zero.
        """
        total = counts.sum()
        if total <= 0:
            return np.full(num_classes, 1.0 / num_classes)
        return counts / total

    @staticmethod
    def _normalize_inverse_prior(counts: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Build a probability distribution proportional to the inverse of the counts.

        Rarer ctypes therefore receive a larger prior. Classes with a zero count
        receive a zero prior, and the result falls back to a uniform distribution
        when all counts are zero.
        """
        inverse = np.zeros(num_classes)
        nonzero = counts > 0
        inverse[nonzero] = 1.0 / counts[nonzero]
        total = inverse.sum()
        if total <= 0:
            return np.full(num_classes, 1.0 / num_classes)
        return inverse / total

    def _estimate_region_archetypes(
        self,
        signatures: np.ndarray,
        count_matrix: np.ndarray,
        num_classes: int,
        region=None,
    ) -> tuple[np.ndarray, dict]:
        """
        Estimate the archetype methylation probabilities mu[c, k] for one region.

        For a (ctype, CpG) pair that is covered (at least one read of ctype c covers
        CpG k), mu[c, k] is the maximum-likelihood estimate o1[c, k] / (o1[c, k] +
        o0[c, k]). For a CpG k that is uncovered for ctype c, mu[c, k] is set to the
        average of the covered mu[c, k'] of that same ctype within the region. The
        result is clipped to (eps, 1 - eps).

        Args:
            signatures: array of signatures (one per unique signature in the region)
            count_matrix: (n_sigs, num_classes) matrix giving, for each signature, the
                number of reads of each ctype carrying it
            num_classes: total number of classes
            region: the name of the region (used only for error messages)

        Returns:
            mu: (num_classes, n_positions) matrix of methylation probabilities
            pos_to_idx: mapping from CpG genomic position to its column index in mu

        Raises:
            ValueError: if a ctype has no covered CpG in the region
                (i.e. it is completely uncovered).
        """
        positions = sorted({pos for sig in signatures for pos, _ in sig})
        pos_to_idx = {pos: i for i, pos in enumerate(positions)}
        n_positions = len(positions)

        # o1[c, k] / o0[c, k]: methylated / unmethylated read counts of ctype c at CpG k
        o1 = np.zeros((num_classes, n_positions))
        o0 = np.zeros((num_classes, n_positions))
        for sig, counts_vec in zip(signatures, count_matrix):
            for pos, state in sig:
                k = pos_to_idx[pos]
                if state == 1:
                    o1[:, k] += counts_vec
                else:
                    o0[:, k] += counts_vec

        coverage = o1 + o0  # (C, K) total reads of ctype c covering CpG k
        covered = coverage > 0

        # a ctype with no covered CpG in the region is completely uncovered: error out
        ctype_covered = covered.any(axis=1)  # (C,)
        if not ctype_covered.all():
            uncovered_ctypes = np.where(~ctype_covered)[0].tolist()
            Warning(
                f"Cell type(s) {uncovered_ctypes} are completely uncovered "
                f"(0 reads) in region {region!r}. The number of fully methylated reads and coverage are artificially set to 1"
            )
        
            o1[uncovered_ctypes, :] = 1
            coverage[uncovered_ctypes, :] =1
            covered = coverage > 0

        # maximum-likelihood estimate where covered, NaN elsewhere
        mu = np.full((num_classes, n_positions), np.nan)
        np.divide(o1, coverage, out=mu, where=covered)

        # fill uncovered CpGs with the per-ctype average of the covered mu
        ctype_means = np.nanmean(mu, axis=1)  # (C,), well-defined: each ctype covered
        uncovered_rows, uncovered_cols = np.where(~covered)
        mu[uncovered_rows, uncovered_cols] = ctype_means[uncovered_rows]

        # keep mu strictly inside (0, 1) so the log-likelihoods stay finite
        mu = np.clip(mu, self.eps, 1.0 - self.eps)
        return mu, pos_to_idx

    @staticmethod
    def _signature_log_likelihood(
        signature, mu: np.ndarray, pos_to_idx: dict
    ) -> np.ndarray:
        """
        Compute the per-ctype log-likelihood log P(sig | c) of a signature.

        log phi[c] = sum_{k in dom(s)} s_k log(mu[c, k]) + (1 - s_k) log(1 - mu[c, k])

        Positions absent from the fitted model (``pos_to_idx``) are dropped, so a
        signature seen only outside the fit subset is still scored on its covered
        CpGs.
        """
        num_classes = mu.shape[0]
        pairs = [(pos, state) for pos, state in signature if pos in pos_to_idx]
        if not pairs:
            return np.zeros(num_classes)

        indices = [pos_to_idx[pos] for pos, _ in pairs]
        states = np.array([state for _, state in pairs], dtype=float)  # (L,)
        mu_sub = mu[:, indices]  # (num_classes, L)
        log_terms = states * np.log(mu_sub) + (1.0 - states) * np.log(1.0 - mu_sub)
        return log_terms.sum(axis=1)

    @staticmethod
    def _bayes_posterior(log_phi: np.ndarray, prior: np.ndarray) -> np.ndarray:
        """
        Compute the posterior P(ctype | sig) from log-likelihoods and a prior.

        Uses a log-sum-exp stabilization to avoid underflow. Ctypes with a zero
        prior keep an exactly zero posterior.
        """
        num_classes = log_phi.shape[0]
        log_post = np.full(num_classes, -np.inf)
        mask = prior > 0
        log_post[mask] = np.log(prior[mask]) + log_phi[mask]

        max_log = log_post.max()
        if not np.isfinite(max_log):
            # all ctypes have zero prior: fall back to a uniform distribution
            return np.full(num_classes, 1.0 / num_classes)

        post = np.exp(log_post - max_log)
        post /= post.sum()
        return post
