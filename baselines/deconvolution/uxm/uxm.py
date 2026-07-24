"""
This module  reimplements pieces of UXM code from original work by Loyfer et. al: https://github.com/nloyfer/UXM_deconv.
The focus is on creating a minimum setup sufficient to run deconvolution in python in a manner compatible with overall pipeline without introducing dependencies to original code.
Some of the methods are directly copied while other are specific to this repo.

Use and distribution of the original UXM_deconv code reproduced here is subject to the
Software Research License included in LICENSE.md alongside this module.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd
import numpy as np
from scipy import optimize
from baselines.deconvolution.base import BaselineDeconvolver
from baselines.deconvolution.utils import rearange_deconvolution_results

# ``mark_records_methyl_state`` is owned by syto core (used both by the UXM
# baseline and by :meth:`UXMMethylationAtlas.from_reads`); re-exported here for
# backward compatibility.
from syto.data.atlases.uxm_atlases import mark_records_methyl_state

if TYPE_CHECKING:
    from syto.data.atlases.uxm_atlases import UXMMethylationAtlas

### Selected original deconvolution code from https://github.com/nloyfer/UXM_deconv ###

_module_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OOP interface
# ---------------------------------------------------------------------------


class UXMDeconvolver(BaselineDeconvolver):
    """UXM read-based deconvolution baseline.

    Parameters
    ----------
    atlas : UXMMethylationAtlas
        Atlas object providing region boundaries and cell-type UXM ratios.
    ref_cells : list of str, optional
        Cell-type columns to use for deconvolution.  Defaults to
        ``atlas.ref_cells`` (all available cell types).
    """

    name = "uxm"

    def __init__(
        self,
        atlas: "UXMMethylationAtlas",
        ref_cells: Optional[List[str]] = None,
    ) -> None:
        self._atlas = atlas
        self._ref_cells = ref_cells if ref_cells is not None else atlas.ref_cells

    @property
    def atlas(self) -> "UXMMethylationAtlas":
        return self._atlas

    def prepare_reads(self, reads: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Overlap reads with atlas, trim to region boundaries, and mark M/U/X state."""
        prepared = self._atlas.prepare_reads(reads, trim=True, **kwargs)
        return mark_records_methyl_state(prepared)

    def _build_uxm_input(self, reads: pd.DataFrame, min_cpgs_count=4) -> dict:
        """
        Build UXM-compatible scaling factors and counts from prepared reads.

        Expects reads to already have NCPGS, record_M, record_U, record_X (from
        mark_records_methyl_state) and a 'name' column identifying the atlas region.
        """
        from copy import deepcopy

        results_agg = (
            reads[reads["NCPGS"] >= min_cpgs_count]
            .groupby("name")
            .aggregate({"record_M": "sum", "record_U": "sum", "record_X": "sum"})
            .reset_index()
        )
        results_agg["count"] = (
            results_agg["record_M"] + results_agg["record_U"] + results_agg["record_X"]
        )
        results_agg["sf"] = results_agg["record_U"] / results_agg["count"]
        results_agg["direction"] = "U"

        sf = deepcopy(results_agg[["name", "direction"]])
        sf["sample"] = results_agg["sf"]
        counts = results_agg[["name", "direction", "count"]].copy()
        counts.columns = ["name", "direction", "sample"]

        return {"scaling_factors": sf, "counts": counts}

    def build_input(self, reads: pd.DataFrame, min_cpgs_count: int = 4) -> dict:
        return self._build_uxm_input(reads, min_cpgs_count=min_cpgs_count)

    def deconvolute_single_sample(self, samp, atlas, counts, verbose, debug=False):
        """
        Deconvolve a single sample, using NNLS, to get the mixture coefficients.
        :param samp: a vector of a single sample
        :param atlas: the atlas DataFrame
        :return: the mixture coefficients
        """

        name = samp.columns[2]
        counts.columns = ["name", "direction", "counts"]

        # remove missing sites from both sample and atlas:
        # TODO: imputation for the atlas?
        nd_cols = ["name", "direction"]
        data = (
            samp.merge(
                atlas.drop_duplicates(nd_cols, ignore_index=True),
                on=nd_cols,
                how="inner",
            )
            .copy()
            .dropna(axis=0)
        )
        data = data.merge(
            counts.drop_duplicates(nd_cols, ignore_index=True), on=nd_cols, how="left"
        )

        if data.empty:
            _module_logger.warning("Skipping an empty sample: %s", name)
            return np.nan, np.nan

        if data.shape[0] > atlas.shape[0]:
            _module_logger.error("Merge went wrong. Validate your atlas")
            return None, None
        if verbose:
            _module_logger.info(
                "%s: %d \\ %d markers", name, data.shape[0], atlas.shape[0]
            )
        del data["name"], data["direction"]

        samp = data.iloc[:, 0]
        counts = data.iloc[:, -1]
        red_atlas = data.iloc[:, 1:-1]

        # apply weights:
        red_atlas = red_atlas * counts.values[:, np.newaxis]
        samp = samp * counts

        # get the mixture coefficients by deconvolution
        # (non-negative least squares)
        mixture, residual = optimize.nnls(red_atlas, samp)
        mixture /= np.sum(mixture)
        return mixture

    def deconvolute_multiple_samples(
        self, atlas, ref_cells, sf, counts, sample_names=["pseudo_bulk_sample"]
    ):
        params = [
            (
                sf[["name", "direction", samp]],
                atlas[["name", "direction"] + ref_cells],
                counts[["name", "direction", samp]],
                False,
                False,
            )
            for samp in sample_names
        ]

        arr = [self.deconvolute_single_sample(*p) for p in params]
        return arr

    def deconvolute_reads(
        self,
        reads: pd.DataFrame,
        labels_dict_reversed: Dict[str, int],
        n_labels: Optional[int] = None,
        prepare: bool = True,
        min_cpgs_count: int = 4,
    ) -> Optional[List[float]]:
        reads_sorted = self._sort_reads(reads)
        prepared = self.prepare_reads(reads_sorted) if prepare else reads_sorted
        uxm_in = self.build_input(prepared, min_cpgs_count=min_cpgs_count)
        proportions = self.deconvolute_multiple_samples(
            self._atlas.atlas,
            self._ref_cells,
            uxm_in["scaling_factors"],
            uxm_in["counts"],
            sample_names=["sample"],
        )[0]
        if not isinstance(proportions, np.ndarray):
            return None
        return rearange_deconvolution_results(
            labels_dict_reversed, proportions, self._ref_cells, n_labels=n_labels
        )
