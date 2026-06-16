"""
Pseudobulk Deconvolution Pipeline

Runs baseline deconvolution models on a pre-generated pseudobulk HDF5 file and
saves predicted cell-type proportions as parquet files for downstream comparison.

Supported models: uxm

Output schema (one row per pseudobulk x cell type):
  pb_index          - pseudobulk identifier (position within the split)
  cell_type         - cell type name from labels_dict
  target_proportion - ground-truth mixing proportion
  {model}           - predicted proportion (column named after the model)

To compare models later, merge on (pb_index, cell_type).
"""

import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from syto.data.pseudobulk_hdf5_utils import PseudobulkHDF5Reader, PseudobulkHDF5Schema

# ---------------------------------------------------------------------------
# Module-level worker state — populated in the parent and inherited by
# child processes via fork (zero-copy, copy-on-write) so large objects
# (input_df, atlas_df, indices) are never serialised per task.
# ---------------------------------------------------------------------------
_WORKER_UXM_STATE: Dict[str, Any] = {}


def _uxm_worker(
    task: Tuple[int, int, np.ndarray, np.ndarray],
) -> Tuple[int, List[Dict[str, Any]]]:
    """Run one pseudobulk end-to-end: reconstruct reads, then run UXM.

    Must be a module-level function so it is picklable.  Large shared objects
    come from ``_WORKER_UXM_STATE``, which forked workers inherit for free.
    """
    pb_index, seed, n_samples_per_class_per_grg, target_proportions = task
    s = _WORKER_UXM_STATE

    from syto.data.pseudobulk_generator import _sample_read_ids_from_grouped_dataframe
    from baselines.deconvolution.uxm.uxm import (
        build_uxm_input,
        mark_records_methyl_state,
        rearange_uxm_deconvolution_results,
        uxm_deconvolution,
    )

    read_ids = _sample_read_ids_from_grouped_dataframe(
        n_samples_per_class_per_grg, s["indices_per_class_and_grg"], seed=seed
    )
    reads = s["input_df"].iloc[read_ids].reset_index(drop=True)
    reads = mark_records_methyl_state(reads)
    uxm_in = build_uxm_input(reads)
    proportions = uxm_deconvolution(
        s["atlas_df"],
        s["ref_cells"],
        uxm_in["scaling_factors"],
        uxm_in["counts"],
        sample_names=["sample"],
    )[0]

    if not isinstance(proportions, np.ndarray):
        return pb_index, []

    aligned = rearange_uxm_deconvolution_results(
        s["labels_dict_reversed"], proportions, s["ref_cells"], n_labels=s["n_labels"]
    )
    return pb_index, [
        {
            "pb_index": pb_index,
            "cell_type": s["labels_dict"][label_idx],
            "target_proportion": float(target_proportions[label_idx]),
            "uxm": float(aligned[label_idx]),
        }
        for label_idx in range(s["n_labels"])
    ]


class PseudobulkDeconvolutionPipeline:
    """Run a baseline deconvolution model over all pseudobulks in an HDF5 file.

    Per split, shared state (input_df, sampling index, atlas) is loaded once
    in the parent process and inherited by worker processes via fork.  Workers
    receive only the tiny per-pseudobulk HDF5 params (seed + n_reads arrays)
    and perform the full reconstruction + deconvolution without GIL contention.

    Results are written to
    ``{output_dir}/{model}/{split}_deconvolution_results.parquet``.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration. See config template for full reference.
    logger : logging.Logger
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config["output_dir"])
        self.h5_path = config["pseudobulk_h5_path"]
        self.model = config["model"]
        self.class_label_column = config.get("class_label_column", "original_label")

        with open(config["labels_dict_path"], "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.labels_dict: Dict[int, str] = {int(k): v for k, v in raw.items()}
        self.labels_dict_reversed: Dict[str, int] = {
            v: k for k, v in self.labels_dict.items()
        }
        self.n_labels = len(self.labels_dict)

    def _resolve_splits(self, reader: PseudobulkHDF5Reader) -> List[str]:
        configured = self.config.get("splits")
        if configured:
            return configured
        return reader.list_splits()

    def _count_split_pseudobulks(self, split: str) -> Optional[int]:
        import h5py

        pbs_group = PseudobulkHDF5Schema.pseudobulks_group(split)
        try:
            with h5py.File(self.h5_path, "r") as f:
                return len(f[pbs_group]) if pbs_group in f else None
        except Exception:
            return None

    def _run_uxm(self) -> None:
        global _WORKER_UXM_STATE

        from baselines.deconvolution.uxm.uxm import load_atlas

        n_workers: int = self.config.get("n_workers", 1)
        # chunksize controls how many tasks are sent to each worker per IPC
        # round-trip. For slow tasks (>100 ms each) 1 is fine; raise it only
        # if IPC overhead becomes measurable.
        chunksize: int = self.config.get("batch_size", 1)

        atlas_path = self.config["atlas_path"]
        ignore_cells = self.config.get("ignore_cells", [])

        atlas_df, ref_cells_all = load_atlas(atlas_path)
        ref_cells = [c for c in ref_cells_all if c not in ignore_cells]
        self.logger.info("UXM: %d reference cell types", len(ref_cells))

        reader = PseudobulkHDF5Reader(self.h5_path, logger=self.logger)
        splits = self._resolve_splits(reader)

        out_dir = self.output_dir / "uxm"
        out_dir.mkdir(parents=True, exist_ok=True)

        for split in splits:
            self.logger.info("Processing split: %s", split)
            total = self._count_split_pseudobulks(split)

            self.logger.info("  Loading shared reconstruction state…")
            input_df, indices_per_class_and_grg = reader.build_reconstruction_state(
                split, class_label_column=self.class_label_column
            )

            # Set module-level state BEFORE forking so workers inherit it.
            _WORKER_UXM_STATE = {
                "input_df": input_df,
                "indices_per_class_and_grg": indices_per_class_and_grg,
                "atlas_df": atlas_df,
                "ref_cells": ref_cells,
                "labels_dict": self.labels_dict,
                "labels_dict_reversed": self.labels_dict_reversed,
                "n_labels": self.n_labels,
            }

            tasks = (
                (pb_idx, seed, n_samples, target_props)
                for pb_idx, (seed, n_samples, target_props) in enumerate(
                    reader.iter_pseudobulk_params(split)
                )
            )

            rows: List[Dict[str, Any]] = []

            if n_workers == 1:
                with tqdm(total=total, desc=split, unit="pb") as pbar:
                    for task in tasks:
                        pb_index, result_rows = _uxm_worker(task)
                        if not result_rows:
                            self.logger.warning(
                                "pseudobulk %d: deconvolution returned NaN, skipping",
                                pb_index,
                            )
                        rows.extend(result_rows)
                        pbar.update(1)
            else:
                ctx = mp.get_context("fork")
                with ctx.Pool(n_workers) as pool, tqdm(
                    total=total, desc=split, unit="pb"
                ) as pbar:
                    for pb_index, result_rows in pool.imap_unordered(
                        _uxm_worker, tasks, chunksize=chunksize
                    ):
                        if not result_rows:
                            self.logger.warning(
                                "pseudobulk %d: deconvolution returned NaN, skipping",
                                pb_index,
                            )
                        rows.extend(result_rows)
                        pbar.update(1)

            df = pd.DataFrame(rows)
            out_path = out_dir / f"{split}_deconvolution_results.parquet"
            df.to_parquet(out_path, index=False)
            self.logger.info(
                "Saved %d rows (%d pseudobulks) for split '%s' → %s",
                len(df),
                df["pb_index"].nunique(),
                split,
                out_path,
            )

    def run(self) -> None:
        if self.model == "uxm":
            self._run_uxm()
        else:
            raise ValueError(
                f"Unknown deconvolution model: '{self.model}'. Supported: uxm"
            )
