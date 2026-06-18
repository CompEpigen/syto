"""
Pseudobulk Deconvolution Pipeline

Runs one or more baseline deconvolution models on a pre-generated pseudobulk
HDF5 file and saves predicted cell-type proportions as parquet files for
downstream comparison.

Supported models: uxm, celfieish, celfie

Output schema (one row per pseudobulk × cell type):
  pb_index          - pseudobulk identifier (position within the split)
  cell_type         - cell type name from labels_dict
  target_proportion - ground-truth mixing proportion
  {model_key}       - predicted proportion (column named after the model/checkpoint)

For EM models with em_checkpoints configured, results are written per
checkpoint:
  {output_dir}/celfieish_50_steps/{split}_deconvolution_results.parquet

To compare models later, merge parquets on (pb_index, cell_type).
"""

import json
import logging
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from syto.data.pseudobulk_hdf5_utils import PseudobulkHDF5Reader, PseudobulkHDF5Schema

# ---------------------------------------------------------------------------
# Module-level worker state — populated in the parent and inherited by
# child processes via fork (zero-copy, copy-on-write) so large objects
# (input_df, atlas objects, indices) are never serialised per task.
# ---------------------------------------------------------------------------
_WORKER_STATE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Per-model read helpers (called inside _baselines_worker)
# ---------------------------------------------------------------------------

def _make_rows(
    pb_index: int,
    col_name: str,
    aligned: List[float],
    target_proportions: np.ndarray,
    labels_dict: Dict[int, str],
    n_labels: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "pb_index": pb_index,
            "cell_type": labels_dict[i],
            "target_proportion": float(target_proportions[i]),
            col_name: float(aligned[i]),
        }
        for i in range(n_labels)
    ]


def _run_uxm_on_reads(
    reads: pd.DataFrame,
    model_cfg: Dict[str, Any],
    s: Dict[str, Any],
    pb_index: int,
    target_proportions: np.ndarray,
) -> Dict[str, List[Dict[str, Any]]]:
    from baselines.deconvolution.uxm.uxm import run_uxm_deconvolution

    aligned = run_uxm_deconvolution(
        reads, model_cfg["atlas_df"], model_cfg["ref_cells"],
        s["labels_dict_reversed"], n_labels=s["n_labels"],
    )
    if aligned is None:
        return {}
    return {
        "uxm": _make_rows(
            pb_index, "uxm", aligned, target_proportions, s["labels_dict"], s["n_labels"]
        )
    }


def _run_celfieish_on_reads(
    reads: pd.DataFrame,
    model_cfg: Dict[str, Any],
    s: Dict[str, Any],
    pb_index: int,
    target_proportions: np.ndarray,
    prepare_reads: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    from baselines.deconvolution.celfieish.celfieish import run_celfieish_deconvolution

    em_checkpoints = model_cfg.get("em_checkpoints")
    result = run_celfieish_deconvolution(
        reads, model_cfg["atlas"], s["labels_dict_reversed"],
        n_labels=s["n_labels"], prepare_reads=prepare_reads,
        num_iterations=model_cfg.get("num_iterations", 50),
        convergence_criteria=model_cfg.get("convergence_criteria", 0.001),
        checkpoints=em_checkpoints,
    )
    if result is None:
        return {}

    if em_checkpoints is not None:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for n_steps, aligned in result:
            col = f"celfieish_{n_steps}_steps"
            out[col] = _make_rows(
                pb_index, col, aligned, target_proportions, s["labels_dict"], s["n_labels"]
            )
        return out

    return {
        "celfieish": _make_rows(
            pb_index, "celfieish", result, target_proportions, s["labels_dict"], s["n_labels"]
        )
    }


def _run_celfie_on_reads(
    reads: pd.DataFrame,
    model_cfg: Dict[str, Any],
    s: Dict[str, Any],
    pb_index: int,
    target_proportions: np.ndarray,
    prepare_reads: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    from baselines.deconvolution.celfie.celfie import run_celfie_deconvolution

    em_checkpoints = model_cfg.get("em_checkpoints")
    result = run_celfie_deconvolution(
        reads, model_cfg["atlas"], s["labels_dict_reversed"],
        n_labels=s["n_labels"], prepare_reads=prepare_reads,
        num_iterations=model_cfg.get("num_iterations", 50),
        convergence_criteria=model_cfg.get("convergence_criteria", 0.001),
        random_restarts=model_cfg.get("random_restarts", 1),
        checkpoints=em_checkpoints,
    )
    if result is None:
        return {}

    if em_checkpoints is not None:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for n_steps, aligned in result:
            col = f"celfie_{n_steps}_steps"
            out[col] = _make_rows(
                pb_index, col, aligned, target_proportions, s["labels_dict"], s["n_labels"]
            )
        return out

    return {
        "celfie": _make_rows(
            pb_index, "celfie", result, target_proportions, s["labels_dict"], s["n_labels"]
        )
    }


# ---------------------------------------------------------------------------
# Module-level worker
# ---------------------------------------------------------------------------

def _baselines_worker(
    task: Tuple[int, int, np.ndarray, np.ndarray],
) -> Tuple[int, Dict[str, List[Dict[str, Any]]]]:
    """Process one pseudobulk through all configured baseline models.

    Reads are sampled once and shared across every model in the run.
    Returns (pb_index, {output_key: [row_dicts]}).
    """
    pb_index, seed, n_samples_per_class_per_grg, target_proportions = task
    s = _WORKER_STATE

    from syto.data.pseudobulk_generator import _sample_read_ids_from_grouped_dataframe

    read_ids = _sample_read_ids_from_grouped_dataframe(
        n_samples_per_class_per_grg, s["indices_per_class_and_grg"], seed=seed
    )
    reads = s["input_df"].iloc[read_ids].reset_index(drop=True)

    results: Dict[str, List[Dict[str, Any]]] = {}
    for model_cfg in s["models"]:
        model_name = model_cfg["name"]
        if model_name == "uxm":
            model_results = _run_uxm_on_reads(reads, model_cfg, s, pb_index, target_proportions)
        elif model_name == "celfieish":
            model_results = _run_celfieish_on_reads(reads, model_cfg, s, pb_index, target_proportions, False)
        elif model_name == "celfie":
            model_results = _run_celfie_on_reads(reads, model_cfg, s, pb_index, target_proportions,False)
        else:
            model_results = {}
        results.update(model_results)

    return pb_index, results


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class PseudobulkDeconvolutionPipeline:
    """Run baseline deconvolution models over all pseudobulks in an HDF5 file.

    Per split, shared state (input_df, sampling index, atlas objects) is
    loaded once in the parent process and inherited by worker processes via
    fork.  Workers receive only the tiny per-pseudobulk params (seed +
    n_reads arrays) and perform full reconstruction + deconvolution without
    GIL contention.

    Config must contain a ``baselines`` list; each entry specifies a model
    name, atlas path, and optional EM parameters.  See the template YAML for
    the full reference.

    Output layout::

        {output_dir}/{output_key}/{split}_deconvolution_results.parquet

    where ``output_key`` is ``uxm``, ``celfieish``, ``celfie_50_steps``, etc.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config["output_dir"])
        self.h5_path = config["pseudobulk_h5_path"]
        self.class_label_column = config.get("class_label_column", "original_label")

        with open(config["labels_dict_path"], "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.labels_dict: Dict[int, str] = {int(k): v for k, v in raw.items()}
        self.labels_dict_reversed: Dict[str, int] = {v: k for k, v in self.labels_dict.items()}
        self.n_labels = len(self.labels_dict)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Atlas loading
    # ------------------------------------------------------------------

    def _load_model_state(self, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Load the atlas for one baseline config and return a worker-ready dict."""
        model_name = model_cfg["model"]

        if model_name == "uxm":
            from baselines.deconvolution.uxm.uxm import load_atlas

            atlas_df, ref_cells_all = load_atlas(model_cfg["atlas_path"])
            ignore_cells = model_cfg.get("ignore_cells", [])
            ref_cells = [c for c in ref_cells_all if c not in ignore_cells]
            self.logger.info("UXM: %d reference cell types", len(ref_cells))
            return {"name": "uxm", "atlas_df": atlas_df, "ref_cells": ref_cells}

        if model_name in ("celfieish", "celfie"):
            from syto.data.atlases.celfieish_atlases import CpGBetaCountsMethylationAtlas

            atlas_path = model_cfg["atlas_path"]
            atlas = CpGBetaCountsMethylationAtlas(
                atlas_name=model_cfg.get("atlas_name", Path(atlas_path).stem),
                reference_genome=model_cfg.get("reference_genome", "hg38"),
                atlas_path=atlas_path,
            )
            ref_cells = atlas.ref_cells
            self.logger.info("%s: %d reference cell types", model_name.upper(), len(ref_cells))
            state: Dict[str, Any] = {
                "name": model_name,
                "atlas": atlas,
                "ref_cells": ref_cells,
                "em_checkpoints": model_cfg.get("em_checkpoints"),
                "num_iterations": model_cfg.get("num_iterations", 50),
                "convergence_criteria": model_cfg.get("convergence_criteria", 0.001),
            }
            if model_name == "celfie":
                state["random_restarts"] = model_cfg.get("random_restarts", 1)
            return state

        raise ValueError(
            f"Unknown model: {model_name!r}. Supported: uxm, celfieish, celfie"
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def _run_baselines(self) -> None:
        global _WORKER_STATE

        n_workers: int = self.config.get("n_workers", 1)
        chunksize: int = self.config.get("batch_size", 1)

        baseline_cfgs = self.config["baselines"]
        model_states = [self._load_model_state(cfg) for cfg in baseline_cfgs]

        reader = PseudobulkHDF5Reader(self.h5_path, logger=self.logger)
        splits = self._resolve_splits(reader)

        for split in splits:
            self.logger.info("Processing split: %s", split)
            total = self._count_split_pseudobulks(split)

            self.logger.info("  Loading shared reconstruction state…")
            input_df, indices_per_class_and_grg = reader.build_reconstruction_state(
                split, class_label_column=self.class_label_column
            )

            # Populate module-level state BEFORE forking so workers inherit it.
            _WORKER_STATE = {
                "input_df": input_df,
                "indices_per_class_and_grg": indices_per_class_and_grg,
                "labels_dict": self.labels_dict,
                "labels_dict_reversed": self.labels_dict_reversed,
                "n_labels": self.n_labels,
                "models": model_states,
            }

            tasks = (
                (pb_idx, seed, n_samples, target_props)
                for pb_idx, (seed, n_samples, target_props) in enumerate(
                    reader.iter_pseudobulk_params(split)
                )
            )

            all_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

            if n_workers == 1:
                with tqdm(total=total, desc=split, unit="pb") as pbar:
                    for task in tasks:
                        pb_index, model_rows = _baselines_worker(task)
                        if not model_rows:
                            self.logger.warning(
                                "pseudobulk %d: all models returned empty results, skipping",
                                pb_index,
                            )
                        for key, rows in model_rows.items():
                            all_rows[key].extend(rows)
                        pbar.update(1)
            else:
                ctx = mp.get_context("fork")
                with ctx.Pool(n_workers) as pool, tqdm(
                    total=total, desc=split, unit="pb"
                ) as pbar:
                    for pb_index, model_rows in pool.imap_unordered(
                        _baselines_worker, tasks, chunksize=chunksize
                    ):
                        if not model_rows:
                            self.logger.warning(
                                "pseudobulk %d: all models returned empty results, skipping",
                                pb_index,
                            )
                        for key, rows in model_rows.items():
                            all_rows[key].extend(rows)
                        pbar.update(1)

            for output_key, rows in all_rows.items():
                out_dir = self.output_dir / output_key
                out_dir.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(rows)
                out_path = out_dir / f"{split}_deconvolution_results.parquet"
                df.to_parquet(out_path, index=False)
                self.logger.info(
                    "Saved %d rows (%d pseudobulks) for split '%s' [%s] → %s",
                    len(df),
                    df["pb_index"].nunique(),
                    split,
                    output_key,
                    out_path,
                )

    def run(self) -> None:
        self._run_baselines()
