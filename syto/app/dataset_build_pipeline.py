"""Recovered-reads / BAM -> atlas-overlapped Parquet dataset build pipeline.

Two phases:
  stage    -- per-file overlap+trim+label+bucket (parallel by file); inputs are
              recovered-reads CSVs (input_type: csv) or BAMs labelled per
              sample via sample_groups_path (input_type: bam)
  finalize -- per-bucket soft/hard labels + split + compaction
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from syto.data.atlases.uxm_atlases import UXMMethylationAtlas
from syto.data.dataset_build.buckets import build_region_index
from syto.data.dataset_build.stage import stage_bam_file, stage_file
from syto.data.dataset_build.splits import plan_splits
from syto.data.dataset_build.finalize import finalize_bucket
from syto.data.dataset_build.filters import staged_counts, load_region_names
from syto.data.labelers.archetype_labeler import ArchetypeLabeler
from syto.data.sequencing.bam_processing import (
    merge_fetch_intervals,
    resolve_bam_parsing_params,
)


def _stage_worker(
    path,
    *,
    reference_genome,
    atlas_path,
    atlas_cell_types,
    region_index,
    labels_dict,
    staged_dir,
    counts_dir,
    sep,
    cell_type_match_dict,
):
    """Module-level staging worker (picklable for ProcessPoolExecutor).

    The atlas is reconstructed inside each worker process rather than pickled
    and shipped across the process boundary.
    """
    atlas = UXMMethylationAtlas(
        atlas_name="build",
        reference_genome=reference_genome,
        atlas_path=atlas_path,
        cell_types=atlas_cell_types,
    )
    return stage_file(
        str(path),
        atlas,
        region_index,
        labels_dict,
        staged_dir,
        counts_dir,
        sep=sep,
        cell_type_match_dict=cell_type_match_dict,
    )


def _stage_bam_worker(
    path_and_ctype,
    *,
    reference_genome,
    atlas_path,
    atlas_cell_types,
    region_index,
    labels_dict,
    staged_dir,
    counts_dir,
    bam_params,
    intervals,
    cell_type_match_dict,
):
    """BAM counterpart of :func:`_stage_worker`; takes a ``(path, ctype)`` pair."""
    path, ctype = path_and_ctype
    atlas = UXMMethylationAtlas(
        atlas_name="build",
        reference_genome=reference_genome,
        atlas_path=atlas_path,
        cell_types=atlas_cell_types,
    )
    return stage_bam_file(
        str(path),
        ctype,
        atlas,
        region_index,
        labels_dict,
        staged_dir,
        counts_dir,
        bam_params=bam_params,
        intervals=intervals,
        cell_type_match_dict=cell_type_match_dict,
    )


class DatasetBuildPipeline:
    """Orchestrate the two-pass recovered-reads dataset build."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config["output_dir"])
        # Where staged shards + counts live. Defaults to output_dir so single-dir
        # runs are unchanged; set it to a previous run's dir to finalize existing
        # staged data into a fresh output_dir (e.g. stage on U250, finalize on U25).
        staged_root = Path(config.get("staged_source_dir", self.output_dir))
        self.staged_dir = staged_root / "staged"
        self.counts_dir = staged_root / "counts"
        self.final_dir = self.output_dir / "final"
        self.n_buckets = int(config["n_buckets"])

    def _atlas(self):
        return UXMMethylationAtlas(
            atlas_name="build",
            reference_genome=self.config["reference_genome"],
            atlas_path=self.config["atlas_path"],
            cell_types=self.config.get("atlas_cell_types"),
        )

    def _labels_dict(self):
        return json.loads(Path(self.config["labels_dict_path"]).read_text())

    def run(self) -> Dict[str, Any]:
        phase = self.config.get("phase", "all")
        summary: Dict[str, Any] = {}
        if phase in ("stage", "all"):
            summary["stage"] = self._run_stage()
        if phase in ("finalize", "all"):
            summary["finalize"] = self._run_finalize()
        return summary

    def _run_stage(self) -> Dict[str, Any]:
        atlas = self._atlas()
        labels_dict = self._labels_dict()
        region_index = build_region_index(atlas.atlas, self.n_buckets)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        region_index.to_parquet(self.output_dir / "regions_index.parquet", index=False)

        cell_match = self.config.get("cell_type_match_dict") or {}
        n_workers = int(self.config.get("n_workers", 1))
        common = dict(
            reference_genome=self.config["reference_genome"],
            atlas_path=self.config["atlas_path"],
            atlas_cell_types=self.config.get("atlas_cell_types"),
            region_index=region_index,
            labels_dict=labels_dict,
            staged_dir=str(self.staged_dir),
            counts_dir=str(self.counts_dir),
            cell_type_match_dict=cell_match,
        )

        input_type = self.config.get("input_type", "csv")
        if input_type == "csv":
            inputs = sorted(Path(self.config["input_dir"]).glob("*.csv"))
            work = partial(_stage_worker, sep=self.config.get("sep", "\t"), **common)
        elif input_type == "bam":
            inputs = self._bam_inputs()
            bam_cfg = self.config.get("bam_processing", {})
            bam_params = resolve_bam_parsing_params(
                bam_path=None,
                data_type=bam_cfg.get("data_type", "wgbs"),
                bam_cfg=bam_cfg,
                chromosomes=bam_cfg.get("chromosomes", "all"),
                reference_path=bam_cfg.get("reference_path"),
            )
            padding = (
                int(bam_cfg.get("mate_fetch_padding", 1000))
                if bam_params["merge_pairs"]
                else 0
            )
            intervals = merge_fetch_intervals(
                atlas.atlas, bam_params["chromosomes"], padding
            )
            work = partial(
                _stage_bam_worker,
                bam_params=bam_params,
                intervals=intervals,
                **common,
            )
        else:
            raise ValueError(f"input_type must be 'csv' or 'bam', got {input_type!r}")

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                stats = list(
                    tqdm(ex.map(work, inputs), total=len(inputs), desc="Staging files")
                )
        else:
            stats = [work(p) for p in tqdm(inputs, desc="Staging files")]

        self.logger.info("Staged %d files", len(stats))
        return {"files": len(stats), "stats": stats}

    def _bam_inputs(self):
        """``(bam_path, group)`` pairs for BAMs listed in ``sample_groups_path``.

        The groups CSV has ``name`` (BAM file name without ``.bam``) and
        ``group`` columns; BAMs not listed there are skipped.
        """
        groups = pd.read_csv(self.config["sample_groups_path"])
        group_of = dict(zip(groups["name"], groups["group"]))
        bams = sorted(Path(self.config["input_dir"]).glob("*.bam"))
        skipped = [b.name for b in bams if b.stem not in group_of]
        if skipped:
            self.logger.info(
                "Skipping %d BAM(s) not in %s: %s",
                len(skipped),
                self.config["sample_groups_path"],
                ", ".join(skipped),
            )
        missing = sorted(set(group_of) - {b.stem for b in bams})
        if missing:
            self.logger.warning(
                "%d sample(s) in the groups file have no BAM: %s",
                len(missing),
                ", ".join(missing),
            )
        return [(b, group_of[b.stem]) for b in bams if b.stem in group_of]

    def _run_finalize(self) -> Dict[str, Any]:
        labels_dict = self._labels_dict()
        num_classes = len(labels_dict)
        labelers_config = self.config["labelers"]
        signature_config = self.config["signature"]
        pattern_column = signature_config["methylation_pattern_column"]

        finalize_atlas_path = self.config.get("finalize_atlas_path")
        min_pattern_length = self.config.get("min_pattern_length")
        fit_splits = self.config.get("fit_splits")
        fallback = self.config.get("label_fallback", "uniform")
        split_remap = self.config.get("split_remap")

        region_names = (
            load_region_names(finalize_atlas_path) if finalize_atlas_path else None
        )
        filtering_active = region_names is not None or bool(min_pattern_length)

        if filtering_active:
            counts = staged_counts(
                str(self.staged_dir),
                region_names=region_names,
                min_pattern_length=min_pattern_length,
                pattern_column=pattern_column,
            )
        else:
            counts = pd.concat(
                [pd.read_parquet(p) for p in sorted(self.counts_dir.glob("*.parquet"))],
                ignore_index=True,
            )

        plan = plan_splits(
            counts,
            train_ratio=self.config.get("train_ratio", 0.7),
            valid_ratio=self.config.get("valid_ratio", 0.15),
            test_ratio=self.config.get("test_ratio", 0.15),
            seed=self.config.get("seed", 42),
        )
        global_prior = self._compute_global_prior(counts, num_classes, labelers_config)
        buckets = sorted(
            int(p.name.split("=")[1]) for p in self.staged_dir.glob("region_bucket=*")
        )
        written = []
        for b in tqdm(buckets, desc="Finalizing buckets"):
            path = finalize_bucket(
                str(self.staged_dir),
                b,
                str(self.final_dir),
                plan,
                num_classes=num_classes,
                labelers_config=labelers_config,
                signature_config=signature_config,
                global_prior=global_prior,
                region_names=region_names,
                min_pattern_length=min_pattern_length,
                pattern_column=pattern_column,
                fit_splits=fit_splits,
                fallback=fallback,
                split_remap=split_remap,
            )
            if path is not None:
                written.append(path)
        self.logger.info("Finalized %d buckets", len(written))
        return {"buckets": len(written), "paths": written}

    def _compute_global_prior(self, counts, num_classes, labelers_config):
        """Dataset-wide inverse-frequency prior, only if an archetype needs it.

        Computed from the per-(file, original_label) counts sidecar so no reads
        are loaded. Returns None when no archetype labeler requests
        'inv_global_freq'.
        """
        needs_global = any(
            entry.get("type") == "archetype"
            and entry.get("ctype_prior_type") == "inv_global_freq"
            for entry in labelers_config.values()
        )
        if not needs_global:
            return None
        global_counts = np.zeros(num_classes)
        summed = counts.groupby("original_label")["n_reads"].sum()
        for label, n_reads in summed.items():
            global_counts[int(label)] = n_reads
        return ArchetypeLabeler._normalize_inverse_prior(global_counts, num_classes)
