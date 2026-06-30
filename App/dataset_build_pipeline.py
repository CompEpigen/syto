"""Recovered-reads -> atlas-overlapped Parquet dataset build pipeline.

Two phases:
  stage    -- per-file overlap+trim+label+bucket (parallel by file)
  finalize -- per-bucket soft/hard labels + split + compaction
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from syto.data.atlases.uxm_atlases import UXMMethylationAtlas
from syto.data.dataset_build.buckets import build_region_index
from syto.data.dataset_build.stage import stage_file
from syto.data.dataset_build.splits import plan_splits
from syto.data.dataset_build.finalize import finalize_bucket
from syto.data.labelers.archetype_labeler import ArchetypeLabeler


class DatasetBuildPipeline:
    """Orchestrate the two-pass recovered-reads dataset build."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config["output_dir"])
        self.staged_dir = self.output_dir / "staged"
        self.counts_dir = self.output_dir / "counts"
        self.final_dir = self.output_dir / "final"
        self.n_buckets = int(config["n_buckets"])

    def _atlas(self):
        return UXMMethylationAtlas(
            atlas_name="build",
            reference_genome=self.config["reference_genome"],
            atlas_path=self.config["atlas_path"],
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

        csvs = sorted(Path(self.config["input_dir"]).glob("*.csv"))
        cell_match = self.config.get("cell_type_match_dict") or {}
        n_workers = int(self.config.get("n_workers", 1))

        def _work(path):
            return stage_file(
                str(path),
                self._atlas(),
                region_index,
                labels_dict,
                str(self.staged_dir),
                str(self.counts_dir),
                sep=self.config.get("sep", "\t"),
                cell_type_match_dict=cell_match,
            )

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                stats = list(
                    tqdm(ex.map(_work, csvs), total=len(csvs), desc="Staging files")
                )
        else:
            stats = [_work(p) for p in tqdm(csvs, desc="Staging files")]

        self.logger.info("Staged %d files", len(stats))
        return {"files": len(stats), "stats": stats}

    def _run_finalize(self) -> Dict[str, Any]:
        labels_dict = self._labels_dict()
        num_classes = len(labels_dict)
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
        labelers_config = self.config["labelers"]
        signature_config = self.config["signature"]
        global_prior = self._compute_global_prior(counts, num_classes, labelers_config)
        buckets = sorted(
            int(p.name.split("=")[1]) for p in self.staged_dir.glob("region_bucket=*")
        )
        written = []
        for b in tqdm(buckets, desc="Finalizing buckets"):
            written.append(
                finalize_bucket(
                    str(self.staged_dir),
                    b,
                    str(self.final_dir),
                    plan,
                    num_classes=num_classes,
                    labelers_config=labelers_config,
                    signature_config=signature_config,
                    global_prior=global_prior,
                )
            )
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
