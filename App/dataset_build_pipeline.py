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

import pandas as pd

from syto.data.atlases.uxm_atlases import UXMMethylationAtlas
from syto.data.dataset_build.buckets import build_region_index
from syto.data.dataset_build.stage import stage_file
from syto.data.dataset_build.splits import plan_splits
from syto.data.dataset_build.finalize import finalize_bucket


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
            atlas_name="build", reference_genome=self.config["reference_genome"],
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
                str(path), self._atlas(), region_index, labels_dict,
                str(self.staged_dir), str(self.counts_dir),
                sep=self.config.get("sep", "\t"), cell_type_match_dict=cell_match,
            )

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                stats = list(ex.map(_work, csvs))
        else:
            stats = [_work(p) for p in csvs]

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
        buckets = sorted(
            int(p.name.split("=")[1]) for p in self.staged_dir.glob("region_bucket=*")
        )
        written = []
        for b in buckets:
            written.append(finalize_bucket(
                str(self.staged_dir), b, str(self.final_dir), plan,
                num_classes=num_classes,
                max_distance=self.config.get("max_distance", 0.5),
                min_reads=self.config.get("min_reads", 30),
            ))
        self.logger.info("Finalized %d buckets", len(written))
        return {"buckets": len(written), "paths": written}
