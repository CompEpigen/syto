"""
Inference Pipeline

End-to-end pipeline for processing BAM files (or pre-processed reads),
running classifier predictions (MethylBERT, Dismir, CancerDetector,
or LookupClassifier), and performing deconvolution using multiple
methods simultaneously.
"""

import copy
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch

import numpy as np
import pandas as pd
from baselines.deconvolution.base import BaselineDeconvolver
from syto.data.atlases.uxm_atlases import mark_records_methyl_state
from baselines.deconvolution.uxm.uxm import UXMDeconvolver
from baselines.deconvolution.celfie.celfie import CelFiEDeconvolver
from baselines.deconvolution.celfieish.celfieish import CelFiEISHDeconvolver
from baselines.deconvolution.epidish.epidish import (
    EpiDishDeconvolver,
    epidish_result_name,
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


from syto.data import LOYFER_CELL_TYPE_MATCH_DICT
from syto.data.sequencing.bam_processing import (
    BamRegionReader,
    process_bam_with_chunking,
    resolve_merge_pairs,
)
from syto.classification.prediction_aggregation import (
    aggregate_predictions_by_grg,
    StreamingGrgAggregator,
)
from syto.classification.classifiers.lazy_classifier_factory import (
    read_classifier_factory,
)
from syto.data.atlases.uxm_atlases import UXMMethylationAtlas
from syto.deconvolution.feature_selection import apply_feature_mask
from syto.deconvolution.least_squares_deconvolvers import (
    PSLSDeconvolver,
    NNLSDeconvolver,
)
from syto.deconvolution.deep_deconvolvers.mlp import MLPDeconvolver
from syto.deconvolution.deep_deconvolvers.swn import SWNDeconvolver
from syto.deconvolution.xgbdeconvolver import (
    XGBoostDeconvolver,
)
from syto.calibration.linear_calibrator import LinearCalibrator
from syto.calibration.vector_scaling_calibrator import VectorScalingCalibrator

from syto.data.dataset import resolve_column

LINEAR_NORM_METHODS = ["clip-normalize", "simplex-projection"]


class _BaselineStreamAccumulator:
    """Collects one baseline's per-chunk inputs during a streamed run.

    Each baseline builds its model input as independent per-region entries, so
    a chunked run can call ``build_input`` on one slice of that baseline's
    atlas at a time and merge the pieces at the end.  Only the merged input
    survives between chunks — never the reads.
    """

    def __init__(self, deconvolver: BaselineDeconvolver, logger: logging.Logger):
        self.deconvolver = deconvolver
        self.name = deconvolver.name
        self.logger = logger
        self.parts: List[dict] = []
        self.n_reads = 0
        self.failed = False

    @property
    def atlas(self):
        return self.deconvolver.atlas

    def update(self, reads: pd.DataFrame, region_names) -> None:
        """Overlap *reads* with this chunk's regions and stash the input."""
        if self.failed:
            return
        try:
            # A copy narrowed to the chunk's regions: the per-region lookups are
            # shared, only the region table is restricted, so a read is never
            # attributed to a region belonging to another chunk.
            sliced = copy.copy(self.deconvolver)
            # pylint: disable=protected-access
            sliced._atlas = self.deconvolver.atlas.subset_regions(region_names)

            prepared = sliced.prepare_reads(BaselineDeconvolver._sort_reads(reads))
            if prepared is None or prepared.empty:
                return
            built = sliced.build_input(prepared)
            if built:
                self.parts.append(built)
                self.n_reads += len(prepared)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.failed = True
            self.logger.error(
                "Baseline '%s' failed while streaming a chunk: %s",
                self.name,
                e,
                exc_info=True,
            )

    def finalize(self, labels_dict_reversed: Dict[str, int], n_labels: int):
        """Merge the chunk inputs and run the model once."""
        if self.failed:
            return None
        if not self.parts:
            self.logger.warning("%s: no overlapping reads, skipping.", self.name)
            return None
        try:
            merged = self.deconvolver.merge_inputs(self.parts)
            self.parts = []
            if merged is None:
                self.logger.warning("%s: no overlapping reads, skipping.", self.name)
                return None
            return self.deconvolver.deconvolute_from_input(
                merged, labels_dict_reversed, n_labels=n_labels
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Baseline method '%s' failed: %s", self.name, e, exc_info=True
            )
            return None


class _StreamingParquetWriter:
    """Append read-level prediction chunks to a single parquet file.

    The schema is fixed by the first chunk and later chunks are aligned to it.
    Alignment is not cosmetic: ``merge_paired_reads`` emits extra columns
    (``read_total_cpgs``, ``mate1_start``, ...) only for fragments it actually
    merged, so the column set genuinely varies from chunk to chunk.  Missing
    columns are filled with nulls and unexpected ones are dropped, each
    reported once, rather than failing the run over a diagnostic output.
    """

    def __init__(self, path: str, logger: logging.Logger):
        self.path = path
        self.logger = logger
        self._writer = None
        self._schema = None
        self._columns: Optional[List[str]] = None
        self._reported_added: set = set()
        self._reported_dropped: set = set()
        self.n_rows = 0

    def write(self, df: pd.DataFrame) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if df is None or len(df) == 0:
            return
        if self._writer is None:
            table = pa.Table.from_pandas(df, preserve_index=False)
            self._schema = table.schema
            self._columns = list(df.columns)
            self._writer = pq.ParquetWriter(self.path, self._schema)
        else:
            table = pa.Table.from_pandas(
                self._align(df), schema=self._schema, preserve_index=False
            )
        self._writer.write_table(table)
        self.n_rows += len(df)

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reshape a chunk to the columns the file was opened with."""
        missing = [col for col in self._columns if col not in df.columns]
        extra = [col for col in df.columns if col not in self._columns]

        new_missing = set(missing) - self._reported_added
        if new_missing:
            self._reported_added |= new_missing
            self.logger.warning(
                "%s: columns absent from a chunk, written as null: %s",
                os.path.basename(self.path),
                ", ".join(sorted(new_missing)),
            )
        new_extra = set(extra) - self._reported_dropped
        if new_extra:
            self._reported_dropped |= new_extra
            self.logger.warning(
                "%s: columns not present in the first chunk, dropped: %s",
                os.path.basename(self.path),
                ", ".join(sorted(new_extra)),
            )

        if not missing and not extra:
            return df
        return df.reindex(columns=self._columns)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self.logger.info(
                "Saved %d read-level predictions to %s", self.n_rows, self.path
            )
            self._writer = None


class InferencePipeline:
    """
    Orchestrates the full inference pipeline.

    Stages:
        1. BAM → processed reads (or load pre-processed)
        2. Reads x atlas → region-overlapped, annotated reads
        3. Annotated reads → classifier predictions
           (MethylBERT / Dismir / CancerDetector / LookupClassifier)
        4. Read-level predictions → DMR-level aggregation
        5. DMR-level predictions → deconvolution proportions

    Parameters
    ----------
    config : dict
        Parsed inference configuration (from inference.yaml).
    logger : logging.Logger
        Logger instance for status messages.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

        # ── Load labels dictionary ──────────────────────────────────────
        labels_dict_path = config["labels_dict_path"]
        with open(labels_dict_path, "r", encoding="utf-8") as f:
            # JSON keys are strings; convert to {int: str}
            raw = json.load(f)
            self.labels_dict: Dict[int, str] = {int(k): v for k, v in raw.items()}
        self.labels_dict_reversed: Dict[str, int] = {
            v: k for k, v in self.labels_dict.items()
        }
        self.logger.info(
            f"Loaded labels dictionary with {len(self.labels_dict)} cell types"
        )

        self.cell_type_match_dict = LOYFER_CELL_TYPE_MATCH_DICT

        # ── Resolve syto deconvolution config ───────────────────────────
        # The syto pipeline (read-level classification + feature aggregation)
        # is driven by ``deconvolution.syto``, a dict of the shape
        # ``{atlas_path, atlas_name, methods: [...]}``. It is enabled only when
        # at least one method is listed; its atlas feeds stages 2-3.
        deconv_cfg = self.config.get("deconvolution", {})
        self.syto_cfg = deconv_cfg.get("syto", {}) or {}
        self.syto_methods_enabled = bool(self.syto_cfg.get("methods"))

        # ── Load the syto atlas (only when syto methods are enabled) ─────
        self.atlas = None
        if self.syto_methods_enabled:
            atlas_path = self.syto_cfg["atlas_path"]
            atlas_name = self.syto_cfg.get("atlas_name", Path(atlas_path).stem)
            self.atlas = UXMMethylationAtlas(
                atlas_name=atlas_name,
                reference_genome="hg38" if "hg38" in atlas_path else "hg19",
                atlas_path=atlas_path,
                sep="\t",
            )
            self.logger.info(
                f"Loaded syto atlas with {len(self.atlas.atlas)} regions from {atlas_path}"
            )

        # ── Load baseline deconvolvers ───────────────────────────────
        self._baseline_deconvolvers = self._load_baseline_deconvolvers()

        # ── Placeholder attributes populated during run() ───────────────
        self.processed_reads: Optional[pd.DataFrame] = None
        self.prepared_reads: Optional[pd.DataFrame] = None
        self._read_classifier = None
        self.predictions_df: Optional[pd.DataFrame] = None
        self.dmr_aggregated: Optional[pd.DataFrame] = None
        self.deconvolution_results: Dict[str, Any] = {}

        # By default the algorithm assumes that we have at least some data for each DMR group.
        self.fill_in_missing_labels = self.config.get("fill_in_missing_labels", False)

        # ── Missing-label substitution strategy ──────────────────────────
        self.missing_label_strategy = self.config.get(
            "missing_label_strategy", "zeroes"
        )
        self.prior_weight = float(self.config.get("prior_weight", 1.0))
        self.uniform_prior: Optional[pd.DataFrame] = None

        if self.missing_label_strategy != "zeroes":
            self.uniform_prior = self._load_or_compute_uniform_prior()

        # Resolve classifier type early so it's available even when
        # classification is skipped (e.g. predicted_reads input).
        classifier_cfg = config.get("classifier", config.get("model", {}))
        self.classifier_type = classifier_cfg.get("classifier_type", "methylbert")

        if self.config.get("num_labels", None) is None:
            self.num_labels = len(self.labels_dict)
        else:
            self.num_labels = self.config["num_labels"]

        # ── Chunked syto processing ─────────────────────────────────────
        # When enabled, stages 2-4 run one slice of the atlas at a time and the
        # DMR matrix is accumulated incrementally, so only the predictions of
        # the current slice are held in memory.
        chunking_cfg = self.config.get("chunked_inference", {}) or {}
        self.chunked_inference = bool(chunking_cfg.get("enabled", False))
        self.chunk_by = chunking_cfg.get("chunk_by", "region")
        if self.chunk_by not in ("region", "grg"):
            raise ValueError(
                f"chunked_inference.chunk_by must be 'region' or 'grg', "
                f"got {self.chunk_by!r}"
            )
        self.regions_per_chunk = int(chunking_cfg.get("regions_per_chunk", 25))
        if self.regions_per_chunk < 1:
            raise ValueError("chunked_inference.regions_per_chunk must be >= 1")
        self.chunk_progress_bar = bool(chunking_cfg.get("progress_bar", True))

        # Stage 1 streaming: parse each chunk's reads straight from the BAM
        # instead of parsing the whole file up front.  Only meaningful for BAM
        # input, and only when the atlas drives the chunks.
        self.stream_bam = (
            self.chunked_inference
            and self.config.get("input", {}).get("type") == "bam"
            and bool(chunking_cfg.get("stream_bam", True))
        )
        if self.stream_bam:
            self._validate_bam_streaming()

        # The feature mask is only needed by the syto feature-based methods.
        if self.syto_methods_enabled:
            self.features_mask = np.load(config["features_mask_path"])["features_mask"]
            self.input_length = int(np.sum(self.features_mask))

    # ═══════════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════════

    def run(self) -> List[Tuple[str, str, np.ndarray]]:
        """Execute the full inference pipeline end-to-end."""
        # pylint: disable=attribute-defined-outside-init
        self.skip_classification = False
        self.skip_reads_processing = False
        self.skip_aggregation_for_syto = False

        if not self.syto_methods_enabled:
            self.logger.info(
                "The config doesn't feature syto methods --> related classification "
                "and feature extraction stages will be skipped"
            )
            self.skip_classification = True
            self.skip_aggregation_for_syto = True

        # ── Stage 1: obtain processed reads ─────────────────────────────
        # A single ``data_path`` is resolved according to ``input.type``.
        input_cfg = self.config["input"]
        self.file_name = Path(input_cfg["data_path"]).name
        if input_cfg["type"] == "bam":
            if self.stream_bam:
                # Stage 1 is folded into the chunk loop below: each slice of the
                # atlas fetches its own reads, so the full read table is never
                # built.  ``processed_reads`` stays None by design.
                self.logger.info(
                    "Stage 1 streams from the BAM: reads are parsed per atlas chunk"
                )
            else:
                self.processed_reads = self._process_bam()
        elif input_cfg["type"] == "parsed_reads":
            self.processed_reads = self._load_parsed_reads()
        elif input_cfg["type"] == "predicted_reads":
            self.predictions_df = self._load_reads_with_predictions()
            self.prepared_reads = self.predictions_df  # For UXM to work
            self.logger.info(
                f"Classified reads were provided: {len(self.predictions_df)} processed reads. Proceeding with deconvolution."
            )
            self.skip_classification = True
            self.skip_reads_processing = True
        else:
            raise ValueError(
                f"Unknown input type: {input_cfg['type']}. "
                "Must be 'bam' or 'parsed_reads' or 'predicted_reads'."
            )
        if self.stream_bam:
            # ── Stages 1-4, one atlas slice at a time ───────────────────
            self.dmr_aggregated = self._run_syto_stages_chunked()
            self.logger.info(
                f"Stages 1-4 complete (chunked, streamed from BAM): "
                f"{len(self.dmr_aggregated)} DMR-level aggregations"
            )
        elif not self.processed_reads is None:
            if not self.skip_reads_processing:
                self.logger.info(
                    f"Stage 1 complete: {len(self.processed_reads)} processed reads"
                )

                # Stages 2-3 only serve syto classification. When classification
                # is skipped (no syto methods, or predicted reads supplied), the
                # syto atlas is not loaded and these stages are bypassed; baseline
                # deconvolvers prepare the raw processed reads themselves.
                if not self.skip_classification and self.chunked_inference:
                    # ── Stages 2-4, one atlas slice at a time ───────────────
                    self.dmr_aggregated = self._run_syto_stages_chunked()
                    self.logger.info(
                        f"Stages 2-4 complete (chunked): "
                        f"{len(self.dmr_aggregated)} DMR-level aggregations"
                    )
                elif not self.skip_classification:
                    # ── Stage 2: overlap reads with atlas regions ───────────────
                    self.prepared_reads = self._prepare_reads()
                    self.logger.info(
                        f"Stage 2 complete: {len(self.prepared_reads)} atlas-overlapped reads"
                    )

                    # ── Stage 3: classifier predictions ─────────────────────────
                    self.predictions_df = self._predict_classifier()
                    self.logger.info(
                        f"Stage 3 complete: predictions for {len(self.predictions_df)} reads"
                    )

        # ── Stage 4: aggregate to DMR level ────────────────────────────
        # (already accumulated chunk by chunk when chunked inference is on)
        if not self.skip_aggregation_for_syto and self.dmr_aggregated is None:
            self.dmr_aggregated = self._aggregate_to_dmr()
            self.logger.info(
                f"Stage 4 complete: {len(self.dmr_aggregated)} DMR-level aggregations"
            )
        elif self.skip_aggregation_for_syto:
            self.logger.info(
                f"Stage 4 is skipped as no prediction aggregation is required for baseline methods"
            )
        # ── Stage 5: deconvolution ─────────────────────────────────────
        self.deconvolution_results = self._run_deconvolution()
        self.logger.info(
            f"Stage 5 complete: ran {len(self.deconvolution_results)} deconvolution methods"
        )
        # ── Save results ────────────────────────────────────────────────
        self._save_results()
        return self.deconvolution_results

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 1: BAM processing / loading pre-processed reads
    # ═══════════════════════════════════════════════════════════════════

    def _bam_parsing_params(self) -> Dict[str, Any]:
        """Resolve every BAM-parsing knob shared by whole-file and chunked reads."""
        input_cfg = self.config["input"]
        bam_cfg = self.config.get("bam_processing", {})

        data_type = input_cfg.get("data_type")
        if data_type == "wgbs":
            require_flags = bam_cfg.get("require_flags", 3)
        elif data_type == "ont":
            self.logger.info(
                f"ont is selected as data_type. The exclude_flags will not be used and require_flags is ovveriden to 0."
            )
            require_flags = 0
        else:
            raise ValueError(
                "The pipeline only supports BAMS originating from WGBS or ONT"
            )

        chromosomes = input_cfg.get("chromosomes", "all")
        if chromosomes == "all":
            chromosomes = [f"chr{i}" for i in range(1, 23)]
            chromosomes += ["chrX"]
            chromosomes += ["chrY"]

        return {
            "bam_path": input_cfg["data_path"],
            "reference_path": input_cfg.get("reference_path"),
            "data_type": data_type,
            "chromosomes": chromosomes,
            "methyl_tr": bam_cfg.get("ont_methyl_tr", 122),
            "unmethyl_tr": bam_cfg.get("ont_unmethyl_tr"),
            "n_jobs": bam_cfg.get("n_jobs", 4),
            "min_mapq": bam_cfg.get("min_mapq", 10),
            "require_flags": require_flags,
            "exclude_flags": bam_cfg.get("exclude_flags", 1796),
            "min_cpgs": bam_cfg.get("min_cpgs", 1),
            # ONT records sharing a read name are supplementary alignments, not
            # mates, so merging is refused for ONT however the config reads.
            "merge_pairs": resolve_merge_pairs(
                data_type, bam_cfg.get("merge_pairs", True)
            ),
        }

    def _atlas_fetch_intervals(self) -> List[Tuple[str, int, int]]:
        """Merged 0-based intervals covering every atlas this run will consult.

        Deconvolution only ever looks at reads overlapping an atlas region, so
        there is no reason to parse the rest of the genome.  The syto atlas and
        each enabled baseline atlas are pooled, padded, and merged, which turns
        a whole-BAM walk into a few thousand indexed fetches.

        Returns an empty list when no atlas is loaded, i.e. when there is
        nothing to restrict to.
        """
        frames = []
        if self.atlas is not None:
            frames.append(self.atlas.atlas[["chr", "start", "end"]])
        for deconvolver in self._baseline_deconvolvers:
            frames.append(deconvolver.atlas.atlas[["chr", "start", "end"]])
        if not frames:
            return []

        params = self._bam_parsing_params()
        wanted = set(params["chromosomes"])
        padding = self._mate_fetch_padding(params)

        regions = pd.concat(frames, ignore_index=True)
        regions = regions[regions["chr"].isin(wanted)]
        regions = regions.sort_values(["chr", "start", "end"], kind="stable")

        merged: List[Tuple[str, int, int]] = []
        for chromosome, start, end in regions.itertuples(index=False):
            # Atlas regions are 1-based inclusive; fetch wants 0-based half-open.
            start = max(0, int(start) - 1 - padding)
            end = int(end) + padding
            if merged and merged[-1][0] == chromosome and start <= merged[-1][2]:
                merged[-1] = (chromosome, merged[-1][1], max(merged[-1][2], end))
            else:
                merged.append((chromosome, start, end))
        return merged

    def _mate_fetch_padding(self, params: Dict[str, Any]) -> int:
        """Extra span fetched around each region so mates come along.

        Only relevant when mates are merged (WGBS): a fragment whose second
        mate falls outside every atlas region would otherwise arrive alone and
        stay unmerged, unlike in a whole-genome parse.
        """
        if not params["merge_pairs"]:
            return 0
        return int(
            self.config.get("bam_processing", {}).get("mate_fetch_padding", 1000)
        )

    def _process_bam(self) -> pd.DataFrame:
        """Process a BAM file into a read-level DataFrame."""
        params = self._bam_parsing_params()
        bam_path = params["bam_path"]
        chromosomes = params["chromosomes"]

        restrict = self.config.get("bam_processing", {}).get("restrict_to_atlas", True)
        intervals = self._atlas_fetch_intervals() if restrict else []
        if intervals:
            return self._process_bam_over_regions(params, intervals)

        self.logger.info(f"Processing BAM: {bam_path} ({len(chromosomes)} chromosomes)")

        df = process_bam_with_chunking(
            bam_path=bam_path,
            chromosomes=chromosomes,
            methyl_tr=params["methyl_tr"],
            unmethyl_tr=params["unmethyl_tr"],
            n_jobs=params["n_jobs"],
            reference_path=params["reference_path"],
            data_type=params["data_type"],
            min_mapq=params["min_mapq"],
            require_flags=params["require_flags"],
            exclude_flags=params["exclude_flags"],
            min_cpgs=params["min_cpgs"],
            merge_pairs=params["merge_pairs"],
        )
        if not len(df):
            self.logger.warning(
                "The dataset has 0 reads after applying all samtools filters. "
                "The attempt will be made to reparse .bam without applying flag filters"
            )
            df = process_bam_with_chunking(
                bam_path=bam_path,
                chromosomes=chromosomes,
                methyl_tr=params["methyl_tr"],
                unmethyl_tr=params["unmethyl_tr"],
                n_jobs=params["n_jobs"],
                reference_path=params["reference_path"],
                data_type=params["data_type"],
                min_mapq=params["min_mapq"],
                require_flags=None,
                exclude_flags=None,
                min_cpgs=params["min_cpgs"],
                merge_pairs=params["merge_pairs"],
            )
        if not len(df):
            self.logger.warning(
                "Setting exclude_flags=None and require_flags=None didn't help. "
                "The processing of this file will be terminated."
            )
            return None

        return df

    def _process_bam_over_regions(
        self, params: Dict[str, Any], intervals: List[Tuple[str, int, int]]
    ) -> Optional[pd.DataFrame]:
        """Parse only the reads overlapping the atlases, in one indexed pass."""
        span = sum(end - start for _, start, end in intervals)
        self.logger.info(
            "Processing BAM: %s (%d merged atlas intervals, %.1f Mb; "
            "set bam_processing.restrict_to_atlas: false to parse the whole file)",
            params["bam_path"],
            len(intervals),
            span / 1e6,
        )

        def read(require_flags, exclude_flags):
            with BamRegionReader(
                bam_path=params["bam_path"],
                data_type=params["data_type"],
                reference_path=params["reference_path"],
                methyl_tr=params["methyl_tr"],
                unmethyl_tr=params["unmethyl_tr"],
                min_mapq=params["min_mapq"],
                require_flags=require_flags,
                exclude_flags=exclude_flags,
                min_cpgs=params["min_cpgs"],
                merge_pairs=params["merge_pairs"],
            ) as reader:
                return reader.read_regions(intervals)

        df = read(params["require_flags"], params["exclude_flags"])
        if not len(df):
            self.logger.warning(
                "The dataset has 0 reads after applying all samtools filters. "
                "The attempt will be made to reparse .bam without applying flag filters"
            )
            df = read(None, None)
        if not len(df):
            self.logger.warning(
                "Setting exclude_flags=None and require_flags=None didn't help. "
                "The processing of this file will be terminated."
            )
            return None

        return df

    def _load_parsed_reads(self) -> pd.DataFrame:
        """Load pre-parsed reads from a pickle file."""
        path = self.config["input"]["data_path"]
        self.logger.info(f"Loading pre-parsed reads from {path}")
        if ".csv" in path or path.endswith(".parquet"):
            # The published read tables carry the same columns in either
            # format; only the container differs.
            if path.endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, sep="\t")
            df.rename(columns={"ref_name": "chromosome"}, inplace=True)
            df.rename(columns={"ref_pos": "read_start"}, inplace=True)
            df.rename(columns={"methyl_seq": "methylation_encoding"}, inplace=True)
            df.rename(columns={"original_seq": "seq"}, inplace=True)
            df["read_end"] = df["read_start"] + df["seq"].apply(len)
            df["read_name"] = range(len(df))
            # TODO: temporary set here to avoid eval loop crashing the predict.
            # MUST FIX IN THE FUTURE IN THE EVAL LOOP!!!
            df["label"] = 0
        else:
            with open(path, "rb") as f:
                df = pickle.load(f)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"Expected a pandas DataFrame in {path}, got {type(df)}"
                )
        return df

    def _load_reads_with_predictions(self) -> pd.DataFrame:
        """Load reads augmented with read-level predictions from a pickle file."""
        path = self.config["input"]["data_path"]
        self.logger.info(f"Loading reads with predictions from {path}")
        with open(path, "rb") as f:
            df = pickle.load(f)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame in {path}, got {type(df)}")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 2: overlap reads with atlas regions
    # ═══════════════════════════════════════════════════════════════════

    def _prepare_reads(self) -> pd.DataFrame:
        """Overlap processed reads with atlas regions and resolve DMR labels.

        Calls ``atlas.prepare_reads`` which:
        - overlaps reads with atlas regions (adds ``name``, region coords)
        - trims reads to region boundaries (coordinates, seq, pattern)
        - resolves ``dmr_ctype_label`` via labels_dict and cell_type_match_dict
        """
        self.logger.info("Overlapping reads with atlas regions ...")
        prepared = self._overlap_reads_with_atlas(self.processed_reads, self.atlas)

        if len(prepared) == 0:
            raise RuntimeError(
                "No reads overlapped with atlas regions. "
                "Check that chromosome naming is consistent between BAM and atlas."
            )

        return prepared

    def _overlap_reads_with_atlas(self, reads: pd.DataFrame, atlas) -> pd.DataFrame:
        """Overlap, trim and annotate *reads* against *atlas* (may be a subset).

        Unlike :meth:`_prepare_reads` this returns an empty frame instead of
        raising when nothing overlaps, which is a normal outcome for an
        individual chunk of atlas regions.
        """
        df = reads.copy()
        df = df.sort_values(["chromosome", "read_start"]).reset_index(drop=True)

        prepared = atlas.prepare_reads(
            df,
            trim=True,
            labels_dict=self.labels_dict,
            cell_type_match_dict=self.cell_type_match_dict,
        )
        if len(prepared) == 0:
            return prepared

        return mark_records_methyl_state(prepared)

    def _build_classifier(self):
        """Instantiate the configured read classifier (loaded once per run)."""
        if getattr(self, "_read_classifier", None) is not None:
            return self._read_classifier

        classifier_cfg = self.config.get("classifier", self.config.get("model", {}))
        self.classifier_type = classifier_cfg.get("classifier_type")
        read_classifier = read_classifier_factory(
            name=self.classifier_type,
            path=self.config["checkpoint_path"],
            labels_dict=self.labels_dict,
            num_labels=self.num_labels,
            seq_length=self.config.get("max_sequence_length", 150),
            foundation_model_path=classifier_cfg.get(
                "foundation_model", "hanyangii/methylbert_hg19_12l"
            ),
            classifier_head_implementation=classifier_cfg.get(
                "classifier_head_implementation", "grg_attention_based"
            ),
            grg_label_column=classifier_cfg.get("grg_label_column", "dmr_ctype_label"),
            dismir_flavor=classifier_cfg.get("dismir_flavor", "lstm"),
            cancer_detector_prior_type=classifier_cfg.get(
                "cancer_detector_prior_type", "uniform"
            ),
            soft_labels=classifier_cfg.get("soft_labels", False),
            batch_size=self.config.get("prediction_batch_size", 2200),
        )
        self._read_classifier = read_classifier
        return read_classifier

    def _predict_classifier(self, reads: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run the configured classifier on prepared reads.

        Parameters
        ----------
        reads : pd.DataFrame, optional
            Prepared reads to classify.  Defaults to ``self.prepared_reads``.

        Returns
        -------
        pd.DataFrame
            The prepared reads augmented with ``prediction_*`` columns
            (one per cell type).
        """
        classifier_cfg = self.config.get("classifier", self.config.get("model", {}))
        read_classifier = self._build_classifier()
        if reads is None:
            self.logger.info("Running classifier predictions ...")
            reads = self.prepared_reads

        result_df = read_classifier.predict_split(reads, **classifier_cfg)

        prediction_cols = [
            c
            for c in result_df.columns
            if c.startswith("prediction_") and c[len("prediction_") :].isdigit()
        ]
        result_df = result_df.dropna(subset=prediction_cols)

        if "M_rate" in result_df.columns:
            result_df.rename(columns={"M_rate": "methylation_level"}, inplace=True)

        return result_df

    # ═══════════════════════════════════════════════════════════════════
    #  Stages 2-4, chunked: one atlas slice at a time
    # ═══════════════════════════════════════════════════════════════════

    def _run_syto_stages_chunked(self) -> pd.DataFrame:
        """Overlap, classify and aggregate one slice of the atlas at a time.

        Predicting every read up front and aggregating afterwards needs the
        whole read x region prediction table (one float column per cell type)
        in memory at once, which is what runs large samples out of memory.
        Here each slice of atlas regions is carried through stages 2-4 on its
        own and only its running sums survive into the next slice, so peak
        memory follows the busiest slice rather than the whole sample.

        Returns the same DMR-aggregated matrix as the unchunked path.
        """
        baseline_accumulators = self._build_baseline_accumulators()
        chunks = self._plan_chunks(baseline_accumulators)
        self.logger.info(
            "Chunked inference: %d chunks (chunk_by=%s%s, source=%s%s)",
            len(chunks),
            self.chunk_by,
            (
                f", regions_per_chunk={self.regions_per_chunk}"
                if self.chunk_by == "region"
                else ""
            ),
            "bam" if self.stream_bam else "processed reads",
            (
                f", streaming {len(baseline_accumulators)} baseline(s)"
                if baseline_accumulators
                else ""
            ),
        )

        aggregator, n_prepared = self._accumulate_chunks(chunks, baseline_accumulators)

        if aggregator.n_reads_seen == 0 and self.stream_bam:
            # Mirrors the whole-file path: an empty result is usually the flag
            # filters, so retry once with them off before giving up.
            params = self._bam_parsing_params()
            if params["require_flags"] or params["exclude_flags"]:
                self.logger.warning(
                    "0 reads after applying all samtools filters. "
                    "Retrying the chunked pass without flag filters"
                )
                baseline_accumulators = self._build_baseline_accumulators()
                aggregator, n_prepared = self._accumulate_chunks(
                    chunks, baseline_accumulators, relax_flag_filters=True
                )

        if aggregator.n_reads_seen == 0:
            raise RuntimeError(
                "No reads overlapped with atlas regions. "
                "Check that chromosome naming is consistent between BAM and atlas."
            )

        self._finalize_streamed_baselines(baseline_accumulators)

        self.logger.info(
            "Chunked stages 2-3: %d atlas-overlapped reads, %d classified",
            n_prepared,
            aggregator.n_reads_seen,
        )
        return aggregator.finalize(
            fill_in_missing_labels=self.fill_in_missing_labels,
            labels_dict=self.labels_dict,
            substitution_strategy=self.missing_label_strategy,
            uniform_prior=self.uniform_prior,
            prior_weight=self.prior_weight,
        )

    def _validate_bam_streaming(self) -> None:
        """Reject configurations that streaming stage 1 cannot serve.

        Streaming never builds the full read table.  The read-based baselines
        are fed from the same pass (their inputs are per-region and merge
        exactly), but anything that needs every read as one frame - the
        processed-reads dump - has nothing to work from.
        """
        if not self.syto_methods_enabled:
            raise ValueError(
                "chunked_inference.stream_bam needs syto methods to drive the "
                "chunks (deconvolution.syto.methods is empty). Set "
                "chunked_inference.stream_bam: false."
            )

        if self._baseline_deconvolvers and self.chunk_by != "region":
            raise ValueError(
                "chunked_inference.chunk_by='grg' cannot stream the read-based "
                "baselines: their chunks follow their own atlases, so the pass is "
                "planned over the union of all atlases in genomic order. Use "
                "chunk_by: region, or disable the baselines."
            )

        if self.config.get("output", {}).get("save_processed_reads", True):
            self.logger.warning(
                "output.save_processed_reads is ignored when "
                "chunked_inference.stream_bam is on: reads are parsed per chunk "
                "and never held as one table. Read-level predictions are still "
                "written (predictions.parquet)."
            )

    def _accumulate_chunks(
        self,
        chunks: List[pd.DataFrame],
        baseline_accumulators: Optional[List[Any]] = None,
        relax_flag_filters: bool = False,
    ) -> Tuple[StreamingGrgAggregator, int]:
        """Run every chunk through stages (1-)2-4 and return the accumulator.

        When *baseline_accumulators* are given, the same chunk of reads also
        feeds each read-based baseline, so one pass over the BAM serves every
        consumer.
        """
        baseline_accumulators = baseline_accumulators or []
        aggregator = StreamingGrgAggregator(
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            weight_col="NCPGS",
            create_weight_from_cpgs=False,
        )
        writer = self._open_chunked_prediction_writer()
        reader = (
            self._open_bam_region_reader(relax_flag_filters=relax_flag_filters)
            if self.stream_bam
            else None
        )
        read_index = (
            None if self.stream_bam else self._build_read_index(self.processed_reads)
        )

        n_prepared = 0
        progress = self._progress_bar(chunks)
        try:
            for regions in progress:
                if reader is not None:
                    reads = reader.read_regions(self._fetch_intervals(regions))
                else:
                    reads = self._reads_for_regions(regions, read_index)
                if reads is None or len(reads) == 0:
                    continue

                # Read-based baselines see the same reads, each restricted to
                # the regions of its own atlas that fall in this chunk.
                for index, accumulator in enumerate(baseline_accumulators):
                    owned = regions.loc[regions["consumer"] == index, "name"]
                    if len(owned):
                        accumulator.update(reads, owned)

                syto_regions = (
                    regions.loc[regions["consumer"] == -1, "name"]
                    if "consumer" in regions.columns
                    else regions["name"]
                )
                if len(syto_regions):
                    prepared = self._overlap_reads_with_atlas(
                        reads, self.atlas.subset_regions(syto_regions)
                    )
                    if len(prepared):
                        n_prepared += len(prepared)
                        predictions = self._predict_classifier(prepared)
                        aggregator.update(predictions)
                        if writer is not None:
                            try:
                                writer.write(predictions)
                            except (
                                Exception
                            ) as e:  # pylint: disable=broad-exception-caught
                                # predictions.parquet is a diagnostic output;
                                # losing it must not cost the whole run.
                                self.logger.error(
                                    "Failed to write read-level predictions, "
                                    "continuing without them: %s",
                                    e,
                                    exc_info=True,
                                )
                                writer.close()
                                writer = None
                        del prepared, predictions

                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(
                        reads=n_prepared, predicted=aggregator.n_reads_seen
                    )
                # Drop the slice before the next one is built.
                del reads
        finally:
            if writer is not None:
                writer.close()
            if reader is not None:
                reader.close()
            if hasattr(progress, "close"):
                progress.close()

        return aggregator, n_prepared

    def _build_baseline_accumulators(self) -> List[Any]:
        """One accumulator per enabled baseline, for streamed runs only.

        Without streaming the baselines keep working from the full read table in
        stage 5, so there is nothing to accumulate.
        """
        if not self.stream_bam or not self._baseline_deconvolvers:
            return []
        return [
            _BaselineStreamAccumulator(deconvolver, self.logger)
            for deconvolver in self._baseline_deconvolvers
        ]

    def _finalize_streamed_baselines(self, accumulators: List[Any]) -> None:
        """Solve each streamed baseline and stash its result for stage 5."""
        self._streamed_baseline_results = []
        for accumulator in accumulators:
            self.logger.info(
                "Running baseline deconvolution: %s (streamed, %d reads)",
                accumulator.name,
                accumulator.n_reads,
            )
            result = accumulator.finalize(self.labels_dict_reversed, self.num_labels)
            if result is None:
                continue
            self._streamed_baseline_results.extend(
                self._format_baseline_result(accumulator.deconvolver, result)
            )

    def _open_bam_region_reader(self, relax_flag_filters: bool = False):
        """Open the BAM once for the whole chunked run."""
        params = self._bam_parsing_params()
        self.logger.info(
            "Streaming reads from %s (%d chromosomes)",
            params["bam_path"],
            len(params["chromosomes"]),
        )
        return BamRegionReader(
            bam_path=params["bam_path"],
            data_type=params["data_type"],
            reference_path=params["reference_path"],
            methyl_tr=params["methyl_tr"],
            unmethyl_tr=params["unmethyl_tr"],
            min_mapq=params["min_mapq"],
            require_flags=None if relax_flag_filters else params["require_flags"],
            exclude_flags=None if relax_flag_filters else params["exclude_flags"],
            min_cpgs=params["min_cpgs"],
            merge_pairs=params["merge_pairs"],
        )

    def _fetch_intervals(self, regions: pd.DataFrame) -> List[Tuple[str, int, int]]:
        """Convert a chunk's atlas rows into 0-based half-open fetch intervals.

        Regions outside ``input.chromosomes`` are dropped, so restricting the
        run to a few chromosomes skips their reads entirely.
        """
        if getattr(self, "_fetch_chromosomes", None) is None:
            params = self._bam_parsing_params()
            self._fetch_chromosomes = set(params["chromosomes"])
            self._fetch_padding = self._mate_fetch_padding(params)
        # Consumers share loci; fetch each interval once per chunk.
        intervals = regions.loc[
            regions["chr"].isin(self._fetch_chromosomes), ["chr", "start", "end"]
        ].drop_duplicates()
        return [
            (
                chromosome,
                max(0, int(start) - 1 - self._fetch_padding),
                int(end) + self._fetch_padding,
            )
            for chromosome, start, end in intervals.itertuples(index=False)
        ]

    def _plan_chunks(
        self, baseline_accumulators: Optional[List[Any]] = None
    ) -> List[pd.DataFrame]:
        """Split the regions to process into the slices handled one by one.

        Every chunk is a frame of regions carrying a ``consumer`` column: ``-1``
        for the syto atlas, otherwise the index of the baseline accumulator that
        owns the region.  Chunks are planned over the *union* of all consumers'
        atlases in genomic order, so one BAM pass serves every consumer and a
        read spanning regions of different atlases is fetched (and its tags
        decoded) once rather than once per atlas.

        ``regions_per_chunk`` counts *distinct loci*, not rows: consumers
        sharing a region (the usual case, since the baseline atlases are built
        from the same markers) land in the same chunk, so its reads are parsed
        once and handed to all of them.

        ``chunk_by="grg"`` yields one slice per GR group (cell type target),
        i.e. exactly one row of the feature matrix per slice.  ``"region"``
        yields fixed-size batches of individual regions, which keeps the working
        set smaller at the cost of more classifier calls.
        """
        columns = ["chr", "start", "end", "name"]
        syto = self.atlas.atlas.assign(consumer=-1)

        if self.chunk_by == "grg":
            return [group for _, group in syto.groupby("target", sort=True)]

        frames = [syto[columns + ["consumer"]]]
        for index, accumulator in enumerate(baseline_accumulators or []):
            frames.append(accumulator.atlas.atlas[columns].assign(consumer=index))

        union = pd.concat(frames, ignore_index=True).sort_values(
            ["chr", "start", "end"], kind="stable"
        )
        if len(union) == 0:
            return []
        # Number the distinct loci in genomic order, then cut every
        # ``regions_per_chunk`` of them; rows sharing a locus stay together.
        locus = union.groupby(["chr", "start", "end"], sort=False).ngroup()
        return [
            group
            for _, group in union.groupby(locus // self.regions_per_chunk, sort=True)
        ]

    @staticmethod
    def _build_read_index(reads: pd.DataFrame) -> Dict[str, Any]:
        """Index reads by chromosome and start, for fast per-region lookup.

        Reads are not fixed length (ONT reads run to tens of kb), so a start
        position alone cannot bound the search.  The per-chromosome maximum
        read length gives the window that must be scanned to the left of a
        region before filtering on ``read_end``.
        """
        index: Dict[str, Any] = {}
        starts_all = reads["read_start"].to_numpy()
        ends_all = reads["read_end"].to_numpy()
        for chromosome, positions in reads.groupby(
            "chromosome", sort=False
        ).indices.items():
            order = positions[np.argsort(starts_all[positions], kind="stable")]
            starts = starts_all[order]
            ends = ends_all[order]
            index[chromosome] = {
                "positions": order,
                "starts": starts,
                "ends": ends,
                "max_length": int((ends - starts).max()) if len(order) else 0,
            }
        return index

    def _reads_for_regions(
        self, regions: pd.DataFrame, read_index: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Select the reads that can overlap any region in this chunk."""
        selected: List[np.ndarray] = []
        for chromosome, start, end in zip(
            regions["chr"], regions["start"], regions["end"]
        ):
            entry = read_index.get(chromosome)
            if entry is None:
                continue
            starts, ends = entry["starts"], entry["ends"]
            # Atlas regions are 1-based; overlap_reads compares against
            # ``start - 1`` (0-based) and an exclusive ``end``.
            region_start = int(start) - 1
            region_end = int(end)
            first = np.searchsorted(
                starts, region_start - entry["max_length"], side="left"
            )
            last = np.searchsorted(starts, region_end, side="left")
            if last <= first:
                continue
            window = slice(first, last)
            hits = np.flatnonzero(ends[window] >= region_start) + first
            if len(hits):
                selected.append(entry["positions"][hits])

        if not selected:
            return None
        return self.processed_reads.iloc[np.unique(np.concatenate(selected))]

    def _progress_bar(self, chunks: List[pd.DataFrame]):
        """Wrap the chunk list in a tqdm bar when progress display is on."""
        if not self.chunk_progress_bar:
            return chunks
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover - tqdm ships with the project
            self.logger.warning("tqdm is not installed; progress bar disabled")
            return chunks
        return tqdm(chunks, desc="Chunked inference", unit="chunk")

    def _open_chunked_prediction_writer(self):
        """Return a streaming parquet writer for read-level predictions.

        In chunked mode the full prediction table never exists in memory, so
        ``output.save_predictions`` is honoured by appending each chunk to a
        parquet file instead of pickling one big DataFrame at the end.
        """
        output_cfg = self.config.get("output", {})
        if not output_cfg.get("save_predictions", True):
            return None

        output_dir = self.config.get("output_dir", "./inference_output")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "predictions.parquet")
        self.logger.info("Streaming read-level predictions to %s", path)
        return _StreamingParquetWriter(path, self.logger)

    def _load_or_compute_uniform_prior(self) -> pd.DataFrame:
        """Load the uniform prior matrix from the pseudobulk HDF5 file.

        Reads ``outputs/{split}/pure_profiles/uniform_prior`` from the HDF5
        produced by the pseudobulk generation pipeline and returns it as a
        DataFrame with a ``dmr_ctype_label`` column, ready for
        :func:`aggregate_predictions_by_grg`.

        Config keys
        -----------
        pseudobulk_path : str
            Path to the pseudobulk HDF5 file containing pure profiles.
        pure_profiles_split : str, optional
            Split name from which to read the prior (default ``"train"``).

        Raises
        ------
        ValueError
            If ``pseudobulk_path`` is not configured.
        KeyError
            If the HDF5 file has no pure profiles for the requested split.
        """
        from syto.data.pseudobulk_store import open_pseudobulk_store

        pseudobulk_path = self.config.get("pseudobulk_path")
        if not pseudobulk_path:
            raise ValueError(
                f"missing_label_strategy='{self.missing_label_strategy}' requires "
                "'pseudobulk_path' in config pointing to a pseudobulk store "
                "with pre-computed pure profiles."
            )

        split = self.config.get("pure_profiles_split", "train")
        self.logger.info(
            "Loading uniform prior from pseudobulk store (split=%r): %s",
            split,
            pseudobulk_path,
        )
        reader = open_pseudobulk_store(pseudobulk_path, logger=self.logger)
        return reader.read_uniform_prior(split)

    # ═════════════════════════════════════════════════════════════════
    #  Stage 4: aggregate predictions by DMR
    # ═════════════════════════════════════════════════════════════════

    def _aggregate_to_dmr(self) -> pd.DataFrame:
        """
        Aggregate read-level predictions to DMR level using
        ``aggregate_predictions_by_grg`` from edautils.
        """
        self.logger.info("Aggregating predictions by DMR ...")

        aggregated = aggregate_predictions_by_grg(
            df=self.predictions_df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            weight_col="NCPGS",
            create_weight_from_cpgs=False,
            fill_in_missing_labels=self.fill_in_missing_labels,
            labels_dict=self.labels_dict,
            substitution_strategy=self.missing_label_strategy,
            uniform_prior=self.uniform_prior,
            prior_weight=self.prior_weight,
        )

        return aggregated

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 5: deconvolution
    # ═══════════════════════════════════════════════════════════════════

    def _run_deconvolution(self) -> List[Tuple[str, str, np.ndarray]]:
        """
        Execute all enabled deconvolution methods and return results.

        For each deconvolution method, if a ``calibrators_dir`` is specified,
        all calibrator files found in that directory are automatically loaded
        and applied to produce additional calibrated result entries.

        Returns
        -------
        list of (deconvolver, calibrator, proportions)
            Each element is a tuple of (deconvolver name, calibrator name
            or "None", proportions array).
        """
        deconv_cfg = self.config.get("deconvolution", {})
        results: List[Tuple[str, str, np.ndarray]] = []

        # ── Syto feature-based methods ──────────────────────────────────
        for method_cfg in (deconv_cfg.get("syto", {}) or {}).get("methods", []):
            if not method_cfg.get("enabled", False):
                continue

            name = method_cfg["name"]
            self.logger.info(f"Running deconvolution method: {name}")

            try:
                base_name = name
                if name == "ls":
                    base_name = method_cfg["flavor"]

                if name == "xgboost":
                    proportions = self._run_xgboost_deconvolution(method_cfg)
                elif name in ["3Layer_MLP", "Shallow_Wide_Network"]:
                    proportions = self._run_nn_deconvolution(method_cfg)
                elif name == "ls":
                    proportions = self._run_ls_deconvolution(method_cfg)
                else:
                    self.logger.warning(
                        f"Unknown deconvolution method: {name}, skipping"
                    )
                    continue

                results.append((base_name, "None", proportions))

                calibrators_dir = method_cfg.get("calibrators_dir", None)
                if calibrators_dir is None and method_cfg.get(
                    "use_callibration", False
                ):
                    calibrators_dir = str(Path(method_cfg["callibrator_path"]).parent)
                if calibrators_dir is not None:
                    results.extend(
                        self._apply_all_calibrators(
                            proportions, calibrators_dir, base_name
                        )
                    )

            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error(
                    f"Deconvolution method '{name}' failed: {e}", exc_info=True
                )

        # ── Read-based baseline methods ─────────────────────────────────
        streamed = getattr(self, "_streamed_baseline_results", None)
        if streamed is not None:
            # Already deconvoluted chunk by chunk during the streamed pass.
            results.extend(streamed)
        else:
            for deconvolver in self._baseline_deconvolvers:
                results.extend(self._run_baseline(deconvolver))

        return results

    def _apply_all_calibrators(
        self,
        proportions: np.ndarray,
        calibrators_dir: str,
        base_name: str,
    ) -> List[Tuple[str, str, np.ndarray]]:
        """Discover and apply all calibrator files in *calibrators_dir*.

        Parameters
        ----------
        proportions : np.ndarray
            Raw (uncalibrated) deconvolution proportions.  May be 1-D
            (single sample) or 2-D.
        calibrators_dir : str
            Path to the directory that contains calibrator ``.npz`` files
            produced by :class:`CalibratorFittingPipeline`.
        base_name : str
            Deconvolver name, e.g. ``"nnls"``.

        Returns
        -------
        list of (deconvolver, calibrator, proportions)
            Each element is a tuple of (deconvolver name, calibrator
            description, calibrated proportions array).
        """
        results: List[Tuple[str, str, np.ndarray]] = []
        calibrators_dir = Path(calibrators_dir)

        if not calibrators_dir.is_dir():
            self.logger.warning(
                f"Calibrators directory does not exist: {calibrators_dir}"
            )
            return results

        # Ensure 2-D input for the calibrators
        props_2d = proportions
        if props_2d.ndim == 1:
            props_2d = np.expand_dims(props_2d, 0)

        # ── Linear calibrator ──────────────────────────────────────
        linear_path = calibrators_dir / "linear_calibrator.joblib"
        if linear_path.exists():
            self.logger.info(f"  Loading linear calibrator from {linear_path}")
            linear_cal = LinearCalibrator.load(linear_path)

            for norm_method in LINEAR_NORM_METHODS:
                short_name = norm_method.replace("-", "_")
                calibrator_label = f"linear_{short_name}"
                try:
                    calib = linear_cal.predict(props_2d, norm_method=norm_method)
                    results.append((base_name, calibrator_label, np.round(calib, 4)))
                    self.logger.info(f"    ✓ {base_name} + {calibrator_label}")
                except Exception as e:
                    self.logger.error(
                        f"    ✗ {base_name} + {calibrator_label} failed: {e}",
                        exc_info=True,
                    )

        # ── Vector-scaling calibrator ──────────────────────────────
        vs_path = calibrators_dir / "vector_scaling_calibrator_with_cv.joblib"
        if vs_path.exists():
            calibrator_label = "vector_scaling"
            self.logger.info(f"  Loading vector-scaling calibrator from {vs_path}")
            try:
                vs_cal = VectorScalingCalibrator.load(vs_path)
                calib = vs_cal.predict(props_2d)
                results.append((base_name, calibrator_label, np.round(calib, 4)))
                self.logger.info(f"    ✓ {base_name} + {calibrator_label}")
            except Exception as e:
                self.logger.error(
                    f"    ✗ {base_name} + {calibrator_label} failed: {e}",
                    exc_info=True,
                )

        if not results:
            self.logger.warning(f"  No calibrator files found in {calibrators_dir}")

        return results

    def _run_ls_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run LS based deconvolution on the aggregated predictions selected feature matrices.

        Note: calibration is now handled centrally by ``_apply_all_calibrators``
        via the ``calibrators_dir`` config key.  The legacy inline calibration
        block has been removed.
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        flavor = method_cfg["flavor"]
        self.logger.info(f"Loading {flavor} from {checkpoint_path}")
        X = apply_feature_mask(
            np.array(
                self.dmr_aggregated[
                    [f"prediction_{i}_wavg" for i in range(self.num_labels)]
                ]
            ),
            self.features_mask,
        )
        if "nnls" in flavor:
            deconvolver = NNLSDeconvolver.load(checkpoint_path)
            proportions, _, _ = deconvolver.predict_single_sample(X)
        elif "psls" in flavor:
            deconvolver = PSLSDeconvolver.load(checkpoint_path)
            proportions = deconvolver.predict_single_sample(X)
        else:
            raise ValueError(
                "LS family of deconvolvers supports only two flavors: nnls and psls"
            )

        proportions = np.round(proportions, 4)
        self.logger.debug(f"{flavor} proportions: {proportions}")

        return proportions

    def _run_xgboost_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run XGBoost-based deconvolution.

        Loads a pre-trained ``XGBoostDeconvolver`` from checkpoint and
        runs prediction on the GR-aggregated matrix.
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        self.logger.info(f"Loading XGBoostDeconvolver from {checkpoint_path}")

        deconvolver = XGBoostDeconvolver.load(checkpoint_path)

        # Build the prediction matrix from GR-aggregated data
        X = apply_feature_mask(
            np.array(
                self.dmr_aggregated[
                    [f"prediction_{i}_wavg" for i in range(self.num_labels)]
                ]
            ),
            self.features_mask,
        )[np.newaxis]
        deconv_preds = deconvolver.predict(X)
        proportions = np.round(deconv_preds, 4)
        self.logger.debug(f"XGBoost proportions: {proportions}")

        return proportions

    def _run_nn_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run torch-nn-based deconvolution.

        Loads a pre-trained model of specified architectur from checkpoint and
        runs prediction on the GR-aggregated matrix.
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        metadata_path = method_cfg.get("metadata_path", None)
        architecture = method_cfg["name"]
        self.logger.info(f"Loading {architecture} from {checkpoint_path}")

        if architecture == "Shallow_Wide_Network":
            deconvolver = SWNDeconvolver.load(
                checkpoint_path, **{"metadata_path": metadata_path}
            )
        elif architecture == "3Layer_MLP":
            deconvolver = MLPDeconvolver.load(
                checkpoint_path, **{"metadata_path": metadata_path}
            )
        else:
            raise ValueError(
                "Architecture for NN method should be either Shallow_Wide_Network or 3Layer_MLP"
            )
        X = apply_feature_mask(
            np.array(
                self.dmr_aggregated[
                    [f"prediction_{i}_wavg" for i in range(self.num_labels)]
                ]
            ),
            self.features_mask,
        )[np.newaxis]

        deconv_preds = deconvolver.predict(X)
        proportions = np.round(deconv_preds, 4)
        self.logger.debug(f"{architecture} proportions: {proportions}")

        return proportions

    def _load_baseline_deconvolvers(self) -> List[BaselineDeconvolver]:
        """Instantiate deconvolvers for all baselines listed under deconvolution.baselines."""
        baselines_cfg = self.config.get("deconvolution", {}).get("baselines", [])
        deconvolvers: List[BaselineDeconvolver] = []

        for cfg in baselines_cfg:
            if not cfg.get("enabled", True):
                continue
            model = cfg["model"]

            if model == "uxm":
                from syto.data.atlases.uxm_atlases import (
                    UXMMethylationAtlas as _UXMAtlas,
                )

                atlas_path = cfg["atlas_path"]
                uxm_atlas = _UXMAtlas(
                    atlas_name=cfg.get("atlas_name", Path(atlas_path).stem),
                    reference_genome=cfg.get("reference_genome", "hg38"),
                    atlas_path=atlas_path,
                )
                ignore_cells = cfg.get("ignore_cells", [])
                ref_cells = [c for c in uxm_atlas.ref_cells if c not in ignore_cells]
                self.logger.info(
                    "UXM baseline: %d reference cell types", len(ref_cells)
                )
                deconvolvers.append(UXMDeconvolver(uxm_atlas, ref_cells=ref_cells))

            elif model in ("celfieish", "celfie"):
                from syto.data.atlases.celfieish_atlases import (
                    CpGBetaCountsMethylationAtlas,
                )

                atlas_path = cfg["atlas_path"]
                atlas = CpGBetaCountsMethylationAtlas(
                    atlas_name=cfg.get("atlas_name", Path(atlas_path).stem),
                    reference_genome=cfg.get("reference_genome", "hg38"),
                    atlas_path=atlas_path,
                )
                self.logger.info(
                    "%s baseline: %d reference cell types",
                    model.upper(),
                    len(atlas.ref_cells),
                )
                em_checkpoints = cfg.get("em_checkpoints")
                num_iterations = cfg.get("num_iterations", 50)
                convergence_criteria = cfg.get("convergence_criteria", 0.001)

                if model == "celfieish":
                    deconvolvers.append(
                        CelFiEISHDeconvolver(
                            atlas,
                            num_iterations=num_iterations,
                            convergence_criteria=convergence_criteria,
                            em_checkpoints=em_checkpoints,
                        )
                    )
                else:
                    deconvolvers.append(
                        CelFiEDeconvolver(
                            atlas,
                            num_iterations=num_iterations,
                            convergence_criteria=convergence_criteria,
                            random_restarts=cfg.get("random_restarts", 10),
                            freeze_gamma=cfg.get("freeze_gamma", True),
                            sum_by_region=cfg.get("sum_by_region", True),
                            em_checkpoints=em_checkpoints,
                        )
                    )

            elif model == "epidish":
                from syto.data.atlases.celfieish_atlases import (
                    CpGBetaCountsMethylationAtlas,
                )

                atlas_path = cfg["atlas_path"]
                atlas = CpGBetaCountsMethylationAtlas(
                    atlas_name=cfg.get("atlas_name", Path(atlas_path).stem),
                    reference_genome=cfg.get("reference_genome", "hg38"),
                    atlas_path=atlas_path,
                )
                method = cfg.get("method", "RPC")
                deconvolver = EpiDishDeconvolver(
                    atlas,
                    method=method,
                    maxit=cfg.get("maxit", 50),
                    nu_v=cfg.get("nu_v", (0.25, 0.5, 0.75)),
                    constraint=cfg.get("constraint", "inequality"),
                )
                # Disambiguate the result label by method (RPC->'epidish',
                # CBS->'epidish_cbs', CP->'epidish_cp'); an explicit config
                # ``name`` overrides it.
                deconvolver.name = epidish_result_name(method, cfg.get("name"))
                self.logger.info(
                    "EpiDISH baseline (%s): %d reference cell types -> result '%s'",
                    method,
                    len(atlas.ref_cells),
                    deconvolver.name,
                )
                deconvolvers.append(deconvolver)

            else:
                self.logger.warning(
                    "Unknown baseline model %r in config; skipping.", model
                )

        return deconvolvers

    def _run_baseline(
        self, deconvolver: BaselineDeconvolver
    ) -> List[Tuple[str, str, np.ndarray]]:
        """Run a single baseline deconvolver and return result tuples."""
        model = deconvolver.name
        self.logger.info(f"Running baseline deconvolution: {model}")

        if self.processed_reads is None:
            self.logger.warning(
                "%s: requires processed_reads (not available for predicted_reads input); skipping.",
                model,
            )
            return []

        try:
            result = deconvolver.deconvolute_reads(
                self.processed_reads,
                self.labels_dict_reversed,
                n_labels=self.num_labels,
                prepare=True,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error(f"Baseline method '{model}' failed: {e}", exc_info=True)
            return []

        if result is None:
            self.logger.warning("%s: no overlapping reads, skipping.", model)
            return []

        return self._format_baseline_result(deconvolver, result)

    def _format_baseline_result(
        self, deconvolver: BaselineDeconvolver, result
    ) -> List[Tuple[str, str, np.ndarray]]:
        """Turn a baseline's proportions into result tuples (EM checkpoints included)."""
        model = deconvolver.name
        em_checkpoints = getattr(deconvolver, "em_checkpoints", None)
        if em_checkpoints is not None:
            return [
                (f"{model}_{n_steps}_steps", "None", np.round(np.array(aligned), 4))
                for n_steps, aligned in result
            ]

        self.logger.debug("%s proportions: %s", model, result)
        return [(model, "None", np.round(np.array(result), 4))]

    # ═══════════════════════════════════════════════════════════════════
    #  Output
    # ═══════════════════════════════════════════════════════════════════

    def _save_results(self) -> None:
        """Save all pipeline outputs to the configured output directory."""
        output_cfg = self.config.get("output", {})
        output_dir = self.config.get("output_dir", "./inference_output")
        os.makedirs(output_dir, exist_ok=True)

        # ── Processed reads (pickle) ───────────────────────────────────
        if (
            output_cfg.get("save_processed_reads", True)
            and self.processed_reads is not None
        ):
            path = os.path.join(output_dir, "processed_reads.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.processed_reads, f)
            self.logger.info(f"Saved processed reads to {path}")

        # ── Predictions with MethylBERT scores (pickle) ────────────────
        # Chunked runs stream these to parquet during stages 2-4 instead; see
        # ``_open_chunked_prediction_writer``.
        if output_cfg.get("save_predictions", True) and self.predictions_df is not None:
            path = os.path.join(output_dir, "predictions.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.predictions_df, f)
            self.logger.info(f"Saved predictions to {path}")

        # ── Deconvolution proportions (single long-format CSV) ──────────
        if output_cfg.get("save_deconvolution", True) and self.deconvolution_results:
            classifier_name = getattr(self, "classifier_type", "unknown")
            file_name = getattr(self, "file_name", "unknown")
            cell_types = list(self.labels_dict.values())

            rows = []
            for deconvolver, calibrator, proportions in self.deconvolution_results:
                props_flat = proportions.flatten()
                for ct, prop in zip(cell_types, props_flat):
                    rows.append(
                        {
                            "FileName": file_name,
                            "CellType": ct,
                            "Classifier": classifier_name,
                            "Deconvolver": deconvolver,
                            "Calibrator": calibrator,
                            "PredictedProportion": prop,
                        }
                    )

            results_df = pd.DataFrame(rows)
            path = os.path.join(output_dir, "deconvolution_results.csv")
            results_df.to_csv(path, index=False)
            self.logger.info(
                f"Saved deconvolution results ({len(self.deconvolution_results)} "
                f"method/calibrator combinations) to {path}"
            )

            # ── Optional visualization report (opt-in) ─────────────────
            viz_cfg = dict(output_cfg.get("visualization", {}))
            if viz_cfg.get("enabled", False):
                try:
                    from syto.visualization.inference import (
                        render_deconvolution_report,
                        collect_baseline_labels,
                    )

                    # Attribute every configured baseline (incl. custom result
                    # names like 'epidish_houseman' and EM-checkpoint suffixes)
                    # to the baseline group, unless the user set it explicitly.
                    if "baseline_deconvolvers" not in viz_cfg:
                        viz_cfg["baseline_deconvolvers"] = collect_baseline_labels(
                            self._baseline_deconvolvers
                        )

                    render_deconvolution_report(
                        results_df, output_dir, viz_cfg, logger=self.logger
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    self.logger.error(
                        f"Deconvolution visualization failed: {e}", exc_info=True
                    )

        # ── GR-aggregated predictions (pickle) ────────────────────────
        if self.dmr_aggregated is not None:
            path = os.path.join(output_dir, "dmr_aggregated.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.dmr_aggregated, f)
            self.logger.info(f"Saved DMR aggregated predictions to {path}")
