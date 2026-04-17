"""
MethylBERT Inference Pipeline

End-to-end pipeline for processing BAM files (or pre-processed reads),
running MethylBERT classifier predictions, and performing deconvolution
using multiple methods simultaneously.
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict
import torch.nn as nn
import torch

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from copy import deepcopy

from methyldl.modelling.classifiers.methylbert import (
    MethylBert,
    MethylVocab,
    MethylBertFinetuneDataset,
    methylbert_finetune_collator,
    default_methylbert_config,
)

from methyldl.data import LOYFER_CELL_TYPE_MATCH_DICT
from methyldl.data.sequencing.bam_processing import process_bam_with_chunking
from methyldl.modelling.data_preprocessing_for_inference import (
    prepare_methylbert_list_inference,
)
from methyldl.modelling.prediction_aggregation import (
    aggregate_predictions_by_dmr,
    aggregate_chuncked_predictions_weighted,
)
from methyldl.data.sequencing.genome import generate_kmer_str_with_overlap

from methyldl.deconvolution.uxm import (
    prepare_reads_for_uxm,
    uxm_deconvolution,
    rearange_uxm_deconvolution_results,
    load_atlas,
)
from methyldl.deconvolution.least_squares_deconvolvers import PSLSDeconvolver, NNLSDeconvolver

from methyldl.deconvolution.xgbdeconvolver import (
    XGBoostDeconvolver,
    XGBDeconvolverConfig,
)

from methyldl.deconvolution.linear_calibrator import LinearCalibrator


class InferencePipeline:
    """
    Orchestrates the full MethylBERT inference pipeline.

    Stages:
        1. BAM → processed reads (or load pre-processed)
        2. Reads × atlas → region-overlapped, annotated reads
        3. Annotated reads → MethylBERT classifier predictions
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
        with open(labels_dict_path, "r") as f:
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

        # ── Load atlas ──────────────────────────────────────────────────
        atlas_path = config["atlas_path"]
        self.atlas = pd.read_csv(atlas_path, sep="\t")
        self.logger.info(
            f"Loaded atlas with {len(self.atlas)} regions from {atlas_path}"
        )

        # ── Placeholder attributes populated during run() ───────────────
        self.processed_reads: Optional[pd.DataFrame] = None
        self.prepared_reads: Optional[pd.DataFrame] = None
        self.predictions_df: Optional[pd.DataFrame] = None
        self.dmr_aggregated: Optional[pd.DataFrame] = None
        self.deconvolution_results: Dict[str, Any] = {}
        self.features_mask = np.load(config["features_mask_path"])["features_mask"]

        # By default the algorithm assumes that we have at least some data for each DMR group.
        self.fill_in_missing_labels = self.config.get("fill_in_missing_labels", False)

        if self.config.get("num_labels", None) is None:
            self.num_labels = len(self.labels_dict)
        else:
            self.num_labels = self.config["num_labels"]

        self.input_length = int(np.sum(self.features_mask))

    # ═══════════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════════

    def run(self) -> Dict[str, Any]:
        """Execute the full inference pipeline end-to-end."""

        self.skip_classification = False
        # ── Stage 1: obtain processed reads ─────────────────────────────
        input_cfg = self.config["input"]
        if input_cfg["type"] == "bam":
            self.file_name = input_cfg["bam_path"].split("/")[-1]
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
        else:
            raise ValueError(
                f"Unknown input type: {input_cfg['type']}. "
                "Must be 'bam' or 'parsed_reads' or 'predicted_reads'."
            )
        if not self.processed_reads is None:
            if not self.skip_classification:
                self.logger.info(
                    f"Stage 1 complete: {len(self.processed_reads)} processed reads"
                )

                # ── Stage 2: overlap reads with atlas regions ───────────────────
                self.prepared_reads, self.prepared_reads_chuncked = (
                    self._prepare_reads()
                )
                self.logger.info(
                    f"Stage 2 complete: {len(self.prepared_reads)} atlas-overlapped reads"
                )

                # ── Stage 3: MethylBERT predictions ─────────────────────────────
                self.predictions_df = self._predict_methylbert()
                self.logger.info(
                    f"Stage 3 complete: predictions for {len(self.predictions_df)} reads"
                )

        # ── Stage 4: aggregate to DMR level ────────────────────────────
        self.dmr_aggregated = self._aggregate_to_dmr()
        self.logger.info(
            f"Stage 4 complete: {len(self.dmr_aggregated)} DMR-level aggregations"
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

    def _process_bam(self) -> pd.DataFrame:
        """Process a BAM file into a read-level DataFrame."""
        input_cfg = self.config["input"]
        bam_cfg = self.config.get("bam_processing", {})

        bam_path = input_cfg["bam_path"]
        reference_path = input_cfg.get("reference_path")
        data_type = input_cfg.get("data_type")

        # Resolve chromosomes
        chromosomes = input_cfg.get("chromosomes", "all")
        if chromosomes == "all":
            chromosomes = [f"chr{i}" for i in range(1, 23)]
            chromosomes += ["chrX"]
            chromosomes += ["chrY"]

        self.logger.info(f"Processing BAM: {bam_path} ({len(chromosomes)} chromosomes)")

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

        df = process_bam_with_chunking(
            bam_path=bam_path,
            chromosomes=chromosomes,
            methyl_tr=bam_cfg.get("ont_methyl_tr", 122),
            n_jobs=bam_cfg.get("n_jobs", 4),
            reference_path=reference_path,
            data_type=data_type,
            min_mapq=bam_cfg.get("min_mapq", 10),
            require_flags=require_flags,
            exclude_flags=bam_cfg.get("exclude_flags", 1796),
            min_cpgs=bam_cfg.get("min_cpgs", 1),
            merge_pairs=bam_cfg.get("merge_pairs", True),
        )
        if not len(df):
            self.logger.warning(
                "The dataset has 0 reads after applying all samtools filters. The attempt will be made to reparse .bam without applying flag filters"
            )
            df = process_bam_with_chunking(
                bam_path=bam_path,
                chromosomes=chromosomes,
                methyl_tr=bam_cfg.get("ont_methyl_tr", 122),
                n_jobs=bam_cfg.get("n_jobs", 4),
                reference_path=reference_path,
                data_type=data_type,
                min_mapq=bam_cfg.get("min_mapq", 10),
                require_flags=None,
                exclude_flags=None,
                min_cpgs=bam_cfg.get("min_cpgs", 1),
                merge_pairs=bam_cfg.get("merge_pairs", True),
            )
        if not len(df):
            self.logger.warning(
                "Setting exclude_flags=None and require_flags=None didn't help. The processing of this file will be terminated."
            )
            return None

        return df

    def _load_parsed_reads(self) -> pd.DataFrame:
        """Load pre-parsed reads from a pickle file."""
        path = self.config["input"]["parsed_reads_path"]
        self.logger.info(f"Loading pre-parsed reads from {path}")
        if ".csv" in path:
            df = pd.read_csv(path, sep="\t")
            df.rename(columns={"ref_name":"chromosome"}, inplace=True)
            df.rename(columns={"ref_pos":"read_start"}, inplace=True)
            df.rename(columns={"methyl_seq":"methylation_encoding"}, inplace=True)
            df.rename(columns={"original_seq":"seq"}, inplace=True)
            df["read_end"] = df["read_start"] + df["seq"].apply(len)
            df["read_name"] = range(len(df))
            df["label"] = 0 # TODO: temporary set here to avoid eval loop crashing the predict. MUST FIX IN THE FUTURE IN THE EVAL LOOP!!!
        else:
            with open(path, "rb") as f:
                df = pickle.load(f)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected a pandas DataFrame in {path}, got {type(df)}")
        return df

    def _load_reads_with_predictions(self) -> pd.DataFrame:
        """Load reads augmented with read-level predictions from a pickle file."""
        path = self.config["input"]["predicted_reads_path"]
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
        """
        Overlap processed reads with atlas regions and resolve DMR labels.

        Uses ``prepare_reads_for_uxm`` from edautils, which:
        - iterates atlas regions and scans sorted reads for overlaps
        - trims reads to region boundaries
        - computes M / U / X counts
        - resolves ``dmr_ctype_label`` via labels_dict_reversed and
          cell_type_match_dict
        """
        df = self.processed_reads.copy()

        # Sort by chromosome and start position (required by prepare_reads_for_uxm)
        df = df.sort_values(["chromosome", "read_start"]).reset_index(drop=True)

        # For inference from BAM, reads don't have ground-truth labels.
        # Add dummy columns so prepare_reads_for_uxm doesn't break.
        if "original_label" not in df.columns:
            df["original_label"] = -1
        if "label" not in df.columns:
            df["label"] = -1

        self.logger.info("Overlapping reads with atlas regions ...")
        prepared_uxm = prepare_reads_for_uxm(
            reads_data=df,
            atlas=self.atlas,
            labels_dict=self.labels_dict,
            cell_type_match_dict=self.cell_type_match_dict,
        )

        if len(prepared_uxm) == 0:
            raise RuntimeError(
                "No reads overlapped with atlas regions. "
                "Check that chromosome naming is consistent between BAM and atlas."
            )
        seq_length = self.config.get("max_sequence_length", 150)
        prepared_uxm.rename(
            columns={"seq": "input_ids", "pattern": "methylation_ids"}, inplace=True
        )
        chunked_inputs = prepare_methylbert_list_inference(
            prepared_uxm,
            "dmr_ctype_label",
            seq_length=seq_length,
            stride=int(seq_length / 2),
        )

        return prepared_uxm, chunked_inputs

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 3: MethylBERT predictions
    # ═══════════════════════════════════════════════════════════════════

    def _predict_methylbert(self) -> pd.DataFrame:
        """
        Convert prepared reads to MethylBERT format and run the classifier.

        Returns
        -------
        pd.DataFrame
            The prepared_reads DataFrame augmented with ``prediction_*``
            columns (one per cell type).
        """
        model_cfg = self.config["model"]
        checkpoint_path = self.config["checkpoint_path"]
        seq_len = self.config.get("max_sequence_length", 150)
        batch_size = self.config.get("prediction_batch_size", 2200)

        # ── Build MethylBERT dataset ────────────────────────────────────
        self.logger.info("Converting reads to MethylBERT format ...")

        vocab = MethylVocab(k=3)
        dataset = MethylBertFinetuneDataset(
            data_source=self.prepared_reads_chuncked,
            vocab=vocab,
            seq_len=seq_len,
            n_cores=10,
            lazy_tokenization=True,
        )

        # ── Load MethylBERT model ──────────────────────────────────────
        self.logger.info(f"Loading MethylBERT from checkpoint: {checkpoint_path}")

        num_dmr_labels = dataset.num_dmrs()

        rrms_config = OrderedDict(
            [
                ("lr", 0.0004),
                ("beta", (0.9, 0.98)),
                ("weight_decay", 0.1),
                ("warmup_step", 100),
                ("eps", 1e-6),
                ("with_cuda", True),
                ("log_freq", 200),
                ("eval_freq", 200),
                ("n_hidden", None),
                ("decrease_steps", 200),
                ("eval", False),
                ("amp", True),
                ("gradient_accumulation_steps", 1),
                ("max_grad_norm", 1.0),
                ("save_freq", None),
                ("loss", "ce"),
                ("adam_beta1", 0.9),
                ("adam_beta2", 0.98),
                ("seed", 950410),
            ]
        )

        model_instance = MethylBert(
            foundation_model_path=model_cfg.get(
                "foundation_model", "hanyangii/methylbert_hg19_12l"
            ),
            seq_len=seq_len,
            custom_config=rrms_config.copy(),
            fine_tuned_model_path=checkpoint_path,
            num_labels=self.num_labels,
            num_dmr_labels=num_dmr_labels,
            output_dir=os.path.join(self.config["output"]["output_dir"], "tmp_trainer"),
            classifier_implementation="dmr_attention_based",
            batch_size=batch_size,
        )

        # ── Run predictions ────────────────────────────────────────────
        self.logger.info("Running MethylBERT predictions ...")

        predictions = model_instance.predict(
            dataset,
            batch_size=batch_size,
        )
        predictions_pd = pd.DataFrame(
            predictions[0], columns=["prediction_" + str(x) for x in range(self.num_labels)]
        )
        predictions_pd["read_name"] = [x[-3] for x in self.prepared_reads_chuncked[1:]]
        predictions_pd["ncpgs_marked"] = [
            x[-2] for x in self.prepared_reads_chuncked[1:]
        ]

        predictions_pd = aggregate_chuncked_predictions_weighted(predictions_pd)

        result_df = pd.merge(self.prepared_reads, predictions_pd, on="read_name")
        result_df = result_df.dropna(
            subset=result_df.columns.difference(["soft_label"])
        )
        result_df.rename(columns={"M_rate": "methylation_level"}, inplace=True)

        return result_df

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 4: aggregate predictions by DMR
    # ═══════════════════════════════════════════════════════════════════

    def _aggregate_to_dmr(self) -> pd.DataFrame:
        """
        Aggregate read-level predictions to DMR level using
        ``aggregate_predictions_by_dmr`` from edautils.
        """
        self.logger.info("Aggregating predictions by DMR ...")

        aggregated = aggregate_predictions_by_dmr(
            df=self.predictions_df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            weight_col="NCPGS",
            create_weight_from_cpgs=False,
            fill_in_missing_labels=self.fill_in_missing_labels,
            labels_dict=self.labels_dict,
        )

        return aggregated

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 5: deconvolution
    # ═══════════════════════════════════════════════════════════════════

    def _run_deconvolution(self) -> Dict[str, Any]:
        """
        Execute all enabled deconvolution methods and return results.

        Returns
        -------
        dict
            Mapping from method name to its output (array of proportions
            or DataFrame).
        """
        deconv_cfg = self.config.get("deconvolution", {})
        methods = deconv_cfg.get("methods", [])

        results: Dict[str, Any] = {}

        for method_cfg in methods:
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
                elif name == "uxm":
                    proportions = self._run_uxm_deconvolution(method_cfg)
                elif name in ["3Layer_MLP", "Shallow_Wide_Network"]:
                    proportions = self._run_nn_deconvolution(method_cfg)
                elif name == "ls":
                    proportions = self._run_ls_deconvolution(method_cfg)
                else:
                    self.logger.warning(
                        f"Unknown deconvolution method: {name}, skipping"
                    )
                    continue

                results[base_name] = proportions

                if method_cfg.get("use_callibration", False):
                    calibrator = LinearCalibrator()
                    calibrator.load_calibration_parameters(method_cfg["callibrator_path"])
                    if proportions.ndim == 1:
                        proportions = np.expand_dims(proportions, 0)
                    calib_proportions = calibrator.predict(proportions)
                    calib_proportions = np.round(calib_proportions, 4)
                    results[f"{base_name}_callibrated"] = calib_proportions

            except Exception as e:
                self.logger.error(
                    f"Deconvolution method '{name}' failed: {e}", exc_info=True
                )

        return results
    
    def _run_ls_deconvolution(self, method_cfg: Dict[str,Any]) -> np.ndarray:
        """
        Run LS based deconvolution on the aggregated predictions selected feature matrices
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        flavor = method_cfg["flavor"]
        self.logger.info(f"Loading {flavor} from {checkpoint_path}")
        X = self._extract_features_by_mask(
            np.array(self.dmr_aggregated[[f"prediction_{i}_wavg" for i in range(self.num_labels)]]),
            self.features_mask,
        )
        X = X.flatten()
        if "nnls" in flavor:
            deconvolver = NNLSDeconvolver.load(checkpoint_path)
            proportions,_,_ = deconvolver.predict_single_sample(X)
        elif "psls" in flavor:
            deconvolver = PSLSDeconvolver.load(checkpoint_path)
            proportions = deconvolver.predict_single_sample(X)
        else:
            raise ValueError("LS fabily of deconvolvers supports only two flavors: nnls and psls")
        
        proportions = np.round(proportions,4)
        self.logger.debug(f"{flavor} proportions: {proportions}")

        return proportions

    def _run_xgboost_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run XGBoost-based deconvolution.

        Loads a pre-trained ``XGBoostDeconvolver`` from checkpoint and
        runs prediction on the DMR-aggregated matrix.
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        self.logger.info(f"Loading XGBoostDeconvolver from {checkpoint_path}")

        deconvolver = XGBoostDeconvolver.load(checkpoint_path)

        # Build the prediction matrix from DMR-aggregated data
        # prediction_matrix = self._build_prediction_matrix()
        X = self._extract_features_by_mask(
            np.array(self.dmr_aggregated[[f"prediction_{i}_wavg" for i in range(self.num_labels)]]),
            self.features_mask,
        )
        deconv_preds = deconvolver._predict_raw(X)
        deconv_preds = deconvolver._transform_output(deconv_preds)[0]
        # proportions = np.round(deconvolver.predict(prediction_matrix),4)
        proportions = np.round(deconv_preds, 4)
        self.logger.debug(f"XGBoost proportions: {proportions}")

        return proportions

    def _run_nn_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run torch-nn-based deconvolution.

        Loads a pre-trained model of specified architectur from checkpoint and
        runs prediction on the DMR-aggregated matrix.
        """
        checkpoint_path = method_cfg["checkpoint_path"]
        architecture = method_cfg["name"]
        self.logger.info(f"Loading {architecture} from {checkpoint_path}")

        if architecture == "Shallow_Wide_Network":
            #     deconvolver  = nn.Sequential(
            #     nn.Linear(78, 1024),
            #     nn.GELU(),
            #     nn.Dropout(0.2),
            #     nn.Linear(1024, 39),
            #     nn.Softmax(dim=-1)
            # )
            deconvolver = nn.Sequential(
                nn.Linear(self.input_length, 1024),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 39),
                nn.Softmax(dim=-1),
            )
        elif architecture == "3Layer_MLP":
            # deconvolver = nn.Sequential(
            #     nn.Linear(78, 128),
            #     nn.GELU(),
            #     nn.Dropout(0.2),
            #     nn.Linear(128, 128),
            #     nn.GELU(),
            #     nn.Dropout(0.2),
            #     nn.Linear(128, 64),
            #     nn.GELU(),
            #     nn.Dropout(0.2),
            #     nn.Linear(64, 39),
            #     nn.Softmax(dim=-1)
            # )
            deconvolver = nn.Sequential(
                nn.Linear(self.input_length, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, self.input_length),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.input_length, 39),
                nn.Softmax(dim=-1),
            )
        else:
            raise ValueError(
                "Architecture for NN method should be either Shallow_Wide_Network or 3Layer_MLP"
            )
        device = "cuda"
        deconvolver.to(device)
        deconvolver.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        deconvolver.eval()
        X = torch.FloatTensor(
            self._extract_features_by_mask(
                np.array(
                    self.dmr_aggregated[[f"prediction_{i}_wavg" for i in range(self.num_labels)]]
                ),
                self.features_mask,
            )
        ).to("cuda")
        # X = torch.unsqueeze(X,0)
        # X = torch.concat([torch.diagonal(X[:, :, :39], dim1=1, dim2=2),X[:, :, -1]], dim=1)
        deconv_preds = deconvolver(X)
        proportions = np.round(deconv_preds.to("cpu").detach().numpy(), 4)
        self.logger.debug(f"{architecture} proportions: {proportions}")

        return proportions

    def _run_uxm_deconvolution(self, method_cfg: Dict[str, Any]) -> np.ndarray:
        """
        Run UXM deconvolution.
        """
        # Load atlas for UXM (uses its own loader)
        atlas_path = self.config["atlas_path"]
        uxm_atlas, ref_cells = load_atlas(atlas_path)

        # Build UXM-compatible input from the prepared reads
        uxm_input = self._build_uxm_input(uxm_atlas, ref_cells)

        # Run UXM deconvolution
        uxm_proportions = uxm_deconvolution(
            atlas=uxm_atlas,
            ref_cells=ref_cells,
            sf=uxm_input["scaling_factors"],
            counts=uxm_input["counts"],
            sample_names=["sample"],
        )[0]

        # Align proportions to match labels_dict order
        ref_cells = [x for x in ref_cells if x != "Megakaryocytes"]

        proportions_aligned = rearange_uxm_deconvolution_results(
            self.labels_dict_reversed, uxm_proportions, ref_cells
        )
        proportions_aligned = np.round(np.array(proportions_aligned), 4)

        self.logger.debug(f"UXM proportions: {proportions_aligned}")
        return proportions_aligned

    @staticmethod
    def _extract_diag_and_rej(matrix):
        return np.reshape(np.concat([np.diag(matrix), matrix[:, -1]], axis=0), (1, 78))

    @staticmethod
    def _extract_features_by_mask(matrix, target_mask):
        return np.expand_dims(
            np.ma.masked_array(
                matrix, ~np.array(target_mask, dtype=np.bool)
            ).compressed(),
            0,
        )

    def _build_prediction_matrix(self) -> np.ndarray:
        """
        Build the deconvolution input matrix from DMR-aggregated predictions.

        Returns
        -------
        np.ndarray
            Shape: (1, n_dmrs, n_cell_types + 1) — a single sample with
            per-DMR predictions and a rejection column.
        """
        n_labels = len(self.labels_dict)
        pred_cols = [f"prediction_{i}" for i in range(n_labels)]

        # Pivot aggregated data: rows = DMR groups, cols = cell-type predictions
        agg = self.dmr_aggregated.copy()

        # Sort by dmr_ctype_label so matrix rows are in consistent order
        agg = agg.sort_values("dmr_ctype_label").reset_index(drop=True)

        # Extract prediction matrix
        if all(col in agg.columns for col in pred_cols):
            matrix = agg[pred_cols].values  # (n_dmrs, n_labels)
        else:
            # Auto-detect prediction columns
            available = [c for c in agg.columns if c.startswith("prediction_")]
            matrix = agg[available].values

        return self._extract_diag_and_rej(matrix)

    def _build_uxm_input(
        self, uxm_atlas: pd.DataFrame, ref_cells: list
    ) -> Dict[str, Any]:
        """
        Build UXM-compatible input from prepared reads.

        Computes per-region scaling factors and methylation counts
        in the format expected by ``decon_single_samp``.
        """
        prepared = self.prepared_reads.copy()

        results_agg = (
            prepared[prepared["NCPGS"] > 3]
            .groupby("name")
            .aggregate({"record_M": "sum", "record_U": "sum", "record_X": "sum"})
            .reset_index()
        )
        results_agg["count"] = (
            results_agg["record_M"] + results_agg["record_U"] + results_agg["record_X"]
        )
        results_agg["sf"] = results_agg["record_U"] / results_agg["count"]
        # TODO: Derrive direction from the source
        results_agg["direction"] = "U"
        sample_name = "sample"

        sf = deepcopy(results_agg[["name", "direction"]])
        sf[sample_name] = results_agg["sf"]
        counts = results_agg[["name", "direction", "count"]]
        counts.columns = ["name", "direction", sample_name]

        return {
            "scaling_factors": sf,
            "counts": counts,
        }

    # ═══════════════════════════════════════════════════════════════════
    #  Output
    # ═══════════════════════════════════════════════════════════════════

    def _save_results(self) -> None:
        """Save all pipeline outputs to the configured output directory."""
        output_cfg = self.config.get("output", {})
        output_dir = output_cfg.get("output_dir", "./inference_output")
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
        if output_cfg.get("save_predictions", True) and self.predictions_df is not None:
            path = os.path.join(output_dir, "predictions.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.predictions_df, f)
            self.logger.info(f"Saved predictions to {path}")

        # ── Deconvolution proportions (CSV) ────────────────────────────
        if output_cfg.get("save_deconvolution", True) and self.deconvolution_results:
            for method_name, proportions in self.deconvolution_results.items():
                path = os.path.join(output_dir, f"deconvolution_{method_name}.csv")

                if isinstance(proportions, np.ndarray):
                    prop_df = pd.DataFrame(
                        [list(self.labels_dict.values()), proportions.flatten()]
                    ).T
                    prop_df.columns = ["CellType", method_name]
                elif isinstance(proportions, pd.DataFrame):
                    prop_df = proportions

                prop_df.to_csv(path, index=False)
                self.logger.info(f"Saved {method_name} deconvolution to {path}")

        # ── DMR-aggregated predictions (pickle) ────────────────────────
        if self.dmr_aggregated is not None:
            path = os.path.join(output_dir, "dmr_aggregated.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.dmr_aggregated, f)
            self.logger.info(f"Saved DMR aggregated predictions to {path}")