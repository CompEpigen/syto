"""
Pseudo-Bulk Generation Pipeline

End-to-end pipeline orchestrating:
    1. Loading / rebalancing train/valid/test splits
    2. Read preparation (derived columns, optional UXM alignment)
    3. Classifier prediction (MethylBERT / Dismir / CancerDetector)
    4. Parallel IO-example generation (random or target-proportion mode)
    5. Consolidation of partial pickles into a single .npz
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from methyldl.data import LOYFER_CELL_TYPE_MATCH_DICT
from methyldl.data.pseudo_bulk_generation import (
    consolidate_ios_pickles,
    run_ios_generation_parallel,
)
from methyldl.data.read_preparation import prepare_splits_for_pseudobulk
from methyldl.data.split_rebalancing import rebalance_splits
from methyldl.modelling.classifier_adapter import ClassifierAdapter


class PseudoBulkPipeline:
    """Orchestrate pseudo-bulk mixture generation end-to-end.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    logger : logging.Logger
        Logger instance.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Labels
        labels_dict_path = config["labels_dict_path"]
        with open(labels_dict_path, "r") as f:
            raw = json.load(f)
            self.labels_dict: Dict[int, str] = {int(k): v for k, v in raw.items()}
        self.num_labels = config.get("num_labels", len(self.labels_dict))

        # Cell-type matching dict
        self.cell_type_match_dict = config.get(
            "cell_type_match_dict", LOYFER_CELL_TYPE_MATCH_DICT
        )

        # Output paths
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.ios_dir = os.path.join(self.output_dir, "ios_deconvolution")
        os.makedirs(self.ios_dir, exist_ok=True)

        self.ios_base_path = os.path.join(
            self.ios_dir, "ios_deconvolution_data.pkl"
        )
        self.consolidated_path = os.path.join(
            self.output_dir, "ios_full_matrices.npz"
        )

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def run(self) -> Dict[str, Any]:
        """Execute the full pseudo-bulk generation pipeline.

        Returns
        -------
        dict
            Consolidated output arrays keyed by ``proportions``,
            ``features_train``, ``features_valid``, ``features_test``.
        """
        if self.config["input_type"] == "raw_splits":
            # ── Stage 1: Load splits ──────────────────────────────────
            self.logger.info("Stage 1: Loading data splits ...")
            train, valid, test = self._load_splits()
            self.logger.info(
                f"  train={len(train)}, valid={len(valid)}, test={len(test)} reads"
        )

            # ── Stage 1b: Optional rebalancing ────────────────────────
            if self.config.get("rebalance_splits", False):
                self.logger.info("Stage 1b: Rebalancing splits ...")
                manual_labels = self.config.get("manual_common_labels", [28, 35])
                train, valid, test = rebalance_splits(
                    train,
                    valid,
                    test,
                    manual_common_labels=manual_labels,
                )
                self.logger.info(
                    f"  After rebalance: train={len(train)}, valid={len(valid)}, "
                    f"test={len(test)}"
                )

            # ── Stage 2: Prepare reads ────────────────────────────────
            self.logger.info("Stage 2: Preparing reads ...")
            generate_uxm = self.config.get("generate_uxm_inputs", True)
            atlas_path = self.config.get("atlas_path")

            train, valid, test = prepare_splits_for_pseudobulk(
                train,
                valid,
                test,
                labels_dict=self.labels_dict,
                num_labels=self.num_labels,
                generate_uxm_inputs=generate_uxm,
                atlas_path=atlas_path,
                cell_type_match_dict=self.cell_type_match_dict,
            )
            self.logger.info(
                f"  After preparation: train={len(train)}, valid={len(valid)}, "
                f"test={len(test)}"
            )
            with open(os.path.join(self.output_dir, "uxm_prepared_reads.pkl"), "wb") as f:
                pickle.dump((train, valid, test), f)
        elif self.config["input_type"] == "uxm_prepared":
            self.logger.info("Skipping Stage 1 and 2: Loading uxm prepared reads ...")
            with open(os.path.join(self.output_dir, "uxm_prepared_reads.pkl"), "rb") as f:
                train, valid, test = pickle.load(f)
            self.logger.info(
                f"  After loading: train={len(train)}, valid={len(valid)}, "
                f"test={len(test)}"
            )
        elif self.config["input_type"] == "pre_predicted":
            self.logger.info("Skipping Stage 1,2,3: Loading reads with predictions ..")
            train, valid, test = self._load_splits()
            self.logger.info(
                f"  train={len(train)}, valid={len(valid)}, test={len(test)} reads"
        )

        # ── Stage 3: Classifier predictions (if needed) ───────────
        if self.config["input_type"] in ["raw_splits", "uxm_prepared"]:
            self.logger.info("Stage 3: Running classifier predictions ...")
            adapter = self._build_classifier_adapter()

            for name, df in [("train", train), ("valid", valid), ("test", test)]:
                self.logger.info(f"  Predicting {name} split ({len(df)} reads) ...")
                predicted = adapter.predict_split(df)

                # Save intermediate predicted split
                intermediate_path = os.path.join(
                    self.output_dir, f"{name}_predicted.pkl"
                )
                with open(intermediate_path, "wb") as f:
                    pickle.dump(predicted, f)
                self.logger.info(f"  Saved predicted {name} to {intermediate_path}")

                if name == "train":
                    train = predicted
                elif name == "valid":
                    valid = predicted
                else:
                    test = predicted

            with open(os.path.join(self.output_dir, "predicted_reads.pkl"), "wb") as f:
                pickle.dump((train, valid, test), f)
        else:
            self.logger.info(
                "Stage 3: Skipped (input already has predictions)"
            )

        # ── Stage 4: Generate IO examples ─────────────────────────
        self.logger.info("Stage 4: Generating pseudo-bulk IO examples ...")
        generation_mode = self.config["generation_mode"]
        generate_uxm_in_ios = self.config.get("generate_uxm_inputs_in_ios", True)

        shared_kwargs = dict(
            train_data=train,
            valid_data=valid,
            test_data=test,
            file_name=self.ios_base_path,
            n_workers=self.config.get("n_workers"),
            batch_size=self.config.get("batch_size", 100),
            checkpoint_interval=self.config.get("checkpoint_interval", 1000),
            start_checkpoint_idx=self.config.get("start_checkpoint_idx", 0),
            n_read_per_split=self.config.get("n_read_per_split"),
            num_labels=self.num_labels,
            generate_uxm_inputs=generate_uxm_in_ios,
        )

        if generation_mode == "random":
            allowed_labels = self.config.get("allowed_labels")
            if allowed_labels is None:
                allowed_labels = list(range(self.num_labels))

            ios, exceptions = run_ios_generation_parallel(
                n_io_examples=self.config.get("n_io_examples", 30000),
                allowed_labels=allowed_labels,
                n_cells_max=self.config.get("n_cells_max", 10),
                **shared_kwargs,
            )
        elif generation_mode == "target_proportions":
            target_proportions_path = self.config["target_proportions_path"]
            target_proportions = np.load(target_proportions_path)
            target_proportions = target_proportions["proportions"]
            ios, exceptions = run_ios_generation_parallel(
                target_proportions=target_proportions,
                **shared_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown generation_mode: '{generation_mode}'. "
                "Must be 'random' or 'target_proportions'."
            )

        if exceptions:
            self.logger.warning(
                f"  {len(exceptions)} example(s) failed during generation."
            )
            for lbl, prop, err in exceptions[:5]:
                self.logger.warning(f"    labels={lbl}, proportions={prop}: {err}")

        # ── Stage 5: Consolidate ──────────────────────────────────
        self.logger.info("Stage 5: Consolidating partial pickles ...")
        result = consolidate_ios_pickles(
            ios_dir=self.ios_dir,
            output_path=self.consolidated_path,
            num_labels=self.num_labels,
        )
        self.logger.info(
            f"  Consolidated {result['proportions'].shape[0]} examples to "
            f"{self.consolidated_path}"
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    #  Internal helpers
    # ═══════════════════════════════════════════════════════════════

    def _load_splits(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/valid/test splits from parquet or pickle."""
        input_type = self.config["input_type"]

        if input_type == "raw_splits":
            data_path = self.config["data_path"]
            train = pd.read_parquet(os.path.join(data_path, "train.parquet"))
            valid = pd.read_parquet(os.path.join(data_path, "valid.parquet"))
            test = pd.read_parquet(os.path.join(data_path, "test.parquet"))
        elif input_type == "pre_predicted":
            pickle_paths = self.config["pickle_paths"]
            with open(pickle_paths["train"], "rb") as f:
                train = pickle.load(f)
            with open(pickle_paths["valid"], "rb") as f:
                valid = pickle.load(f)
            with open(pickle_paths["test"], "rb") as f:
                test = pickle.load(f)

        else:
            raise ValueError(
                f"Unknown input_type: '{input_type}'. "
                "Must be 'parquet' or 'pickle'."
            )

        return train, valid, test

    def _build_classifier_adapter(self) -> ClassifierAdapter:
        """Build a ClassifierAdapter from config."""
        classifier_cfg = self.config.get("classifier_config", {})

        return ClassifierAdapter(
            classifier_type=self.config["classifier_type"],
            checkpoint_path=self.config["classifier_checkpoint"],
            labels_dict=self.labels_dict,
            num_labels=self.num_labels,
            seq_length=classifier_cfg.get("seq_length", 150),
            foundation_model_path=classifier_cfg.get(
                "foundation_model", "hanyangii/methylbert_hg19_12l"
            ),
            classifier_head_implementation=classifier_cfg.get(
                "classifier_head_implementation", "dmr_attention_based"
            ),
            dmr_label_column=classifier_cfg.get(
                "dmr_label_column", "dmr_ctype_label"
            ),
            soft_labels=classifier_cfg.get("soft_labels", True),
            dismir_flavor=classifier_cfg.get("dismir_flavor", "lstm"),
            batch_size=classifier_cfg.get("batch_size", 2200),
        )
