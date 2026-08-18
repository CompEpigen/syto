#!/usr/bin/env python3
"""
MethylDL Main Application Entry Point
Supports pretraining, fine-tuning, and inference for multiple model architectures
"""

import logging
import argparse
import sys
import os
from typing import Dict, Any
import shutil

import yaml


def setup_logging(verbose: bool = False, log_file: str = None):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict[str, Any], task: str) -> None:
    """Validate configuration for the specified task."""
    required_fields = {
        "classifier_fit": ["model", "data_path", "max_sequence_length"],
        "pretrain": ["model", "data_path"],  # Add pretrain requirements
        # The classifier/checkpoint and the syto atlas are only required when
        # syto classification-based deconvolution is enabled (deconvolution.syto).
        # A config may run baseline deconvolvers only, or supply pre-classified
        # reads, in which case those fields are absent.
        "inference": [
            "labels_dict_path",
            "input",
        ],
        "generate_pseudobulk": [
            "output_dir",
            "labels_dict_path",
        ],
        "fit_deconvolution": [
            "pseudobulk_path",
            "output_dir",
            "labels_dict_path",
        ],
        # pseudobulk_path / features_mask_path / deconvolvers_dir are only
        # required for deconvolvers that must be loaded and evaluated; see the
        # conditional check below.
        "fit_calibration": [
            "output_dir",
            "labels_dict_path",
        ],
        "confidence_intervals": [
            "calibration_results_dir",
            "labels_dict_path",
        ],
        "deconvolute_pseudobulk": [
            "pseudobulk_path",
            "output_dir",
            "labels_dict_path",
        ],
        "build_dataset": [
            "input_dir",
            "output_dir",
            "atlas_path",
            "reference_genome",
            "labels_dict_path",
            "n_buckets",
            "signature",
            "labelers",
        ],
    }

    if task not in required_fields:
        raise ValueError(f"Unknown task: {task}")

    for field in required_fields[task]:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' for task '{task}'")

    # Calibrating deconvolvers whose predictions are already stored on disk
    # (the baselines) needs no pseudobulk file, feature mask, or saved model.
    if task == "fit_calibration":
        model_backed = [
            cfg
            for cfg in config.get("deconvolvers", [])
            if not cfg.get("predictions_dir")
        ]
        if model_backed:
            for field in (
                "pseudobulk_path",
                "features_mask_path",
                "deconvolvers_dir",
            ):
                if field not in config:
                    names = [cfg.get("name") for cfg in model_backed]
                    raise ValueError(
                        f"Missing required field '{field}' for task '{task}': "
                        f"deconvolver(s) {names} have no 'predictions_dir' and "
                        "must be loaded and evaluated"
                    )

    # Validate model-specific configuration
    # (not needed for generate_pseudobulk or fit_deconvolution)
    if task not in (
        "generate_pseudobulk",
        "fit_deconvolution",
        "fit_calibration",
        "confidence_intervals",
        "deconvolute_pseudobulk",
        "build_dataset",
    ):
        if task in ("classifier_fit", "pretrain"):
            model = config["model"]["architecture"].lower()
        elif "classifier" in config:
            model = config["classifier"]["classifier_type"].lower()
        else:
            # Inference without a classifier (baseline-only or pre-classified
            # reads): no architecture to validate.
            model = None
        if model is not None and model not in [
            "methylbert",
            "dismir",
            "cancer_detector",
            "lookup",
            "epigenbert2",
        ]:
            raise ValueError(f"Unknown model architecture: {model}")


def run_classifier_fit(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run classifier fitting workflow."""
    from classifier_fit_pipeline import ClassifierFittingPipeline

    logger.info("Starting classifier fitting pipeline")
    pipeline = ClassifierFittingPipeline(config=config, logger=logger)
    pipeline.run()


def run_inference(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run inference based on configuration."""
    from inference import InferencePipeline

    logger.debug("Starting inference pipeline")
    pipeline = InferencePipeline(config=config, logger=logger)
    results = pipeline.run()

    # Log summary
    logger.debug("=" * 60)
    logger.debug("INFERENCE RESULTS SUMMARY")
    logger.debug("=" * 60)
    for deconvolver, calibrator, proportions in results:
        # Flatten the proportions array if it is 2D (e.g., shape (1, 39))
        flat_props = proportions.flatten()
        # 1. Pair the labels with their values
        # pipeline.labels_dict is {int: str}, so we match by index
        cell_contributions = []
        for idx, name in pipeline.labels_dict.items():
            if idx < len(flat_props):
                cell_contributions.append((name, flat_props[idx]))
        # 2. Sort by proportion (the second element of the tuple) in descending order
        cell_contributions.sort(key=lambda x: x[1], reverse=True)
        # 3. Take the top 5
        top_5 = cell_contributions[:5]
        # 4. Format and log
        label = f"{deconvolver} (calibrator={calibrator})"
        top5_str = ", ".join([f"{name}: {val:.4f}" for name, val in top_5])
        logger.debug(f"  {label} Top 5: {top5_str}")
        logger.debug("=" * 60)


def run_pretraining(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run pretraining based on configuration."""
    logger.info("Pretraining mode not yet implemented")
    # TODO: Implement pretraining logic
    raise NotImplementedError("Pretraining mode is not yet implemented")


def run_pseudobulk_generation(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run pseudo-bulk generation using the HDF5-based PseudobulkGenerator."""
    from App.pseudobulk_pipeline import PseudoBulkPipeline

    logger.info("Starting pseudo-bulk generation pipeline (HDF5-based)")
    pipeline = PseudoBulkPipeline(config=config, logger=logger)
    output_path = pipeline.run()

    # Log summary
    logger.info("=" * 60)
    logger.info("PSEUDO-BULK GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Output HDF5: {output_path}")
    logger.info("=" * 60)


def run_deconvolution_fitting(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run deconvolution model fitting based on configuration."""
    from deconvolution_pipeline import DeconvolutionFittingPipeline

    logger.info("Starting deconvolution fitting pipeline")
    pipeline = DeconvolutionFittingPipeline(config=config, logger=logger)
    pipeline.run()


def run_calibration_fitting(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run calibrator fitting based on configuration."""
    from calibration_pipeline import CalibratorFittingPipeline

    logger.info("Starting calibration fitting pipeline")
    pipeline = CalibratorFittingPipeline(config=config, logger=logger)
    pipeline.run()


def run_pseudobulk_deconvolution(
    config: Dict[str, Any], logger: logging.Logger
) -> None:
    """Run baseline deconvolution on a pre-generated pseudobulk HDF5 file."""
    from App.pseudobulk_deconvolution_pipeline import PseudobulkDeconvolutionPipeline

    logger.info("Starting pseudobulk deconvolution pipeline")
    pipeline = PseudobulkDeconvolutionPipeline(config=config, logger=logger)
    pipeline.run()


def run_build_dataset(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run the recovered-reads dataset build pipeline."""
    from App.dataset_build_pipeline import DatasetBuildPipeline

    logger.info("Starting recovered-reads dataset build pipeline")
    summary = DatasetBuildPipeline(config=config, logger=logger).run()
    logger.info("Dataset build summary: %s", summary)


def run_confidence_intervals(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Recompute metrics with bootstrap confidence intervals."""
    from conf_interval_pipeline import ConfidenceIntervalPipeline

    logger.info("Starting confidence interval pipeline")
    pipeline = ConfidenceIntervalPipeline(config=config, logger=logger)
    pipeline.run()


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="MethylDL Training Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune using configuration file
  python main.py --task classifier_fit --config config/classifier_fit_epigenbert2.yaml
  
  # Fine-tune with command-line overrides
  python main.py --task classifier_fit --config config/base.yaml \\
    --model epigenbert2 --max-seq-length 1000 --data-path /data/methylation
  
  # Run inference
  python main.py --task inference --config config/inference.yaml --checkpoint /path/to/model
        """,
    )

    # Required arguments
    parser.add_argument(
        "--task",
        choices=[
            "pretrain",
            "classifier_fit",
            "inference",
            "generate_pseudobulk",
            "fit_deconvolution",
            "fit_calibration",
            "confidence_intervals",
            "deconvolute_pseudobulk",
            "build_dataset",
            "create_config",
        ],
        required=True,
        help="Task to perform",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to configuration file (YAML). Not used by create_config.",
    )

    # Optional overrides
    parser.add_argument(
        "--model",
        choices=["dismir", "epigenbert2", "methylbert"],
        help="Model architecture (overrides config)",
    )
    parser.add_argument(
        "--data-path", type=str, help="Path to data directory (overrides config)"
    )
    parser.add_argument(
        "--max-seq-length", type=int, help="Maximum sequence length (overrides config)"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to model checkpoint for inference"
    )
    parser.add_argument(
        "--bam", type=str, help="Path to input BAM file (inference mode)"
    )
    parser.add_argument(
        "--atlas", type=str, help="Path to atlas TSV file (inference mode)"
    )
    parser.add_argument(
        "--labels-dict",
        type=str,
        help="Path to labels dictionary JSON (inference mode)",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for inference results"
    )
    parser.add_argument(
        "--datasets", nargs="+", help="Specific datasets to train on (default: all)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. If not set, logs go to stdout only.",
    )

    # MLflow overrides
    parser.add_argument("--mlflow-uri", type=str, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, help="MLflow experiment name")

    # Other options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate configuration without running"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose, args.log_file)
    logger.info("Starting MethylDL application - Task: %s", args.task)

    # The config-creation wizard is interactive and takes no --config file.
    if args.task == "create_config":
        from wizard import run_wizard

        run_wizard()
        return 0

    if not args.config:
        parser.error("--config is required for this task")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("Loaded configuration from %s", args.config)

        # Apply command-line overrides
        if args.model:
            if "model" not in config:
                config["model"] = {}
            config["model"]["architecture"] = args.model
            logger.info("Override: model = %s", args.model)

        if args.data_path:
            config["data_path"] = args.data_path
            logger.info("Override: data_path = %s", args.data_path)

        if args.max_seq_length:
            config["max_sequence_length"] = args.max_seq_length
            logger.info("Override: max_sequence_length = %s", args.max_seq_length)

        if args.checkpoint:
            config["checkpoint_path"] = args.checkpoint
            logger.info("Override: checkpoint_path = %s", args.checkpoint)

        # Inference-specific overrides
        if hasattr(args, "bam") and args.bam:
            if "input" not in config:
                config["input"] = {}
            config["input"]["type"] = "bam"
            config["input"]["bam_path"] = args.bam
            logger.info("Override: input.bam_path = %s", args.bam)

        if hasattr(args, "atlas") and args.atlas:
            config["atlas_path"] = args.atlas
            logger.info("Override: atlas_path = %s", args.atlas)

        if hasattr(args, "labels_dict") and args.labels_dict:
            config["labels_dict_path"] = args.labels_dict
            logger.info("Override: labels_dict_path = %s", args.labels_dict)

        if hasattr(args, "output_dir") and args.output_dir:
            config["output_dir"] = args.output_dir
            logger.info("Override: output_dir = %s", args.output_dir)

        if args.datasets:
            config["datasets"] = args.datasets
            logger.info("Override: datasets = %s", args.datasets)

        if args.mlflow_uri:
            if "mlflow" not in config:
                config["mlflow"] = {}
            config["mlflow"]["tracking_uri"] = args.mlflow_uri
            logger.info("Override: MLflow URI = %s", args.mlflow_uri)

        if args.experiment_name:
            if "mlflow" not in config:
                config["mlflow"] = {}
            config["mlflow"]["experiment_name"] = args.experiment_name
            logger.info("Override: experiment_name = %s", args.experiment_name)

        # Validate configuration
        validate_config(config, args.task)
        logger.info("Configuration validated successfully")

        if args.dry_run:
            logger.info(
                "Dry run mode - Configuration is valid, exiting without execution"
            )
            return 0

        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(args.config, output_dir)
        logger.info(f"Copied config to {output_dir}")

        # Execute task
        if args.task == "classifier_fit":
            run_classifier_fit(config, logger)
        elif args.task == "inference":
            run_inference(config, logger)
        elif args.task == "pretrain":
            run_pretraining(config, logger)
        elif args.task == "generate_pseudobulk":
            run_pseudobulk_generation(config, logger)
        elif args.task == "fit_deconvolution":
            run_deconvolution_fitting(config, logger)
        elif args.task == "fit_calibration":
            run_calibration_fitting(config, logger)
        elif args.task == "confidence_intervals":
            run_confidence_intervals(config, logger)
        elif args.task == "deconvolute_pseudobulk":
            run_pseudobulk_deconvolution(config, logger)
        elif args.task == "build_dataset":
            run_build_dataset(config, logger)

        logger.info("Task completed successfully")
        return 0

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
