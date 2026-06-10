#!/usr/bin/env python3
"""
MethylDL Main Application Entry Point
Supports pretraining, fine-tuning, and inference for multiple model architectures
"""

import logging
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any

import yaml
from syto.classification.experiment_wrappers import (
    AbstractMLFlowExperiment,
    TransformersMLFLowExperiment,
)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from syto.classification.experiment_wrappers import (
    DismirMLflowExperiment,
    EpigenBERT2MLflowExperiment,
    MethylBertMLflowExperiment,
)
from syto.classification.classifiers.dnabert2 import (
    TrainingArguments,
)  # TODO - must be different for MethylBERT


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
        "fine_tune": ["model", "data_path", "max_sequence_length"],
        "pretrain": ["model", "data_path"],  # Add pretrain requirements
        "inference": [
            "classifier",
            "checkpoint_path",
            "labels_dict_path",
            "atlas_path",
            "input",
        ],
        "generate_pseudobulk": [
            "output_dir",
            "labels_dict_path",
        ],
        "fit_deconvolution": [
            "predicted_splits",
            "ios_full_matrices_path",
            "output_dir",
            "labels_dict_path",
        ],
        "fit_calibration": [
            "ios_feature_selected_path",
            "deconvolvers_dir",
            "output_dir",
            "labels_dict_path",
        ],
        "confidence_intervals": [
            "calibration_results_dir",
            "labels_dict_path",
        ],
    }

    if task not in required_fields:
        raise ValueError(f"Unknown task: {task}")

    for field in required_fields[task]:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' for task '{task}'")

    # Validate model-specific configuration (not needed for generate_pseudobulk or fit_deconvolution)
    if task not in (
        "generate_pseudobulk",
        "fit_deconvolution",
        "fit_calibration",
        "confidence_intervals",
    ):
        model = config["classifier"]["classifier_type"].lower()
        if model not in ["methylbert", "dismir", "cancer_detector", "lookup"]:
            raise ValueError(f"Unknown model architecture: {model}")


def create_experiment(
    config: Dict[str, Any], logger: logging.Logger
) -> AbstractMLFlowExperiment:
    """Create the appropriate experiment based on configuration."""
    model_arch = config["model"]["architecture"].lower()
    data_path = config["data_path"]
    max_seq_length = config["max_sequence_length"]

    # MLflow configuration
    mlflow_config = config.get("mlflow", {})

    experiment_name = None
    tracking_uri = None

    experiment_name = mlflow_config.get("experiment_name")
    tracking_uri = mlflow_config.get("tracking_uri")

    if not experiment_name:
        raise ValueError("MLflow is enabled but 'experiment_name' is not provided")

    logger.info(f"MLflow enabled - Experiment: {experiment_name}")
    if tracking_uri:
        logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Model-specific parameters
    model_config = config["model"]

    if model_arch == "dismir":
        return DismirMLflowExperiment(
            data_path=data_path,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            max_sequence_length=max_seq_length,
            model_flavor=model_config.get("flavor", "lstm"),
            splits=config.get("splits", ["train", "valid", "test"]),
        )

    elif model_arch == "epigenbert2":
        return EpigenBERT2MLflowExperiment(
            data_path=data_path,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            max_sequence_length=max_seq_length,
            use_cpg_methylation=model_config.get("use_cpg_methylation", True),
            use_m6a_methylation=model_config.get("use_m6a_methylation", False),
            foundation_model_huggingface=model_config.get(
                "foundation_model", "zhihan1996/DNABERT-2-117M"
            ),
            splits=config.get("splits", ["train", "valid", "test"]),
            use_triton=model_config.get("use_triton", False),
        )

    elif model_arch == "methylbert":
        return MethylBertMLflowExperiment(
            data_path=data_path,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            max_sequence_length=max_seq_length,
            foundation_model_huggingface=model_config.get(
                "foundation_model", "hanyangii/methylbert_hg19_12l"
            ),
            splits=config.get("splits", ["train", "valid", "test"]),
        )

    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")


def run_fine_tuning(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run fine-tuning based on configuration."""

    # Create experiment
    experiment = create_experiment(config, logger)

    # Get training configuration
    training_config = config.get("training", {})
    model_arch = config["model"]["architecture"].lower()

    # Dataset selection
    datasets = config.get("datasets", "all")

    if datasets == "all":
        # Train on all available datasets
        logger.info("Training on all available datasets")
        if model_arch == "dismir":
            experiment.run_full_experiment(**training_config)
        else:
            # For transformer models, check if custom TrainingArguments provided
            if "training_arguments" in training_config:
                args_dict = training_config["training_arguments"]
                training_args = TrainingArguments(**args_dict)
                experiment.run_full_experiment(training_args=training_args)
            else:
                experiment.run_full_experiment(**training_config)
    else:
        # Train on specific datasets
        if isinstance(datasets, str):
            datasets = [datasets]

        for dataset_name in datasets:
            logger.info(f"Training on dataset: {dataset_name}")

            if model_arch == "dismir":
                experiment.train_dataset(dataset_name, **training_config)
            else:
                # For transformer models
                assert isinstance(
                    experiment, TransformersMLFLowExperiment
                ), "Expected a transformer experiment instance"
                if "training_arguments" in training_config:
                    args_dict = training_config["training_arguments"]
                    training_args = TrainingArguments(**args_dict)
                    # pylint: disable-next:unexpected-keyword-arg
                    experiment.train_dataset(dataset_name, training_args=training_args)
                else:
                    experiment.train_dataset(dataset_name, **training_config)


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
    """Run pseudo-bulk mixture generation based on configuration."""
    from App.pseudobulk_pipeline import PseudoBulkPipeline

    logger.info("Starting pseudo-bulk generation pipeline")
    pipeline = PseudoBulkPipeline(config=config, logger=logger)
    result = pipeline.run()

    # Log summary
    logger.info("=" * 60)
    logger.info("PSEUDO-BULK GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total examples: {result[list(result.keys())[0]].shape[0]}")
    # splits_cfg = config.get("splits", ["train", "valid", "test"])
    # for split_name in splits_cfg:
    #     logger.info(f"Features shape {split_name}: {result[f'features_{split_name}'].shape}")
    logger.info(f"  Output saved to: {config['output_dir']}")
    logger.info("=" * 60)


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
  python main.py --task fine_tune --config config/fine_tune_epigenbert2.yaml
  
  # Fine-tune with command-line overrides
  python main.py --task fine_tune --config config/base.yaml \\
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
            "fine_tune",
            "inference",
            "generate_pseudobulk",
            "fit_deconvolution",
            "fit_calibration",
            "confidence_intervals",
        ],
        required=True,
        help="Task to perform",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file (YAML)"
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
    logger.info(f"Starting MethylDL application - Task: {args.task}")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Apply command-line overrides
        if args.model:
            if "model" not in config:
                config["model"] = {}
            config["model"]["architecture"] = args.model
            logger.info(f"Override: model = {args.model}")

        if args.data_path:
            config["data_path"] = args.data_path
            logger.info(f"Override: data_path = {args.data_path}")

        if args.max_seq_length:
            config["max_sequence_length"] = args.max_seq_length
            logger.info(f"Override: max_sequence_length = {args.max_seq_length}")

        if args.checkpoint:
            config["checkpoint_path"] = args.checkpoint
            logger.info(f"Override: checkpoint_path = {args.checkpoint}")

        # Inference-specific overrides
        if hasattr(args, "bam") and args.bam:
            if "input" not in config:
                config["input"] = {}
            config["input"]["type"] = "bam"
            config["input"]["bam_path"] = args.bam
            logger.info(f"Override: input.bam_path = {args.bam}")

        if hasattr(args, "atlas") and args.atlas:
            config["atlas_path"] = args.atlas
            logger.info(f"Override: atlas_path = {args.atlas}")

        if hasattr(args, "labels_dict") and args.labels_dict:
            config["labels_dict_path"] = args.labels_dict
            logger.info(f"Override: labels_dict_path = {args.labels_dict}")

        if hasattr(args, "output_dir") and args.output_dir:
            if "output" not in config:
                config["output"] = {}
            config["output"]["output_dir"] = args.output_dir
            logger.info(f"Override: output.output_dir = {args.output_dir}")

        if args.datasets:
            config["datasets"] = args.datasets
            logger.info(f"Override: datasets = {args.datasets}")

        if args.mlflow_uri:
            if "mlflow" not in config:
                config["mlflow"] = {}
            config["mlflow"]["tracking_uri"] = args.mlflow_uri
            logger.info(f"Override: MLflow URI = {args.mlflow_uri}")

        if args.experiment_name:
            if "mlflow" not in config:
                config["mlflow"] = {}
            config["mlflow"]["experiment_name"] = args.experiment_name
            logger.info(f"Override: experiment_name = {args.experiment_name}")

        # Validate configuration
        validate_config(config, args.task)
        logger.info("Configuration validated successfully")

        if args.dry_run:
            logger.info(
                "Dry run mode - Configuration is valid, exiting without execution"
            )
            return 0

        # Execute task
        if args.task == "fine_tune":
            run_fine_tuning(config, logger)
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

        logger.info("Task completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
