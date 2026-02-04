#!/usr/bin/env python3
"""
MethylDL Main Application Entry Point
Supports pretraining, fine-tuning, and inference for multiple model architectures
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from methyldl.modelling.experiment_wrappers import (
    DismirMLflowExperiment,
    EpigenBERT2MLflowExperiment, 
    MethylBertMLflowExperiment
)
from methyldl.modelling.dnabert2 import TrainingArguments #TODO - must be different for MethylBERT 

def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('methyldl_training.log')
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config: Dict[str, Any], task: str) -> None:
    """Validate configuration for the specified task."""
    required_fields = {
        'fine_tune': ['model', 'data_path', 'max_sequence_length'],
        'pretrain': ['model', 'data_path'],  # Add pretrain requirements
        'inference': ['model', 'checkpoint_path', 'data_path']  # Add inference requirements
    }
    
    if task not in required_fields:
        raise ValueError(f"Unknown task: {task}")
    
    for field in required_fields[task]:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' for task '{task}'")
    
    # Validate model-specific configuration
    model = config['model']['architecture'].lower()
    if model not in ['dismir', 'epigenbert2', 'methylbert']:
        raise ValueError(f"Unknown model architecture: {model}")

def create_experiment(config: Dict[str, Any], logger: logging.Logger) -> Any:
    """Create the appropriate experiment based on configuration."""
    model_arch = config['model']['architecture'].lower()
    data_path = config['data_path']
    max_seq_length = config['max_sequence_length']
    
    # MLflow configuration
    mlflow_config = config.get('mlflow', {})
    
    experiment_name = None
    tracking_uri = None
    

    experiment_name = mlflow_config.get('experiment_name')
    tracking_uri = mlflow_config.get('tracking_uri')
    
    if not experiment_name:
        raise ValueError("MLflow is enabled but 'experiment_name' is not provided")
    
    logger.info(f"MLflow enabled - Experiment: {experiment_name}")
    if tracking_uri:
        logger.info(f"MLflow tracking URI: {tracking_uri}")

    
    # Model-specific parameters
    model_config = config['model']
    
    if model_arch == 'dismir':
        return DismirMLflowExperiment(
            data_path=data_path,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            max_sequence_length=max_seq_length,
            model_flavor=model_config.get('flavor', 'lstm'),
            splits=config.get('splits', ['train', 'valid', 'test'])
        )
    
    elif model_arch == 'epigenbert2':
        return EpigenBERT2MLflowExperiment(
            data_path=data_path,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            max_sequence_length=max_seq_length,
            use_cpg_methylation=model_config.get('use_cpg_methylation', True),
            use_m6a_methylation=model_config.get('use_m6a_methylation', False),
            foundation_model_huggingface=model_config.get('foundation_model', 'zhihan1996/DNABERT-2-117M'),
            splits=config.get('splits', ['train', 'valid', 'test']),
            use_triton=model_config.get('use_triton', False)
        )
    
    elif model_arch == 'methylbert':
        return MethylBertMLflowExperiment(
            data_path=data_path,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            max_sequence_length=max_seq_length,
            foundation_model_huggingface=model_config.get('foundation_model', 'hanyangii/methylbert_hg19_12l'),
            splits=config.get('splits', ['train', 'valid', 'test'])
        )
    
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

def run_fine_tuning(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run fine-tuning based on configuration."""
    
    # Create experiment
    experiment = create_experiment(config, logger)
    
    # Get training configuration
    training_config = config.get('training', {})
    model_arch = config['model']['architecture'].lower()
    
    # Dataset selection
    datasets = config.get('datasets', 'all')
    
    if datasets == 'all':
        # Train on all available datasets
        logger.info("Training on all available datasets")
        if model_arch == 'dismir':
            experiment.run_full_experiment(**training_config)
        else:
            # For transformer models, check if custom TrainingArguments provided
            if 'training_arguments' in training_config:
                args_dict = training_config['training_arguments']
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
            
            if model_arch == 'dismir':
                experiment.train_dataset(dataset_name, **training_config)
            else:
                # For transformer models
                if 'training_arguments' in training_config:
                    args_dict = training_config['training_arguments']
                    training_args = TrainingArguments(**args_dict)
                    experiment.train_dataset(dataset_name, training_args=training_args)
                else:
                    experiment.train_dataset(dataset_name, **training_config)

def run_inference(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run inference based on configuration."""
    logger.info("Inference mode not yet implemented")
    # TODO: Implement inference logic
    raise NotImplementedError("Inference mode is not yet implemented")

def run_pretraining(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run pretraining based on configuration."""
    logger.info("Pretraining mode not yet implemented")
    # TODO: Implement pretraining logic
    raise NotImplementedError("Pretraining mode is not yet implemented")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='MethylDL Training Application',
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
        """
    )
    
    # Required arguments
    parser.add_argument('--task', 
                       choices=['pretrain', 'fine_tune', 'inference'],
                       required=True,
                       help='Task to perform')
    parser.add_argument('--config', 
                       type=str, 
                       required=True,
                       help='Path to configuration file (YAML)')
    
    # Optional overrides
    parser.add_argument('--model', 
                       choices=['dismir', 'epigenbert2', 'methylbert'],
                       help='Model architecture (overrides config)')
    parser.add_argument('--data-path', 
                       type=str,
                       help='Path to data directory (overrides config)')
    parser.add_argument('--max-seq-length', 
                       type=int,
                       help='Maximum sequence length (overrides config)')
    parser.add_argument('--checkpoint', 
                       type=str,
                       help='Path to model checkpoint for inference')
    parser.add_argument('--datasets', 
                       nargs='+',
                       help='Specific datasets to train on (default: all)')
    
    # MLflow overrides
    parser.add_argument('--mlflow-uri', 
                       type=str,
                       help='MLflow tracking URI')
    parser.add_argument('--experiment-name', 
                       type=str,
                       help='MLflow experiment name')
    
    # Other options
    parser.add_argument('--verbose', 
                       action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Validate configuration without running')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info(f"Starting MethylDL application - Task: {args.task}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Apply command-line overrides
        if args.model:
            if 'model' not in config:
                config['model'] = {}
            config['model']['architecture'] = args.model
            logger.info(f"Override: model = {args.model}")
        
        if args.data_path:
            config['data_path'] = args.data_path
            logger.info(f"Override: data_path = {args.data_path}")
        
        if args.max_seq_length:
            config['max_sequence_length'] = args.max_seq_length
            logger.info(f"Override: max_sequence_length = {args.max_seq_length}")
        
        if args.checkpoint:
            config['checkpoint_path'] = args.checkpoint
            logger.info(f"Override: checkpoint_path = {args.checkpoint}")
        
        if args.datasets:
            config['datasets'] = args.datasets
            logger.info(f"Override: datasets = {args.datasets}")
        
        if args.mlflow_uri:
            if 'mlflow' not in config:
                config['mlflow'] = {}
            config['mlflow']['tracking_uri'] = args.mlflow_uri
            logger.info(f"Override: MLflow URI = {args.mlflow_uri}")
        
        if args.experiment_name:
            if 'mlflow' not in config:
                config['mlflow'] = {}
            config['mlflow']['experiment_name'] = args.experiment_name
            logger.info(f"Override: experiment_name = {args.experiment_name}")
        
        # Validate configuration
        validate_config(config, args.task)
        logger.info("Configuration validated successfully")
        
        if args.dry_run:
            logger.info("Dry run mode - Configuration is valid, exiting without execution")
            return 0
        
        # Execute task
        if args.task == 'fine_tune':
            run_fine_tuning(config, logger)
        elif args.task == 'inference':
            run_inference(config, logger)
        elif args.task == 'pretrain':
            run_pretraining(config, logger)
        
        logger.info("Task completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())