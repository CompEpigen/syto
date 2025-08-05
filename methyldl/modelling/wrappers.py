import mlflow
import mlflow.pytorch
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,roc_curve, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')
from methyldl.modelling.dismir import Dismir
from transformers import TrainingArguments, EarlyStoppingCallback
from transformers import TrainerCallback, TrainerControl, TrainerState
from tqdm import tqdm
import gc
from methyldl.modelling.dnabert2 import EpigenDnabert2 
from methyldl.data.dataset import SupervisedDataset 
from typing import List, Dict
from methyldl.data.genome import generate_kmer_str_with_overlap
from methyldl.modelling.methylbert import MethylVocab,MethylBertFinetuneDataset
from methyldl.modelling.methylbert import MethylBert,default_methylbert_config
import random
import time


class RestartOnPoorPerformanceCallback(TrainerCallback):
    """
    Simpler version: Only checks performance once at a specific step.
    If performance is poor, triggers restart. Otherwise, never checks again.
    """
    
    def __init__(self, 
                 eval_loss_threshold: float,
                 check_at_step: int,
                 max_retries: int = 3):
        """
        Args:
            eval_loss_threshold: Maximum acceptable eval loss at check_at_step
            check_at_step: Exact step at which to check performance (only checks once)
            max_retries: Maximum number of training restarts allowed
        """
        self.eval_loss_threshold = eval_loss_threshold
        self.check_at_step = check_at_step
        self.max_retries = max_retries
        
        # Internal state
        self.retry_count = 0
        self.should_restart = False
        self.has_checked = False  # Only check once
        
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Check eval loss only once at the specified step."""
        
        # Skip if we've already checked
        if self.has_checked:
            return control
        
        # Only proceed if we have eval loss in the current log
        if state.log_history and 'eval_loss' in state.log_history[-1]:
            current_step = state.global_step
            
            # Check only at the specific step
            if current_step >= self.check_at_step:
                self.has_checked = True  # Mark as checked
                eval_loss = state.log_history[-1]['eval_loss']
                
                print(f"\n=== Performance Check at Step {current_step} ===")
                print(f"Eval loss: {eval_loss:.4f}, Threshold: {self.eval_loss_threshold:.4f}")
                
                # Check if eval loss is above threshold
                if eval_loss > self.eval_loss_threshold:
                    if self.retry_count < self.max_retries:
                        self.should_restart = True
                        control.should_training_stop = True
                        print(f"Poor initial performance. Triggering restart (attempt {self.retry_count + 1}/{self.max_retries})")
                    else:
                        print(f"Poor performance but max retries ({self.max_retries}) reached. Continuing.")
                else:
                    print(f"Good performance! Continuing training without further checks.")
        
        return control
    
    def reset_for_retry(self):
        """Reset internal state for a new training attempt."""
        self.retry_count += 1
        self.should_restart = False
        self.has_checked = False

# class RestartOnPoorPerformanceCallback(TrainerCallback):
#     """
#     A callback that monitors eval loss and triggers training restart if performance
#     is poor after a specified number of logging steps.
#     """
    
#     def __init__(self, 
#                  eval_loss_threshold: float,
#                  check_after_n_steps: int,
#                  max_retries: int = 3,
#                  patience_before_restart: int = 2):
#         """
#         Args:
#             eval_loss_threshold: Maximum acceptable eval loss after check_after_n_steps
#             check_after_n_steps: Number of logging steps after which to check performance
#             max_retries: Maximum number of training restarts allowed
#             patience_before_restart: Number of consecutive checks before restart
#         """
#         self.eval_loss_threshold = eval_loss_threshold
#         self.check_after_n_steps = check_after_n_steps
#         self.max_retries = max_retries
#         self.patience_before_restart = patience_before_restart
        
#         # Internal state
#         self.retry_count = 0
#         self.poor_performance_count = 0
#         self.last_check_step = 0
#         self.should_restart = False
#         self.eval_losses = []
        
#     def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         """Check eval loss after specified number of logging steps."""
        
#         # Only proceed if we have eval loss in the current log
#         if state.log_history and 'eval_loss' in state.log_history[-1]:
#             current_step = state.global_step
#             eval_loss = state.log_history[-1]['eval_loss']
#             self.eval_losses.append(eval_loss)
            
#             # Check if we've reached the check point
#             if current_step >= self.check_after_n_steps and current_step > self.last_check_step:
#                 self.last_check_step = current_step
                
#                 # Check if eval loss is above threshold
#                 if eval_loss > self.eval_loss_threshold:
#                     self.poor_performance_count += 1
#                     print(f"Poor performance detected: eval_loss={eval_loss:.4f} > threshold={self.eval_loss_threshold:.4f}")
#                     print(f"Poor performance count: {self.poor_performance_count}/{self.patience_before_restart}")
                    
#                     # Check if we should restart
#                     if self.poor_performance_count >= self.patience_before_restart:
#                         if self.retry_count < self.max_retries:
#                             self.should_restart = True
#                             control.should_training_stop = True
#                             print(f"Triggering restart (attempt {self.retry_count + 1}/{self.max_retries})")
#                         else:
#                             print(f"Max retries ({self.max_retries}) reached. Continuing with current training.")
#                 else:
#                     # Reset poor performance count if we're doing well
#                     self.poor_performance_count = 0
#                     #print(f"Performance OK: eval_loss={eval_loss:.4f} <= threshold={self.eval_loss_threshold:.4f}")
        
#         return control
    
#     def reset_for_retry(self):
#         """Reset internal state for a new training attempt."""
#         self.retry_count += 1
#         self.poor_performance_count = 0
#         self.last_check_step = 0
#         self.should_restart = False
#         self.eval_losses = []

def split_long_reads(df: pd.DataFrame, max_read_length: int) -> pd.DataFrame:
    """
    Split DNA reads longer than max_read_length into smaller chunks.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing DNA read data
    max_read_length : int
        Maximum allowed read length for splitting
    
    Returns:
    --------
    pd.DataFrame
        Transformed dataset with split reads and calculated CpG counts
    """
    
    def count_cpgs(methylation_string: str) -> int:
        """Count CpG sites (represented by '1' or '2' in methylation_ids)
        0 - unmethylated C, 1 - methylated C, 2 - methylation status unknown"""
        return max(sum(1 for char in methylation_string if char in ['0', '1']),1) # Setting to at least one so reads without CpGs can also take part 
    
    def split_single_read(row: pd.Series) -> List[Dict]:
        """Split a single read into chunks if it exceeds max_read_length"""
        # Extract relevant fields directly from the pandas Series
        read_name = row['read_name']
        input_ids = row['input_ids']
        methylation_ids = row['methylation_ids']
        chromosome = row['chromosome']
        original_file = row['original_file']
        label = row['label']
        if "dmr_id" in row.keys():
            dmr_id = row["dmr_id"]
        else:
            dmr_id = 0
        # Get the actual sequence length
        sequence_length = len(input_ids)
        
        # If the read is within the max length, return as is
        if sequence_length <= max_read_length:
            return [{
                'read_name': read_name,
                'input_ids': input_ids,
                'methylation_ids': methylation_ids,
                'chromosome': chromosome,
                'original_file': original_file,
                'label': label,
                'num_cpgs': count_cpgs(methylation_ids),
                'dmr_id':dmr_id
            }]
        
        # Split the read into chunks
        chunks = []
        for i in range(0, sequence_length, max_read_length):
            end_idx = min(i + max_read_length, sequence_length)
            
            # Extract the chunk
            chunk_input_ids = input_ids[i:end_idx]
            chunk_methylation_ids = methylation_ids[i:end_idx]
            
            chunks.append({
                'read_name': read_name,
                'input_ids': chunk_input_ids,
                'methylation_ids': chunk_methylation_ids,
                'chromosome': chromosome,
                'original_file': original_file,
                'label': label,
                'num_cpgs': count_cpgs(chunk_methylation_ids),
                'dmr_id':dmr_id
            })
        
        return chunks
    
    # Process all rows
    all_chunks = []
    for _, row in df.iterrows():
        chunks = split_single_read(row)
        all_chunks.extend(chunks)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(all_chunks)
    
    return result_df

class DismirMLflowExperiment:
    """
    MLflow wrapper for training Dismir models across multiple chromosomes.
    Handles experiment tracking, metrics logging, and artifact storage.
    """
    
    def __init__(self, 
                 data_path,
                 experiment_name=None,
                 tracking_uri=None,
                 max_sequence_length=1000,
                 model_flavor="minigru",
                 splits = ["train", "valid", "test", "rest"]
                 ):
        """
        Initialize the MLflow experiment wrapper.
        
        Args:
            data_path (str): Path to the main data directory containing chromosome folders
            experiment_name (str): Name for the MLflow experiment
            tracking_uri (str): MLflow tracking URI (optional)
            max_sequence_length (int): Maximum sequence length for the model
            model_flavor (str): Model flavor ("minigru" or "lstm")
        """
        self.data_path = Path(data_path)
        self.max_sequence_length = max_sequence_length
        self.model_flavor = model_flavor
        self.splits = splits
        
        # Set up MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if experiment_name is None:
            experiment_name = f"Dismir_Chromosome_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
        # Find all chromosome directories
        self.chromosome_dirs = self._find_chromosome_directories()
        print(f"Found {len(self.chromosome_dirs)} chromosome directories: {list(self.chromosome_dirs.keys())}")
    
    def _find_chromosome_directories(self):
        """Find all directories containing the required parquet files."""
        chromosome_dirs = {}
        
        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")
        

        for item in self.data_path.iterdir():
            if item.is_dir():
                # if item.name in _chr:
                # print(item.name)
                # if item.name not in processed_chr:
                #     if "cpg_counts_selected" in item.name:
                    # Check if this directory contains the required files
                        required_files = [x+".parquet" for x in self.splits]
                        if all((item / f).exists() for f in required_files):
                            chromosome_dirs[item.name] = item
        
        if not chromosome_dirs:
            raise ValueError(f"No directories with required parquet files found in {self.data_path}")
        print(chromosome_dirs)
        return chromosome_dirs
    
    def _calculate_data_stats(self, data_path):
        """Calculate statistics for a dataset."""
        try:
            df = pd.read_parquet(data_path)
            stats = {
                'num_samples': len(df),
                'num_positive': int(df['label'].sum()) if 'label' in df.columns else 0,
                'num_negative': int(len(df) - df['label'].sum()) if 'label' in df.columns else 0,
                'positive_ratio': float(df['label'].mean()) if 'label' in df.columns else 0.0,
                'avg_sequence_length': float(df['input_ids'].str.len().mean()) if 'input_ids' in df.columns else 0.0
            }
            return stats
        except Exception as e:
            print(f"Error calculating stats for {data_path}: {e}")
            return {}
    
    def _calculate_metrics(self, y_true, y_pred_proba, y_pred_binary,best_treshold):
        """Calculate comprehensive metrics."""
        try:
            
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred_binary)),
                'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred_binary, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred_binary, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0,
                "mcc":float(matthews_corrcoef(y_true,y_pred_binary)),
                "threshold": best_treshold
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp),
                    'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                    'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                })
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    def _make_predictions_and_calculate_metrics(self, model, split_name, chromosome, max_sequence_length):
        """Make predictions and calculate metrics for a given split."""
        try:
            path = self.chromosome_dirs[chromosome] / f'{split_name}.parquet'
            if split_name not in ["train", "valid","test"]: 
                df = pd.read_parquet(str(path).replace("_cpg_counts_selected", ""))    
                data_chunked = split_long_reads(df,max_sequence_length)
                dna, methylation, labels = data_chunked["input_ids"],data_chunked["methylation_ids"], data_chunked["label"]
            else:
                df = pd.read_parquet(path)   
                dna, methylation, labels = df["input_ids"], df["methylation_ids"], df["label"]
            
            predictions, predictions_binary = model.predict(dna, methylation_sequences=methylation)
            if split_name not in ["train", "valid", "test"]:      
                data_chunked["predictions_proba"] = predictions
                data_chunked["predictions_weighted"] = data_chunked["predictions_proba"] * data_chunked["num_cpgs"]
                data_chunked_agg = data_chunked.groupby("read_name").agg(
                    predictions_weighted=pd.NamedAgg(column="predictions_weighted", aggfunc="sum"),
                    num_cpgs=pd.NamedAgg(column="num_cpgs", aggfunc="sum"),
                    label=pd.NamedAgg(column="label", aggfunc="min"),
                    )
                data_chunked_agg["predictions_weighted"] = data_chunked_agg["predictions_weighted"]/data_chunked_agg["num_cpgs"]
                fpr, tpr, thresholds = roc_curve(data_chunked_agg["label"], data_chunked_agg["predictions_weighted"])
                best_treshold = thresholds[np.argmax(tpr-fpr)]
                data_chunked_agg["predictions"] = data_chunked_agg["predictions_weighted"]>best_treshold
                data_chunked_agg = data_chunked_agg.loc[data_chunked_agg["num_cpgs"]>0,]
                labels, predictions_binary, predictions = data_chunked_agg["label"], data_chunked_agg["predictions"], data_chunked_agg["predictions_weighted"]
            else:
                best_treshold = 0.5

            # Calculate metrics
            metrics = self._calculate_metrics(labels, predictions, predictions_binary,best_treshold)
            
            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'true_label': labels,
                'predicted_probability': predictions,
                'predicted_label': predictions_binary
            })
            
            return metrics, predictions_df
            
        except Exception as e:
            print(f"Error making predictions for {split_name}: {e}")
            return {}, pd.DataFrame()
    
    def train_chromosome(self, 
                        chromosome,
                        epochs=200,
                        batch_size=128,
                        patience=12,
                        optimizer_type="SGD",
                        lr=0.01,
                        momentum=0.9,
                        weight_decay=1e-6):
        """
        Train a model for a specific chromosome and log everything to MLflow.
        
        Args:
            chromosome (str): Chromosome name (directory name)
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            patience (int): Early stopping patience
            optimizer_type (str): Optimizer type ("SGD" or "Adam")
            lr (float): Learning rate
            momentum (float): Momentum (for SGD)
            weight_decay (float): Weight decay
            
        Returns:
            dict: Training results and metrics
        """
        
        with mlflow.start_run(run_name=f"chromosome_{chromosome}"):
            # try:
                print(f"\n=== Training chromosome {chromosome} ===")
                
                # Log parameters
                mlflow.log_param("chromosome", chromosome)
                mlflow.log_param("max_sequence_length", self.max_sequence_length)
                mlflow.log_param("model_flavor", self.model_flavor)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("patience", patience)
                mlflow.log_param("optimizer_type", optimizer_type)
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("momentum", momentum)
                mlflow.log_param("weight_decay", weight_decay)
                
                # Get file paths
                chr_dir = self.chromosome_dirs[chromosome]
                file_paths = [
                    str(chr_dir / 'train.parquet'),
                    str(chr_dir / 'test.parquet'), 
                    str(chr_dir / 'valid.parquet')
                ]
                
                # Calculate and log data statistics
                data_stats = {}
                for split in self.splits:
                    file_path = chr_dir / f'{split}.parquet'
                    stats = self._calculate_data_stats(file_path)
                    data_stats[split] = stats
                    
                    # Log data statistics
                    for key, value in stats.items():
                        mlflow.log_metric(f"data_{split}_{key}", value)
                
                # Create temporary directory for model weights
                temp_dir = f"./temp_weights_{chromosome}"
                os.makedirs(temp_dir, exist_ok=True)
                
                # Initialize and train model
                print(f"Initializing Dismir model for chromosome {chromosome}")
                dismir_instance = Dismir(
                    self.max_sequence_length,
                    *file_paths,
                    flavour=self.model_flavor
                )
                
                print(f"Starting training for chromosome {chromosome}")
                dismir_instance.train(
                    epochs=epochs,
                    batch_size=batch_size,
                    train_dir=temp_dir,
                    patience=patience,
                    optimizer_type=optimizer_type,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    verbose=1
                )
                
                # Load best model weights
                best_weights_path = os.path.join(temp_dir, "weight.pt")
                if os.path.exists(best_weights_path):
                    dismir_instance.model.load_state_dict(torch.load(best_weights_path, map_location=dismir_instance.device))
                    print(f"Loaded best weights for chromosome {chromosome}")
                
                # Calculate metrics and make predictions for each split
                all_predictions = {}
                all_metrics = {}
                
                for split in self.splits:
                    print(f"Calculating metrics for {split} split...")
                    metrics, predictions_df = self._make_predictions_and_calculate_metrics(
                        dismir_instance, split, chromosome, dismir_instance.max_sequence_length
                    )
                    
                    all_metrics[split] = metrics
                    all_predictions[split] = predictions_df
                    
                    # Log metrics to MLflow
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"{split}_{metric_name}", metric_value)
                
                # Save model
                print(f"Saving model artifacts for chromosome {chromosome}")
                mlflow.pytorch.log_model(
                    dismir_instance.model,
                    f"model_chromosome_{chromosome}",
                    registered_model_name=f"{self.experiment_name}_chromosome_{chromosome}"
                )
                
                # Save predictions as artifacts
                predictions_dir = f"predictions_{chromosome}"
                os.makedirs(predictions_dir, exist_ok=True)
                
                for split, pred_df in all_predictions.items():
                    pred_file = os.path.join(predictions_dir, f"{split}_predictions.pkl")
                    with open(pred_file, "wb") as f:
                       pickle.dump(pred_df, f)
                
                mlflow.log_artifacts(predictions_dir, "predictions")
                
                # Save comprehensive results
                results = {
                    'chromosome': chromosome,
                    'data_stats': data_stats,
                    'metrics': all_metrics,
                    'model_params': {
                        'max_sequence_length': self.max_sequence_length,
                        'model_flavor': self.model_flavor,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'patience': patience,
                        'optimizer_type': optimizer_type,
                        'learning_rate': lr,
                        'momentum': momentum,
                        'weight_decay': weight_decay
                    }
                }
                
                # Save results as JSON artifact
                results_file = f"results_{chromosome}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                mlflow.log_artifact(results_file, "results")
                
                # Cleanup temporary files
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                if os.path.exists(predictions_dir):
                    shutil.rmtree(predictions_dir)
                if os.path.exists(results_file):
                    os.remove(results_file)
                
                print(f"✓ Completed training for chromosome {chromosome}")
                return results
                
            # except Exception as e:
            #     print(f"✗ Error training chromosome {chromosome}: {e}")
            #     mlflow.log_param("error", str(e))
            #     raise e
    
    def run_full_experiment(self, **training_kwargs):
        """
        Run the complete experiment across all chromosomes.
        
        Args:
            **training_kwargs: Arguments to pass to train_chromosome method
        """
        print(f"Starting full experiment: {self.experiment_name}")
        print(f"Training {len(self.chromosome_dirs)} chromosome folders")
        print(f"Chromosome folders: {list(self.chromosome_dirs.keys())}")
        
        all_results = {}
        failed_chromosomes = []
        
        for chromosome in self.chromosome_dirs.keys():
            # try:

            results = self.train_chromosome(chromosome, **training_kwargs)
            all_results[chromosome] = results
            # except Exception as e:
            #     print(f"Failed to train chromosome {chromosome}: {e}")
            #     failed_chromosomes.append(chromosome)
            #     continue
        
    
class EpigenBERT2MLflowExperiment:
    """
    MLflow wrapper for training EpigenBERT2 models across multiple chromosomes.
    Handles experiment tracking, metrics logging, and artifact storage with 
    progressive sequence length prediction strategy.
    """
    
    def __init__(self, 
                 data_path,
                 experiment_name=None,
                 tracking_uri=None,
                 max_sequence_length=1000,
                 use_cpg_methylation=True,
                 use_m6a_methylation=False,
                 foundation_model_huggingface="zhihan1996/DNABERT-2-117M",
                 splits = ['train', 'valid', 'test', 'rest']):
        """
        Initialize the MLflow experiment wrapper.
        
        Args:
            data_path (str): Path to the main data directory containing chromosome folders
            experiment_name (str): Name for the MLflow experiment
            tracking_uri (str): MLflow tracking URI (optional)
            max_sequence_length (int): Maximum sequence length for the model
            use_cpg_methylation (bool): Whether to use CpG methylation
            use_m6a_methylation (bool): Whether to use m6A methylation
            foundation_model_huggingface (str): Hugging Face model path
        """
        self.data_path = Path(data_path)
        self.max_sequence_length = max_sequence_length
        self.use_cpg_methylation = use_cpg_methylation
        self.use_m6a_methylation = use_m6a_methylation
        self.foundation_model_huggingface = foundation_model_huggingface
        self.splits = splits
        
        # Set up MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if experiment_name is None:
            experiment_name = f"EpigenBERT2_Chromosome_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
        # Find all chromosome directories
        self.chromosome_dirs = self._find_chromosome_directories()
        print(f"Found {len(self.chromosome_dirs)} chromosome directories: {list(self.chromosome_dirs.keys())}")
    
    def _find_chromosome_directories(self):
        """Find all directories containing the required parquet files."""
        chromosome_dirs = {}
        
        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")
        
        # processed_chr = [f"chr{x}_cpg_counts_selected" for x in range(1, 23, 1) if not x==19]
        # processed_chr.extend([f"chr{x}" for x in range(2, 23, 1)])
        # _chr = [f"chr{x}"for x in [4,18,19,21]]
        # _chr = [f"chr{x}"for x in [13]]
        for item in self.data_path.iterdir():
            if item.is_dir():
                # if item.name in _chr:
                #     if "cpg_counts_selected" in item.name:
                        # Check if this directory contains the required files
                        required_files = [x+ '.parquet' for x in self.splits]
                        if all((item / f).exists() for f in required_files):
                            chromosome_dirs[item.name] = item
        
        if not chromosome_dirs:
            raise ValueError(f"No directories with required parquet files found in {self.data_path}")
        
        print(chromosome_dirs)
        return chromosome_dirs
    
    def _calculate_data_stats(self, data_path):
        """Calculate statistics for a dataset."""
        try:
            df = pd.read_parquet(data_path)
            stats = {
                'num_samples': len(df),
                'num_positive': int(df['label'].sum()) if 'label' in df.columns else 0,
                'num_negative': int(len(df) - df['label'].sum()) if 'label' in df.columns else 0,
                'positive_ratio': float(df['label'].mean()) if 'label' in df.columns else 0.0,
                'avg_sequence_length': float(df['input_ids'].str.len().mean()) if 'input_ids' in df.columns else 0.0,
                'max_sequence_length': int(df['input_ids'].str.len().max()) if 'input_ids' in df.columns else 0
            }
            return stats
        except Exception as e:
            print(f"Error calculating stats for {data_path}: {e}")
            return {}
    
    def _calculate_metrics(self, y_true, y_pred_proba, y_pred_binary):
        """Calculate comprehensive metrics."""
        try:
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred_binary)),
                'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred_binary, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred_binary, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp),
                    'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                    'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                })
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    def _progressive_predict(self, data_df, checkpoint_path, split_name):
        """
        Make predictions using progressive sequence length strategy for long sequences.
        
        Args:
            data_df (pd.DataFrame): DataFrame with sequences to predict
            checkpoint_path (str): Path to the fine-tuned model checkpoint
            split_name (str): Name of the split for logging
            
        Returns:
            tuple: (predictions, labels, sequence_lengths)
        """
        print(f"Starting progressive prediction for {split_name} split...")
        
        # Calculate max sequence length in the data
        max_read_length = np.max([len(x) for x in data_df["input_ids"]])
        print(f"Max sequence length in {split_name}: {max_read_length}")
        
        # For train and valid, use standard prediction if sequences are reasonable
        if split_name in ['train', 'valid'] and max_read_length <= self.max_sequence_length:
            print(f"Using standard prediction for {split_name} (max_length <= {self.max_sequence_length})")
            epigenbert = EpigenDnabert2(
                fine_tuned_model_path=checkpoint_path,
                max_sequence_length=self.max_sequence_length,
                use_cpg_methylation=self.use_cpg_methylation,
                use_m6a_methylation=self.use_m6a_methylation,
                foundation_model_huggingface=self.foundation_model_huggingface
            )
            
            test_dataset = SupervisedDataset(
                tokenizer=epigenbert.tokenizer,
                data_path_or_list=data_df,
                kmer=-1,
                data_interface="pandas"
            )
            
            result = epigenbert.predict(test_dataset, batch_size=200)
            predictions = result.predictions if hasattr(result, 'predictions') else result[0]
            labels = result.label_ids if hasattr(result, 'label_ids') else result[1]
            
            # Clean up
            del epigenbert
            gc.collect()
            torch.cuda.empty_cache()
            
            return predictions, labels, [self.max_sequence_length] * len(predictions)
        
        # Progressive prediction for test and rest splits or long sequences
        print(f"Using progressive prediction for {split_name}")
        all_predictions = []
        all_labels = []
        all_max_sequence_lengths = []
        
        # Generate sequence length ranges
        sequence_ranges = list(reversed(range(1000, max_read_length + 1000, 1000)))
        
        for max_seq_len in tqdm(sequence_ranges, desc=f"Processing {split_name}"):
            # Filter data for current sequence length range
            if max_seq_len == sequence_ranges[0]:  # First iteration (longest sequences)
                subset_data = data_df[data_df["sequence_length"] > max_seq_len - 1000]
            else:
                subset_data = data_df[
                    (data_df["sequence_length"] > max_seq_len - 1000) & 
                    (data_df["sequence_length"] <= max_seq_len)
                ]
            
            if subset_data.shape[0] == 0:
                continue
                
            print(f"Processing {subset_data.shape[0]} sequences with max_length={max_seq_len}")
            
            # Initialize model for this sequence length
            epigenbert = EpigenDnabert2(
                fine_tuned_model_path=checkpoint_path,
                max_sequence_length=max_seq_len,
                use_cpg_methylation=self.use_cpg_methylation,
                use_m6a_methylation=self.use_m6a_methylation,
                foundation_model_huggingface=self.foundation_model_huggingface
            )
            
            # Create dataset for this subset
            test_dataset = SupervisedDataset(
                tokenizer=epigenbert.tokenizer,
                data_path_or_list=subset_data,
                kmer=-1,
                data_interface="pandas"
            )
            
            # Make predictions
            result = epigenbert.predict(test_dataset, batch_size=200)
            predictions = result.predictions if hasattr(result, 'predictions') else result[0]
            labels = result.label_ids if hasattr(result, 'label_ids') else result[1]
            
            # Store results
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_max_sequence_lengths.extend([max_seq_len] * subset_data.shape[0])
            
            # Clean up
            del epigenbert, test_dataset
            gc.collect()
            torch.cuda.empty_cache()
        
        return np.array(all_predictions), np.array(all_labels), all_max_sequence_lengths
    
    def _make_predictions_and_calculate_metrics(self, checkpoint_path, split_name, chromosome):
        """Make predictions and calculate metrics for a given split."""
        try:
            path = self.chromosome_dirs[chromosome] / f'{split_name}.parquet'
            if split_name not in ["train","valid"]:
                path = str(path).replace("_cpg_counts_selected", "")
            data_df = pd.read_parquet(path)
            
            # Add sequence_length column if not present
            if 'sequence_length' not in data_df.columns:
                data_df['sequence_length'] = data_df['input_ids'].str.len()
            
            # Use progressive prediction strategy
            predictions, labels, seq_lengths = self._progressive_predict(
                data_df, checkpoint_path, split_name
            )
            
            # Convert probabilities to binary predictions
            predictions_binary = (predictions > 0.5).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_metrics(labels, predictions, predictions_binary)
            
            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'true_label': labels,
                'predicted_probability': predictions,
                'predicted_label': predictions_binary,
                'max_sequence_length_used': seq_lengths
            })
            
            return metrics, predictions_df
            
        except Exception as e:
            print(f"Error making predictions for {split_name}: {e}")
            return {}, pd.DataFrame()
        
    def _get_best_checkpoint(self, trainer, output_dir: str) -> str:
        """
        Return the path of the best checkpoint according to eval_loss.
        Falls back to the latest checkpoint if – for any reason – the best one
        cannot be located.
        """
        # 1️⃣ try to pick it directly from the Trainer state ------------------
        if trainer is not None and trainer.state.best_model_checkpoint:
            best_ckpt = trainer.state.best_model_checkpoint
            if os.path.isdir(best_ckpt):
                return os.path.join(os.path.abspath(best_ckpt), "model.safetensors")

        # 2️⃣ look inside trainer_state.json ----------------------------------
        state_file = Path(output_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            best_ckpt = state.get("best_model_checkpoint")
            if best_ckpt and os.path.isdir(best_ckpt):
                return os.path.join(os.path.abspath(best_ckpt), "model.safetensors")

        # 3️⃣ graceful fallback: use the most-recent checkpoint ---------------
        ckpts = sorted(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1])
        )
        if ckpts:
            return os.path.join(os.path.abspath(str(ckpts[-1])), "model.safetensors")


    
    def train_chromosome(self, 
                        chromosome,
                        training_args=None,
                        num_train_epochs=250,
                        per_device_train_batch_size=50,
                        per_device_eval_batch_size=50,
                        learning_rate=3e-5,
                        early_stopping_patience=10,
                        early_stopping_threshold=0.001,
                        save_steps=50,
                        eval_steps=50,
                        warmup_steps=100,
                        logging_steps=100,
                        eval_loss_threshold=0.6,
                        check_at_step=500,
                        max_retries=3):
        """
        Train a model for a specific chromosome and log everything to MLflow.
        
        Args:
            chromosome (str): Chromosome name (directory name)
            training_args (TrainingArguments, optional): Custom training arguments object
            num_train_epochs (int): Number of training epochs (used if training_args is None)
            per_device_train_batch_size (int): Training batch size per device
            per_device_eval_batch_size (int): Evaluation batch size per device
            learning_rate (float): Learning rate
            early_stopping_patience (int): Early stopping patience
            early_stopping_threshold (float): Early stopping threshold
            save_steps (int): Save model every N steps
            eval_steps (int): Evaluate every N steps
            warmup_steps (int): Number of warmup steps
            logging_steps (int): Log every N steps
            
        Returns:
            dict: Training results and metrics
        """
        
        with mlflow.start_run(run_name=f"chromosome_{chromosome}"):
            print(f"\n=== Training chromosome {chromosome} ===")
            
            # Log parameters
            mlflow.log_param("chromosome", chromosome)
            mlflow.log_param("max_sequence_length", self.max_sequence_length)
            mlflow.log_param("use_cpg_methylation", self.use_cpg_methylation)  
            mlflow.log_param("use_m6a_methylation", self.use_m6a_methylation)
            mlflow.log_param("foundation_model", self.foundation_model_huggingface)
            # Log parameters based on training_args or individual parameters
            output_dir =  os.path.abspath(f"output/epigenbert2_{chromosome}")
            if training_args is not None:
                training_args.output_dir = output_dir
                mlflow.log_param("num_train_epochs", training_args.num_train_epochs)
                mlflow.log_param("per_device_train_batch_size", training_args.per_device_train_batch_size)
                mlflow.log_param("per_device_eval_batch_size", training_args.per_device_eval_batch_size)
                mlflow.log_param("learning_rate", training_args.learning_rate)
                mlflow.log_param("save_steps", training_args.save_steps)
                mlflow.log_param("eval_steps", training_args.eval_steps)
                mlflow.log_param("warmup_steps", training_args.warmup_steps)
                mlflow.log_param("logging_steps", training_args.logging_steps)
                mlflow.log_param("output_dir", training_args.output_dir)
                mlflow.log_param("run_name", training_args.run_name)
                
                # Extract values for use in training
                num_train_epochs = training_args.num_train_epochs
                per_device_train_batch_size = training_args.per_device_train_batch_size
                per_device_eval_batch_size = training_args.per_device_eval_batch_size
                learning_rate = training_args.learning_rate
                save_steps = training_args.save_steps
                eval_steps = training_args.eval_steps
                warmup_steps = training_args.warmup_steps
                logging_steps = training_args.logging_steps
            else:
                mlflow.log_param("num_train_epochs", num_train_epochs)
                mlflow.log_param("per_device_train_batch_size", per_device_train_batch_size)
                mlflow.log_param("per_device_eval_batch_size", per_device_eval_batch_size)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("save_steps", save_steps)
                mlflow.log_param("eval_steps", eval_steps)
                mlflow.log_param("warmup_steps", warmup_steps)
                mlflow.log_param("logging_steps", logging_steps)
                mlflow.log_param("output_dir", output_dir)
            
            mlflow.log_param("early_stopping_patience", early_stopping_patience)
            mlflow.log_param("early_stopping_threshold", early_stopping_threshold)
            
            # Get chromosome directory
            chr_dir = self.chromosome_dirs[chromosome]
            
            # Calculate and log data statistics
            data_stats = {}
            for split in self.splits:
                file_path = chr_dir / f'{split}.parquet'
                stats = self._calculate_data_stats(file_path)
                data_stats[split] = stats
                
                # Log data statistics
                for key, value in stats.items():
                    mlflow.log_metric(f"data_{split}_{key}", value)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            
            # Set up training arguments
            if training_args is None:
                training_args = TrainingArguments(
                    run_name=f"epigenbert2_{chromosome}",
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    gradient_accumulation_steps=1,
                    learning_rate=learning_rate,
                    fp16=True,
                    save_steps=save_steps,
                    output_dir=output_dir,
                    evaluation_strategy="steps",
                    eval_steps=eval_steps,
                    warmup_steps=warmup_steps,
                    logging_steps=logging_steps,
                    num_train_epochs=num_train_epochs,
                    overwrite_output_dir=True,
                    log_level="info",
                    find_unused_parameters=False,
                    batch_eval_metrics=False,
                    eval_and_save_results=True,
                    remove_unused_columns=False,
                    eval_accumulation_steps=8,
                    torch_empty_cache_steps=10,
                    prediction_loss_only=False,
                    gradient_checkpointing=False,
                    skip_memory_metrics=True,
                    auto_find_batch_size=False,
                    save_strategy="steps",
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                )
            else:
                # Use the provided training_args but ensure some key settings
                if not hasattr(training_args, 'save_strategy'):
                    training_args.save_strategy = "steps"
                if not hasattr(training_args, 'load_best_model_at_end'):
                    training_args.load_best_model_at_end = True
                if not hasattr(training_args, 'metric_for_best_model'):
                    training_args.metric_for_best_model = "eval_loss"
                if not hasattr(training_args, 'greater_is_better'):
                    training_args.greater_is_better = False
            
            # Add early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )

            restart_callback = RestartOnPoorPerformanceCallback(
                eval_loss_threshold=eval_loss_threshold,
                check_at_step=check_at_step,
                max_retries=max_retries
            )

            # Fine-tune the model
            print(f"Starting fine-tuning for chromosome {chromosome}")
            attempt = 0
            while attempt <= max_retries:
                print(f"\n=== Training attempt {attempt + 1}/{max_retries + 1} for chromosome {chromosome} ===")
                            # Initialize model
                print(f"Initializing EpigenBERT2 model for chromosome {chromosome}")
                seed = int(time.time() * 1000) % 2**32
                # seed = 894526933
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                model_instance = EpigenDnabert2(
                    use_cpg_methylation=self.use_cpg_methylation,
                    use_m6a_methylation=self.use_m6a_methylation,
                    max_sequence_length=self.max_sequence_length,
                    foundation_model_huggingface=self.foundation_model_huggingface
                )
                print(f"Weights sum for -2 layer: {np.sum(list(model_instance.model.parameters())[-2].to("cpu").detach().numpy())}")
                training_args.seed = seed
                print(f"Seed for training args is {training_args.seed}")
                model_instance.fine_tune(
                    data_path=str(chr_dir),
                    training_args=training_args,
                    data_interface="pandas",
                    callbacks = [early_stopping_callback,restart_callback]
                )

                            
                # Check if we should restart
                if restart_callback.should_restart and attempt < max_retries:
                    print(f"Restart triggered. Eval loss did not meet criteria.")
                    mlflow.log_metric("restart_triggered", 1)
                    restart_callback.reset_for_retry()
                    attempt += 1
                    if model_instance is not None:
                        del model_instance
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                else:
                    # Training completed successfully or max retries reached
                    print(f"Training completed successfully or max retries reached.")
                    mlflow.log_metric("restart_triggered", 0)
                    break
            mlflow.log_metric("total_attempts", attempt + 1)             

            
            # -----------------------------------------------------------------------
            # ✓ pick *best* checkpoint (absolute path, verified to exist)
            # -----------------------------------------------------------------------
            checkpoint_path = self._get_best_checkpoint(
                trainer=model_instance.trainer,       
                output_dir=training_args.output_dir
            )
            print(f"Using checkpoint: {checkpoint_path}")
            mlflow.log_param("best_checkpoint", checkpoint_path)
                        # Log model artifacts
            print(f"Logging model artifacts for chromosome {chromosome}")
            checkpoint_folder_path = str(checkpoint_path).replace("\model.safetensors", "")
            if os.path.exists(checkpoint_folder_path):
                mlflow.log_artifacts(checkpoint_folder_path, f"model")

            # Calculate metrics and make predictions for each split
            all_predictions = {}
            all_metrics = {}
            
            for split in self.splits:
                print(f"Calculating metrics for {split} split...")
                metrics, predictions_df = self._make_predictions_and_calculate_metrics(
                    checkpoint_path, split, chromosome
                )
                
                all_metrics[split] = metrics
                all_predictions[split] = predictions_df
                
                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{split}_{metric_name}", metric_value)
            
            # Save predictions as artifacts
            predictions_dir = f"predictions_{chromosome}"
            os.makedirs(predictions_dir, exist_ok=True)
            
            for split, pred_df in all_predictions.items():
                pred_file = os.path.join(predictions_dir, f"{split}_predictions.pkl")
                with open(pred_file, "wb") as f:
                    pickle.dump(pred_df, f)
            
            mlflow.log_artifacts(predictions_dir, "predictions")
            
            # Save comprehensive results
            results = {
                'chromosome': chromosome,
                'data_stats': data_stats,
                'metrics': all_metrics,
                'model_params': {
                    'max_sequence_length': self.max_sequence_length,
                    'use_cpg_methylation': self.use_cpg_methylation,
                    'use_m6a_methylation': self.use_m6a_methylation,
                    'foundation_model': self.foundation_model_huggingface,
                    'num_train_epochs': num_train_epochs,
                    'per_device_train_batch_size': per_device_train_batch_size,
                    'per_device_eval_batch_size': per_device_eval_batch_size,
                    'learning_rate': learning_rate,
                    'early_stopping_patience': early_stopping_patience,
                    'early_stopping_threshold': early_stopping_threshold
                },
                'checkpoint_path': checkpoint_path
            }
            
            # Save results as JSON artifact
            results_file = f"results_{chromosome}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            mlflow.log_artifact(results_file, "results")
            
            # Cleanup temporary files
            import shutil
            if os.path.exists(predictions_dir):
                shutil.rmtree(predictions_dir)
            if os.path.exists(results_file):
                os.remove(results_file)
            
            print(f"✓ Completed training for chromosome {chromosome}")
            return results
    
    def run_full_experiment(self, training_args=None, **training_kwargs):
        """
        Run the complete experiment across all chromosomes.
        
        Args:
            training_args (TrainingArguments, optional): Custom training arguments object
            **training_kwargs: Arguments to pass to train_chromosome method (used if training_args is None)
        """
        print(f"Starting full experiment: {self.experiment_name}")
        print(f"Training {len(self.chromosome_dirs)} chromosome folders")
        print(f"Chromosome folders: {list(self.chromosome_dirs.keys())}")
        
        all_results = {}
        failed_chromosomes = []
        
        for chromosome in self.chromosome_dirs.keys():
            try:
                if training_args is not None:
                    results = self.train_chromosome(chromosome, training_args=training_args, **training_kwargs)
                else:
                    results = self.train_chromosome(chromosome, **training_kwargs)
                all_results[chromosome] = results
            except Exception as e:
                print(f"Failed to train chromosome {chromosome}: {e}")
                failed_chromosomes.append(chromosome)
                continue
        
    
class MethylBertMLflowExperiment:
    """
    MLflow wrapper for training  MethylBert model across multiple chromosomes.
    Handles experiment tracking, metrics logging, and artifact storage.
    """
    
    def __init__(self, 
                 data_path,
                 experiment_name=None,
                 tracking_uri=None,
                 max_sequence_length=150,
                 foundation_model_huggingface="hanyangii/methylbert_hg19_12l",
                 splits = ["train", "valid", "test", "rest"],
                 ):
        """
        Initialize the MLflow experiment wrapper.
        
        Args:
            data_path (str): Path to the main data directory containing chromosome folders
            experiment_name (str): Name for the MLflow experiment
            tracking_uri (str): MLflow tracking URI (optional)
            max_sequence_length (int): Maximum sequence length for the model
            model_flavor (str): Model flavor ("minigru" or "lstm")
        """
        self.data_path = Path(data_path)
        if max_sequence_length > 510: 
            return ValueError("max_sequence_length for MethylBert cannot be bigger than 510 bp")
        self.max_sequence_length = max_sequence_length
        self.splits = splits
        self.foundation_model_huggingface = foundation_model_huggingface
        
        # Set up MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if experiment_name is None:
            experiment_name = f"MethylBert_Chromosome_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
        # Find all chromosome directories
        self.chromosome_dirs = self._find_chromosome_directories()
        print(f"Found {len(self.chromosome_dirs)} chromosome directories: {list(self.chromosome_dirs.keys())}")
    
    def _get_best_checkpoint(self, trainer, output_dir: str) -> str:
        """
        Return the path of the best checkpoint according to eval_loss.
        Falls back to the latest checkpoint if – for any reason – the best one
        cannot be located.
        """
        # 1️⃣ try to pick it directly from the Trainer state ------------------
        if trainer is not None and trainer.state.best_model_checkpoint:
            best_ckpt = trainer.state.best_model_checkpoint
            if os.path.isdir(best_ckpt):
                return os.path.join(os.path.abspath(best_ckpt), "model.safetensors")

        # 2️⃣ look inside trainer_state.json ----------------------------------
        state_file = Path(output_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            best_ckpt = state.get("best_model_checkpoint")
            if best_ckpt and os.path.isdir(best_ckpt):
                return os.path.join(os.path.abspath(best_ckpt), "model.safetensors")

        # 3️⃣ graceful fallback: use the most-recent checkpoint ---------------
        ckpts = sorted(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1])
        )
        if ckpts:
            return os.path.join(os.path.abspath(str(ckpts[-1])), "model.safetensors")
    def _find_chromosome_directories(self):
        """Find all directories containing the required parquet files."""
        chromosome_dirs = {}
        
        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")
        
        # _chr = [f"chr{x}"for x in range(2,10,1)]
        # _chr.extend([f"chr{x}"for x in range(20,23,1)])
        # _chr = ["chr7","chr8","chr4"]
        for item in self.data_path.iterdir():
            if item.is_dir():
                # if item.name in _chr:
                #if item.name not in processed_chr:
                #     if "cpg_counts_selected" in item.name:
                    # Check if this directory contains the required files
                        required_files = [x+".parquet" for x in self.splits]
                        if all((item / f).exists() for f in required_files):
                            chromosome_dirs[item.name] = item
        
        if not chromosome_dirs:
            raise ValueError(f"No directories with required parquet files found in {self.data_path}")
        print(chromosome_dirs)
        return chromosome_dirs
    
    def _calculate_data_stats(self, data_path):
        """Calculate statistics for a dataset."""
        try:
            df = pd.read_parquet(data_path)
            stats = {
                'num_samples': len(df),
                'num_positive': int(df['label'].sum()) if 'label' in df.columns else 0,
                'num_negative': int(len(df) - df['label'].sum()) if 'label' in df.columns else 0,
                'positive_ratio': float(df['label'].mean()) if 'label' in df.columns else 0.0,
                'avg_sequence_length': float(df['input_ids'].str.len().mean()) if 'input_ids' in df.columns else 0.0
            }
            return stats
        except Exception as e:
            print(f"Error calculating stats for {data_path}: {e}")
            return {}
    
    def _calculate_metrics(self, y_true, y_pred_proba, y_pred_binary,best_treshold):
        """Calculate comprehensive metrics."""
        try:
            
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred_binary)),
                'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred_binary, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred_binary, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0,
                "threshold": best_treshold
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp),
                    'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                    'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                })
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
        
    
    def _prepare_methylbert_list(self, data_path, split, dmrs, split_to_chunks=False):
        data = pd.read_parquet(os.path.join(data_path, f"{split}.parquet"))
            # Handle different DMR column names
        if "dmr_name" in data.columns:
            # Standard case: single dmr_name per row
            data = data.merge(dmrs, on=["dmr_name"])
        elif "dmr_names" in data.columns:
            # Special case: multiple dmr_names per row (e.g., test_whole_reads)
            # Extract the first DMR name from the list and create dmr_id mapping
            def get_first_dmr_id(dmr_names_str):
                # Assuming dmr_names is stored as a string representation of a list
                # or as an actual list. Handle both cases.
                if isinstance(dmr_names_str, str):
                    # Parse string representation of list if needed
                    import ast
                    try:
                        dmr_list = ast.literal_eval(dmr_names_str)
                    except:
                        # If it's a comma-separated string
                        dmr_list = dmr_names_str.split(',')
                else:
                    dmr_list = dmr_names_str
                
                # Get the first DMR name
                first_dmr = dmr_list[0] if dmr_list else None
                
                # Find the corresponding dmr_id
                if first_dmr:
                    matching_dmr = dmrs[dmrs["dmr_name"] == first_dmr]
                    if not matching_dmr.empty:
                        return matching_dmr.iloc[0]["dmr_id"]
                return 0  # Default DMR ID if not found
            
            # Apply the function to create dmr_id column
            data["dmr_id"] = data["dmr_names"].apply(get_first_dmr_id)
        else:
            # No DMR information available, use default DMR ID
            data["dmr_id"] = 0
        
        if split_to_chunks:
            data = split_long_reads(data, self.max_sequence_length)
            data.reset_index(drop=True, inplace=True)  # drop=True to avoid keeping old index
        
        data_list = [['dna_seq', 'methyl_seq', 'dmr_ctype', 'dmr_label', 'ctype']]
        valid_indices = []  # Track indices of valid rows
        
        for i, row in data.iterrows():
            dna = generate_kmer_str_with_overlap(row["input_ids"][:(self.max_sequence_length+2)])
            methyl = row["methylation_ids"][1:-1][:self.max_sequence_length]
            
            if not len(dna) or not len(methyl):
                # Skip this row and don't add its index to valid_indices
                continue
                
            label = row["label"]
            dmr_label = row["dmr_id"]
            data_list.append([dna, methyl, 1, dmr_label, label])
            valid_indices.append(i)  # Track this as a valid row
        
        # Filter the original data to keep only valid rows
        data_filtered = data.loc[valid_indices].copy()
        data_filtered.reset_index(drop=True, inplace=True)
        
        if split_to_chunks:
            return data_list, data_filtered
        return data_list
    
    def _make_predictions_and_calculate_metrics(self, model, split_name, chromosome,train_dataset, valid_dataset,dmrs):
        """Make predictions and calculate metrics for a given split."""
        try:
            1+1
        finally:
            path = self.chromosome_dirs[chromosome] / f'{split_name}.parquet'
            if split_name not in ["train", "valid"]: 
                df = pd.read_parquet(str(path).replace("_cpg_counts_selected", ""))
                df = df.loc[df["input_ids"].apply(len)>0] # Preventing zero length sequences --> TODO: fix in the source!!!  
                #split_data_list, df = self._prepare_methylbert_list(self.chromosome_dirs[chromosome], split_name, dmrs, split_to_chunks=True)
                split_data_list = self._prepare_methylbert_list(self.chromosome_dirs[chromosome], split_name, dmrs, split_to_chunks=False)
                vocab = MethylVocab(k=3)
                split_dataset = MethylBertFinetuneDataset(
                    data_source=split_data_list,
                    vocab=vocab,
                    seq_len=self.max_sequence_length
                )
            elif split_name=="train":
                split_dataset = train_dataset
            elif split_name=="valid":
                split_dataset = valid_dataset
                
            result = model.predict(split_dataset,batch_size=200)
            predictions = result.predictions if hasattr(result, 'predictions') else result[0]
            labels = result.label_ids if hasattr(result, 'label_ids') else result[1]

            if split_name not in ["train", "valid", "test"]:      
                df["predictions_proba"] = predictions
                df["predictions_weighted"] = df["predictions_proba"] * df["num_cpgs"]
                data_chunked_agg = df.groupby("read_name").agg(
                    predictions_weighted=pd.NamedAgg(column="predictions_weighted", aggfunc="sum"),
                    num_cpgs=pd.NamedAgg(column="num_cpgs", aggfunc="sum"),
                    label=pd.NamedAgg(column="label", aggfunc="min"),
                    )
                data_chunked_agg["predictions_weighted"] = data_chunked_agg["predictions_weighted"]/data_chunked_agg["num_cpgs"]
                fpr, tpr, thresholds = roc_curve(data_chunked_agg["label"], data_chunked_agg["predictions_weighted"])
                best_treshold = thresholds[np.argmax(tpr-fpr)]
                data_chunked_agg["predictions"] = data_chunked_agg["predictions_weighted"]>best_treshold
                data_chunked_agg = data_chunked_agg.loc[data_chunked_agg["num_cpgs"]>0,]
                labels, predictions_binary, predictions = data_chunked_agg["label"], data_chunked_agg["predictions"], data_chunked_agg["predictions_weighted"]
            else:
                best_treshold = 0.5

            # # Calculate metrics
            predictions_binary = predictions > best_treshold
            metrics = self._calculate_metrics(labels, predictions, predictions_binary,best_treshold)
            
            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'true_label': labels,
                'predicted_probability': predictions,
                'predicted_label': predictions_binary
            })
            
            return metrics, predictions_df
            
        # except Exception as e:
        #     print(f"Error making predictions for {split_name}: {e}")
        #     return {}, pd.DataFrame()
    
    def train_chromosome(self, 
                        chromosome,
                        epochs=250,
                        batch_size=500,
                        lr=0.0004,
                        warmup_steps=100,
                        weight_decay=0.1,
                        log_freq=20,
                        eval_freq=20,
                        training_args=None,
                        early_stopping_patience=10,
                        early_stopping_threshold=0.001):
        """
        Train a model for a specific chromosome and log everything to MLflow.
        
        Args:
            chromosome (str): Chromosome name (directory name)
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            lr (float): Learning rate
            warmup_steps (int): Number of warmup steps
            weight_decay (float): Weight decay
            log_freq (int): Logging frequency
            eval_freq (int): Evaluation frequency
            training_args (TrainingArguments, optional): Custom training arguments
            
        Returns:
            dict: Training results and metrics
        """
        
        with mlflow.start_run(run_name=f"chromosome_{chromosome}"):
            try:
                print(f"\n=== Training chromosome {chromosome} ===")
                
                # Log parameters
                mlflow.log_param("chromosome", chromosome)
                mlflow.log_param("max_sequence_length", self.max_sequence_length)
                mlflow.log_param("foundation_model", self.foundation_model_huggingface)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("warmup_steps", warmup_steps)
                mlflow.log_param("weight_decay", weight_decay)
                mlflow.log_param("log_freq", log_freq)
                mlflow.log_param("eval_freq", eval_freq)
                mlflow.log_param("early_stopping_patience", early_stopping_patience)
                mlflow.log_param("early_stopping_threshold", early_stopping_threshold)
                
                # Get chromosome directory
                chr_dir = self.chromosome_dirs[chromosome]
                
                # Calculate and log data statistics
                data_stats = {}
                for split in self.splits:
                    file_path = chr_dir / f'{split}.parquet'
                    stats = self._calculate_data_stats(file_path)
                    data_stats[split] = stats
                    
                    # Log data statistics
                    for key, value in stats.items():
                        mlflow.log_metric(f"data_{split}_{key}", value)
                
                # Create DMR mapping from training data
                print(f"Creating DMR mapping for chromosome {chromosome}")
                dmr_dfs = []
                for split in self.splits:
                    df = pd.read_parquet(chr_dir / f"{split}.parquet")
                    if "dmr_name" in df.columns:
                        dmr_dfs.append(df)
                
                if dmr_dfs:
                    all_dmrs = pd.concat(dmr_dfs)["dmr_name"].unique()
                else:
                    # Fallback if dmr_name doesn't exist
                    all_dmrs = ["dmr_0"]  # Default DMR
                    
                dmrs = pd.DataFrame(enumerate(all_dmrs))
                dmrs.columns = ["dmr_id", "dmr_name"]
                num_dmr_labels = len(dmrs)
                
                mlflow.log_param("num_dmr_labels", num_dmr_labels)
                
                # Prepare datasets
                print(f"Preparing datasets for chromosome {chromosome}")
                train_data = self._prepare_methylbert_list(chr_dir, "train", dmrs)
                valid_data = self._prepare_methylbert_list(chr_dir, "valid", dmrs)
                
                # Create MethylBert datasets
                vocab = MethylVocab(k=3)
                train_dataset = MethylBertFinetuneDataset(
                    data_source=train_data,
                    vocab=vocab,
                    seq_len=self.max_sequence_length
                )
                valid_dataset = MethylBertFinetuneDataset(
                    data_source=valid_data,
                    vocab=vocab,
                    seq_len=self.max_sequence_length
                )
                
                # Create output directory
                output_dir = f"./output/methylbert_{chromosome}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Set up model config
                from collections import OrderedDict
                methylbert_config = OrderedDict([
                    ("lr", lr),
                    ("beta", (0.9, 0.98)),
                    ("weight_decay", weight_decay),
                    ("warmup_step", warmup_steps),
                    ("eps", 1e-6),
                    ("with_cuda", True),
                    ("log_freq", log_freq),
                    ("eval_freq", eval_freq),
                    ("n_hidden", None),
                    ("decrease_steps", 200),
                    ("eval", False),
                    ("amp", True),
                    ("gradient_accumulation_steps", 1),
                    ("max_grad_norm", 1.0),
                    ("save_freq", None),
                    ("loss", "bce"),
                    ("adam_beta1", 0.9),
                    ("adam_beta2", 0.98),
                    ("seed", 950410),
                ])
                
                # Initialize MethylBert model
                print(f"Initializing MethylBert model for chromosome {chromosome}")
                model_instance = MethylBert(
                    custom_config=methylbert_config,
                    foundation_model_path=self.foundation_model_huggingface,
                    load_weights=True,
                    num_labels=2,
                    num_dmr_labels=num_dmr_labels,
                    seq_len=self.max_sequence_length,
                    output_dir=output_dir,
                    batch_size=batch_size
                )
                
                # Override training args if provided
                if training_args is not None:
                    model_instance.training_args = training_args
                    model_instance.training_args.output_dir = output_dir
                else:
                    # Update training args with our parameters
                    model_instance.training_args.num_train_epochs = epochs
                    model_instance.training_args.per_device_train_batch_size = batch_size
                    model_instance.training_args.per_device_eval_batch_size = batch_size
                    model_instance.training_args.learning_rate = lr
                    model_instance.training_args.warmup_steps = warmup_steps
                    model_instance.training_args.weight_decay = weight_decay
                    model_instance.training_args.logging_steps = log_freq
                    model_instance.training_args.eval_steps = eval_freq
                    model_instance.training_args.save_steps = eval_freq
                
                # Add early stopping callback
                early_stopping_callback = EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=early_stopping_threshold
                )

                # Fine-tune the model
                print(f"Starting fine-tuning for chromosome {chromosome}")
                model_instance.fine_tune(
                    data_path=None,
                    train_dataset=train_dataset,
                    val_dataset=valid_dataset,
                    test_dataset=None,
                    training_args=model_instance.training_args,
                    callbacks = [early_stopping_callback]
                )

                # -----------------------------------------------------------------------
                # ✓ pick *best* checkpoint (absolute path, verified to exist)
                # -----------------------------------------------------------------------
                checkpoint_path = self._get_best_checkpoint(
                    trainer=model_instance.trainer,       
                    output_dir=output_dir
                )
                print(f"Using checkpoint: {checkpoint_path}")
                mlflow.log_param("best_checkpoint", checkpoint_path)
                            # Log model artifacts
                print(f"Logging model artifacts for chromosome {chromosome}")
                checkpoint_folder_path = str(checkpoint_path).replace("\model.safetensors", "")
                if os.path.exists(checkpoint_folder_path):
                    mlflow.log_artifacts(checkpoint_folder_path, f"model")

                # Calculate metrics and make predictions for each split
                all_predictions = {}
                all_metrics = {}
                
                self.current_vocab = vocab
                
                # Reinstaintiating model from the best checkpoint
                model_instance = MethylBert(
                    custom_config=methylbert_config,
                    foundation_model_path=self.foundation_model_huggingface,
                    load_weights=True,
                    num_labels=2,
                    num_dmr_labels=num_dmr_labels,
                    seq_len=self.max_sequence_length,
                    output_dir=output_dir,
                    batch_size=batch_size,
                    fine_tuned_model_path = checkpoint_path
                )

                for split in self.splits:
                    print(f"Calculating metrics for {split} split...")
                    metrics, predictions_df = self._make_predictions_and_calculate_metrics(
                        model_instance, split, chromosome,
                        train_dataset, 
                        valid_dataset,
                        dmrs
                    )
                    
                    all_metrics[split] = metrics
                    all_predictions[split] = predictions_df
                    
                    # Log metrics to MLflow
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"{split}_{metric_name}", metric_value)
                
                # Save predictions as artifacts
                predictions_dir = f"predictions_{chromosome}"
                os.makedirs(predictions_dir, exist_ok=True)
                
                for split, pred_df in all_predictions.items():
                    pred_file = os.path.join(predictions_dir, f"{split}_predictions.pkl")
                    with open(pred_file, "wb") as f:
                        pickle.dump(pred_df, f)
                
                mlflow.log_artifacts(predictions_dir, "predictions")
                
                # Save comprehensive results
                results = {
                    'chromosome': chromosome,
                    'data_stats': data_stats,
                    'metrics': all_metrics,
                    'model_params': {
                        'max_sequence_length': self.max_sequence_length,
                        'foundation_model': self.foundation_model_huggingface,
                        'num_dmr_labels': num_dmr_labels,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': lr,
                        'warmup_steps': warmup_steps,
                        'weight_decay': weight_decay,
                        'log_freq': log_freq,
                        'eval_freq': eval_freq
                    },
                    'checkpoint_path': checkpoint_path
                }
                
                # Save results as JSON artifact
                results_file = f"results_{chromosome}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                mlflow.log_artifact(results_file, "results")
                
                # Cleanup temporary files
                import shutil
                if os.path.exists(predictions_dir):
                    shutil.rmtree(predictions_dir)
                if os.path.exists(results_file):
                    os.remove(results_file)
                
                # Clean up model from memory
                del model_instance
                gc.collect()
                torch.cuda.empty_cache()
                
                print(f"✓ Completed training for chromosome {chromosome}")
                return results
                
            except Exception as e:
                print(f"✗ Error training chromosome {chromosome}: {e}")
                mlflow.log_param("error", str(e))
                raise e
    
    def run_full_experiment(self, **training_kwargs):
        """
        Run the complete experiment across all chromosomes.
        
        Args:
            **training_kwargs: Arguments to pass to train_chromosome method
        """
        print(f"Starting full experiment: {self.experiment_name}")
        print(f"Training {len(self.chromosome_dirs)} chromosome folders")
        print(f"Chromosome folders: {list(self.chromosome_dirs.keys())}")
        
        all_results = {}
        failed_chromosomes = []
        
        for chromosome in self.chromosome_dirs.keys():
            # try:

            results = self.train_chromosome(chromosome, **training_kwargs)
            all_results[chromosome] = results
            # except Exception as e:
            #     print(f"Failed to train chromosome {chromosome}: {e}")
            #     failed_chromosomes.append(chromosome)
            #     continue
        