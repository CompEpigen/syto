# MethylDL Application

This directory contains the main application entry point and configuration for training methylation classification models.

## Directory Structure

```
App/
├── main.py                 # Main application entry point
├── config/                 # Configuration files
│   ├── fine_tune_epigenbert2.yaml
│   ├── fine_tune_dismir.yaml
│   └── fine_tune_methylbert.yaml
├── scripts/                # Helper scripts
│   ├── run_fine_tuning.sh
│   ├── setup_mlflow.sh
│   ├── validate_data.sh
│   ├── check_gpu.py
│   └── compare_experiments.py
└── README.md              # This file
```

## Supported Models

1. **Dismir** - Lightweight RNN-based model
   - Flavors: miniGRU, LSTM
   - Fast training and lower memory requirements for shorter sequences

2. **MethylBERT** - Specialized methylation BERT
   - Pre-trained 
   - Maximum sequence length: 510bp

3. **EpigenBERT2** - DNABERT-2 based transformer
   - Supports CpG and m6A methylation
   - High accuracy, requires GPU

## Quick Start

### 1. Setup Environment

```bash
# Check GPU availability
python scripts/check_gpu.py

# Validate data structure
./scripts/validate_data.sh /path/to/data

# Setup MLflow (optional)
./scripts/setup_mlflow.sh
```

### 2. Fine-tuning

#### Using Configuration Files

```bash
# Fine-tune EpigenBERT2
python main.py --task fine_tune --config config/fine_tune_epigenbert2.yaml

# Fine-tune Dismir
python main.py --task fine_tune --config config/fine_tune_dismir.yaml

# Fine-tune MethylBERT
python main.py --task fine_tune --config config/fine_tune_methylbert.yaml
```

#### With Command-line Overrides

```bash
# Override specific parameters
python main.py --task fine_tune \
    --config config/fine_tune_epigenbert2.yaml \
    --max-seq-length 2000 \
    --data-path /custom/data \
    --mlflow \
    --experiment-name "custom_experiment"

# Train specific datasets only
python main.py --task fine_tune \
    --config config/fine_tune_dismir.yaml \
    --datasets chr1 chr2 chr3

# Disable MLflow tracking
python main.py --task fine_tune \
    --config config/fine_tune_methylbert.yaml \
    --no-mlflow

# Dry run to validate configuration
python main.py --task fine_tune \
    --config config/fine_tune_epigenbert2.yaml \
    --dry-run
```

### 3. Monitor Training

#### With MLflow UI

```bash
# If MLflow is enabled, view results at:
mlflow ui --port 5000

# Or use the setup script:
./scripts/setup_mlflow.sh
```

#### Compare Experiments

```bash
# Compare different model experiments
python scripts/compare_experiments.py \
    "epigenbert2_methylation_classification" \
    "dismir_methylation_classification" \
    "methylbert_methylation_classification" \
    --metric test_f1_score
```

## Configuration Guide

### Basic Configuration Structure

```yaml
# Model configuration
model:
  architecture: <model_name>  # epigenbert2, dismir, methylbert
  # Model-specific parameters...

# Data configuration  
data_path: "/path/to/data"
max_sequence_length: 1000
datasets: all  # or list specific datasets

# MLflow configuration
mlflow:
  enabled: true
  experiment_name: "experiment_name"
  tracking_uri: "http://localhost:5000"  # optional

# Training configuration
training:
  # Training hyperparameters...
```

### Model-Specific Parameters

#### EpigenBERT2
- `foundation_model`: Hugging Face model path
- `use_cpg_methylation`: Enable CpG methylation (default: true)
- `use_m6a_methylation`: Enable m6A methylation (default: false)
- `use_triton`: Enable Triton optimization (default: false)

#### Dismir
- `flavor`: Model variant ("minigru" or "lstm")
- `optimizer_type`: "SGD" or "Adam"

#### MethylBERT
- `foundation_model`: Pre-trained model path
- Note: max_sequence_length ≤ 510

## Data Requirements

Your data directory should follow this structure:

```
data_path/
├── dataset1/
│   ├── train.parquet
│   ├── valid.parquet
│   ├── test.parquet
│   └── rest.parquet  # optional
├── dataset2/
│   ├── train.parquet
│   ├── valid.parquet
│   └── test.parquet
└── ...
```

Each parquet file should contain:
- `input_ids`: DNA sequences
- `methylation_ids`: Methylation states
- `label`: Binary classification labels
- `dmr_name` (optional): DMR identifiers for MethylBERT

## Container Usage

When running in the Singularity container:

```bash
# Inside container
cd /workspace/methyldl/App
python main.py --task fine_tune --config config/fine_tune_epigenbert2.yaml

# From host with container
singularity exec --nv methyldl.sif \
    python /workspace/methyldl/App/main.py \
    --task fine_tune \
    --config /workspace/methyldl/App/config/fine_tune_epigenbert2.yaml
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `per_device_train_batch_size`
   - Reduce `max_sequence_length`
   - Enable gradient checkpointing (EpigenBERT2)

2. **Slow Training**
   - Ensure GPU is available: `python scripts/check_gpu.py`
   - Check data loading: reduce `num_workers` if needed
   - Use mixed precision training (fp16=true)

3. **Poor Performance**
   - Adjust learning rate
   - Increase warmup steps
   - Check data quality with EDA scripts

4. **MLflow Connection Issues**
   - Ensure MLflow server is running
   - Check tracking_uri in configuration
   - Verify network connectivity

### Logging

Training logs are saved to:
- Console output (use `--verbose` for detailed logs)
- `methyldl_training.log` file
- MLflow tracking (if enabled)

## Advanced Usage

### Custom Training Arguments

For fine-grained control, use custom TrainingArguments:

```yaml
training:
  training_arguments:
    output_dir: "./custom_output"
    num_train_epochs: 100
    per_device_train_batch_size: 32
    gradient_accumulation_steps: 2
    learning_rate: 5.0e-5
    warmup_ratio: 0.1
    logging_strategy: "steps"
    logging_steps: 10
    evaluation_strategy: "epoch"
    save_strategy: "epoch"
    save_total_limit: 3
    load_best_model_at_end: true
    metric_for_best_model: "eval_loss"
    greater_is_better: false
    fp16: true
    dataloader_num_workers: 4
    seed: 42
```

### Multi-GPU Training

For distributed training across multiple GPUs:

```bash
# Using PyTorch distributed
torchrun --nproc_per_node=4 main.py \
    --task fine_tune \
    --config config/fine_tune_epigenbert2.yaml

# Or with Accelerate
accelerate launch main.py \
    --task fine_tune \
    --config config/fine_tune_epigenbert2.yaml
```

## Integration with HPC

For SLURM-based HPC systems, see `../deployments/hpc/slurm/` for job submission scripts.

Example SLURM submission:

```bash
sbatch ../deployments/hpc/slurm/single_gpu.sbatch
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `methyldl_training.log`
3. Check MLflow UI for training metrics
4. Consult the main project README