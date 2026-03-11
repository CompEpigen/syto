# Refactoring Plan solving issue #66

## Assignment

### Task

- Compose a refactoring plan involving transferring code from edautils.py into main methyldl source code and optionally featuring other proposed improvements.
- Execute refactoring plan.
- Write additional tests.

### Acceptance criteria

- The refactoring plan is approved by Dmytro R.
- The refactored code is committed to the repo via pull request.
- All files in the methyldl source code having test coverage at least 85% (with all tests passing).
- All EDAs, Tutorials and APP must run without errors.

## Step 0

- check that the tests are runnable

```shell
# Should specify in the doc that the git LFS extension is necessary
sudo apt install git-lfs
git lfs install
git lfs pull

# then install the dependencies
poetry install --with dev

# then run the tests (with triton use conditional on GPU accepting it)
poetry run coverage run
```

## Step 1

- Apply the [Black](https://black.readthedocs.io/en/stable/) module on all the code (it correct inconsistent indentation and inconsistent use of quotes, making the code more readable)

## Step 2

The goal is to run all the EDAs and tutorials and APP to check that they run without errors, and to check the test coverage of the code.

- the tests run correctly : 67.27% coverage with 1 skipped test (because of Triton not being supported on my GPU)
- Tutorials:
  - only necessary to pass fine_tuning_* and training_with_dismir. But anyway Dmytro told me that if the test (as of now) pass, then the aforementioned tutorials should pass too, so I will not losse time checking the tutorials for now.
- Trying to run the APP, it seems like the containers are needed
  - trying to run the containers

## Step 3 (in parallel with step 2)

Mapping functions in edautils.py to the methyldl locations:

- **BAM/Read processing** → `methyldl/data/` (new `bam.py` or `reads.py`)
  - `process_bam_with_chunking` — Top-level BAM processing pipeline: reads a BAM file in parallel genomic chunks and returns a DataFrame of read-level methylation data
  - `process_single_read` — Extracts methylation data (CpG positions, states, encoding) from a single pysam read (ONT or WGBS)
  - `process_tabular_chunk` — Worker function that processes all reads in one genomic chunk of a BAM file
  - `detect_data_type` — Auto-detects whether a BAM file is ONT (has ML tags) or WGBS by sampling reads
  - `clean_cigar_sequence` — Processes CIGAR string to align read sequence to reference coordinates (handles insertions, deletions, soft clips)
  - `parse_mm_tag` — Parses BAM MM tag to extract modification types (e.g., C+m) and their positions for ML array indexing
  - `extract_cpg_ml_values` — Extracts only the ML probability values corresponding to C+m (CpG methylation) from the full ML array
  - `cpg_scan` — Numba-accelerated scanner that finds CpG positions and their methylation states in an ONT read
  - `cpg_scan_wgbs_forward` — Numba-accelerated CpG scanner for WGBS forward-strand reads (C→T = unmethylated)
  - `cpg_scan_wgbs_reverse` — Numba-accelerated CpG scanner for WGBS reverse-strand reads (G→A = unmethylated)
  - `merge_paired_reads` — Merges paired-end read mates into single fragment entries, handling overlapping CpG consensus
  - `_merge_mate_pair` — Merges two mates of a paired-end read into a single fragment dict with combined methylation
  - `_merge_methylation_encodings` — Merges methylation encodings from two mates with consensus logic (agree/disagree/unknown)
  - `_merge_comma_separated` — Utility to merge two comma-separated strings keeping unique values
  - `count_reference_cpgs` — Counts CpG dinucleotides in a reference genome region (used for unknown CpG calculation)
  - `add_reference_cpg_counts` — Adds reference-based CpG counts to a DataFrame, computing unknown CpGs in insert regions
  - `chunk_read_data` — Splits a read into fixed-size chunks, clips to DMR boundaries, and returns one entry per overlapping DMR
  - `analyze_read_dmr_overlap` — Computes overlap metrics (bp, percentage) between a read segment and DMR interval trees
  - `_empty_overlap_result` — Returns a default empty-overlap dict when a read doesn't overlap any DMR
  - `_get_overlapping_dmrs` — Returns a list of individual overlapping DMR dicts (name, type, overlap) for iteration
- **Prediction aggregation** → `methyldl/modelling/` or new `methyldl/inference/`
  - `aggregate_predictions_by_dmr` — Aggregates read-level prediction probabilities to DMR level using simple and weighted averages
  - `aggregate_predictions_by_dmr_optimized` — Faster vectorized version of DMR-level prediction aggregation (used in pseudo-bulk generation)
  - `get_final_prediction` — Computes final class prediction (argmax) and confidence from aggregated probability columns
  - `aggregate_chuncked_predictions_weighted` — Aggregates chunked read predictions back to whole-read level using weighted averaging
- **Deconvolution models & training** → `methyldl/deconvolution/`
  - `FlattenMLPDeconvolver` (class) — Simple MLP that flattens the DMR×class prediction matrix and predicts cell type proportions
  - `DMRAttentionDeconvolver` (class) — Attention-based deconvolver that uses multi-head attention over DMR embeddings
  - `DiagonalAwareDeconvolver` (class) — Multi-pathway deconvolver with separate encoders for diagonal, off-diagonal confusion, and rejection features
  - `CNNDeconvolver` (class) — Treats the DMR×class matrix as a 2D image and applies CNN convolutions for deconvolution
  - `DeconvolverOutput` (class) — Dataclass holding deconvolver outputs: proportions, per-pathway features, logits
  - `TrainingHistory` (class) — Dataclass storing per-epoch training metrics (loss, MAE, MSE, KL, cosine sim, etc.)
  - `EarlyStopping` (class) — Early stopping handler with configurable patience, min_delta, and min/max mode
  - `train_matrix_deconvolver` — Full training loop for matrix-based deconvolvers: handles batching, optimization, validation, and early stopping
  - `compute_deconvolution_metrics` — Computes MAE, MSE, KL divergence, max error, and cosine similarity between predicted and true proportions
- **Pseudo-bulk generation** → `methyldl/deconvolution/` or `methyldl/data/`
  - `generate_pseudo_bulk` — Creates synthetic bulk samples by mixing reads from different cell types at specified proportions, with UXM deconvolution
  - `generate_pseudo_bulk_optimized` — Faster version using pre-computed grouped DataFrames (used in parallel workers)
  - `run_ios_generation_parallel` — Orchestrates parallel generation of many pseudo-bulk input/output examples for deconvolver training
  - `init_worker` — Initializes multiprocessing worker with shared data and pre-grouped DataFrames
  - `worker_task` — Worker function that generates a batch of pseudo-bulk samples with random cell type mixtures
  - `random_select_with_weights` (defined twice, lines 586 and 906) — Randomly selects up to n elements and generates normalized random weights summing to 1
- **Inference prep** → `methyldl/data/` or `methyldl/modelling/`
  - `chunk_tokens` — Splits a token sequence into overlapping windows of fixed size with a given stride
  - `generate_valid_tokens` — Generates (kmer, methylation_char) token pairs from a read, filtering out N-containing kmers
  - `prepare_methylbert_list_inference` — Prepares sliding-window chunked inference data in the format expected by MethylBERT
- **Visualization** → stays in `EDA/` or new `methyldl/visualization/`
  - `plot_filled_kde` — Plots overlapping filled KDE curves for two groups with Cohen's d annotation
  - `create_methylation_boxplots` — Creates grouped boxplots of methylation levels by prediction outcome (correct/incorrect/rejected)
  - `create_methylation_violinplots` — Creates violin plots of methylation level distribution by prediction outcome
  - `plot_predictions_by_celltype` — Plots predicted vs true proportions as scatter/bar per cell type across samples
  - `plot_predictions_by_mixture_complexity` — Plots deconvolution accuracy stratified by number of cell types in the mixture
  - `plot_deconvolution_results` — Multi-panel summary plot of deconvolution performance (scatter, error bars, heatmap)
  - `bland_altman_plot` — Creates a Bland-Altman agreement plot with limits of agreement, regression, and confidence intervals
  - `bland_altman_grid` — Creates a grid of per-cell-type or per-complexity Bland-Altman subplots with shared axes
  - `bland_altman_comparison` — Compares Bland-Altman statistics across groups (complexity levels or cell type groups) as bar charts
  - `plot_methylation_with_predictions` — Plots methylation encoding along a read with prediction label overlay
  - `plot_methylation_with_predictions_grouped` — Grouped version of methylation-with-predictions plot for multiple reads
  - `plot_methylation_vs_prediction_scatter` — Scatter plot of methylation level vs prediction confidence
  - `plot_confusion_matrix_for_target_confidence` — Plots a confusion matrix filtered to a specific confidence threshold
  - `DeconvolverVisualizer` (class) — Comprehensive visualization toolkit for DiagonalAwareDeconvolver interpretability (ablation, feature space, pathway contributions)
- **Statistics / metrics** → `methyldl/deconvolution/evaluation.py` or `methyldl/modelling/evaluation.py`
  - `cohen_d` — Computes Cohen's d effect size between two distributions using pooled standard deviation
  - `print_deconvolution_metrics_summary` — Pretty-prints a dictionary of deconvolution metrics (MAE, cosine sim, etc.)
  - `print_bland_altman_stats` — Pretty-prints formatted Bland-Altman statistics (bias, LoA, proportional bias check)
- **Constants** → `methyldl/data/` (constants file or `__init__.py`)
  - `cell_type_match_dict` — Maps ~80 fine-grained cell type names to ~39 canonical cell type labels used by the models
