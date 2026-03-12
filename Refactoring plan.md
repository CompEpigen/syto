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

- check that the tests and the App are runnable : OK

```shell
poetry run coverage run
poetry run python main.py --task fine_tune --config ./config/dnabert2base.yaml --data-path ../Data
```

## Step 1 (?)

- Apply the [Black](https://black.readthedocs.io/en/stable/) module on all the code (it corrects inconsistent indentation and inconsistent use of quotes, making the code more readable)

## Step 2 : actual refactoring plan

### Functions moved from edautils.py

Mapping functions in edautils.py to their new locations in the methyldl source code.
They are grouped by their main functionality and where they sould go.

- **Constants** → `methyldl/data/__init__.py`
  - `cell_type_match_dict` — Maps ~80 fine-grained cell type names to ~39 canonical cell type labels used by the models
    - Renamed into: "CELL_TYPE_MATCH_DICT"
- **Miscellaneous utilities** → `methyldl/data/misc_utils.py`
  - `_merge_comma_separated` — Utility to merge two comma-separated strings keeping unique values
- **BAM processing** → `methyldl/data/sequencing/bam_processing.py`
  - `clean_cigar_sequence` — Processes CIGAR string to align read sequence to reference coordinates (handles insertions, deletions, soft clips)
  - `parse_mm_tag` — Parses BAM MM tag to extract modification types (e.g., C+m) and their positions for ML array indexing
  - `extract_cpg_ml_values` — Extracts only the ML probability values corresponding to C+m (CpG methylation) from the full ML array
  - `process_bam_with_chunking`  — Top-level BAM processing pipeline: reads a BAM file in parallel genomic chunks and returns a DataFrame of read-level methylation data
  - `process_tabular_chunk` — Worker function that processes all reads in one genomic chunk of a BAM file
  - `process_single_read` — Extracts methylation data (CpG positions, states, encoding) from a single pysam read (ONT or WGBS)
  - `detect_data_type` — Auto-detects whether a BAM file is ONT (has ML tags) or WGBS by sampling reads.
    - Renamed into: "detect_bam_data_type"
  - `cpg_scan` — Numba-accelerated scanner that finds CpG positions and their methylation states in an ONT read
  - `cpg_scan_wgbs_forward` — Numba-accelerated CpG scanner for WGBS forward-strand reads (C→T = unmethylated)
  - `cpg_scan_wgbs_reverse` — Numba-accelerated CpG scanner for WGBS reverse-strand reads (G→A = unmethylated)
  - `merge_paired_reads` — Merges paired-end read mates into single fragment entries, handling overlapping CpG consensus
  - `_merge_mate_pair` — Merges two mates of a paired-end read into a single fragment dict with combined methylation
  - `_merge_methylation_encodings` — Merges methylation encodings from two mates with consensus logic (agree/disagree/unknown)
- **DMR overlap analysis** → `methyldl/data/sequencing/dmr_overlap_analysis.py`
  - `_get_overlapping_dmrs` — Returns a list of individual overlapping DMR dicts (name, type, overlap) for iteration
  - `analyze_read_dmr_overlap` — Computes overlap metrics (bp, percentage) between a read segment and DMR interval trees
  - `_empty_overlap_result` — Returns a default empty-overlap dict when a read doesn't overlap any DMR
- **Genome analysis utilities** → `methyldl/data/sequencing/genome.py`
  - `count_reference_cpgs` — Counts CpG dinucleotides in a reference genome region (used for unknown CpG calculation)
  - `add_reference_cpg_counts` — Adds reference-based CpG counts to a DataFrame, computing unknown CpGs in insert regions
- **Pseudo-bulk generation** → `methyldl/data/pseudo_bulk_generation.py`
  - `generate_pseudo_bulk` — Creates synthetic bulk samples by mixing reads from different cell types at specified proportions, with UXM deconvolution
  - `generate_pseudo_bulk_optimized` — Faster version using pre-computed grouped DataFrames (used in parallel workers)
  - `run_ios_generation_parallel` — Orchestrates parallel generation of many pseudo-bulk input/output examples for deconvolver training
  - `init_worker` — Initializes multiprocessing worker with shared data and pre-grouped DataFrames
  - `worker_task` — Worker function that generates a batch of pseudo-bulk samples with random cell type mixtures
  - `random_select_with_weights` — Randomly selects up to n elements and generates normalized random weights summing to 1
- **Data preprocessing for inference** → `methyldl/modelling/data_inference_preprocessing.py`
  - `chunk_read_data` — Splits a read into fixed-size chunks, clips to DMR boundaries, and returns one entry per overlapping DMR
  - `chunk_tokens` — Splits a token sequence into overlapping windows of fixed size with a given stride
  - `generate_valid_tokens` — Generates (kmer, methylation_char) token pairs from a read, filtering out N-containing kmers
  - `prepare_methylbert_list_inference` — Prepares sliding-window chunked inference data in the format expected by MethylBERT
- **Deconvolution models** → `methyldl/deconvolution/deep_deconvolvers/`
  - `FlattenMLPDeconvolver` (class) — Simple MLP that flattens the DMR×class prediction matrix and predicts cell type proportions
    - Moved to `methyldl/deconvolution/deep_deconvolvers/flatten_mlp_deconvolver.py`
  - `DMRAttentionDeconvolver` (class) — Attention-based deconvolver that uses multi-head attention over DMR embeddings
    - Moved to `methyldl/deconvolution/deep_deconvolvers/dmr_attention_deconvolver.py`
  - `DiagonalAwareDeconvolver` (class) — Multi-pathway deconvolver with separate encoders for diagonal, off-diagonal confusion, and rejection features
    - Moved to `methyldl/deconvolution/deep_deconvolvers/diagonal_aware_deconvolver.py`
  - `CNNDeconvolver` (class) — Treats the DMR×class matrix as a 2D image and applies CNN convolutions for deconvolution
    - Moved to `methyldl/deconvolution/deep_deconvolvers/cnn_deconvolver.py`
- **Deconvolution models training** → `methyldl/deconvolution/deep_deconvolvers_training.py`
  - `DeconvolverOutput` (class) — Dataclass holding deconvolver outputs: proportions, per-pathway features, logits
  - `train_matrix_deconvolver` — Full training loop for matrix-based deconvolvers: handles batching, optimization, validation, and early stopping
  - `TrainingHistory` (class) — Dataclass storing per-epoch training metrics (loss, MAE, MSE, KL, cosine sim, etc.)
  - `EarlyStopping` (class) — Early stopping handler with configurable patience, min_delta, and min/max mode
- **Deconvolution models evaluation** → `methyldl/deconvolution/evaluation.py`
  - `compute_deconvolution_metrics` — Computes MAE, MSE, KL divergence, max error, and cosine similarity between predicted and true proportions
  - `print_deconvolution_metrics_summary` — Pretty-prints a dictionary of deconvolution metrics (MAE, cosine sim, etc.)
- **Prediction aggregation** → `methyldl/modelling/prediction_aggregation.py`
  - `aggregate_predictions_by_dmr` — Aggregates read-level prediction probabilities to DMR level using simple and weighted averages
  - `aggregate_predictions_by_dmr_optimized` — Faster vectorized version of DMR-level prediction aggregation (used in pseudo-bulk generation)
  - `get_final_prediction` — Computes final class prediction (argmax) and confidence from aggregated probability columns

### Modules moved from modelling into a subfolder `classifiers`

- **Actual classifier models** → `methyldl/modelling/classifiers/`
  - minirnns (folder)
  - `dismir.py`
  - `methylbert.py`
  - `dnabert2.py`
