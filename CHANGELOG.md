# Changelog

All notable changes to syto are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2026-10-13

### Added

- Classifier fitting on de-duplicated reads. `deduplicate_train` collapses
  identical reads of each training part into one row carrying a `read_count`;
  Dismir turns the counts into sampling weights (`count_weighting: sampler`,
  with `samples_per_epoch`) or loss weights (`count_weighting: loss`).
  `deduplicate_valid` does the same for the validation split, with
  count-weighted loss and metrics that equal the read-by-read values. The
  de-duplication key (`dedup_keys`) includes every scalar classifier column,
  such as a GRG label, so reads seen differently are never merged. MethylBERT
  and EpigenBERT2 support the same options (MethylBERT weighs each chunk by its
  read's count; with `use_balanced_trainer`, sampler mode draws on-target and
  background rows in proportion to their counts).
- Pseudobulk generation on de-duplicated reads (`deduplicate_reads`): raw
  splits are loaded de-duplicated, reads are sampled in proportion to their
  counts, and each unique read is predicted once. `prediction_block_size`
  bounds how many reads the classifier encodes per call.
- `valid_subsample` evaluates classifier fitting on a seeded random subset of
  the validation reads.
- `best_metric` and `best_mode` select the epoch reported as best by the
  classifier fit (default `val_loss`, minimised).
- Dismir options: `dropout_prob` and `penalty_scale` (defaults match the
  previous fixed values), `monitor_metric` and `monitor_mode` (the metric that
  drives early stopping and the saved checkpoint), and `compact_inputs`
  (`uint8` codes instead of one-hot tensors).
- `UXMMethylationAtlas.with_region_groups` splits each target's regions into
  groups of similar atlas profile (e.g. with
  `BalancedRegionClusterer(group_size=25)`) and writes them as an atlas
  column, optionally with integer codes matching `build_dataset`'s
  `region_annotations` encoding. Groups are named `<target>__k`, where `__0`
  holds the regions most specific to the target.
- Inference carries extra atlas columns onto each read
  (`region_annotation_columns`, e.g. a GR-group label the classifier head keys
  on) and passes `num_grg_labels` to the classifier.

### Changed

- Classifier fitting seeds Python, NumPy and torch from `seed` (default 42), so
  repeated fits of one config give the same model.
- `BalancedRegionClusterer.fit_predict` takes a feature matrix and returns
  cluster ids; the per-target atlas logic it used to contain moved to
  `UXMMethylationAtlas.with_region_groups`.
- Lower memory use on large datasets such as U250: reduced parquet parts no
  longer hold references to whole-part buffers, HDF5 consolidation writes in row
  chunks instead of copying whole splits, and block-wise prediction fills
  preallocated columns.

### Fixed

- Resuming a pseudobulk run no longer recreates existing split checkpoints,
  which made it forget completed batches and consolidate none of them.
  Consolidation now refuses to run while a split is incomplete, so its batch
  files are kept.
- Region groups are ranked on imputed atlas values. Previously a single missing
  value in a target's own column sent its group last, so group numbers did not
  follow specificity for most U250 targets. Group membership is unchanged.

## [1.2.0] - 2026-10-05

### Added

- `UXMMethylationAtlas.from_block_table` builds atlas regions from one or more
  block tables: `wgbstools find_markers` output (`#chr` and `region` are read
  as `chr` and `name`) or post-processed tables whose target and direction
  live in custom columns (`target_column`, `direction_column`; `hypo`/`hyper`
  are accepted for `U`/`M`). Extra columns such as cluster ids or scores are
  kept in the atlas.
- `region_annotations` option for the `build_dataset` finalize phase: maps
  atlas columns to dataset columns (e.g. `cluster: cluster_label`), encoded as
  integers starting at 0 and written onto every read. The code-to-value
  mapping is saved as `region_annotations.json` in the output directory.
- The pseudobulk pipeline accepts `atlas_cell_types`, for atlases whose groups
  are not the 39 Loyfer cell types, and passes `classifier_config.num_grg_labels`
  to the classifier, so checkpoints trained with a different number of GRG
  groups load.
- BAM parsing resolves contig names whichever way the `chr` prefix is spelled,
  so hs37d5-named BAMs (`1`) work against `chrN` atlases and hg19-named
  reference FASTAs in inference as well as in dataset building.

### Changed

- `build_dataset` stages a BAM one region bucket at a time instead of in a
  single pass, so only one bucket's reads are held in memory. As a result, a
  BAM sample's `n_reads` counts staged rows rather than every parsed read, and
  `n_in` counts a read once per bucket that fetches it.
- `UXMMethylationAtlas.from_reads` keeps the extra region columns of `markers`
  (e.g. cluster ids), so they stay available for filtering and read
  annotation.

### Deprecated

- `UXMMethylationAtlas.from_wgbstools_markers` is now an alias of
  `from_block_table` and warns on use. Its cell-type columns are left empty
  instead of being estimated from find_markers' mean betas; build the atlas
  with `from_block_table`, then fill the columns from reads with `from_reads`.

### Fixed

- WGBS reads were silently dropped when the reference FASTA and the BAM
  spelled contig names differently (e.g. hs37d5 BAM, hg19 FASTA): every
  reference lookup failed and each read was discarded.
- Inference on a BAM that yields no reads now stops with an error naming the
  likely causes (unsorted or unindexed BAM, contigs not overlapping the atlas,
  wrong `input.reference_path`) instead of failing further downstream.
- MethylBERT training with `grg_label_column` pointing at a column other than
  `dmr_ctype_label` now also loads the `grg_ctype_label` column from columnar
  datasets.


## [1.1.0] - 2026-09-15

### Added

- `build_dataset` can stage reads directly from BAM files (`input_type: bam`).
  Each BAM in `input_dir` is labelled with its group from `sample_groups_path`
  (CSV with `name` and `group` columns); BAMs not listed there are skipped.
  Parsing is configured under `bam_processing` and is restricted to the atlas
  regions.
- `atlas_cell_types` option for `build_dataset`, so atlases cell-types can be 
  overwritten at the runtime - useful for bringing custom cohorts for 
  training.
- `UXMMethylationAtlas.from_wgbstools_markers` builds an atlas from the
  `Markers.*.bed` files written by `wgbstools find_markers`.
- Chromosome names are harmonised between BAMs and the atlas, so BAMs aligned to
  references without the `chr` prefix (e.g. hs37d5) work with `chrN` atlases.
- `CHANGELOG.md` documenting releases from 0.5.0 onward.

### Changed

- BAM parsing helpers shared by inference and dataset building now live in
  `syto.data.sequencing.bam_processing`.

### Fixed

- WGBS methylation calls for read 2 of paired-end reads were
  taken from the wrong strand, which made read 2 appear almost fully
  methylated. While this doesn't affect published experiment, where all
  reads were recovered from the .pat with pairs information already
  accounted for, it would have influenced downstream use. Therefore, 
  the users willing to deconvolve their .bam files must use version 1.1.0
  and higher.

### Removed

- `tests/data/test_pseudobulk_conversion.py`, which depended on a script that is
  not part of the repository.

## [1.0.0] - 2026-09-09

First public release under the name syto (previously methyldl).

### Added

- `syto` command-line entry point, replacing the old `App/` scripts.
- Interactive configuration wizard, including a lazy mode, with schemas for
  pseudobulk generation and `fit_deconvolution`.
- `build_dataset` pipeline that turns recovered reads into atlas-overlapped
  Parquet datasets in two phases (`stage`, `finalize`), with `split_remap`, read
  filters, and a separate `staged_source_dir` for finalizing existing data.
- Config-driven labelers for dataset finalization: data-driven soft labels (with
  optional pooling and superset counts), archetype soft labels (
    in development and not published), and hard labels
  with background and smoothing.
- EpiDISH deconvolution baseline (RPC, CBS, CP) integrated into inference, the
  pseudobulk pipeline, and the wizard.
- Abstract base classes for atlases, signatures, distances, and labelers.
- Building a UXM atlas from labelled reads.
- Columnar (Parquet) pseudobulk store and Parquet input at inference, alongside
  HDF5.
- Balanced training (`BalancedTrainer`) and early stopping for MethylBERT and
  EpigenBERT.
- Classifier fitting on columnar data.
- Inference without labels, BAM filtering and chunked processing at inference,
  and non-overlapping chunking of reads longer than the model input length.
- Configurable lower and upper methylation thresholds for ONT data.
- Deconvolution report at inference (opt-in): CLI summary and PDF report.
- Calibration of baseline deconvolvers.
- CPU-only support.
- Classifier calibration metrics and a hyperparameter for
  `ConfidenceWeightedCrossEntropy`.
- Quick-start script, license, and README.

### Changed

- Renamed the package from `methyldl` to `syto`; the `modelling` module is now
  `classification`.
- Renamed `dmr_label_column` to `grg_label_column`.
- Reworked pseudobulk generation and propagated the new pseudobulk format.
- Replaced the classifier adapter with an abstract base class and harmonised the
  classifier API.
- Decoupled baselines from the core pipeline and refactored the experiment
  wrapper.
- Harmonised metric logging for CancerDetector and the lookup classifier.
- MLflow reporting logs train metrics and best-checkpoint metrics, and stores the
  config and log as artifacts.
- Optimized CelFiE EM; new `freeze_gamma` and `sum_by_region` options.

### Fixed

- MethylBERT no longer crashes at inference when the original label column is
  missing.
- Files with inconsistent target names are no longer skipped when a match
  dictionary is provided.
- Shape of the feature matrices when the number of prediction classes differs
  from the number of labels to deconvolve.
- Loading of calibrators and deconvolvers.

### Removed

- `dmr_overlap_analysis` and `split_rebalancing`.
- R code and exploratory data analysis, which moved to a separate repository.
- Obsolete tutorials and local job scripts.

## [0.5.0] - 2026-05-18

First tagged release (as methyldl).

### Added

- Read-level classifiers: core model, DISMIR, MethylBERT, mini-RNNs, and a lookup
  classifier with hard labels.
- CancerDetector baseline.
- Soft labels for training.
- Deconvolvers: NNLS, SVM, MLE, MLP, and SWN, with a harmonised API.
- Calibrators (including Dirichlet calibration) in a separate module.
- Independent cross-validation engine and a combined deconvolution loss.
- Bootstrap confidence intervals.
- End-to-end inference integration.

[Unreleased]: https://github.com/CompEpigen/syto/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/CompEpigen/syto/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/CompEpigen/syto/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/CompEpigen/syto/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/CompEpigen/syto/compare/v0.5.0...v1.0.0
[0.5.0]: https://github.com/CompEpigen/syto/releases/tag/v0.5.0
